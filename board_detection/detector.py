import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

BOARD_PX = 800
SQUARE_PX = BOARD_PX // 8  # 100 px


@dataclass
class DetectionResult:
    """Result of chessboard detection."""
    corners: np.ndarray    # (4, 2) float32: TL, TR, BR, BL in original image coords
    board_image: np.ndarray  # BOARD_PX × BOARD_PX BGR top-down view
    squares: list            # 64 BGR images, row-major from top to bottom


def detect_board(image: np.ndarray, debug: bool = False) -> DetectionResult | None:
    """
    Detect chessboard and extract 64 squares.

    Pipeline:
      1. Canny → estimate board rotation (Sobel histogram) → axis-align board
      2. Localise the board region with a cheap HoughLinesP pass (extreme
         intersections) so projection happens on the board area only
      3. **Edge projection** onto each axis inside that region: row-sum →
         horizontal grid lines, col-sum → vertical grid lines.  This catches
         lines even when chess pieces occlude parts of them, because grid
         lines span the full width/height of the board while pieces produce
         only short, localised edge clusters
      4. Pick the 9 strongest peaks per axis (or fewer if not enough found)
      5. Index the detected lines centred on the board middle (row/col 4) –
         so a partial 7-line detection is assumed to be rows 1–7, not 0–6
      6. cv2.findHomography(RANSAC) over the full intersection grid
      7. Invert → board corners; warp; cut into 64 squares
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    angle = _dominant_grid_angle(gray, edges)
    h, w = image.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot_edges = cv2.warpAffine(edges, M_rot, (w, h), flags=cv2.INTER_NEAREST)

    # Stage 1: locate the rough board region using HoughLinesP
    board_roi = _locate_board_roi(rot_edges, (h, w))
    if board_roi is None:
        return None
    x0, y0, x1, y1 = board_roi

    # Stage 2: detect grid lines by edge projection within the ROI
    v_xs, h_ys = _grid_lines_by_projection(rot_edges, x0, y0, x1, y1)
    if len(v_xs) < 5 or len(h_ys) < 5:
        return None

    if debug:
        _show_debug_lines(rot_edges, v_xs, h_ys)

    # Stage 3: build correspondences (intersection point ↔ board grid index)
    # Center-based indexing: the detected lines straddle the board midpoint
    src_pts, dst_pts = _build_correspondences(v_xs, h_ys, (h, w))
    if len(src_pts) < 9:
        return None

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                               ransacReprojThreshold=8)
    if H is None:
        return None

    # Recover board corners in rotated frame, then undo the rotation
    board_box = np.array([[0, 0], [BOARD_PX, 0],
                           [BOARD_PX, BOARD_PX], [0, BOARD_PX]], dtype=np.float32)
    try:
        corners_rot = cv2.perspectiveTransform(
            board_box.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(corners_rot)):
        return None

    M_inv = cv2.invertAffineTransform(M_rot)
    corners = cv2.transform(corners_rot.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
    corners = _order_corners(corners)

    if not _is_valid_quad(corners, (h, w)):
        return None

    board_image = _warp(image, corners)
    return DetectionResult(corners=corners, board_image=board_image,
                           squares=_extract_squares(board_image))


# ── Rough ROI from HoughLinesP intersections ─────────────────────────────────

def _locate_board_roi(rot_edges: np.ndarray, shape: tuple) -> tuple | None:
    """Return (x0, y0, x1, y1) bounding box of the board area, padded."""
    h, w = shape
    lines = cv2.HoughLinesP(rot_edges, 1, np.pi / 180,
                             threshold=100, minLineLength=50, maxLineGap=15)
    if lines is None:
        return None

    h_lines = sorted(_filter_lines(lines, 'horizontal'),
                     key=_line_length, reverse=True)[:40]
    v_lines = sorted(_filter_lines(lines, 'vertical'),
                     key=_line_length, reverse=True)[:40]
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    pts = _cluster(np.array(_intersections(h_lines, v_lines, (h, w))), eps=20)
    if len(pts) < 6:
        return None

    # Pad generously (≈ 2 grid steps) – the Hough extreme corners are often
    # 1 step inside the true board, so we need room to find the missed
    # outermost grid line during the projection pass.
    pad_x = max(150, int((pts[:, 0].max() - pts[:, 0].min()) * 0.30))
    pad_y = max(150, int((pts[:, 1].max() - pts[:, 1].min()) * 0.30))
    x0 = max(0, int(pts[:, 0].min()) - pad_x)
    x1 = min(w, int(pts[:, 0].max()) + pad_x)
    y0 = max(0, int(pts[:, 1].min()) - pad_y)
    y1 = min(h, int(pts[:, 1].max()) + pad_y)
    return x0, y0, x1, y1


# ── Edge projection → grid lines ─────────────────────────────────────────────

def _grid_lines_by_projection(rot_edges: np.ndarray, x0: int, y0: int,
                               x1: int, y1: int, n_lines: int = 9):
    """
    Sum edge intensities along each row and column inside the board ROI.
    Peaks in the row-sum are horizontal grid lines; peaks in the col-sum are
    vertical grid lines.  Return the strongest peaks (up to n_lines per axis).

    When fewer than n_lines peaks survive, fill in the missing positions by
    extrapolating from the median step – preserves the full 9×9 grid even
    when one or two outer grid lines have weak edge support.
    """
    roi = rot_edges[y0:y1, x0:x1].astype(np.float32)

    row_sum = gaussian_filter1d(roi.sum(axis=1), sigma=3)
    col_sum = gaussian_filter1d(roi.sum(axis=0), sigma=3)

    # Expected step ≈ ROI_size / 8 ; require peaks to be at least 60 % of that apart
    min_step_y = max(15, (y1 - y0) // 12)
    min_step_x = max(15, (x1 - x0) // 12)

    h_peaks, _ = find_peaks(row_sum, distance=min_step_y,
                             prominence=row_sum.max() * 0.05)
    v_peaks, _ = find_peaks(col_sum, distance=min_step_x,
                             prominence=col_sum.max() * 0.05)

    h_peaks = _top_n_peaks(h_peaks, row_sum, n_lines)
    v_peaks = _top_n_peaks(v_peaks, col_sum, n_lines)

    h_peaks = _fill_missing_lines(h_peaks, row_sum, n_lines)
    v_peaks = _fill_missing_lines(v_peaks, col_sum, n_lines)

    return np.sort(v_peaks + x0), np.sort(h_peaks + y0)


def _fill_missing_lines(peaks: np.ndarray, profile: np.ndarray,
                         n_target: int = 9) -> np.ndarray:
    """
    Top up *peaks* to *n_target* by extrapolating from the median step.  At
    each round we try a candidate position one step beyond the current min
    and beyond the current max, then pick whichever sits on a stronger spot
    in *profile* (still preferring the chess-grid edge over empty space).
    """
    if len(peaks) >= n_target or len(peaks) < 3:
        return peaks
    peaks = np.sort(peaks)
    diffs = np.diff(peaks)
    step = float(np.median(diffs))

    while len(peaks) < n_target:
        left = peaks[0] - step
        right = peaks[-1] + step
        candidates = []
        if 0 <= left < len(profile):
            candidates.append((left, profile[int(left)]))
        if 0 <= right < len(profile):
            candidates.append((right, profile[int(right)]))
        if not candidates:
            break
        # Pick the candidate with higher local edge support
        best_pos, _ = max(candidates, key=lambda c: c[1])
        peaks = np.sort(np.append(peaks, int(round(best_pos))))
    return peaks


def _top_n_peaks(peaks: np.ndarray, profile: np.ndarray, n: int) -> np.ndarray:
    if len(peaks) <= n:
        return peaks
    heights = profile[peaks]
    keep = np.argsort(heights)[-n:]
    return peaks[keep]


# ── Correspondences (image points ↔ board grid positions) ────────────────────

def _build_correspondences(v_xs: np.ndarray, h_ys: np.ndarray,
                            image_shape: tuple):
    """
    Map detected vertical-line x positions and horizontal-line y positions to
    integer board indices.

    First we reject outliers whose spacing to neighbours is wildly different
    from the median step (e.g. a wood-frame edge picked up as a "grid line").
    For partial detections (n < 9 lines) we infer which board indices the
    detected lines occupy by looking at the distance from the image borders:
    a large gap above the first line means row 0 is missing, so the detected
    block starts at row 1 instead of row 0.
    """
    v_xs = _drop_step_outliers(np.array(v_xs, dtype=np.float64))
    h_ys = _drop_step_outliers(np.array(h_ys, dtype=np.float64))

    step = BOARD_PX / 8.0
    img_h, img_w = image_shape
    col_shift = _detect_index_shift(v_xs, 0, img_w)
    row_shift = _detect_index_shift(h_ys, 0, img_h)
    col_idx = np.arange(len(v_xs)) + col_shift
    row_idx = np.arange(len(h_ys)) + row_shift

    src, dst = [], []
    for r, y in enumerate(h_ys):
        for c, x in enumerate(v_xs):
            src.append([x, y])
            dst.append([col_idx[c] * step, row_idx[r] * step])
    return (np.array(src, dtype=np.float32),
            np.array(dst, dtype=np.float32))


def _detect_index_shift(positions: np.ndarray, axis_min: int,
                         axis_max: int) -> int:
    """
    Decide which board index the first detected line occupies (0..8-n+1).

    Heuristic: count how many full grid steps fit between the image border
    and the first detected line.  That many lines are assumed to be missing
    "above" (= shift).  Symmetrical check for the trailing side keeps the
    placement consistent.
    """
    n = len(positions)
    if n >= 9 or n < 2:
        return max(0, (8 - (n - 1)) // 2)
    step = float(np.median(np.diff(positions)))
    if step <= 0:
        return (8 - (n - 1)) // 2

    space_before = positions[0] - axis_min
    space_after = axis_max - positions[-1]
    # How many extra grid lines could fit before / after?
    missing_before = max(0, int(round(space_before / step) - 0))
    missing_after = max(0, int(round(space_after / step) - 0))
    total_missing = 9 - n
    # Trim so the sum exactly equals total_missing
    if missing_before + missing_after > total_missing:
        # Distribute proportionally
        scale = total_missing / (missing_before + missing_after)
        missing_before = int(round(missing_before * scale))
        missing_after = total_missing - missing_before
    elif missing_before + missing_after < total_missing:
        missing_before += (total_missing - missing_before - missing_after) // 2
    return max(0, min(8 - (n - 1), missing_before))


def _drop_step_outliers(positions: np.ndarray) -> np.ndarray:
    """
    Drop a leading/trailing position whose distance to its only neighbour is
    >1.7× the median inner step.  Catches wood-frame edges that masquerade as
    grid lines and would skew the indexing.
    """
    if len(positions) < 4:
        return positions
    positions = np.sort(positions)
    diffs = np.diff(positions)
    median_step = float(np.median(diffs))

    while len(positions) > 4 and diffs[0] > 1.7 * median_step:
        positions = positions[1:]
        diffs = np.diff(positions)
        median_step = float(np.median(diffs))
    while len(positions) > 4 and diffs[-1] > 1.7 * median_step:
        positions = positions[:-1]
        diffs = np.diff(positions)
        median_step = float(np.median(diffs))
    return positions


# ── Image processing helpers ──────────────────────────────────────────────────

def _dominant_grid_angle(gray: np.ndarray, edges: np.ndarray) -> float:
    """Estimate board rotation via Sobel gradient angle histogram."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.rad2deg(np.arctan2(gy, gx))
    magnitudes = np.sqrt(gx ** 2 + gy ** 2)

    mask = edges > 0
    if not np.any(mask):
        return 0.0

    hist, bins = np.histogram(angles[mask], bins=180, range=(-90, 90),
                               weights=magnitudes[mask])
    peaks, _ = find_peaks(hist, distance=15, height=hist.max() * 0.05)
    if len(peaks) < 2:
        return 0.0

    peak_angles = bins[peaks]
    best_angle, best_score = 0.0, 0.0
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            if abs(abs(peak_angles[i] - peak_angles[j]) - 90) < 15:
                score = hist[peaks[i]] + hist[peaks[j]]
                if score > best_score:
                    best_score = score
                    a1, a2 = peak_angles[i], peak_angles[j]
                    best_angle = a1 if abs(a1) <= abs(a2) else a2
    return float(best_angle)


def _line_length(line) -> float:
    x1, y1, x2, y2 = line[0]
    return float(np.hypot(x2 - x1, y2 - y1))


def _filter_lines(lines, orientation: str, tol: float = 20.0) -> list:
    result = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180) % 180
        if orientation == 'horizontal' and (angle < tol or angle > 180 - tol):
            result.append(line)
        elif orientation == 'vertical' and abs(angle - 90) < tol:
            result.append(line)
    return result


def _intersections(h_lines: list, v_lines: list, shape: tuple) -> list:
    h, w = shape
    pts = []
    for hl in h_lines:
        x1, y1, x2, y2 = map(float, hl[0])
        for vl in v_lines:
            x3, y3, x4, y4 = map(float, vl[0])
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            if 0 <= px <= w and 0 <= py <= h:
                pts.append((px, py))
    return pts


def _cluster(pts: np.ndarray, eps: float = 20.0) -> np.ndarray:
    if len(pts) == 0:
        return pts
    labels = DBSCAN(eps=eps, min_samples=1).fit(pts).labels_
    return np.array([pts[labels == lbl].mean(axis=0) for lbl in np.unique(labels)])


def _order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmax(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmin(d)],   # BL
    ], dtype=np.float32)


def _is_valid_quad(corners: np.ndarray, shape: tuple,
                   area_min: float = 0.02) -> bool:
    h, w = shape
    quad_area = cv2.contourArea(corners.astype(np.float32))
    if quad_area < area_min * h * w:
        return False
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    if max(sides) / max(min(sides), 1e-6) > 5.0:
        return False
    return True


def _warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [BOARD_PX, 0], [BOARD_PX, BOARD_PX], [0, BOARD_PX]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, H, (BOARD_PX, BOARD_PX))


def _extract_squares(board: np.ndarray) -> list:
    sq = SQUARE_PX
    return [board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq].copy()
            for r in range(8) for c in range(8)]


def _show_debug_lines(rot_edges: np.ndarray, v_xs, h_ys) -> None:
    vis = cv2.cvtColor(rot_edges, cv2.COLOR_GRAY2BGR)
    for x in v_xs:
        cv2.line(vis, (int(x), 0), (int(x), vis.shape[0]), (0, 255, 0), 2)
    for y in h_ys:
        cv2.line(vis, (0, int(y)), (vis.shape[1], int(y)), (0, 0, 255), 2)
    cv2.imshow('board_detection_debug', vis)
    cv2.waitKey(1)
