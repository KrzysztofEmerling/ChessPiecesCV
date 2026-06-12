"""
Generate a per-square dataset from chessboard images + corner annotations.

For each annotated image we warp the board with the homography that maps the
4 annotated corners onto an 800×800 square, slice the result into 64 cells,
and write each cell out as its own JPG.

Output layout
-------------
If ChessReD's annotations.json is supplied (--chessred-annotations) each
square is filed under its piece class:

    output/
      empty/
      white_pawn/  white_knight/  white_bishop/  white_rook/  white_queen/  white_king/
      black_pawn/  black_knight/  black_bishop/  black_rook/  black_queen/  black_king/

Without piece labels every square goes to output/_unlabeled/.

Filenames encode the source image and the chess square, e.g.
    G000_IMG000_e4.jpg

Usage
-----
    python -m dataset_generation.generate_squares \\
        --image-root dataset/chessred2k/images \\
        --chessred-annotations dataset/chessred2k/annotations.json \\
        --output dataset_squares

    # Without ChessReD labels – every square in _unlabeled/
    python -m dataset_generation.generate_squares --output dataset_squares
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from board_detection import BOARD_PX, SQUARE_PX, detect_board  # noqa: E402

PIECE_CLASSES = (
    "empty",
    "white_pawn", "white_knight", "white_bishop",
    "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop",
    "black_rook", "black_queen", "black_king",
)
UNLABELED = "_unlabeled"


# ── corner annotation loader ─────────────────────────────────────────────────

def load_corners(ann_path: Path) -> np.ndarray | None:
    with open(ann_path) as f:
        data = json.load(f)
    corners = data.get("corners", [])
    if len(corners) != 4:
        return None
    return np.array([[c["x"], c["y"]] for c in corners], dtype=np.float32)


def load_source_image(ann_data: dict, image_root: Path,
                       repo_root: Path) -> Path | None:
    """Resolve the source_image path stored in our annotation JSON."""
    src = ann_data.get("source_image", "")
    if not src:
        return None
    # Try absolute, repo-relative, then image-root-relative paths
    for cand in (Path(src), repo_root / src, image_root / Path(src).name,
                 image_root / src):
        if cand.exists():
            return cand
    return None


# ── ChessReD label loader (optional) ─────────────────────────────────────────

def load_chessred_labels(annotations_json: Path) -> dict:
    """
    Return {image_stem: {(file_idx, rank_idx): class_name}} where
    file_idx in 0..7 (a..h) and rank_idx in 0..7 (1..8 from bottom).
    """
    with open(annotations_json) as f:
        data = json.load(f)

    id_to_stem = {img["id"]: Path(img["file_name"]).stem
                  for img in data.get("images", [])}
    cat_to_name = {c["id"]: _normalise_class_name(c["name"])
                   for c in data.get("categories", [])}

    pieces_list = data.get("annotations", {}).get("pieces", [])
    if not pieces_list:  # alt structure: flat annotations list
        pieces_list = [a for a in data.get("annotations", [])
                       if isinstance(a, dict) and "chessboard_position" in a]

    labels: dict = {}
    for ann in pieces_list:
        stem = id_to_stem.get(ann.get("image_id"))
        if not stem:
            continue
        pos = (ann.get("chessboard_position") or "").lower()
        if len(pos) != 2 or pos[0] not in "abcdefgh" or pos[1] not in "12345678":
            continue
        file_idx = ord(pos[0]) - ord("a")
        rank_idx = int(pos[1]) - 1
        name = cat_to_name.get(ann.get("category_id"))
        if name:
            labels.setdefault(stem, {})[(file_idx, rank_idx)] = name
    return labels


def _normalise_class_name(name: str) -> str:
    """Map ChessReD category names to our PIECE_CLASSES naming."""
    n = name.lower().replace("-", "_").replace(" ", "_")
    return n if n in PIECE_CLASSES else n


# ── square <-> chess-square mapping ──────────────────────────────────────────

def square_class(col_px: int, row_px_from_top: int,
                  pieces: dict | None) -> str:
    """
    col_px:           column of the square in the warped image (0..7 = file a..h)
    row_px_from_top:  row of the square in the warped image  (0 = top)
    pieces:           {(file_idx, rank_idx): class} or None for unlabeled
    """
    if pieces is None:
        return UNLABELED
    rank_idx = 7 - row_px_from_top  # warped row 0 (top) is rank 8 = rank_idx 7
    return pieces.get((col_px, rank_idx), "empty")


def square_filename(image_stem: str, col_px: int, row_px_from_top: int) -> str:
    file_char = chr(ord("a") + col_px)
    rank = 8 - row_px_from_top
    return f"{image_stem}_{file_char}{rank}.jpg"


# ── core pipeline ────────────────────────────────────────────────────────────

def warp_board(image: np.ndarray, corners: np.ndarray,
                orientation_flip: bool = False) -> np.ndarray:
    """
    Warp *image* so the 4 corners map to an 800×800 board.

    corners order in our annotations: TL, TR, BR, BL.  Standard orientation
    assumes TL ↔ a8 (so file a is on the left, rank 8 on top).  Pass
    orientation_flip=True to rotate 180° when the annotator picked a1 as TL.
    """
    if orientation_flip:
        corners = np.roll(corners, 2, axis=0)  # rotate corner order by 180°
    dst = np.array([[0, 0], [BOARD_PX, 0],
                    [BOARD_PX, BOARD_PX], [0, BOARD_PX]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, H, (BOARD_PX, BOARD_PX))


def extract_squares(board: np.ndarray, padding: int = 0):
    """Yield (col, row_from_top, square_image) for the 64 cells."""
    sq = SQUARE_PX
    for r in range(8):
        for c in range(8):
            y0 = r * sq + padding
            y1 = (r + 1) * sq - padding
            x0 = c * sq + padding
            x1 = (c + 1) * sq - padding
            yield c, r, board[y0:y1, x0:x1].copy()


def process_image(image: np.ndarray, corners: np.ndarray | None,
                   stem: str, labels_map: dict | None,
                   output_dir: Path, padding: int,
                   orientation_flip: bool, fallback_detect: bool) -> int:
    """Warp + slice + write one image.  Returns number of squares written."""
    if corners is None:
        if not fallback_detect:
            return 0
        result = detect_board(image)
        if result is None:
            return 0
        corners = result.corners

    board = warp_board(image, corners, orientation_flip=orientation_flip)

    pieces = None if labels_map is None else labels_map.get(stem)
    written = 0
    for col, row, square in extract_squares(board, padding=padding):
        cls = square_class(col, row, pieces)
        out_path = output_dir / cls / square_filename(stem, col, row)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(out_path), square):
            written += 1
    return written


def generate(corners_dir: Path, image_root: Path | None, output_dir: Path,
             chessred_annotations: Path | None, padding: int,
             orientation_flip: bool, limit: int | None,
             fallback_detect: bool) -> None:
    repo_root = ROOT
    labels_map = (load_chessred_labels(chessred_annotations)
                  if chessred_annotations else None)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_classes = PIECE_CLASSES if labels_map is not None else (UNLABELED,)
    for cls in base_classes:
        (output_dir / cls).mkdir(exist_ok=True)

    annotation_files = sorted(corners_dir.glob("*.json"))
    if limit:
        annotation_files = annotation_files[:limit]

    print(f"Found {len(annotation_files)} annotation files")
    if labels_map is not None:
        print(f"Loaded labels for {len(labels_map)} images from "
              f"{chessred_annotations}")
    print(f"Writing to {output_dir}")
    print()

    ok = skip = 0
    for i, ann_path in enumerate(annotation_files, 1):
        with open(ann_path) as f:
            ann_data = json.load(f)

        img_path = load_source_image(
            ann_data, image_root or repo_root, repo_root)
        if img_path is None:
            skip += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            skip += 1
            continue

        corners = load_corners(ann_path)
        written = process_image(
            image, corners, ann_path.stem, labels_map, output_dir,
            padding=padding, orientation_flip=orientation_flip,
            fallback_detect=fallback_detect)
        if written:
            ok += 1
        else:
            skip += 1

        if i % 50 == 0:
            print(f"  [{i}/{len(annotation_files)}] ok={ok} skip={skip}")

    print(f"\nDone. ok={ok} skip={skip}")
    print(f"Output: {output_dir}")
    if labels_map is not None:
        for cls in PIECE_CLASSES:
            n = len(list((output_dir / cls).glob("*.jpg")))
            if n:
                print(f"  {cls:<14} {n:>6}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corners-dir", type=Path,
                    default=ROOT / "annotations",
                    help="Folder with corner annotation JSONs (default: annotations/)")
    p.add_argument("--image-root", type=Path, default=None,
                    help="Folder where source images live (default: resolved from "
                         "the 'source_image' path in each annotation)")
    p.add_argument("--output", type=Path, default=ROOT / "dataset_squares",
                    help="Output directory for per-square images")
    p.add_argument("--chessred-annotations", type=Path, default=None,
                    help="Path to ChessReD annotations.json for piece labels "
                         "(optional)")
    p.add_argument("--padding", type=int, default=0,
                    help="Crop N pixels off each square edge (helps drop grid lines)")
    p.add_argument("--flip", action="store_true",
                    help="Rotate the warped board 180° (use when the annotator "
                         "picked a1 as TL instead of a8)")
    p.add_argument("--limit", type=int, default=None,
                    help="Process at most N annotation files (useful for testing)")
    p.add_argument("--fallback-detect", action="store_true",
                    help="Run detect_board() when a corner annotation is missing")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    generate(
        corners_dir=args.corners_dir,
        image_root=args.image_root,
        output_dir=args.output,
        chessred_annotations=args.chessred_annotations,
        padding=args.padding,
        orientation_flip=args.flip,
        limit=args.limit,
        fallback_detect=args.fallback_detect,
    )


if __name__ == "__main__":
    main()
