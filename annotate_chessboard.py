import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
# Default image root uses Dataset when available, otherwise the repository root.
DEFAULT_IMAGE_ROOT = REPO_ROOT / 'Dataset'
# Simpler default output folder for JSON annotations.
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'annotations'


def find_image_paths(root_path, extensions=None):
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Image root path does not exist: {root}")

    image_paths = [p for p in root.rglob("*") if p.suffix.lower() in extensions]
    image_paths.sort()
    return image_paths


def permute_image_paths(image_paths, seed=None):
    paths = list(image_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths


def line_length(line):
    x1, y1, x2, y2 = line[0]
    return np.hypot(x2 - x1, y2 - y1)


def line_intersection(line1, line2):
    x1, y1, x2, y2 = map(float, line1[0])
    x3, y3, x4, y4 = map(float, line2[0])

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    if not (min(x1, x2) - 1 <= px <= max(x1, x2) + 1 and min(y1, y2) - 1 <= py <= max(y1, y2) + 1):
        return None
    if not (min(x3, x4) - 1 <= px <= max(x3, x4) + 1 and min(y3, y4) - 1 <= py <= max(y3, y4) + 1):
        return None

    return int(round(px)), int(round(py))


def filter_unique_points(points, min_dist=10):
    if len(points) == 0:
        return []

    points = [tuple(p) for p in points]
    points.sort(key=lambda p: (p[0], p[1]))
    unique = []

    for pt in points:
        if all(np.hypot(pt[0] - ux, pt[1] - uy) >= min_dist for ux, uy in unique):
            unique.append(pt)
    return unique


def classify_lines(lines, orientation='horizontal', angle_tolerance=15):
    classified = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = (angle + 180) % 180
        if orientation == 'horizontal':
            if abs(angle) < angle_tolerance or abs(angle - 180) < angle_tolerance:
                classified.append(line)
        elif orientation == 'vertical':
            if abs(angle - 90) < angle_tolerance:
                classified.append(line)
    return classified


def reduce_lines(lines, max_lines=30):
    if lines is None:
        return []
    lines = sorted(lines, key=line_length, reverse=True)
    return lines[:max_lines]


def detect_chessboard_intersections(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=60, maxLineGap=20)
    if lines is None:
        return []

    h_lines = classify_lines(lines, orientation='horizontal')
    v_lines = classify_lines(lines, orientation='vertical')

    h_lines = reduce_lines(h_lines, max_lines=30)
    v_lines = reduce_lines(v_lines, max_lines=30)

    intersections = []
    for h in h_lines:
        for v in v_lines:
            pt = line_intersection(h, v)
            if pt is not None:
                intersections.append(pt)

    intersections = filter_unique_points(intersections, min_dist=15)

    if debug:
        debug_img = image.copy()
        for x, y in intersections:
            cv2.circle(debug_img, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow('debug_intersections', debug_img)
        cv2.waitKey(1)

    return intersections


def pick_extreme_corners(points, frame_shape):
    if len(points) == 0:
        return []

    h, w = frame_shape[:2]
    arr = np.array(points)
    x = arr[:, 0]
    y = arr[:, 1]

    tl = tuple(arr[np.argmin(x + y)])
    tr = tuple(arr[np.argmin((w - 1 - x) + y)])
    br = tuple(arr[np.argmin((w - 1 - x) + (h - 1 - y))])
    bl = tuple(arr[np.argmin(x + (h - 1 - y))])

    corners = [tl, tr, br, bl]
    unique = []
    for c in corners:
        if c not in unique:
            unique.append(c)
    return unique


def format_point(pt):
    return {'x': int(pt[0]), 'y': int(pt[1])}


def save_annotation(output_path, source_path, corners, candidates):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'source_image': str(source_path),
        'corners': [format_point(pt) for pt in corners],
        'candidates': [format_point(pt) for pt in candidates],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def copy_image_file(source_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source_path.name
    shutil.copy2(source_path, destination)
    return destination


class AnnotationSession:
    ORDER_LABELS = ['top-left', 'top-right', 'bottom-right', 'bottom-left']

    def __init__(self, window_name, image, candidates, snap_distance=20):
        self.window_name = window_name
        self.original = image.copy()
        self.display = image.copy()
        self.candidates = [tuple(map(int, p)) for p in candidates]
        self.snap_distance = snap_distance
        self.selected = []
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = self.snap_point((x, y))
        if len(self.selected) < 4:
            self.selected.append(point)
        else:
            self.selected[-1] = point
        self._redraw()

    def snap_point(self, point):
        if not self.candidates:
            return tuple(point)
        dist = None
        nearest = None
        px, py = point
        for cx, cy in self.candidates:
            d = np.hypot(cx - px, cy - py)
            if dist is None or d < dist:
                dist = d
                nearest = (cx, cy)
        if dist is not None and dist <= self.snap_distance:
            return nearest
        return tuple(point)

    def _redraw(self):
        self.display = self.original.copy()
        for cx, cy in self.candidates:
            cv2.circle(self.display, (cx, cy), 4, (0, 255, 255), -1)
        for idx, pt in enumerate(self.selected):
            color = (0, 255, 0) if idx < 4 else (0, 0, 255)
            cv2.circle(self.display, pt, 6, color, -1)
            label = self.ORDER_LABELS[idx] if idx < len(self.ORDER_LABELS) else str(idx + 1)
            cv2.putText(self.display, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if len(self.selected) < 4:
            text = f"Click {4 - len(self.selected)} corners, s=save, r=reset, d=delete last, n=skip, q=quit"
        else:
            text = "4 corners selected. Press s=save, r=reset, d=delete last, n=skip, q=quit"
        cv2.putText(self.display, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        self._redraw()
        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                return 'quit', []
            if key == ord('n'):
                return 'skip', []
            if key == ord('r'):
                self.selected = []
                self._redraw()
            if key == ord('d'):
                if self.selected:
                    self.selected.pop()
                    self._redraw()
            if key == ord('s'):
                if len(self.selected) == 4:
                    return 'save', list(self.selected)
                else:
                    print('Need 4 selected corners before saving.')

    def close(self):
        cv2.destroyWindow(self.window_name)


def annotate_images(image_paths, output_root, seed=None, limit=None, debug=False, copy_images=False):
    output_root = Path(output_root)
    if limit is not None:
        image_paths = image_paths[:limit]

    image_paths = permute_image_paths(image_paths, seed)

    for idx, path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {path}")
        image = cv2.imread(str(path))
        if image is None:
            print(f"Unable to read image {path}")
            continue

        candidates = detect_chessboard_intersections(image, debug=debug)
        auto_corners = pick_extreme_corners(candidates, image.shape)

        session = AnnotationSession('Chessboard annotation', image, candidates)
        if auto_corners and len(auto_corners) == 4:
            for pt in auto_corners:
                session.selected.append(pt)
            session._redraw()
            print('Auto-estimated corner positions added. Adjust or accept with s.')

        action, corners = session.run()
        session.close()
        if debug:
            cv2.destroyWindow('debug_intersections')

        if action == 'quit':
            print('User requested quit. Stopping annotation.')
            break
        if action == 'skip':
            print('Skipped this image.')
            continue

        if action == 'save' and corners:
            annotation_path = output_root / f"{path.stem}.json"
            save_annotation(annotation_path, path, corners, candidates)
            if copy_images:
                copied = copy_image_file(path, output_root)
                print(f"Copied image to {copied}")
            print(f"Saved annotation {annotation_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Chessboard corner validation annotation tool')
    parser.add_argument('image_root', nargs='?', default=None,
                        help='Root folder of input chessboard images (optional; defaults to repository Dataset or repo root)')
    parser.add_argument('--output', default=None,
                        help='Output folder for annotation JSON files (defaults to repo/annotations)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for permutation')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of images to annotate')
    parser.add_argument('--debug', action='store_true', help='Show debug intersections window')
    parser.add_argument('--copy-images', action='store_true', help='Copy original images into the output folder alongside JSON')
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Use the repo Dataset folder if available, otherwise use the repository root.
    image_root = Path(args.image_root) if args.image_root else (DEFAULT_IMAGE_ROOT if DEFAULT_IMAGE_ROOT.exists() else REPO_ROOT)
    # Save annotations into a simple repo/annotations folder by default.
    output_root = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT

    print(f'Running annotation from: {image_root}')
    print(f'Saving annotations to: {output_root}')

    image_paths = find_image_paths(image_root)
    if len(image_paths) == 0:
        raise SystemExit(f"No image files found under {image_root}")

    annotate_images(
        image_paths,
        output_root,
        seed=args.seed,
        limit=args.limit,
        debug=args.debug,
        copy_images=args.copy_images,
    )
    print('Annotation session finished.')


if __name__ == '__main__':
    main()
