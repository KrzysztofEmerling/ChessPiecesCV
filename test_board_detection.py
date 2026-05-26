# to generował chat bo to tylko skrypt do testu, nie bijcie, wydaje się git

"""
Quick visual test for the board_detection module.

Usage:
    python test_board_detection.py                          # auto-pick first annotated image
    python test_board_detection.py path/to/image.jpg       # specific image
    python test_board_detection.py --compare-annotation     # overlay GT corners from JSON
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from board_detection import detect_board


# ── helpers ───────────────────────────────────────────────────────────────────

def load_annotation(image_path: Path) -> list[tuple[int, int]] | None:
    ann_path = ROOT / "annotations" / (image_path.stem + ".json")
    if not ann_path.exists():
        return None
    with open(ann_path) as f:
        data = json.load(f)
    return [(c["x"], c["y"]) for c in data["corners"]]


def corner_error(detected: np.ndarray, gt: list) -> float:
    """Mean pixel distance between detected and GT corners (after matching order)."""
    gt_arr = np.array(gt, dtype=np.float32)
    return float(np.linalg.norm(detected - gt_arr, axis=1).mean())


def find_default_image() -> Path | None:
    ann_dir = ROOT / "annotations"
    for json_file in sorted(ann_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        img_path = ROOT / data["source_image"]
        if img_path.exists():
            return img_path
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to chessboard image")
    parser.add_argument("--compare-annotation", action="store_true",
                        help="Draw GT corners from the matching JSON annotation")
    parser.add_argument("--debug", action="store_true",
                        help="Show intermediate intersection points (opens cv2 window)")
    args = parser.parse_args()

    # resolve image path
    if args.image:
        image_path = Path(args.image)
    else:
        image_path = find_default_image()
        if image_path is None:
            sys.exit("No image found. Pass an image path as argument.")
        print(f"Auto-selected: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"Cannot read image: {image_path}")

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # run detection
    result = detect_board(image, debug=args.debug)

    if result is None:
        print("Detection FAILED — no board found.")
        sys.exit(1)

    print("Detection OK")
    print("Corners (TL TR BR BL):")
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), result.corners):
        print(f"  {label}: ({x:.0f}, {y:.0f})")

    # optional annotation comparison
    gt_corners = load_annotation(image_path) if args.compare_annotation else None
    if gt_corners is not None:
        err = corner_error(result.corners, gt_corners)
        print(f"Mean corner error vs annotation: {err:.1f} px")

    # ── visualisation ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))

    # panel 1 — original image with detected corners
    ax1 = fig.add_subplot(1, 3, 1)
    vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(vis)
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    labels = ["TL", "TR", "BR", "BL"]
    pts = result.corners
    ax1.add_patch(plt.Polygon(pts, fill=False, edgecolor="lime", linewidth=2))
    for (x, y), lbl, col in zip(pts, labels, colors):
        ax1.plot(x, y, "o", color=col, markersize=10)
        ax1.text(x + 10, y - 10, lbl, color=col, fontsize=9, fontweight="bold")
    if gt_corners:
        gt = np.array(gt_corners)
        ax1.add_patch(plt.Polygon(gt, fill=False, edgecolor="yellow",
                                   linewidth=2, linestyle="--"))
        ax1.set_title("Detected (green) vs GT (yellow)")
    else:
        ax1.set_title("Detected corners")
    ax1.axis("off")

    # panel 2 — warped top-down board
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cv2.cvtColor(result.board_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Warped board (800×800)")
    # draw grid
    for i in range(9):
        ax2.axhline(i * 100, color="cyan", linewidth=0.5, alpha=0.7)
        ax2.axvline(i * 100, color="cyan", linewidth=0.5, alpha=0.7)
    ax2.axis("off")

    # panel 3 — all 64 squares
    ax3 = fig.add_subplot(1, 3, 3)
    grid = np.zeros((800, 800, 3), dtype=np.uint8)
    for idx, sq in enumerate(result.squares):
        r, c = divmod(idx, 8)
        grid[r * 100:(r + 1) * 100, c * 100:(c + 1) * 100] = sq
    ax3.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    ax3.set_title("64 extracted squares")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()

    if args.debug:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
