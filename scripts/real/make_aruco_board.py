"""Generate a printable A4 ArUco grid board (PDF at exact physical size + PNG preview).

Board: DICT_4X4_50, 3 x 4 markers, marker 50 mm, gap 10 mm (170 x 230 mm), centred on A4.
Print at 100% ("actual size", no fit-to-page), then measure one marker edge with a ruler / calipers
and pass that length to scripts/real/scale_from_aruco.py -- printers often scale by 1-3%.

Usage: python scripts/real/make_aruco_board.py --out docs/aruco_board
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLS, ROWS, MARKER_MM, GAP_MM = 3, 4, 50.0, 10.0
DICT = cv2.aruco.DICT_4X4_50
A4_MM = (210.0, 297.0)
DPI = 600


def board():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.GridBoard((COLS, ROWS), MARKER_MM / 1000, GAP_MM / 1000, d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/aruco_board")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    w_mm = COLS * MARKER_MM + (COLS - 1) * GAP_MM
    h_mm = ROWS * MARKER_MM + (ROWS - 1) * GAP_MM
    px = lambda mm: int(round(mm / 25.4 * DPI))  # noqa: E731
    img = board().generateImage((px(w_mm), px(h_mm)), marginSize=0, borderBits=1)
    cv2.imwrite(str(out.with_suffix(".png")), img)

    fig = plt.figure(figsize=(A4_MM[0] / 25.4, A4_MM[1] / 25.4))
    x0, y0 = (A4_MM[0] - w_mm) / 2 / A4_MM[0], (A4_MM[1] - h_mm) / 2 / A4_MM[1]
    ax = fig.add_axes([x0, y0, w_mm / A4_MM[0], h_mm / A4_MM[1]])
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.axis("off")
    fig.text(0.5, 0.03, f"ArUco DICT_4X4_50 · {COLS}x{ROWS} · marker {MARKER_MM:.0f} mm · gap {GAP_MM:.0f} mm · "
             "print at 100% (actual size)", ha="center", fontsize=8)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.parent / (out.name + "_a4.png"), dpi=300)  # full page, used as a texture in simulation
    print(f"board {w_mm:.0f} x {h_mm:.0f} mm -> {out.with_suffix('.pdf')}, {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
