#!/usr/bin/env python3
"""
maze_solver.py — MIT‑licensed shortest‑path solver for black‑and‑white maze images.

Key features
============
* **Auto colour sense** – detects whether corridors are white or black (override with `--invert`).
* **Noise‑friendly** – optional blur + morphological open/close before thresholding.
* **Adaptive down‑scale** – if image exceeds `--max‑pixels` (default 4 000 000), resizes with
  nearest‑neighbour so RAM stays sane.
* **Multiple start/goal pairs** – pass `--points "y1,x1 y2,x2 y3,x3 y4,x4 …"`; each adjacent
  pair forms one solve request. If omitted, a GUI pops up: click start/end, start/end … then
  press *Q* to finish.
* **Outputs** – for each pair:
    • `<stem>_pair<i>_solved.png`  – original maze with path in red
    • `<stem>_pair<i>.csv`         – CSV of path pixel coordinates (row,col)
    • Optional GIF of BFS frontier (`--animate`)

Requires Python ≥ 3.9 and:
    pip install opencv-python-headless networkx numpy matplotlib pillow imageio
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import cv2  # type: ignore
import imageio.v3 as iio  # type: ignore  # for GIF
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np

###############################################################################
# Image helpers
###############################################################################

def auto_should_invert(gray: np.ndarray, sample: int = 10_000) -> bool:
    """Return True if corridors appear darker than walls (i.e. want invert)."""
    h, w = gray.shape
    rng = np.random.default_rng(0)
    ys = rng.integers(0, h, size=min(sample, h * w))
    xs = rng.integers(0, w, size=ys.size)
    sample_vals = gray[ys, xs]
    # assume majority of sampled pixels are background (walls)
    return sample_vals.mean() < 128  # darker overall ⇒ corridors likely dark


def preprocess(gray: np.ndarray, *, thresh: int, invert: bool, max_pixels: int) -> np.ndarray:
    """Blur → threshold → optional invert → optional downscale."""
    # Gaussian blur smooths scanner noise; cheap if small sigma
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold – Otsu if thresh < 0 else manual
    if thresh < 0:
        _, bin_img = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bin_img = cv2.threshold(blur, thresh, 1, cv2.THRESH_BINARY)

    # Invert if corridors are dark
    if invert:
        bin_img = 1 - bin_img

    # Morphological opening (remove isolated dots), then closing (bridge gaps)
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    # Downscale if needed
    h, w = bin_img.shape
    while h * w > max_pixels:
        bin_img = cv2.resize(bin_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        h, w = bin_img.shape
    return bin_img.astype(np.uint8)

###############################################################################
# Graph + path
###############################################################################

def build_graph(bin_img: np.ndarray) -> nx.Graph:
    G: nx.Graph = nx.Graph()
    h, w = bin_img.shape
    white = np.transpose(np.nonzero(bin_img))
    for r, c in white:
        node = (r, c)
        G.add_node(node)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bin_img[nr, nc]:
                G.add_edge(node, (nr, nc))
    return G


def bfs_path(G: nx.Graph, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    try:
        return nx.shortest_path(G, start, end)
    except nx.NetworkXNoPath as e:
        raise ValueError("No path between given points") from e

###############################################################################
# Visuals
###############################################################################

def overlay(gray: np.ndarray, path: list[tuple[int, int]]):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r, c in path:
        rgb[r, c] = (0, 0, 255)  # red
    return rgb

###############################################################################
# Click helper
###############################################################################

def collect_points_gui(gray: np.ndarray) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []

    def on_click(event):
        if event.inaxes:
            pts.append((int(event.ydata), int(event.xdata)))
            plt.scatter(event.xdata, event.ydata, c="red", s=10)
            plt.draw()

    fig, ax = plt.subplots()
    ax.imshow(gray, cmap="gray")
    ax.set_title("Click START, END, START, END … then press Q to finish")

    cid = fig.canvas.mpl_connect("button_press_event", on_click)

    def on_key(event):
        if event.key.lower() == "q":
            plt.close(fig)

    kid = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(kid)
    return pts

###############################################################################
# Main solve routine per pair
###############################################################################

def solve_pair(gray: np.ndarray, bin_img: np.ndarray, start: tuple[int, int], end: tuple[int, int], *, pair_idx: int, stem: str, animate: bool):
    G = build_graph(bin_img)

    frames = []
    if animate:
        # BFS frontier capture – convert generator from networkx bfs_tree
        from collections import deque
        frontier = deque([start])
        visited = {start: None}
        steps = 0
        while frontier:
            current = frontier.popleft()
            if current == end:
                break
            for neigh in G.neighbors(current):
                if neigh not in visited:
                    visited[neigh] = current
                    frontier.append(neigh)
            if steps % 50 == 0:
                frame = overlay(gray, list(visited.keys()))
                frames.append(frame)
            steps += 1
        # rebuild final path
        path = [end]
        while path[-1] != start:
            path.append(visited[path[-1]])
        path.reverse()
    else:
        path = bfs_path(G, start, end)

    out_png = f"{stem}_pair{pair_idx}_solved.png"
    cv2.imwrite(out_png, overlay(gray, path))

    # CSV of pixels
    out_csv = f"{stem}_pair{pair_idx}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col"])
        w.writerows(path)

    if animate and frames:
        frames.append(overlay(gray, path))  # final overlay
        iio.imwrite(f"{stem}_pair{pair_idx}.gif", frames, duration=0.08, loop=0)

    print(f"pair {pair_idx}: length={len(path)-1} → {out_png}")

###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Solve shortest path in a B/W maze image")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--points", nargs="*", help="Pairs of y,x coords e.g. '10,10 150,150'")
    p.add_argument("--threshold", type=int, default=-1, help="Manual threshold (0‑255) or <0 for Otsu")
    p.add_argument("--invert", action="store_true", help="Force invert colors")
    p.add_argument("--max-pixels", type=int, default=4_000_000, help="Resize until pixels ≤ this")
    p.add_argument("--animate", action="store_true", help="Save a BFS GIF animation")
    return p.parse_args()


def main():
    args = parse_args()

    gray0 = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray0 is None:
        sys.exit(f"Cannot read image {args.image}")

    invert = args.invert or auto_should_invert(gray0)
    bin_img = preprocess(gray0, thresh=args.threshold, invert=invert, max_pixels=args.max_pixels)

    # collect start/end pairs
    if args.points:
        coords: list[tuple[int, int]] = []
        for token in args.points:
            try:
                y, x = map(int, token.split(","))
                coords.append((y, x))
            except ValueError:
                sys.exit(f"Bad --points token '{token}', expected 'y,x'")
        if len(coords) % 2:
            sys.exit("Need an even number of coordinates (start/end pairs)")
    else:
        coords = collect_points_gui(gray0)
        if len(coords) % 2:
            sys.exit("Clicked odd number of points — must be pairs")

    stem = args.image.stem
    for i in range(0, len(coords), 2):
        start, end = coords[i], coords[i + 1]
        solve_pair(gray0, bin_img, start, end, pair_idx=i // 2 + 1, stem=stem, animate=args.animate)

    print("Done!")


if __name__ == "__main__":
    main()
