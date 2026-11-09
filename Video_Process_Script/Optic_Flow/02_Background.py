"""Optic flow track, step 2: background removal on the stabilized Expo3 by a rolling masked mean.

Input: the output of 01_Stabilize.py (stabilized TIFF, 0 = no data, + the shifts CSV with the reliable flag).
The video is cut into SEGMENTS, the stable passages processed separately (frames 387-414 show another field and
focal plane: the operator moved the stage and came back). Inside a segment, the frames flagged unreliable by
step 1 (out of focus) are kept in the output but never used to estimate a background.

Every frame is normalized as in Phase_Extraction_v2: ÷ its own median (over the pixels with data) × 100, so all
values are in % of the frame brightness whatever the lamp level. Then, per segment:
  - object masks, as in the Phase_Extraction_v2 masked mean: the pixels that differ from the per-pixel mean
    over the segment by more than MASK_NOISE_K * noise (grown by MASK_DILATION px) are objects
  - rolling masked mean (chosen, 2026-11-09): the background of a frame is the mean of the object-free pixels
    over the ROLLING_WINDOW frames centered on it (window cut at the segment ends). It follows slow changes
    (focus drift, light, cells that settle) but erases a bacterium that stays still for more than about half
    the window. Where a pixel is never free inside the window, the masked mean of the whole segment is used
Compared with one masked mean per segment (v2), the residual noise drops from 1.31 to 0.80 % (segment 0, fast
drift and focus changes) and from 0.69 to 0.58 % (segment 1); see the method log.
Output: frame - background (float32, % of brightness, signed: dark bacteria < 0). The no-data pixels of the
stabilized frame are 0 (neutral) there, so the next filters do not have to handle NaN.
Per frame, a CSV gives the noise std (MAD), the mean residual offset and the share of pixels beyond 3 noise std
(what is left as "objects"); napari shows them as curves.

Outputs (external drive, never overwritten): if they already exist they are loaded, not recomputed. The big
stacks are opened lazily (zarr), only the frames shown are read.

Run with: python 02_Background.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = Path("/Volumes/Extreme SSD/TIFFIle/Optic_Flow")
STABILIZED_TIFF = DATA_DIR / "Expo3_1_stabilized.tif"
SHIFTS_CSV = DATA_DIR / "Expo3_1_shifts.csv"
SEGMENTS = [(0, 386), (415, 962)]  # stable passages, first and last frame included
SHOW_SEGMENT = 1                   # segment opened in napari (index in SEGMENTS)
MASK_NOISE_K = 3.0                 # a pixel is an object if |frame - mean| > this * noise std
MASK_DILATION = 3                  # object masks are grown by this many pixels (catches faint edges)
ROLLING_WINDOW = 51                # frames in the rolling mean (~14 s at 3.6 fps), odd

PANEL = "#262930"                  # napari dark theme background


def segment_name(segment: int) -> str:
    start, end = SEGMENTS[segment]
    return f"Expo3_2_seg{segment}_{start}-{end}"


def output_paths(segment: int) -> dict:
    name = segment_name(segment)
    paths = {"diff": DATA_DIR / f"{name}_diff_rolling.tif"}
    paths["background"] = DATA_DIR / f"{name}_background_masked_mean.tif"  # fallback of the rolling mean
    paths["stats"] = DATA_DIR / f"{name}_stats.csv"
    return paths


def noise_std(values: np.ndarray) -> float:
    """Robust std of the background noise (MAD * 1.4826, insensitive to the objects)."""
    sample = values.ravel()[::7]
    return float(1.4826 * np.median(np.abs(sample - np.median(sample))))


def normalize(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stabilized frame -> (float32 in % of its median brightness, pixels with data)."""
    valid = frame > 0
    scale = np.float32(100 / np.median(frame[valid][::7]))
    return frame.astype(np.float32) * scale, valid


class Segment:
    """Frames of one segment, their reliable flag, and the per-frame object-free masks."""

    def __init__(self, stack: np.ndarray, reliable: np.ndarray):
        self.stack, self.reliable = stack, reliable
        self.mean, self.has_data = self.plain_mean()
        mid = len(stack) // 2
        frame, valid = normalize(stack[mid])
        self.noise = noise_std((frame - self.mean)[valid])
        print(f"  noise std {self.noise:.2f} %, object if |frame - mean| > {MASK_NOISE_K * self.noise:.2f} %")
        self.free = np.zeros(stack.shape, dtype=bool)  # data and no object (only for reliable frames)
        for k in np.flatnonzero(reliable):
            frame, valid = normalize(stack[k])
            objects = np.abs(frame - self.mean) > MASK_NOISE_K * self.noise
            if MASK_DILATION:
                objects = ndimage.binary_dilation(objects, iterations=MASK_DILATION)
            self.free[k] = valid & ~objects
            print(f"\r  object masks: frame {k + 1}/{len(stack)}", end="", flush=True)
        print()

    def plain_mean(self) -> np.ndarray:
        total = np.zeros(self.stack.shape[1:])
        count = np.zeros(self.stack.shape[1:])
        for k in np.flatnonzero(self.reliable):
            frame, valid = normalize(self.stack[k])
            total += np.where(valid, frame, 0)
            count += valid
        return np.where(count > 0, total / np.maximum(count, 1), 100).astype(np.float32), count > 0

    def add(self, total: np.ndarray, count: np.ndarray, k: int, sign: int) -> None:
        """Add (sign +1) or remove (sign -1) the object-free pixels of frame k from running sums."""
        if self.reliable[k]:
            frame, _ = normalize(self.stack[k])
            total += sign * np.where(self.free[k], frame, 0)
            count += sign * self.free[k]

    def masked_mean(self) -> np.ndarray:
        total = np.zeros(self.stack.shape[1:])
        count = np.zeros(self.stack.shape[1:])
        for k in range(len(self.stack)):
            self.add(total, count, k, +1)
        covered = count == 0
        print(f"  masked mean: {(~self.has_data).mean():.2%} pixels without data in every frame (stabilization "
              f"border), {(covered & self.has_data).sum() / self.has_data.sum():.3%} of the others never free "
              f"(plain mean kept there)")
        return np.where(covered, self.mean, total / np.maximum(count, 1)).astype(np.float32)

    def rolling_backgrounds(self, fallback: np.ndarray):
        """Yield the rolling masked-mean background of every frame (running sums: each frame read twice)."""
        half = ROLLING_WINDOW // 2
        n = len(self.stack)
        total = np.zeros(self.stack.shape[1:])
        count = np.zeros(self.stack.shape[1:])
        for k in range(min(half, n - 1) + 1):
            self.add(total, count, k, +1)
        for k in range(n):
            yield np.where(count > 0, total / np.maximum(count, 1), fallback).astype(np.float32)
            if k + half + 1 < n:
                self.add(total, count, k + half + 1, +1)
            if k - half >= 0:
                self.add(total, count, k - half, -1)


def frame_stats(diff: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    values = diff[valid]
    noise = noise_std(values)
    return noise, float(np.median(values)), float((np.abs(values) > 3 * noise).mean())


def process(segment: int, stabilized: np.ndarray, reliable_all: np.ndarray, times: np.ndarray,
            meta: dict) -> None:
    paths = output_paths(segment)
    existing = [p for p in paths.values() if p.exists()]
    if existing:
        raise FileExistsError(f"not overwritten (delete all the outputs of this segment to recompute): {existing}")
    start, end = SEGMENTS[segment]
    print(f"segment {segment}: frames {start}-{end} ({end - start + 1} frames, "
          f"{(~reliable_all[start:end + 1]).sum()} unreliable kept in output, not used for the background)")
    seg = Segment(stabilized[start:end + 1], reliable_all[start:end + 1])
    background = seg.masked_mean()
    n, shape = len(seg.stack), seg.stack.shape
    stats = np.zeros((n, 3))

    def diffs():
        for k, bg in enumerate(seg.rolling_backgrounds(background)):
            frame, valid = normalize(seg.stack[k])
            diff = np.where(valid, frame - bg, 0).astype(np.float32)
            stats[k] = frame_stats(diff, valid)
            print(f"\r  rolling mean: frame {k + 1}/{n}", end="", flush=True)
            yield diff

    meta = {**meta, "step": "2 frame - background (% of frame median brightness)", "segment": [start, end],
            "mask_noise_k": MASK_NOISE_K, "mask_dilation": MASK_DILATION, "rolling_window": ROLLING_WINDOW}
    tifffile.imwrite(paths["background"], background, compression="zlib", metadata=meta)
    with tifffile.TiffWriter(paths["diff"], bigtiff=True) as tif:
        tif.write(diffs(), shape=shape, dtype=np.float32, compression="zlib", photometric="minisblack",
                  metadata={**meta, "method": f"rolling masked mean ({ROLLING_WINDOW} frames)"})
    print()  # after the write: the writer stops pulling frames before the generator could print it
    with open(paths["stats"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "reliable", "rolling_noise", "rolling_offset", "rolling_object_share"])
        for k in range(n):
            writer.writerow([start + k, times[start + k], int(seg.reliable[k])]
                            + [round(float(v), 5) for v in stats[k]])
    ok = seg.reliable
    print(f"  noise median {np.median(stats[ok, 0]):.2f} %, |offset| median {np.median(np.abs(stats[ok, 1])):.2f} %,"
          f" objects median {np.median(stats[ok, 2]):.2%}")
    for p in paths.values():
        print(f"  saved: {p}")


def lazy(path: Path):
    """Stack read on demand (only the frames shown in napari are decoded)."""
    import zarr
    return zarr.open(tifffile.imread(path, aszarr=True), mode="r")


def stats_widget(viewer, table: dict):
    """Napari dock panel: per-frame noise, offset and object share, marker on the shown frame."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QVBoxLayout, QWidget

    t = table["time_s"]
    figure = Figure(figsize=(5, 6), facecolor=PANEL)
    markers = []
    for row, (stat, label) in enumerate([("noise", "noise std (%)"), ("offset", "median offset (%)"),
                                         ("object_share", "pixels > 3 noise std")]):
        axes = figure.add_subplot(3, 1, row + 1)
        axes.set_facecolor(PANEL)
        axes.tick_params(colors="#c3c2b7")
        for spine in axes.spines.values():
            spine.set_color("#5a5d66")
        axes.grid(color="#3a3d45", linewidth=0.5)
        axes.plot(t, table[f"rolling_{stat}"], color="#d95926", linewidth=1.0)
        bad = table["reliable"] == 0
        axes.plot(t[bad], table[f"rolling_{stat}"][bad], "x", color="#f0efec", markersize=4,
                  label="unreliable frame" if row == 0 else None)
        axes.set_ylabel(label, color="#c3c2b7", fontsize=8)
        markers.append(axes.axvline(t[0], color="#f0efec", linewidth=0.8))
        if row == 0:
            axes.legend(facecolor=PANEL, labelcolor="#f0efec", edgecolor="#5a5d66", fontsize=7)
    axes.set_xlabel("time (s)", color="#c3c2b7")
    figure.tight_layout()

    widget = QWidget()
    layout = QVBoxLayout(widget)
    canvas = FigureCanvasQTAgg(figure)
    layout.addWidget(NavigationToolbar2QT(canvas, widget))
    layout.addWidget(canvas)

    def follow(event=None) -> None:
        k = min(viewer.dims.current_step[0], len(t) - 1)
        for marker in markers:
            marker.set_xdata([t[k], t[k]])
        canvas.draw_idle()

    viewer.dims.events.current_step.connect(follow)
    return widget


def main() -> None:
    import napari  # imported here so the functions can be used without a GUI

    with open(SHIFTS_CSV) as f:
        rows = list(csv.DictReader(f))
    reliable = np.array([r["reliable"] == "1" for r in rows])
    times = np.array([float(r["time_s"]) for r in rows])
    stabilized = None
    for segment in range(len(SEGMENTS)):
        if not all(p.exists() for p in output_paths(segment).values()):
            if stabilized is None:
                with tifffile.TiffFile(STABILIZED_TIFF) as tif:
                    stabilized = tif.asarray()
                    # "shape" is the stabilized stack's (written by tifffile): not the shape of the new files
                    meta = {k: v for k, v in json.loads(tif.pages[0].description).items() if k != "shape"}
            process(segment, stabilized, reliable, times, meta)
    del stabilized

    paths = output_paths(SHOW_SEGMENT)
    start, end = SEGMENTS[SHOW_SEGMENT]
    with open(paths["stats"]) as f:
        rows = list(csv.DictReader(f))
    table = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
    span = 4 * np.median(table["rolling_noise"])

    viewer = napari.Viewer(title=f"Expo3 segment {SHOW_SEGMENT}: frames {start}-{end} (slider 0 = frame {start})")
    viewer.add_image(lazy(STABILIZED_TIFF)[start:end + 1], name="1 stabilized", colormap="gray",
                     contrast_limits=(0, 255), visible=False)
    viewer.add_image(tifffile.imread(paths["background"]), name="2 fallback background (segment masked mean, %)",
                     colormap="gray", visible=False)
    viewer.add_image(lazy(paths["diff"]), name=f"2 frame - background (rolling mean, {ROLLING_WINDOW} frames)",
                     colormap="gray", contrast_limits=(-span, span))
    viewer.window.add_dock_widget(stats_widget(viewer, table), name="background stats", area="right")
    napari.run()


if __name__ == "__main__":
    main()
