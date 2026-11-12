"""Optic flow track, step 3: bacteria detection with trackpy (tp.locate), first part of the tracking.

Input: frame - background of step 2 (rolling masked mean, one stack per stable segment, % of brightness,
signed). The bacteria of interest are dark (< 0): the frame is inverted (-diff) so they become bright spots,
which is what trackpy looks for. Bright shapes (out-of-focus cells, halos) become negative and are ignored.

tp.locate, on every frame of the segment:
  - band-pass (noise_size 1 px, background scale DIAMETER px), then local maxima at least SEPARATION px apart
  - each maximum gives a feature: sub-pixel centroid (y, x) over a disk of DIAMETER px, its integrated
    brightness "mass", its radius of gyration "size", its eccentricity "ecc", its peak "signal"
  - DIAMETER = 9 px (odd, a bit wider than a bacterium: ~7 px at 1080p). Pilot (5 frames per segment,
    2026-11-09): 7 px splits a cell into fragments (9000 features / frame), 11 px merges neighbors
Most maxima are noise: the threshold is on the mass. Its floor is measured on synthetic noise with the same
std and the same spatial correlation as the frames (neighbor-pixel correlation 0.85, from the 1080p
downscale): the largest noise mass is ~44 x the noise std (25 in segment 1, 35 in segment 0). Features are
saved down to SAVE_MASS_PER_NOISE x noise (below the floor, so the noise can be seen in napari) and the
napari panel sets the real threshold with a slider, starting at MINMASS_PER_NOISE x noise.
The rod shape is not modelled: the centroid of a rod is its center, which is enough to link positions.
The frames flagged unreliable by step 1 (out of focus) are detected too but flagged in the CSV.

Output (external drive, never overwritten): one CSV of features per segment. If it exists it is loaded.
napari shows SHOW_SEGMENT: the frame - background, the features above the threshold (color = mass), and a
dock panel with the mass histogram (noise floor, threshold) and the number of features per frame.

Run with: python 03_Trackpy.py
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import trackpy as tp

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = Path("/Volumes/Extreme SSD/TIFFIle/Optic_Flow")
SEGMENTS = [(0, 386), (415, 962)]  # same as step 2
SHOW_SEGMENT = 1                   # segment opened in napari (index in SEGMENTS)
DIAMETER = 9                       # px, odd; a bit wider than a bacterium (~7 px)
SEPARATION = 11                     # px, smallest distance between two features
NOISE_MASS_PER_NOISE = 44.0        # largest mass found on synthetic correlated noise, / noise std (pilot)
SAVE_MASS_PER_NOISE = 30.0         # features saved above this mass / noise std (below the noise floor)
MINMASS_PER_NOISE = 60.0           # starting threshold of the napari slider, / noise std

PANEL = "#262930"                  # napari dark theme background
COLUMNS = ["frame", "time_s", "reliable", "y", "x", "mass", "size", "ecc", "signal"]

tp.quiet()


def segment_name(segment: int) -> str:
    start, end = SEGMENTS[segment]
    return f"Expo3_2_seg{segment}_{start}-{end}"


def features_csv(segment: int) -> Path:
    start, end = SEGMENTS[segment]
    return DATA_DIR / f"Expo3_3_seg{segment}_{start}-{end}_features.csv"


def lazy(path: Path):
    """Stack read on demand (only the frames used are decoded)."""
    import zarr
    return zarr.open(tifffile.imread(path, aszarr=True), mode="r")


def read_stats(segment: int) -> dict:
    with open(DATA_DIR / f"{segment_name(segment)}_stats.csv") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in ("frame", "time_s", "reliable", "rolling_noise")}


def segment_noise(stats: dict) -> float:
    """One noise std per segment (median over the reliable frames): one mass threshold for the whole segment."""
    return float(np.median(stats["rolling_noise"][stats["reliable"] == 1]))


def detect(segment: int) -> pd.DataFrame:
    """tp.locate on every inverted frame of the segment -> features above SAVE_MASS_PER_NOISE x noise."""
    path = features_csv(segment)
    if path.exists():
        print(f"features exist, loaded (delete them to recompute): {path}")
        return pd.read_csv(path)
    stack = lazy(DATA_DIR / f"{segment_name(segment)}_diff_rolling.tif")
    stats = read_stats(segment)
    minmass = SAVE_MASS_PER_NOISE * segment_noise(stats)
    found = []
    for k in range(stack.shape[0]):
        f = tp.locate(-np.asarray(stack[k]), DIAMETER, minmass=minmass, separation=SEPARATION)
        f["frame"], f["time_s"], f["reliable"] = int(stats["frame"][k]), stats["time_s"][k], int(stats["reliable"][k])
        found.append(f)
        print(f"\rsegment {segment}: frame {k + 1}/{stack.shape[0]}, {len(f)} features", end="", flush=True)
    features = pd.concat(found, ignore_index=True)[COLUMNS].round(
        {"y": 3, "x": 3, "mass": 2, "size": 3, "ecc": 3, "signal": 3})
    features.to_csv(path, index=False)
    print(f"\nsaved: {path}")
    return features


def threshold_widget(viewer, layer, features: pd.DataFrame, noise: float, start: int):
    """Napari dock panel: a mass threshold slider (shows / hides the features), the mass histogram with the
    noise floor and the threshold, and the number of features per frame against time."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget

    mass = features["mass"].to_numpy()
    frames = features["frame"].to_numpy()
    all_frames = np.arange(frames.min(), frames.max() + 1)
    times = features.groupby("frame")["time_s"].first().reindex(all_frames).interpolate().to_numpy()
    floor = NOISE_MASS_PER_NOISE * noise

    figure = Figure(figsize=(5, 5), facecolor=PANEL)
    hist_axes, count_axes = figure.add_subplot(2, 1, 1), figure.add_subplot(2, 1, 2)
    for axes in (hist_axes, count_axes):
        axes.set_facecolor(PANEL)
        axes.tick_params(colors="#c3c2b7", labelsize=7)
        for spine in axes.spines.values():
            spine.set_color("#5a5d66")
        axes.grid(color="#3a3d45", linewidth=0.5)
    bins = np.geomspace(mass.min(), mass.max(), 80)
    hist_axes.hist(mass, bins=bins, color="#3987e5")
    hist_axes.set_xscale("log")
    hist_axes.set_yscale("log")
    hist_axes.axvline(floor, color="#c3c2b7", linestyle="--", linewidth=0.8, label=f"noise floor {floor:.0f}")
    line = hist_axes.axvline(floor, color="#d95926", linewidth=1.2, label="threshold")
    hist_axes.set_xlabel("mass (% x px)", color="#c3c2b7", fontsize=8)
    hist_axes.set_ylabel("features", color="#c3c2b7", fontsize=8)
    hist_axes.legend(facecolor=PANEL, labelcolor="#f0efec", edgecolor="#5a5d66", fontsize=7)
    (curve,) = count_axes.plot(times, np.zeros_like(times), color="#d95926", linewidth=1.0)
    marker = count_axes.axvline(times[0], color="#f0efec", linewidth=0.8)
    count_axes.set_xlabel("time (s)", color="#c3c2b7", fontsize=8)
    count_axes.set_ylabel("features kept / frame", color="#c3c2b7", fontsize=8)
    figure.tight_layout()

    widget = QWidget()
    box = QVBoxLayout(widget)
    label = QLabel()
    label.setStyleSheet("color: #f0efec")
    slider = QSlider(Qt.Horizontal)
    slider.setRange(int(mass.min()), int(np.percentile(mass, 99.5)))
    canvas = FigureCanvasQTAgg(figure)
    for item in (label, slider, canvas):
        box.addWidget(item)

    def apply(value: int) -> None:
        kept = mass >= value
        layer.shown = kept
        counts = np.bincount(frames[kept] - all_frames[0], minlength=len(all_frames))
        curve.set_ydata(counts)
        count_axes.set_ylim(0, max(1, counts.max()) * 1.1)
        line.set_xdata([value] * 2)
        label.setText(f"mass threshold {value} ({value / noise:.0f} x noise std, floor {floor:.0f}): "
                      f"{counts.mean():.0f} features / frame")
        canvas.draw_idle()

    def follow(event=None) -> None:
        k = min(viewer.dims.current_step[0], len(times) - 1)
        marker.set_xdata([times[k]] * 2)
        canvas.draw_idle()

    slider.valueChanged.connect(apply)
    viewer.dims.events.current_step.connect(follow)
    slider.setValue(int(MINMASS_PER_NOISE * noise))
    apply(slider.value())
    return widget


def main() -> None:
    import napari  # imported here so the functions can be used without a GUI

    for segment in range(len(SEGMENTS)):
        detect(segment)

    start, end = SEGMENTS[SHOW_SEGMENT]
    stats = read_stats(SHOW_SEGMENT)
    noise = segment_noise(stats)
    features = pd.read_csv(features_csv(SHOW_SEGMENT))
    span = 4 * noise
    viewer = napari.Viewer(title=f"Expo3 segment {SHOW_SEGMENT}: frames {start}-{end} (slider 0 = frame {start})")
    viewer.add_image(lazy(DATA_DIR / f"{segment_name(SHOW_SEGMENT)}_diff_rolling.tif"),
                     name="frame - background (rolling mean)", colormap="gray", contrast_limits=(-span, span))
    points = np.stack([features["frame"] - start, features["y"], features["x"]], axis=1)
    layer = viewer.add_points(points, name=f"trackpy features (diameter {DIAMETER} px, color = mass)",
                              size=DIAMETER, symbol="ring", border_width=0.15,
                              features=features[["mass", "size", "ecc", "signal"]], face_color="transparent",
                              border_color="mass", border_colormap="plasma",
                              border_contrast_limits=tuple(np.percentile(features["mass"], [5, 99])))
    viewer.window.add_dock_widget(threshold_widget(viewer, layer, features, noise, start), name="mass threshold",
                                  area="right")
    napari.run()


if __name__ == "__main__":
    main()
