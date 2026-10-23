"""Microscopy video processing v2 (Exponential phase): no threshold, intensity is kept and measured.

Pipeline v2 (one step at a time, each step is validated before the next one is added):
  1. brightness normalization + background removal with a masked mean                  (validated)
  2. DoG filter                                                                         (validated)
  3. mean intensity and std of every shape, no threshold                                (validated)
  4. TIFF output of the whole video                                                     (postponed)

Step 1: the background is estimated from BG_SAMPLE_FRAMES frames spread evenly from FRAME_START to the end of
the video (the full video does not fit in memory).
  - normalization: the lamp brightness jumps during the videos (Stationary 163 -> 76 mean gray level, Expo
    142 <-> 190), so every frame is divided by its own median and multiplied by 100: the background is then
    ~100 in every frame and all values are in % of the frame median brightness (bright-field light is
    proportional to the lamp, so the correction is a ratio). The bacteria cover a small part of the image, so
    they barely move the median
  - pass 1: per-pixel mean over time
  - pass 2: pixels that differ from it by more than MASK_NOISE_K * noise (grown by MASK_DILATION pixels) are
            objects; the background is the mean of the remaining pixels only, so the bacteria passing over a
            pixel do not leave a dark "ghost" in it
This mask is only used to estimate the background: it is never applied to the output.
The result is frame - background (signed float, no clipping) and the intensity |frame - background|, whose
histogram (expected bimodal: background peak + bacteria tail) is drawn in a napari panel.
A line drawn in the "profile line" layer shows the intensity profile of every image in another panel
(exportable: graph as PNG / PDF / SVG, values as CSV).

Step 2: DoG (difference of Gaussians) on frame - background: blur(DOG_SIGMA_SMALL) - blur(DOG_SIGMA_SMALL *
DOG_RATIO). It keeps the structures of the bacteria size and removes the pixel noise and the slow illumination
changes. Side effect: a ring of opposite sign around every object, and the inside of a bacterium wider than the
band is flattened (the DoG values are not the bacteria intensity).
Sizes are set from the measured bacteria (Expo: width ~7 px, length ~19 px).

Step 3: shapes, no threshold.
  - A. regions where the DoG keeps the same sign (edges = DoG zero crossings, Marr-Hildreth), each one split
       around its DoG peaks by a watershed: every local maximum of |DoG| (one per PEAK_SPACING window) seeds a
       shape, so touching bacteria and the huge regions joined through the background are cut apart
  - B'. the bacteria of interest are dark (BACTERIA_SIGN = -1). The DoG puts a ring of opposite sign around every
       object; bright shapes are rings or out-of-focus bacteria (contrast inverted out of the focal plane). They
       are NOT removed: they stay in the table with matches_bacteria_sign = 0 (examples for the YOLO step)
Every shape gets its sign, position, area, the mean and std of frame - background inside it (real intensity, the
DoG values are distorted) and its DoG peak, also divided by the DoG noise std (signal-to-noise ratio). The
histogram of that ratio over the dark shapes is expected bimodal (noise vs bacteria). Table exportable as CSV.

The processing does not classify the growth stage: each video is one stage, and the Expo and Stationary scripts
run the same code; only the CONFIG differs (video, magnification-dependent sizes).

Image processing is done with NumPy / SciPy (+ the masked watershed of scikit-image, SciPy has none with a mask).
OpenCV is only used to decode the mp4 (SciPy cannot read video).

Run with: python Napari_Expo_Process_v2.py
"""

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH = "/Volumes/Extreme SSD/Video_test/Expo.mp4"
TEST_FRAME = 1700          # which frame to test on
FRAME_START = 0             # first frame used (the whole video is at x40)
BG_SAMPLE_FRAMES = 200     # frames spread evenly over the video used to estimate the background

MASK_NOISE_K = 3.0         # a pixel is an object if |frame - mean| > this * noise std (background estimate only)
MASK_DILATION = 3          # object masks are grown by this many pixels (catches faint edges)

DOG_SIGMA_SMALL = 2.0      # DoG: small blur (pixels), ~ bacteria width / (2 * sqrt(3))
DOG_RATIO = 1.6            # DoG: large blur = small blur * this ratio (1.6 = closest to a Laplacian of Gaussian)

PEAK_SPACING = 10          # shapes: one DoG peak (watershed seed) per window of this size (pixels), ~ half a bacterium length
BACTERIA_SIGN = -1         # shapes: the bacteria of interest are dark (-1); bright shapes are flagged, not removed

HIST_BINS = 256            # bins of the pixel histograms
SHAPE_HIST_BINS = 50       # bins of the shape histogram (few shapes per bin with more bins: jagged curve)
PROJECT_DIR = Path(__file__).resolve().parents[2]  # Internship/ (this file is in Video_Process_Script/Phase_Extraction_v2/)
# ---------------------------------------------------------------------------


def to_gray(frame: np.ndarray) -> np.ndarray:
    """BGR frame (as decoded) -> grayscale float32 (ITU-R 601 luma)."""
    if frame.ndim == 2:
        return frame.astype(np.float32)
    return frame[..., 2] * np.float32(0.299) + frame[..., 1] * np.float32(0.587) + frame[..., 0] * np.float32(0.114)


def frame_count(path: str) -> int:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return n


def read_frames(path: str, indices: np.ndarray) -> np.ndarray:
    """Read the given frames -> grayscale uint8 array (len(indices), height, width)."""
    capture = cv2.VideoCapture(path)
    frames = []
    for k, i in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"cannot read frame {i} of {path}")
        frames.append(np.clip(np.rint(to_gray(frame)), 0, 255).astype(np.uint8))
        print(f"\rreading frame {k + 1}/{len(indices)}", end="", flush=True)
    print()
    capture.release()
    return np.stack(frames)


def noise_std(difference: np.ndarray) -> float:
    """Robust std of the background noise of a difference image (MAD * 1.4826, insensitive to the objects)."""
    sample = difference.ravel()[::7]
    return float(1.4826 * np.median(np.abs(sample - np.median(sample))))


def normalize(frame: np.ndarray) -> np.ndarray:
    """Frame -> float32 in % of its own median brightness (background ~100 whatever the lamp level)."""
    return frame.astype(np.float32) * np.float32(100 / np.median(frame.ravel()[::7]))


def masked_mean_background(stack: np.ndarray) -> np.ndarray:
    """Mean over time of the pixels that are NOT objects (objects found against the plain mean) -> float32.

    Every frame is normalized on the fly (a normalized copy of the whole stack would not fit in memory).
    """
    levels = np.array([np.median(frame.ravel()[::7]) for frame in stack])
    print(f"frame median brightness: min {levels.min():.0f}, max {levels.max():.0f} (gray levels)")
    mean = np.zeros(stack.shape[1:], dtype=np.float64)
    for frame in stack:
        mean += normalize(frame)
    mean = (mean / len(stack)).astype(np.float32)
    noise = noise_std(normalize(stack[len(stack) // 2]) - mean)
    print(f"masked mean: noise std {noise:.2f}, object if |frame - mean| > {MASK_NOISE_K * noise:.2f}")
    total = np.zeros(mean.shape, dtype=np.float64)
    count = np.zeros(mean.shape, dtype=np.uint32)
    for k, frame in enumerate(stack):
        frame = normalize(frame)
        objects = np.abs(frame - mean) > MASK_NOISE_K * noise
        if MASK_DILATION:
            objects = ndimage.binary_dilation(objects, iterations=MASK_DILATION)
        total += np.where(objects, 0, frame)
        count += ~objects
        print(f"\rmasked mean: frame {k + 1}/{len(stack)}", end="", flush=True)
    print()
    never_free = count == 0
    print(f"masked mean: {never_free.mean():.3%} pixels are covered in every frame (plain mean kept there)")
    return np.where(never_free, mean, total / np.maximum(count, 1)).astype(np.float32)


def dog_filter(image: np.ndarray) -> np.ndarray:
    """Difference of Gaussians: blur(DOG_SIGMA_SMALL) - blur(DOG_SIGMA_SMALL * DOG_RATIO) -> float32."""
    small = ndimage.gaussian_filter(image, DOG_SIGMA_SMALL)
    large = ndimage.gaussian_filter(image, DOG_SIGMA_SMALL * DOG_RATIO)
    return (small - large).astype(np.float32)


COLORS = {"frame - background": "#3987e5", "DoG": "#d95926", "shapes: DoG peak / noise std": "#199e70"}  # one fixed color per image
PANEL = "#262930"          # napari dark theme background


def split_sign_region(dog: np.ndarray, sign: int) -> np.ndarray:
    """Pixels where the DoG has `sign` -> label image (0 elsewhere), each region split around its |DoG| peaks.

    Seeds: local maxima of |DoG| inside the region (one per PEAK_SPACING window), plus the maximum of any region
    that got no seed. The watershed is masked to the region, so a shape never crosses a zero crossing.
    """
    inside = np.sign(dog) == sign
    strength = np.where(inside, np.abs(dog), 0)
    seeds, n_seeds = ndimage.label(inside & (strength == ndimage.maximum_filter(strength, PEAK_SPACING)))
    regions, n_regions = ndimage.label(inside)
    missing = np.setdiff1d(np.arange(1, n_regions + 1), regions[seeds > 0])
    for k, position in enumerate(ndimage.maximum_position(strength, regions, index=missing)):
        seeds[position] = n_seeds + k + 1
    return watershed(-strength, seeds, mask=inside)               # floods from the peaks down to the zero crossings


def dog_regions(dog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DoG -> (label image, sign per label): dark (DoG < 0) and bright (DoG > 0) shapes, split around their peaks.

    Labels are renumbered 1..n with no gap; sign[0] = 0 (label 0 = pixels with DoG exactly 0, none in practice).
    """
    labels = np.zeros(dog.shape, dtype=np.int32)
    sign = [0]
    for s in (-1, 1):
        part = split_sign_region(dog, s)
        present = np.unique(part[part > 0])
        lookup = np.zeros(part.max() + 1, dtype=np.int32)
        lookup[present] = np.arange(len(sign), len(sign) + len(present))
        labels[part > 0] = lookup[part[part > 0]]
        sign += [s] * len(present)
    return labels, np.array(sign, dtype=np.int8)


def measure_shapes(labels: np.ndarray, sign: np.ndarray, difference: np.ndarray, dog: np.ndarray) -> dict:
    """Per-shape measures -> {column: array indexed by label}, row 0 = no shape.

    mean / std are measured on frame - background; dog_peak is the DoG value of largest magnitude in the shape
    (same sign as the shape), dog_snr = |dog_peak| / DoG noise std.
    """
    n = len(sign)
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n).astype(np.float64)
    total = np.bincount(flat, weights=difference.ravel(), minlength=n)
    squares = np.bincount(flat, weights=difference.ravel().astype(np.float64) ** 2, minlength=n)
    safe = np.maximum(area, 1)
    mean = total / safe
    std = np.sqrt(np.maximum(squares / safe - mean ** 2, 0))
    peak = sign * ndimage.maximum(np.abs(dog), labels, index=np.arange(n))
    rows, cols = np.indices(labels.shape)
    return {"label": np.arange(n), "sign": sign,
            "row": np.bincount(flat, weights=rows.ravel(), minlength=n) / safe,
            "col": np.bincount(flat, weights=cols.ravel(), minlength=n) / safe,
            "area": area, "mean": mean, "std": std, "dog_peak": peak, "dog_snr": np.abs(peak) / noise_std(dog),
            "matches_bacteria_sign": (sign == BACTERIA_SIGN).astype(np.int8)}


def shapes_widget(shapes: dict):
    """Napari dock panel: button to export the shape table (one row per shape) as CSV."""
    from qtpy.QtWidgets import QFileDialog, QPushButton

    button = QPushButton("export shapes CSV")

    def export_csv() -> None:
        default = PROJECT_DIR / "Processing_Results" / f"{Path(VIDEO_PATH).stem}_frame{TEST_FRAME}_shapes.csv"
        path, _ = QFileDialog.getSaveFileName(button, "export shapes", str(default), "CSV (*.csv)")
        if path:  # the dialog asks before replacing an existing file
            np.savetxt(path, np.column_stack([v[1:] for v in shapes.values()]), delimiter=",", fmt="%.6g",
                       header=",".join(shapes), comments="")
            print(f"shapes saved: {path}")

    button.clicked.connect(export_csv)
    return button


def style_axes(axes) -> None:
    """Dark style matching the napari theme."""
    axes.set_facecolor(PANEL)
    axes.tick_params(colors="#c3c2b7")
    for spine in axes.spines.values():
        spine.set_color("#5a5d66")
    axes.grid(color="#3a3d45", linewidth=0.5)


def figure_panel(figure, extra_buttons=()):
    """Matplotlib figure -> Qt widget with the matplotlib toolbar (its save button exports PNG / PDF / SVG)."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from qtpy.QtWidgets import QVBoxLayout, QWidget

    widget = QWidget()
    layout = QVBoxLayout(widget)
    canvas = FigureCanvasQTAgg(figure)
    layout.addWidget(NavigationToolbar2QT(canvas, widget))
    layout.addWidget(canvas)
    for button in extra_buttons:
        layout.addWidget(button)
    return widget, canvas


def histogram_widget(images: dict):
    """Napari dock panel: one histogram of |value| per image (log scale), one panel per image."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(5, 2.2 * len(images)), facecolor=PANEL)
    for k, (name, image) in enumerate(images.items()):
        values = np.abs(image)
        bins = HIST_BINS if image.ndim == 2 else SHAPE_HIST_BINS  # 1D = one value per shape
        edges = np.linspace(0, np.percentile(values, 99.9), bins + 1)
        counts, _ = np.histogram(values, bins=edges)
        axes = figure.add_subplot(len(images), 1, k + 1)
        style_axes(axes)
        axes.stairs(counts, edges, color=COLORS[name], linewidth=1.5)
        axes.set_yscale("log")
        axes.set_ylabel("pixels (log)", color="#c3c2b7")
        axes.set_title(f"|{name}|", color="#f0efec", fontsize=10)
    axes.set_xlabel("value (% of median brightness, or x noise std for the shapes)", color="#c3c2b7")
    figure.tight_layout()
    return figure_panel(figure)[0]


def sample_line(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Values of `image` along the segment start -> end (row, col), one sample per pixel -> (distance, values)."""
    length = float(np.hypot(*(end - start)))
    t = np.linspace(0, 1, max(int(round(length)) + 1, 2))
    coords = start[:, None] + (end - start)[:, None] * t
    return t * length, ndimage.map_coordinates(image, coords, order=1, mode="nearest")


def profile_widget(viewer, images: dict):
    """Napari dock panel: intensity profile of every image along a line drawn in the "profile line" layer.

    Each profile is divided by the noise std of its own image, so the curves share one axis and read
    directly as a signal-to-noise ratio (the DoG values are ~10x smaller). Export: the toolbar
    save button (PNG / PDF / SVG) for the graph, the "export CSV" button for the values.
    """
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QFileDialog, QPushButton

    noise = {name: noise_std(image) for name, image in images.items()}
    figure = Figure(figsize=(5, 3.5), facecolor=PANEL)
    axes = figure.add_subplot()
    csv_button = QPushButton("export CSV")
    widget, canvas = figure_panel(figure, [csv_button])
    last = {}  # profile currently shown, for the CSV export

    def draw(event=None) -> None:
        axes.clear()
        style_axes(axes)
        axes.set_xlabel("distance along the line (pixels)", color="#c3c2b7")
        axes.set_ylabel("value / noise std", color="#c3c2b7")
        if len(lines.data):
            start, end = np.asarray(lines.data[-1][0])[-2:], np.asarray(lines.data[-1][-1])[-2:]
            last.clear()
            for name, image in images.items():
                distance, values = sample_line(image, start, end)
                last["distance (px)"] = distance
                last[f"{name} (% brightness)"] = values
                axes.plot(distance, values / noise[name], color=COLORS[name], linewidth=2,
                          label=f"{name} (noise std {noise[name]:.2f})")
            axes.axhline(0, color="#8a8d96", linewidth=0.8)
            axes.set_title(f"profile ({start[0]:.0f}, {start[1]:.0f}) -> ({end[0]:.0f}, {end[1]:.0f})",
                           color="#f0efec", fontsize=10)
            legend = axes.legend(fontsize=8, facecolor=PANEL, edgecolor="#5a5d66")
            for text in legend.get_texts():
                text.set_color("#f0efec")
        else:
            axes.set_title("draw a line in the 'profile line' layer", color="#f0efec", fontsize=10)
        figure.tight_layout()
        canvas.draw_idle()

    def export_csv() -> None:
        if not last:
            return
        default = PROJECT_DIR / "Processing_Results" / f"{Path(VIDEO_PATH).stem}_frame{TEST_FRAME}_profile.csv"
        path, _ = QFileDialog.getSaveFileName(widget, "export profile", str(default), "CSV (*.csv)")
        if path:  # the dialog asks before replacing an existing file
            np.savetxt(path, np.column_stack(list(last.values())), delimiter=",", fmt="%.4f",
                       header=",".join(last), comments="")
            print(f"profile saved: {path}")

    lines = viewer.add_shapes(ndim=2, shape_type="line", name="profile line", edge_color="cyan", edge_width=2)
    lines.mode = "add_line"
    lines.events.data.connect(draw)
    csv_button.clicked.connect(export_csv)
    draw()
    return widget


def main() -> None:
    import napari  # imported here so the functions can be used without a GUI

    n = frame_count(VIDEO_PATH)
    if not FRAME_START <= TEST_FRAME < n:
        raise ValueError(f"TEST_FRAME must be between FRAME_START ({FRAME_START}) and {n - 1}")
    indices = np.linspace(FRAME_START, n - 1, BG_SAMPLE_FRAMES).astype(int)
    print(f"video: {n} frames, background from {len(indices)} frames (one every {n / len(indices):.0f})")
    background = masked_mean_background(read_frames(VIDEO_PATH, indices))
    frame = read_frames(VIDEO_PATH, np.array([TEST_FRAME]))[0]
    difference = normalize(frame) - background  # signed: darker AND lighter objects are kept
    intensity = np.abs(difference)
    print(f"noise std of frame - background: {noise_std(difference):.2f}")
    dog = dog_filter(difference)

    span = np.percentile(intensity, 99.9)
    viewer = napari.Viewer()
    viewer.add_image(frame, name="original (grayscale, not normalized)", colormap="gray", visible=False)
    viewer.add_image(background, name="background (masked mean, % brightness)", colormap="gray", visible=False)
    viewer.add_image(difference, name="1 frame - background", colormap="gray", contrast_limits=(-span, span),
                     visible=False)
    viewer.add_image(intensity, name="1 intensity |frame - background|", colormap="magma",
                     contrast_limits=(0, span))
    limit = np.percentile(np.abs(dog), 99.9)
    viewer.add_image(dog, name="2 DoG (signed)", colormap="gray", contrast_limits=(-limit, limit), visible=False)
    viewer.add_image(np.abs(dog), name="2 DoG |intensity|", colormap="magma", contrast_limits=(0, limit))
    labels, sign = dog_regions(dog)
    shapes = measure_shapes(labels, sign, difference, dog)
    bacteria = shapes["matches_bacteria_sign"].astype(bool)
    snr = shapes["dog_snr"][bacteria]
    print(f"shapes: {len(sign) - 1} ({bacteria.sum()} with the bacteria sign, {(~bacteria[1:]).sum()} opposite), "
          f"bacteria-sign DoG peak / noise std: median {np.median(snr):.1f}, p99 {np.percentile(snr, 99):.1f}, "
          f"max {snr.max():.1f}, largest area {shapes['area'][bacteria].max():.0f} px")
    features = {name: values for name, values in shapes.items() if name != "label"}
    features["index"] = shapes["label"]
    for name, colormap in [("mean", "twilight_shifted"), ("std", "magma"), ("dog_snr", "magma")]:
        painted = np.where(bacteria[labels], shapes[name][labels], 0).astype(np.float32)
        limit = np.percentile(np.abs(painted), 99.9)
        viewer.add_image(painted, name=f"3 shape {name} (bacteria sign)", colormap=colormap, visible=False,
                         contrast_limits=(-limit, limit) if name == "mean" else (0, limit))
    # opposite sign colored by signal-to-noise: the noise shapes stay dark, the out-of-focus bacteria light up
    painted = np.where(bacteria[labels], 0, shapes["dog_snr"][labels]).astype(np.float32)
    viewer.add_image(painted, name="3 shape dog_snr (opposite sign)", colormap="magma", visible=False,
                     contrast_limits=(0, np.percentile(painted, 99.9)))
    # outlines only, to be laid over "1 frame - background" (hover shows the shape measures)
    for name, keep, visible in [("bacteria sign (dark)", bacteria, True),
                                ("opposite sign (rings, out of focus)", ~bacteria, False)]:
        layer = viewer.add_labels(np.where(keep[labels], labels, 0), name=f"3 shapes, {name}", features=features,
                                  visible=visible)
        layer.contour = 2
    images = {"frame - background": difference, "DoG": dog}
    histograms = {**images, "shapes: DoG peak / noise std": snr}
    viewer.window.add_dock_widget(histogram_widget(histograms), name="bimodal histograms", area="right")
    viewer.window.add_dock_widget(shapes_widget(shapes), name="shapes table", area="right")
    viewer.window.add_dock_widget(profile_widget(viewer, images), name="intensity profile", area="right")
    napari.run()


if __name__ == "__main__":
    main()
