"""Optic flow track, step 1: image stabilizer (stage drift removed by a translation per frame).

Same idea as the Fiji plugin Image Stabilizer in "Translation" mode, written with NumPy / SciPy / scikit-image:
  - every frame is normalized (÷ its own median × 100: the lamp level jumps, see Phase_Extraction_v2) and
    high-passed (frame - Gaussian blur of HIGHPASS_SIGMA px): the uneven illumination of the field and the dust
    of the optics do not move with the stage, they would pull the estimated shift towards zero
  - its shift against a template is measured by phase correlation (scikit-image phase_cross_correlation,
    sub-pixel to 1 / UPSAMPLE px); a Hann window tapers the image edges, which would otherwise give a false peak
  - the template is updated after every frame, as in Fiji: template = TEMPLATE_UPDATE * template
    + (1 - TEMPLATE_UPDATE) * aligned frame. It follows slow changes (focus, cells that grow) instead of
    staying stuck on the first frame
  - a shift that jumps by more than MAX_STEP px from the previous one is rejected: on a frame out of focus
    (Expo3 frame 42) the bacteria vanish and only the dust of the optics is left, which does not move with
    the stage. Such a frame does not enter the template; its shift is interpolated in time between the
    reliable frames around it, and it is flagged reliable = 0 in the CSV
  - the shift is applied to the original 8-bit frame (cubic spline). The pixels that come from outside the
    field are set to 0 (no data: the real gray levels are >= 1), so the next steps can ignore them
Limit: phase correlation uses every pixel, so many bacteria swimming the same way could pull the shift a little.
The fixed structures (debris, stuck cells) anchor it; the shift curve in the napari panel shows what was removed.

Input: the 1080p grayscale TIFF of Expo3 (video_to_tiff.py). Output (external drive, never overwritten): the
stabilized TIFF + a CSV of the shifts. If the output already exists it is loaded, not recomputed.

Run with: python 01_Stabilize.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import window
from skimage.registration import phase_cross_correlation

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_TIFF = Path("/Volumes/Extreme SSD/TIFFIle/Expo3_1080p_gray.tif")
OUTPUT_DIR = Path("/Volumes/Extreme SSD/TIFFIle/Optic_Flow")
OUTPUT_TIFF = OUTPUT_DIR / "Expo3_1_stabilized.tif"
SHIFTS_CSV = OUTPUT_DIR / "Expo3_1_shifts.csv"
HIGHPASS_SIGMA = 20.0      # px; removes the illumination pattern, keeps the bacteria (~7 px wide) and debris
UPSAMPLE = 20              # sub-pixel precision of the shift: 1 / UPSAMPLE px
TEMPLATE_UPDATE = 0.9      # weight of the old template (Fiji default 0.90); 1 = always the first frame
MAX_STEP = 8.0            # px; largest believable shift change between two frames (measured median 0.9 px)
SPLINE_ORDER = 3          # interpolation of the shifted frame (3 = cubic, least blur)

PANEL = "#262930"          # napari dark theme background


def prepare(frame: np.ndarray, taper: np.ndarray) -> np.ndarray:
    """Frame -> normalized, high-passed and edge-tapered float32 image used to measure the shift."""
    frame = frame.astype(np.float32) * np.float32(100 / np.median(frame.ravel()[::7]))
    return (frame - ndimage.gaussian_filter(frame, HIGHPASS_SIGMA)) * taper


def estimate_shifts(stack: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift (dy, dx) that aligns every frame on the running template, the registration error and a
    reliable flag (False: jump > MAX_STEP, shift interpolated in time from the reliable frames)."""
    taper = window("hann", stack.shape[1:]).astype(np.float32)
    template = prepare(stack[0], taper)
    shifts = np.zeros((len(stack), 2))
    errors = np.zeros(len(stack))
    reliable = np.ones(len(stack), dtype=bool)
    last = np.zeros(2)  # shift of the last reliable frame
    for k in range(1, len(stack)):
        moving = prepare(stack[k], taper)
        shift, errors[k], _ = phase_cross_correlation(template, moving, upsample_factor=UPSAMPLE,
                                                      normalization=None)
        shifts[k] = shift
        if np.hypot(*(shift - last)) > MAX_STEP:
            reliable[k] = False
        else:
            last = shift
            aligned = ndimage.shift(moving, shift, order=1, mode="constant", cval=0)
            template = TEMPLATE_UPDATE * template + (1 - TEMPLATE_UPDATE) * aligned
        print(f"\rshift: frame {k + 1}/{len(stack)}  dy {shift[0]:+7.2f}  dx {shift[1]:+7.2f} px"
              f"  ({(~reliable).sum()} rejected)", end="", flush=True)
    print()
    for axis in range(2):
        shifts[~reliable, axis] = np.interp(times[~reliable], times[reliable], shifts[reliable, axis])
    print(f"rejected frames (jump > {MAX_STEP} px): {np.flatnonzero(~reliable).tolist()}")
    return shifts, errors, reliable


def apply_shifts(stack: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift every original frame; pixels coming from outside the field -> 0 (no data)."""
    out = np.zeros_like(stack)
    ones = np.ones(stack.shape[1:], dtype=np.float32)
    for k, (frame, shift) in enumerate(zip(stack, shifts)):
        moved = ndimage.shift(frame.astype(np.float32), shift, order=SPLINE_ORDER, mode="nearest")
        inside = ndimage.shift(ones, shift, order=1, mode="constant", cval=0) > 0.999
        out[k] = np.where(inside, np.clip(np.rint(moved), 1, 255), 0).astype(np.uint8)
        print(f"\rapply: frame {k + 1}/{len(stack)}", end="", flush=True)
    print()
    return out


def save(stabilized: np.ndarray, shifts: np.ndarray, errors: np.ndarray, reliable: np.ndarray,
         meta: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in (OUTPUT_TIFF, SHIFTS_CSV):
        if path.exists():
            raise FileExistsError(f"not overwritten: {path}")
    times = meta["timestamps_s"]
    with open(SHIFTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "dy_px", "dx_px", "error", "reliable"])
        for k, ((dy, dx), err, ok) in enumerate(zip(shifts, errors, reliable)):
            writer.writerow([k, times[k], round(dy, 3), round(dx, 3), round(err, 5), int(ok)])
    meta = {**meta, "step": "1 stabilized (translation, phase correlation)", "highpass_sigma": HIGHPASS_SIGMA,
            "template_update": TEMPLATE_UPDATE, "max_step": MAX_STEP, "no_data_value": 0}
    tifffile.imwrite(OUTPUT_TIFF, stabilized, bigtiff=True, compression="zlib", photometric="minisblack",
                     metadata=meta)
    print(f"saved: {OUTPUT_TIFF}\nsaved: {SHIFTS_CSV}")


def load_shifts() -> np.ndarray:
    with open(SHIFTS_CSV) as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r[k]) for k in ("time_s", "dy_px", "dx_px", "reliable")] for r in rows])


def shifts_widget(viewer, table: np.ndarray):
    """Napari dock panel: dy and dx against time (rejected frames as crosses), with a marker on the frame
    shown in the viewer."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QVBoxLayout, QWidget

    t, dy, dx, reliable = table.T
    rejected = reliable == 0
    figure = Figure(figsize=(5, 3), facecolor=PANEL)
    axes = figure.add_subplot()
    axes.set_facecolor(PANEL)
    axes.tick_params(colors="#c3c2b7")
    for spine in axes.spines.values():
        spine.set_color("#5a5d66")
    axes.grid(color="#3a3d45", linewidth=0.5)
    axes.plot(t, dx, color="#3987e5", linewidth=1.2, label="dx")
    axes.plot(t, dy, color="#d95926", linewidth=1.2, label="dy")
    axes.plot(t[rejected], dx[rejected], "x", color="#f0efec", markersize=5,
              label=f"rejected ({rejected.sum()}, interpolated)")
    axes.plot(t[rejected], dy[rejected], "x", color="#f0efec", markersize=5)
    marker = axes.axvline(t[0], color="#f0efec", linewidth=0.8)
    axes.set_xlabel("time (s)", color="#c3c2b7")
    axes.set_ylabel("shift applied (px)", color="#c3c2b7")
    axes.legend(facecolor=PANEL, labelcolor="#f0efec", edgecolor="#5a5d66")
    speed = np.hypot(np.diff(dy), np.diff(dx)) / np.diff(t)
    axes.set_title(f"drift: total {np.hypot(dy[-1], dx[-1]):.1f} px, median speed {np.median(speed):.2f} px/s",
                   color="#f0efec", fontsize=10)
    figure.tight_layout()

    widget = QWidget()
    layout = QVBoxLayout(widget)
    canvas = FigureCanvasQTAgg(figure)
    layout.addWidget(NavigationToolbar2QT(canvas, widget))
    layout.addWidget(canvas)

    def follow(event=None) -> None:
        k = min(viewer.dims.current_step[0], len(t) - 1)
        marker.set_xdata([t[k], t[k]])
        canvas.draw_idle()

    viewer.dims.events.current_step.connect(follow)
    return widget


def main() -> None:
    import napari  # imported here so the functions can be used without a GUI

    with tifffile.TiffFile(INPUT_TIFF) as tif:
        original = tif.asarray()
        meta = json.loads(tif.pages[0].description)
    print(f"input: {original.shape} {original.dtype}, {meta['mean_fps']:.2f} fps (mean)")

    if OUTPUT_TIFF.exists() and SHIFTS_CSV.exists():
        print(f"output exists, loaded (delete it to recompute): {OUTPUT_TIFF}")
        stabilized = tifffile.imread(OUTPUT_TIFF)
    else:
        shifts, errors, reliable = estimate_shifts(original, np.array(meta["timestamps_s"]))
        stabilized = apply_shifts(original, shifts)
        save(stabilized, shifts, errors, reliable, meta)
    table = load_shifts()

    viewer = napari.Viewer()
    viewer.add_image(original, name="0 original", colormap="gray", contrast_limits=(0, 255), visible=False)
    viewer.add_image(stabilized, name="1 stabilized (0 = no data)", colormap="gray", contrast_limits=(0, 255))
    viewer.window.add_dock_widget(shifts_widget(viewer, table), name="shifts", area="right")
    napari.run()


if __name__ == "__main__":
    main()
