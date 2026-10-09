"""Microscopy video processing: keep only the in-focus bacteria, shown in black and white.

Pipeline (one step at a time, each step is validated before the next one is added):
  1. background removal with a median filter           (validated)
  2. remove round objects, min-area rectangle elongation (validated)
  3. remove out-of-focus bacteria                      (validated)
       - hole ratio: object area / area inside its outer contour
       - two-threshold area ratio: area at the strict threshold / area at a lower threshold. Out-of-focus hollow
         bacteria show a faint interior that only passes the lower threshold, so the two rims join into one big
         object with a low ratio and are removed
  4. black and white output: plain mask of the kept objects, bacteria in black on a white background  (validated)

Step 1: the median of every pixel over ALL frames of the video is the background (bacteria move, so they
are not in the median). That background is subtracted from every frame. The result is a mid-gray image
(background = 128) where objects are darker or lighter than the gray.

Run with: python Napari_Stationary_Process.py
"""

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH = "/Users/juwul/Desktop/Internship/Videosplit/Stationary/Stationary2_split_152.mp4"   # path to the microscopy video
MEDIAN_ROW_CHUNK = 64      # image rows processed at once when computing the median (lower = less memory)
GRAY_LEVEL = 128           # gray level the background gets after subtraction (0-255)
TEST_FRAME = 14            # which frame to test on

# temporary segmentation, only used to find the objects (it is not the output)
SEG_BLUR_SIGMA = 1
SEG_THRESHOLD = None       # how far a pixel must differ from the background gray level (0-255). None = automatic (Otsu)
SEG_MIN_AREA = 30          # objects smaller than this (pixels) are ignored

# E. min-area rectangle: objects that fail either condition are removed
RECT_MIN_ELONGATION = 3.0  # long side / short side of the rectangle. A round object is close to 1
RECT_MIN_FILL = 0.0        # object area / rectangle area (0-1). A disk is ~0.79, a straight rod ~0.9. 0 = not used

# 3. hole ratio
HOLE_RATIO_MIN = 0.9       # objects below this ratio (0-1) are removed. 1 = no hole at all

# 3. two-threshold area ratio
LOOSE_THRESHOLD_RATIO = 0.5    # lower threshold = this ratio * the segmentation threshold
TWO_THRESHOLD_MIN_RATIO = 0.6  # objects whose strict area / loose area (0-1) is below this are removed

# 4. video output: the plain mask of every frame, bacteria in black on a white background
PROCESS_VIDEO = True       # False = only the test frame is processed (fast, to tune the settings)
OUTPUT_PATH = None         # where the video is saved. None = "Results/<video name>_bacteria.mp4" next to this script
# ---------------------------------------------------------------------------


def load_video(path: str) -> np.ndarray:
    """Video file -> grayscale uint8 array (frames, height, width)."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame)
    capture.release()
    if not frames:
        raise ValueError(f"no frame read from: {path}")
    print(f"video loaded: {len(frames)} frames of {frames[0].shape[1]}x{frames[0].shape[0]}")
    return np.stack(frames)


def median_background(video: np.ndarray) -> np.ndarray:
    """Median over time of every pixel -> background image (height, width), uint8.

    Done by chunks of rows so the full video is never duplicated in memory.
    """
    background = np.empty(video.shape[1:], dtype=np.uint8)
    for start in range(0, video.shape[1], MEDIAN_ROW_CHUNK):
        rows = slice(start, start + MEDIAN_ROW_CHUNK)
        background[rows] = np.median(video[:, rows], axis=0).astype(np.uint8)
    return background


def subtract_background(video: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Frame minus background, re-centered on GRAY_LEVEL -> uint8 (frames, height, width)."""
    difference = video.astype(np.int16) - background.astype(np.int16) + GRAY_LEVEL
    return np.clip(difference, 0, 255).astype(np.uint8)


def deviation_image(frame: np.ndarray) -> np.ndarray:
    """Background-removed frame (height, width) -> how far each pixel is from the background gray level (uint8).

    The background is mid-gray, so objects are the pixels that differ from GRAY_LEVEL (darker OR lighter).
    """
    blurred = cv2.GaussianBlur(frame, (0, 0), SEG_BLUR_SIGMA).astype(np.float32)
    return np.clip(np.abs(blurred - GRAY_LEVEL), 0, 255).astype(np.uint8)


def segmentation_threshold(deviation: np.ndarray) -> float:
    """SEG_THRESHOLD, or the Otsu threshold of the deviation image when SEG_THRESHOLD is None."""
    threshold = SEG_THRESHOLD
    if threshold is None:
        threshold, _ = cv2.threshold(deviation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"segmentation threshold: {threshold}")
    return threshold


def label_objects(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask -> (label image, stats). Objects smaller than SEG_MIN_AREA are erased from the label image.

    stats has one row per label (0 = background), columns: x, y, width, height, area.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < SEG_MIN_AREA
    small[0] = True
    labels[small[labels]] = 0
    return labels, stats


def paint(labels: np.ndarray, per_object: np.ndarray) -> np.ndarray:
    """Give every pixel of an object the value computed for that object (0 elsewhere)."""
    return per_object[labels].astype(np.float32)


def rectangle_measures(labels: np.ndarray, stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the min-area rotated rectangle around each object -> (elongation, fill), arrays indexed by label.

    elongation = long side / short side, fill = object area / rectangle area.
    """
    elongation = np.zeros(len(stats), dtype=np.float32)
    fill = np.zeros(len(stats), dtype=np.float32)
    for i in np.unique(labels[labels > 0]):
        x, y, w, h, area = stats[i]
        crop = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(max(contours, key=len))
        side_a, side_b = side_a + 1, side_b + 1  # contour runs through pixel centers: add the pixel width back
        elongation[i] = max(side_a, side_b) / min(side_a, side_b)
        fill[i] = min(area / (side_a * side_b), 1.0)
    return elongation, fill


def rectangle_rods(labels: np.ndarray, elongation: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Boolean image of the objects that are elongated enough (and fill their rectangle enough)."""
    keep = (elongation >= RECT_MIN_ELONGATION) & (fill >= RECT_MIN_FILL)
    keep[0] = False
    present = np.unique(labels[labels > 0])
    print(f"rectangle: {keep[present].sum()} objects kept out of {len(present)}")
    return keep[labels]


def keep_above(labels: np.ndarray, scores: np.ndarray, cutoff: float, name: str) -> np.ndarray:
    """Boolean image of the objects whose score (array indexed by label) is at least `cutoff`."""
    keep = scores >= cutoff
    keep[0] = False
    present = np.unique(labels[labels > 0])
    print(f"{name}: {keep[present].sum()} objects kept out of {len(present)}")
    return keep[labels]


def hole_ratio(labels: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """Object area / area inside its outer contour -> array indexed by label (1 = solid, lower = hollow)."""
    ratio = np.zeros(len(stats), dtype=np.float32)
    for i in np.unique(labels[labels > 0]):
        x, y, w, h, area = stats[i]
        crop = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(crop)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        ratio[i] = area / max(int(filled.sum()), 1)
    return ratio


def two_threshold_ratio(deviation: np.ndarray, threshold: float, objects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Objects found with the lower threshold -> (label image, strict share), the share indexed by label.

    strict share = area of the strict-threshold objects inside it / its own area (1 = solid, lower = faint interior).
    """
    loose_labels, loose_stats = label_objects(deviation > threshold * LOOSE_THRESHOLD_RATIO)
    n = len(loose_stats)
    total = np.bincount(loose_labels.ravel(), minlength=n)
    strict = np.bincount(loose_labels.ravel(), weights=objects.ravel(), minlength=n)
    return loose_labels, strict / np.maximum(total, 1)


def process_frame(frame: np.ndarray) -> SimpleNamespace:
    """Background-removed frame -> every intermediate result of steps 2 and 3. `kept` is the plain mask."""
    deviation = deviation_image(frame)
    threshold = segmentation_threshold(deviation)
    labels, stats = label_objects(deviation > threshold)
    elongation, fill = rectangle_measures(labels, stats)
    rods = rectangle_rods(labels, elongation, fill)
    rod_labels = np.where(rods, labels, 0)  # step 3 only scores the objects that survived step 2
    ratio = hole_ratio(rod_labels, stats)
    loose_labels, strict_share = two_threshold_ratio(deviation, threshold, labels > 0)
    kept = (keep_above(rod_labels, ratio, HOLE_RATIO_MIN, "hole ratio")
            & keep_above(loose_labels, strict_share, TWO_THRESHOLD_MIN_RATIO, "two thresholds"))
    return SimpleNamespace(deviation=deviation, threshold=threshold, labels=labels, elongation=elongation, fill=fill,
                           rods=rods, rod_labels=rod_labels, ratio=ratio, loose_labels=loose_labels,
                           strict_share=strict_share, kept=kept)


def process_video(subtracted: np.ndarray) -> np.ndarray:
    """Background-removed video (frames, height, width) -> black and white video, bacteria = 0 on white (255), uint8."""
    result = np.full_like(subtracted, 255)
    for i, frame in enumerate(subtracted):
        with contextlib.redirect_stdout(io.StringIO()):  # the per-frame messages would flood the terminal
            result[i][process_frame(frame).kept] = 0
        print(f"\rprocessing frame {i + 1}/{len(subtracted)}", end="", flush=True)
    print()
    return result


def video_fps(path: str) -> float:
    """Frame rate of a video file (25 if it is not stored in the file)."""
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if fps > 0 else 25.0


def save_video(frames: np.ndarray, fps: float) -> None:
    """Save a grayscale video (frames, height, width) as an mp4 file."""
    path = Path(OUTPUT_PATH) if OUTPUT_PATH else Path(__file__).resolve().parent / "Results" / f"{Path(VIDEO_PATH).stem}_bacteria.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    print(f"video saved: {path}")


def main() -> None:
    import napari  # imported here so the functions can be used without a GUI

    video = load_video(VIDEO_PATH)
    background = median_background(video)
    subtracted = subtract_background(video, background)
    frame = subtracted[TEST_FRAME]
    p = process_frame(frame)

    result = None
    if PROCESS_VIDEO:
        result = process_video(subtracted)
        save_video(result, video_fps(VIDEO_PATH))

    viewer = napari.Viewer()
    viewer.add_image(video[TEST_FRAME], name="original (grayscale)", colormap="gray", visible=False)
    viewer.add_image(frame, name="1 background removed", colormap="gray", contrast_limits=(0, 255))
    viewer.add_image(p.labels > 0, name="temporary mask", colormap="gray", visible=False)
    viewer.add_image(paint(p.labels, p.elongation), name="2 elongation score", colormap="magma", visible=False)
    viewer.add_image(paint(p.labels, p.fill), name="2 fill score", colormap="magma", visible=False)
    viewer.add_image(p.rods, name="2 rectangle kept", colormap="gray", visible=False)
    viewer.add_image(paint(p.rod_labels, p.ratio), name="3 hole ratio score", colormap="magma", visible=False)
    viewer.add_image(paint(p.loose_labels, p.strict_share), name="3 two-threshold score", colormap="magma", visible=False)
    viewer.add_image(p.kept, name="4 plain mask", colormap="gray", visible=False)
    if result is not None:  # added last so it is shown on top of the single-frame layers
        viewer.add_image(result, name="RESULT video (all frames)", colormap="gray", contrast_limits=(0, 255))
    napari.run()


if __name__ == "__main__":
    main()
