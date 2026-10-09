"""Expo pipeline (median background -> subtract -> black/white -> remove round objects) applied to a Stationary
split, once for each of two backgrounds, so the two medians can be compared side by side in napari:

  A. median of the selected split and of the splits BACKGROUND_OFFSETS after it
  B. median of a video that contains only background (Stationary2_split_013)

Final video: bacteria in black on a white background.

Run with: python Napari_Stationary_Medians.py
"""

import re
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label, regionprops

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH = "/Users/juwul/Desktop/Internship/Videosplit/Stationary/Stationary2_split_150.mp4"
CONTRAST_LIMITS = (0, 60)  # display range of the subtracted video (differences are small)

# A. the background also uses the splits this many places AFTER the selected one (3 s each, so 5 = 15 s later).
#    Use at least two offsets: with one, the median of two backgrounds is just their average.
BACKGROUND_OFFSETS = [5, 10]

# B. video that contains only background: its median is used as the background of the selected video.
BACKGROUND_VIDEO_PATH = "/Users/juwul/Desktop/Internship/Videosplit/Stationary/Stationary2_split_013.mp4"

# Splits recorded with a different exposure are much brighter or darker (e.g. split 150 is ~164, splits 155/160 ~121).
# True = scale every background so its brightness matches the selected video before subtracting.
MATCH_BRIGHTNESS = True

BLUR_SIGMA = 1          # smoothing before thresholding (higher = less noise, fatter objects)
THRESHOLD = None        # None = automatic (Otsu on the whole video), or a number 0-255
MIN_AREA = 40           # objects smaller than this (pixels) are noise and removed
MIN_ECCENTRICITY = 0.95 # 0 = circle, 1 = line. Objects BELOW this value (round) are removed
# ---------------------------------------------------------------------------


def load_video(path: str) -> np.ndarray:
    """Read every frame -> array (frames, height, width, RGB)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def to_grayscale(video: np.ndarray) -> np.ndarray:
    """RGB video (frames, height, width, 3) -> grayscale video (frames, height, width)."""
    return np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video])


def median_background(video: np.ndarray) -> np.ndarray:
    """Background = per-pixel median over time (moving objects disappear)."""
    return np.median(video, axis=0).astype(np.uint8)


def neighbour_path(path: str, offset: int) -> str:
    """Path of the split `offset` places after the selected one (..._split_131 -> ..._split_141)."""
    stem = re.search(r"_split_(\d+)", path)
    number = int(stem.group(1)) + offset
    return path[:stem.start(1)] + f"{number:0{len(stem.group(1))}d}" + path[stem.end(1):]


def video_level(video: np.ndarray) -> float:
    """Typical gray level of a video (median of a subsample of its pixels)."""
    return float(np.median(video[:, ::8, ::8]))


def match_brightness(background: np.ndarray, level: float) -> np.ndarray:
    """Scale a background so that its typical gray level equals `level` (an exposure change is multiplicative)."""
    if not MATCH_BRIGHTNESS:
        return background
    ratio = level / max(float(np.median(background)), 1.0)
    return np.clip(background.astype(np.float32) * ratio, 0, 255).astype(np.uint8)


def offsets_background(path: str, video: np.ndarray) -> np.ndarray:
    """A. Median of the median backgrounds of the selected split and of the splits BACKGROUND_OFFSETS later.

    A bacterium that sits still during the selected split is unlikely to sit at the same place in
    the later splits, so it is outvoted. Fixed dirt is at the same place everywhere and stays.
    """
    level = video_level(video)
    backgrounds = [median_background(video)]
    for offset in BACKGROUND_OFFSETS:
        other = neighbour_path(path, offset)
        if not Path(other).is_file():
            print(f"skipped (not found): {other}")
            continue
        other_background = median_background(to_grayscale(load_video(other)))
        print(f"background A built with: {Path(other).name} (gray level {np.median(other_background):.0f}, "
              f"selected video {level:.0f})")
        backgrounds.append(match_brightness(other_background, level))
    return np.median(backgrounds, axis=0).astype(np.uint8)


def reference_background(path: str) -> np.ndarray:
    """B. Median of a video that contains only background (no bacteria)."""
    return median_background(to_grayscale(load_video(path)))


def subtract_background(video: np.ndarray, background: np.ndarray) -> np.ndarray:
    """|frame - background| keeps objects both darker and brighter than the background."""
    return np.stack([cv2.absdiff(frame, background) for frame in video])


def foreground_mask(subtracted: np.ndarray) -> np.ndarray:
    """Subtracted grayscale video -> boolean mask (frames, height, width) of what differs from the background."""
    gray = np.stack([cv2.GaussianBlur(f, (0, 0), BLUR_SIGMA) for f in subtracted])
    threshold = THRESHOLD
    if threshold is None:
        threshold, _ = cv2.threshold(gray.reshape(-1, 1)[::50], 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"threshold used: {threshold}")
    return gray > threshold


def remove_round_objects(mask: np.ndarray) -> np.ndarray:
    """Keep only elongated objects (eccentricity >= MIN_ECCENTRICITY and area >= MIN_AREA)."""
    kept = np.zeros_like(mask)
    for i, frame in enumerate(mask):
        labels = label(frame)
        keep = np.zeros(labels.max() + 1, dtype=bool)  # lookup table: label -> keep?
        for obj in regionprops(labels):
            keep[obj.label] = obj.area >= MIN_AREA and obj.eccentricity >= MIN_ECCENTRICITY
        kept[i] = keep[labels]
    return kept


def process(video: np.ndarray, background: np.ndarray, name: str):
    """Expo pipeline for one background -> (subtracted, final black-on-white video)."""
    subtracted = subtract_background(video, background)
    print(f"[{name}] max difference {subtracted.max()}")
    mask = foreground_mask(subtracted)
    rods = remove_round_objects(mask)
    print(f"[{name}] objects pixels: {mask.mean():.2%} -> {rods.mean():.2%} after removing round ones")
    final = np.where(rods, 0, 255).astype(np.uint8)  # bacteria black, background white
    return subtracted, final


def main() -> None:
    import napari  # imported here so the processing functions can be used without a GUI

    color_video = load_video(VIDEO_PATH)
    video = to_grayscale(color_video)

    background_a = offsets_background(VIDEO_PATH, video)
    background_b = match_brightness(reference_background(BACKGROUND_VIDEO_PATH), video_level(video))
    subtracted_a, final_a = process(video, background_a, "A offsets")
    subtracted_b, final_b = process(video, background_b, "B split 013")

    viewer = napari.Viewer()
    viewer.add_image(color_video, name="original (color)", rgb=True, visible=False)
    viewer.add_image(video, name="original (grayscale)", colormap="gray", visible=False)
    for tag, background, subtracted, final in (("A offsets", background_a, subtracted_a, final_a),
                                               ("B split 013", background_b, subtracted_b, final_b)):
        viewer.add_image(background, name=f"{tag} - background (median)", colormap="gray", visible=False)
        viewer.add_image(subtracted, name=f"{tag} - background removed", colormap="gray",
                         contrast_limits=CONTRAST_LIMITS, visible=False)
        viewer.add_image(final, name=f"{tag} - bacteria (black on white)", colormap="gray",
                         contrast_limits=(0, 255), visible=False)
    viewer.layers["A offsets - bacteria (black on white)"].visible = True
    napari.run()


if __name__ == "__main__":
    main()
