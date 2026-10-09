"""Remove the background of a video (temporal median), drop round objects (eccentricity),
and view the result in napari. Final video: bacteria in black on a white background.

Run with: python Napari_Process.py
"""

import cv2
import numpy as np
from skimage.measure import label, regionprops

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH = "/Users/juwul/Desktop/Internship/Videosplit/Expo/Expo copie_split_018.mp4"
CONTRAST_LIMITS = (0, 60)  # display range of the subtracted video (differences are small)

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


def main() -> None:
    import napari  # imported here so the processing functions can be used without a GUI

    color_video = load_video(VIDEO_PATH)
    video = to_grayscale(color_video)
    background = median_background(video)
    subtracted = subtract_background(video, background)
    print(f"video {video.shape}, max difference {subtracted.max()}")

    mask = foreground_mask(subtracted)
    rods = remove_round_objects(mask)
    print(f"objects pixels: {mask.mean():.2%} -> {rods.mean():.2%} after removing round ones")
    final = np.where(rods, 0, 255).astype(np.uint8)  # bacteria black, background white

    viewer = napari.Viewer()
    viewer.add_image(color_video, name="original (color)", rgb=True, visible=False)
    viewer.add_image(video, name="original (grayscale)", colormap="gray", visible=False)
    viewer.add_image(background, name="background (median)", colormap="gray", visible=False)
    viewer.add_image(subtracted, name="background removed", colormap="gray",
                     contrast_limits=CONTRAST_LIMITS, visible=False)
    viewer.add_image(final, name="bacteria (black on white)", colormap="gray",
                     contrast_limits=(0, 255))
    napari.run()


if __name__ == "__main__":
    main()
