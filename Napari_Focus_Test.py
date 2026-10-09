"""Test ways to separate in-focus bacteria (solid, sharp) from out-of-focus ones (transparent, blurry edges).

Works on the "background removed (gamma)" image of ONE frame, and shows every option as a napari layer:
  1. band-pass filter (difference of Gaussians)
  2. sharpness score per object  (mean gradient on the object's edge)
  3. Laplacian variance per object
  4. erosion test (objects that disappear when eroded are thin rims)

Per-object options need objects, so the gamma image is thresholded into a temporary mask (only used to find
the objects, it is not the output). Every "score" layer paints each object with its score: hover over an object
in napari to read the value, then set the cut-off in CONFIG.

Run with: python Napari_Focus_Test.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label

sys.path.insert(0, str(Path(__file__).resolve().parent / "Video_Process_Script" / "Phase_Extraction"))
import Napari_Stationary_Process as base

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TEST_FRAME = 14            # which frame of the split to test on

# temporary segmentation, only used to find the objects that get scored
SEG_BLUR_SIGMA = 1
SEG_THRESHOLD = None       # how far a pixel must differ from the background gray level (0-255). None = automatic (Otsu)
SEG_MIN_AREA = 30          # objects smaller than this (pixels) are ignored (score 0)

# 1. band-pass (difference of Gaussians): keeps details between the two sizes
BANDPASS_LOW_SIGMA = 1.0   # removes pixel noise
BANDPASS_HIGH_SIGMA = 6.0  # removes the smooth, blurry structures

# 2. / 3. cut-offs: objects with a score below the cut-off are removed. None = median score of the objects.
SHARPNESS_MIN = None
LAPLACIAN_VAR_MIN = None

# 4. erosion test
EROSION_RADIUS = 3         # pixels; objects thinner than 2 * radius disappear and are removed
# ---------------------------------------------------------------------------


def segment_objects(gamma_frame: np.ndarray) -> np.ndarray:
    """Gamma image (height, width) -> label image (0 = background, 1..N = objects). Temporary mask.

    The gamma image has a mid-gray background, so objects are the pixels that differ from that gray level
    (darker OR lighter), not the pixels above a fixed brightness.
    """
    blurred = cv2.GaussianBlur(gamma_frame, (0, 0), SEG_BLUR_SIGMA).astype(np.float32)
    deviation = np.clip(np.abs(blurred - np.median(blurred)), 0, 255).astype(np.uint8)
    threshold = SEG_THRESHOLD
    if threshold is None:
        threshold, _ = cv2.threshold(deviation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"segmentation threshold: {threshold}")
    labels = label(deviation > threshold)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    labels[areas[labels] < SEG_MIN_AREA] = 0
    return labels


def bandpass(gamma_frame: np.ndarray) -> np.ndarray:
    """Difference of Gaussians: fine detail minus smooth structure. Sharp edges stay, blur disappears."""
    f = gamma_frame.astype(np.float32)
    return cv2.GaussianBlur(f, (0, 0), BANDPASS_LOW_SIGMA) - cv2.GaussianBlur(f, (0, 0), BANDPASS_HIGH_SIGMA)


def sharpness_per_object(gamma_frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean gradient magnitude (Sobel) on each object's edge pixels -> array indexed by label."""
    f = gamma_frame.astype(np.float32)
    gradient = np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))
    objects = labels > 0
    edge = objects & ~cv2.erode(objects.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    total = np.bincount(labels[edge], weights=gradient[edge], minlength=labels.max() + 1)
    count = np.bincount(labels[edge], minlength=labels.max() + 1)
    return total / np.maximum(count, 1)


def laplacian_variance_per_object(gamma_frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Variance of the Laplacian inside each object (high = sharp, low = blurred) -> array indexed by label."""
    lap = cv2.Laplacian(gamma_frame.astype(np.float32), cv2.CV_32F, ksize=3)
    n = labels.max() + 1
    count = np.maximum(np.bincount(labels.ravel(), minlength=n), 1)
    mean = np.bincount(labels.ravel(), weights=lap.ravel(), minlength=n) / count
    mean_sq = np.bincount(labels.ravel(), weights=(lap ** 2).ravel(), minlength=n) / count
    return mean_sq - mean ** 2


def erosion_survivors(labels: np.ndarray) -> np.ndarray:
    """Boolean array indexed by label: True if the object still exists after eroding it by EROSION_RADIUS."""
    size = 2 * EROSION_RADIUS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    eroded = cv2.erode((labels > 0).astype(np.uint8), kernel).astype(bool)
    survivors = np.zeros(labels.max() + 1, dtype=bool)
    survivors[np.unique(labels[eroded])] = True
    survivors[0] = False
    return survivors


def paint(labels: np.ndarray, per_object: np.ndarray) -> np.ndarray:
    """Give every pixel of an object the value computed for that object (0 elsewhere)."""
    return per_object[labels].astype(np.float32)


def keep_above(labels: np.ndarray, scores: np.ndarray, cutoff) -> np.ndarray:
    """Boolean image of the objects whose score is at least `cutoff` (None = median of the objects' scores)."""
    present = np.unique(labels[labels > 0])
    if cutoff is None:
        cutoff = float(np.median(scores[present]))
    print(f"cut-off used: {cutoff:.2f}  ({len(present)} objects, {(scores[present] >= cutoff).sum()} kept)")
    keep = scores >= cutoff
    keep[0] = False
    return keep[labels]


def main() -> None:
    import napari  # imported here so the functions can be used without a GUI

    video = base.load_video(base.VIDEO_PATH)  # already grayscale
    background = base.median_background(video)
    frame = base.subtract_background(video, background)[TEST_FRAME]  # mid-gray background, no gamma transform

    labels = segment_objects(frame)
    sharp = sharpness_per_object(frame, labels)
    lap_var = laplacian_variance_per_object(frame, labels)
    survivors = erosion_survivors(labels)

    viewer = napari.Viewer()
    viewer.add_image(video[TEST_FRAME], name="original (grayscale)", colormap="gray", visible=False)
    viewer.add_image(frame, name="background removed", colormap="gray")
    viewer.add_image(bandpass(frame), name="1 band-pass", colormap="gray", visible=False)
    viewer.add_image(paint(labels, sharp), name="2 sharpness score", colormap="magma", visible=False)
    viewer.add_image(keep_above(labels, sharp, SHARPNESS_MIN), name="2 sharpness kept", colormap="gray", visible=False)
    viewer.add_image(paint(labels, lap_var), name="3 laplacian variance score", colormap="magma", visible=False)
    viewer.add_image(keep_above(labels, lap_var, LAPLACIAN_VAR_MIN), name="3 laplacian kept", colormap="gray", visible=False)
    viewer.add_image(survivors[labels], name="4 erosion kept", colormap="gray", visible=False)
    napari.run()


if __name__ == "__main__":
    main()
