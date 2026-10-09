"""Test ways to stop an out-of-focus hollow bacterium from surviving as two thin "rims".

A hollow bacterium out of focus shows two thin dark edges with a lighter interior. The temporary mask keeps the
edges and drops the interior, so the bacterium is split into two elongated objects that both pass the elongation
and hole-ratio tests of Napari_Stationary_Process.py.

Uses the background method (and the validated steps) of Napari_Stationary_Process.py on ONE frame, then saves one
image per option in OUTPUT_DIR. Every option only removes objects from the current pipeline result (the baseline):
  1. closing, then gap ratio      objects merged by a closing that are mostly empty gap are removed
  2. two-threshold area ratio     objects whose area at the strict threshold is a small part of their area at a lower one
  3. stronger blur                mask made on a more blurred image so the rims merge, then a width limit
  4. parallel-pair grouping       two close, parallel objects are treated as the two rims of one bacterium
  5. double-dip profile           objects (merged by the closing) whose intensity across their width has two dips

Image colors: white = kept, red = removed by the option compared to the baseline (SHOW_REMOVED_IN_RED).

Run with: python Hollow_Bacteria_Test.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "Phase Extraction"))  # where Napari_Stationary_Process.py is
import Napari_Stationary_Process as base

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TEST_FRAME = base.TEST_FRAME   # same frame as Napari_Stationary_Process.py. Write a number to use another frame
OUTPUT_DIR = Path(__file__).resolve().parent / "Hollow_Bacteria_Test_Output"
SHOW_REMOVED_IN_RED = True     # False = pure black and white images

# 1. closing, then gap ratio (also used by option 5)
CLOSE_GAP_RADIUS = 6           # pixels; gaps narrower than 2 * radius are bridged. Should be about half the gap between the rims
GAP_RATIO_MIN = 0.8            # objects with less than this share (0-1) of real mask pixels after closing are removed

# 2. two-threshold area ratio
LOOSE_THRESHOLD_RATIO = 0.5    # lower threshold = this ratio * the segmentation threshold
TWO_THRESHOLD_MIN_RATIO = 0.6  # objects whose strict area / loose area (0-1) is below this are removed

# 3. stronger blur
BLUR_SIGMA = 3.0               # blur of the temporary mask (the normal one is base.SEG_BLUR_SIGMA)
BLUR_MAX_WIDTH = 12            # pixels; wider objects are removed

# 4. parallel-pair grouping
PAIR_MAX_ANGLE = 10            # degrees; max difference of orientation between the two rims
PAIR_MAX_DISTANCE = 25         # pixels; max distance between the two rims (center to center, across the bacterium)
PAIR_MIN_OVERLAP = 0.5         # share (0-1) of the shorter rim that must face the other rim

# 5. double-dip profile
DIP_PEAK_MIN = 0.5             # a peak counts if it reaches this share of the highest value of the profile
DIP_VALLEY_MAX = 0.7           # hollow if the valley between two peaks is below this share of the lower peak
# ---------------------------------------------------------------------------


def disk(radius: int) -> np.ndarray:
    """Round structuring element of the given radius."""
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def object_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Pixel coordinates (n, 2) as (x, y) -> (center, unit vector along the long axis, length)."""
    center = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
    return center, eigenvectors[:, 1], float(np.sqrt(12 * eigenvalues[1]))


def object_points(labels: np.ndarray, stats: np.ndarray, i: int) -> np.ndarray:
    """Pixel coordinates (n, 2) as (x, y) of object i."""
    x, y, w, h, _ = stats[i]
    ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
    return np.stack([xs + x, ys + y], axis=1).astype(np.float64)


def run_pipeline(frame: np.ndarray) -> dict:
    """The validated steps of Napari_Stationary_Process.py on one background-removed frame."""
    deviation = base.deviation_image(frame)
    threshold = base.segmentation_threshold(deviation)
    labels, stats = base.label_objects(deviation > threshold)
    elongation, fill = base.rectangle_measures(labels, stats)
    rods = base.rectangle_rods(labels, elongation, fill)
    rod_labels = np.where(rods, labels, 0)
    kept = base.keep_above(rod_labels, base.hole_ratio(rod_labels, stats), base.HOLE_RATIO_MIN, "hole ratio")
    return dict(deviation=deviation, threshold=threshold, labels=labels, stats=stats, rod_labels=rod_labels, kept=kept)


def option1_closing_gap_ratio(objects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close the mask so the rims merge, then score each merged object = real mask pixels / merged area.

    Returns (image of the objects to keep, closed label image, closed stats).
    """
    closed = cv2.morphologyEx(objects.astype(np.uint8), cv2.MORPH_CLOSE, disk(CLOSE_GAP_RADIUS))
    closed_labels, closed_stats = base.label_objects(closed)
    n = len(closed_stats)
    total = np.bincount(closed_labels.ravel(), minlength=n)
    real = np.bincount(closed_labels.ravel(), weights=objects.ravel(), minlength=n)
    keep = base.keep_above(closed_labels, real / np.maximum(total, 1), GAP_RATIO_MIN, "option 1 gap ratio")
    return keep, closed_labels, closed_stats


def option2_two_threshold(deviation: np.ndarray, threshold: float, objects: np.ndarray) -> np.ndarray:
    """Objects of the loose mask whose strict-threshold area is a large enough part of them."""
    loose_labels, loose_stats = base.label_objects(deviation > threshold * LOOSE_THRESHOLD_RATIO)
    n = len(loose_stats)
    total = np.bincount(loose_labels.ravel(), minlength=n)
    strict = np.bincount(loose_labels.ravel(), weights=objects.ravel(), minlength=n)
    return base.keep_above(loose_labels, strict / np.maximum(total, 1), TWO_THRESHOLD_MIN_RATIO, "option 2 two thresholds")


def object_width(labels: np.ndarray) -> np.ndarray:
    """Width of each object (twice the largest distance to the background) -> array indexed by label."""
    distance = cv2.distanceTransform((labels > 0).astype(np.uint8), cv2.DIST_L2, 5)
    peak = np.zeros(labels.max() + 1, dtype=np.float32)
    inside = labels > 0
    np.maximum.at(peak, labels[inside], distance[inside])
    return 2 * peak - 1


def option3_stronger_blur(frame: np.ndarray) -> np.ndarray:
    """Temporary mask made on a more blurred image (rims merge), then elongation and width tests."""
    blurred = cv2.GaussianBlur(frame, (0, 0), BLUR_SIGMA).astype(np.float32)
    deviation = np.clip(np.abs(blurred - base.GRAY_LEVEL), 0, 255).astype(np.uint8)
    labels, stats = base.label_objects(deviation > base.segmentation_threshold(deviation))
    elongation, fill = base.rectangle_measures(labels, stats)
    rods = base.rectangle_rods(labels, elongation, fill)
    narrow = (object_width(labels) <= BLUR_MAX_WIDTH)[labels]
    print(f"option 3 width: {np.unique(labels[rods & ~narrow]).size} elongated objects removed for being wider than {BLUR_MAX_WIDTH}")
    return rods & narrow


def option4_parallel_pairs(rod_labels: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """Remove every object that has a close, parallel, facing neighbor (the other rim of the same bacterium)."""
    ids = np.unique(rod_labels[rod_labels > 0])
    if len(ids) < 2:
        return rod_labels > 0
    axes = [object_axes(object_points(rod_labels, stats, i)) for i in ids]
    center = np.array([a[0] for a in axes])
    direction = np.array([a[1] for a in axes])
    length = np.array([a[2] for a in axes])

    offset = center[None, :, :] - center[:, None, :]                    # [i, j] = center j - center i
    along = np.abs(np.einsum("ijk,ik->ij", offset, direction))          # distance along the axis of i
    across = np.abs(direction[:, None, 0] * offset[..., 1] - direction[:, None, 1] * offset[..., 0])
    parallel = np.abs(direction @ direction.T) >= np.cos(np.radians(PAIR_MAX_ANGLE))
    overlap = (length[:, None] + length[None, :]) / 2 - along
    facing = overlap >= PAIR_MIN_OVERLAP * np.minimum(length[:, None], length[None, :])
    pair = parallel & facing & (across <= PAIR_MAX_DISTANCE)
    np.fill_diagonal(pair, False)

    paired = pair.any(axis=1)
    print(f"option 4 parallel pairs: {(~paired).sum()} objects kept out of {len(ids)}")
    keep = np.zeros(rod_labels.max() + 1, dtype=bool)
    keep[ids] = ~paired
    return keep[rod_labels]


def has_double_dip(profile: np.ndarray) -> bool:
    """True if the profile has two clear peaks separated by a clear valley."""
    profile = np.pad(profile, 1)
    peaks = [k for k in range(1, len(profile) - 1)
             if profile[k] > profile[k - 1] and profile[k] >= profile[k + 1] and profile[k] >= DIP_PEAK_MIN * profile.max()]
    if len(peaks) < 2:
        return False
    first, second = sorted(sorted(peaks, key=lambda k: profile[k])[-2:])
    valley = profile[first:second + 1].min()
    return bool(valley <= DIP_VALLEY_MAX * min(profile[first], profile[second]))


def option5_double_dip(deviation: np.ndarray, closed_labels: np.ndarray, closed_stats: np.ndarray) -> np.ndarray:
    """Remove the merged objects whose mean intensity across their width has two dips (a hollow bacterium)."""
    keep = np.zeros(len(closed_stats), dtype=bool)
    ids = np.unique(closed_labels[closed_labels > 0])
    for i in ids:
        points = object_points(closed_labels, closed_stats, i)
        if len(points) < 3:
            keep[i] = True
            continue
        center, along, _ = object_axes(points)
        across = np.rint((points - center) @ np.array([-along[1], along[0]])).astype(int)
        across -= across.min()
        values = deviation[points[:, 1].astype(int), points[:, 0].astype(int)].astype(np.float64)
        profile = np.bincount(across, weights=values) / np.maximum(np.bincount(across), 1)
        keep[i] = not has_double_dip(np.convolve(profile, np.ones(3) / 3, mode="same"))
    print(f"option 5 double dip: {keep[ids].sum()} objects kept out of {len(ids)}")
    return keep[closed_labels]


def run_options(frame: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Background-removed frame -> ({image name: boolean image of the kept objects}, temporary mask).

    The first entry of the dictionary is the baseline.
    """
    p = run_pipeline(frame)
    objects = p["labels"] > 0
    keep1, closed_labels, closed_stats = option1_closing_gap_ratio(objects)
    keep2 = option2_two_threshold(p["deviation"], p["threshold"], objects)
    keep3 = option3_stronger_blur(frame)
    keep4 = option4_parallel_pairs(p["rod_labels"], p["stats"])
    keep5 = option5_double_dip(p["deviation"], closed_labels, closed_stats)
    kept = p["kept"]
    results = {
        "02_baseline_current_pipeline": kept,
        "03_option1_closing_gap_ratio": kept & keep1,
        "04_option2_two_threshold": kept & keep2,
        "05_option3_stronger_blur": kept & keep3,
        "06_option4_parallel_pairs": kept & keep4,
        "07_option5_double_dip": kept & keep5,
    }
    return results, objects


def save_image(name: str, kept: np.ndarray, baseline: np.ndarray) -> None:
    """Save kept objects in white, and (optionally) the objects removed compared to the baseline in red."""
    image = np.zeros((*kept.shape, 3), dtype=np.uint8)
    image[kept] = 255
    if SHOW_REMOVED_IN_RED:
        image[baseline & ~kept] = (0, 0, 255)  # BGR
    cv2.imwrite(str(OUTPUT_DIR / f"{name}.png"), image)


def main() -> None:
    video = base.load_video(base.VIDEO_PATH)
    background = base.median_background(video)
    frame = base.subtract_background(video, background)[TEST_FRAME]

    results, objects = run_options(frame)
    baseline = results["02_baseline_current_pipeline"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "00_background_removed.png"), frame)
    cv2.imwrite(str(OUTPUT_DIR / "01_temporary_mask.png"), objects.astype(np.uint8) * 255)
    for name, kept in results.items():
        save_image(name, kept, baseline)
        print(f"{name}: {int(kept.sum())} px kept, {int((baseline & ~kept).sum())} px removed vs baseline")
    print(f"images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
