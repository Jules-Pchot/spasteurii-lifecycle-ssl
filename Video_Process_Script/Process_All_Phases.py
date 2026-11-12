"""Batch processing: run the pipeline of each growth phase on every video of that phase.

For each phase, every video in Videosplit/<Phase>/ goes through the phase's own pipeline (imported from
Phase_Extraction/, not copied), and the black-on-white result is saved as
Processing_Results/<Phase>/<video name>_bacteria.mp4.

  Expo        -> Napari_Expo_Process.py        (median background, |difference|, Otsu, eccentricity)
  Stationary  -> Napari_Stationary_Process.py  (median background, min-area rectangle, hole ratio, two thresholds)

The settings of each pipeline are the ones in its own file. One video failing does not stop the batch;
a summary is printed at the end.

Run with: python Process_All_Phases.py
"""

import contextlib
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "Phase_Extraction"))
import Napari_Expo_Process as expo
import Napari_Stationary_Process as stationary

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_DIR / "Videosplit"
OUTPUT_DIR = PROJECT_DIR / "Processing_Results"
PHASES = ["Expo", "Stationary"]  # phases to process (remove one to process only the other)
SKIP_EXISTING = True             # True = videos whose result already exists are not processed again
LIMIT = None                     # None = every video; a number = only the first N videos of each phase (for a test)
# ---------------------------------------------------------------------------


def process_expo(path: Path) -> np.ndarray:
    """Expo pipeline -> black-on-white uint8 video (frames, height, width)."""
    video = expo.to_grayscale(expo.load_video(str(path)))
    subtracted = expo.subtract_background(video, expo.median_background(video))
    rods = expo.remove_round_objects(expo.foreground_mask(subtracted))
    return np.where(rods, 0, 255).astype(np.uint8)


def process_stationary(path: Path) -> np.ndarray:
    """Stationary pipeline -> black-on-white uint8 video (frames, height, width)."""
    video = stationary.load_video(str(path))
    subtracted = stationary.subtract_background(video, stationary.median_background(video))
    return stationary.process_video(subtracted)


PIPELINES = {"Expo": process_expo, "Stationary": process_stationary}


def save_video(frames: np.ndarray, fps: float, path: Path) -> None:
    """Save a grayscale video (frames, height, width) as an mp4 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


def run_phase(phase: str) -> dict:
    """Process every video of one phase -> counts of done / skipped / failed videos."""
    videos = sorted((INPUT_DIR / phase).glob("*.mp4"))[:LIMIT]
    counts = {"done": 0, "skipped": 0, "failed": []}
    print(f"\n=== {phase}: {len(videos)} videos -> {OUTPUT_DIR / phase}")
    for i, path in enumerate(videos, 1):
        output = OUTPUT_DIR / phase / f"{path.stem}_bacteria.mp4"
        if SKIP_EXISTING and output.exists():
            counts["skipped"] += 1
            continue
        start = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # the pipelines print a lot per video
                result = PIPELINES[phase](path)
            save_video(result, stationary.video_fps(str(path)), output)
            counts["done"] += 1
            print(f"[{i}/{len(videos)}] {path.name} ({time.time() - start:.0f} s)")
        except Exception as exc:  # one bad video must not stop the batch
            counts["failed"].append(path.name)
            print(f"[{i}/{len(videos)}] {path.name} FAILED: {exc}")
    return counts


def main() -> None:
    summary = {phase: run_phase(phase) for phase in PHASES}
    print("\n=== summary")
    for phase, c in summary.items():
        print(f"{phase}: {c['done']} processed, {c['skipped']} already done, {len(c['failed'])} failed")
        for name in c["failed"]:
            print(f"   failed: {name}")


if __name__ == "__main__":
    main()
