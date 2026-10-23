"""Add a batch of labeling tiles to the existing Label Studio set (made by Make_Labeling_Tasks.py).

Goal of batch 2: more in_focus boxes in Stationary (×100), the weak class of training v1 (151 labeled boxes).
N_FRAMES new frames are spread evenly over the video, skipping the frames already used and every frame within
VAL_MARGIN of a fixed validation frame (neighboring frames look almost the same: they would leak into the score).
Only the TILES_PER_FRAME tiles of each frame with the most pre-drawn in_focus boxes are kept (most Stationary tiles
hold no in-focus bacterium). Same tiles, pre-drawn boxes and display as Make_Labeling_Tasks.py.

Output (OUTPUT_DIR, next to batch 1): tiles/*.png, tasks_<BATCH>.json (to import into the existing project),
frames_<BATCH>.csv. Nothing is overwritten.

Run with: python Add_Labeling_Tasks.py
"""

import csv
import json
from urllib.parse import quote

import numpy as np
from PIL import Image

import Make_Labeling_Tasks as base

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BATCH = "batch2"
VIDEO = ("Stationary", base.stationary)
N_FRAMES = 40              # new frames spread over the video
TILES_PER_FRAME = 6        # tiles kept per frame: the richest in pre-drawn in_focus boxes (at least one)
VAL_FRAMES = [2626, 6014]  # fixed validation frames of Train_YOLO.ipynb: new frames stay away from them
VAL_MARGIN = 100           # frames
# ---------------------------------------------------------------------------


def new_frames(module, used: set[int]) -> np.ndarray:
    """N_FRAMES frames spread over the video, away from the used and validation frames."""
    n = module.frame_count(module.VIDEO_PATH)
    candidates = np.array([i for i in range(module.FRAME_START, n)
                           if i not in used and all(abs(i - v) > VAL_MARGIN for v in VAL_FRAMES)])
    return candidates[np.linspace(0, len(candidates) - 1, N_FRAMES + 2).astype(int)[1:-1]]  # skip both ends


def main() -> None:
    name, module = VIDEO
    tasks_path, frames_path = base.OUTPUT_DIR / f"tasks_{BATCH}.json", base.OUTPUT_DIR / f"frames_{BATCH}.csv"
    for path in (tasks_path, frames_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists: change BATCH (nothing is overwritten)")
    with open(base.OUTPUT_DIR / "frames.csv") as file:
        used = {int(row["frame"]) for row in csv.DictReader(file) if row["video"] == name}

    n = module.frame_count(module.VIDEO_PATH)
    background = module.masked_mean_background(module.read_frames(
        module.VIDEO_PATH, np.linspace(module.FRAME_START, n - 1, module.BG_SAMPLE_FRAMES).astype(int)))
    tasks, frame_rows = [], []
    for i in new_frames(module, used):
        frame = module.read_frames(module.VIDEO_PATH, np.array([i]))[0]
        normalized = module.normalize(frame)
        difference = normalized - background
        dog = module.dog_filter(difference)
        labels, sign = module.dog_regions(dog)
        boxes = base.predrawn_boxes(labels, sign, module.measure_shapes(labels, sign, difference, dog))
        image = base.to_png(normalized)
        candidates = []
        for row in base.tile_origins(image.shape[0]):
            for col in base.tile_origins(image.shape[1]):
                result = base.tile_predictions(boxes, row, col)
                n_in = sum(r["value"]["rectanglelabels"][0] == "in_focus" for r in result)
                if n_in:
                    candidates.append((n_in, row, col, result))
        candidates.sort(key=lambda c: -c[0])
        for n_in, row, col, result in candidates[:TILES_PER_FRAME]:
            tile_name = f"{name}_f{i:05d}_r{row:04d}_c{col:04d}.png"
            if (base.OUTPUT_DIR / "tiles" / tile_name).exists():
                raise FileExistsError(f"{tile_name} already exists")
            Image.fromarray(image[row:row + base.TILE, col:col + base.TILE]).save(base.OUTPUT_DIR / "tiles" / tile_name)
            tasks.append({"data": {"image": "/data/local-files/?d=" + quote(f"tiles/{tile_name}"),
                                   "video": name, "frame": int(i), "row": row, "col": col, "batch": BATCH},
                          "predictions": [{"model_version": "v2_dog_shapes", "result": result}]})
        kept = candidates[:TILES_PER_FRAME]
        frame_rows.append([name, int(i), f"{np.median(frame):.0f}", len(boxes), len(kept), sum(c[0] for c in kept)])
        print(f"{name} frame {i}: median brightness {np.median(frame):.0f}, {len(kept)} tiles, "
              f"{sum(c[0] for c in kept)} pre-drawn in_focus")

    tasks_path.write_text(json.dumps(tasks))
    with open(frames_path, "w", newline="") as file:
        csv.writer(file).writerows([["video", "frame", "median_brightness", "predrawn_boxes", "tiles",
                                     "predrawn_in_focus"], *frame_rows])
    print(f"{len(tasks)} tiles, {sum(r[5] for r in frame_rows)} pre-drawn in_focus -> {tasks_path}")


if __name__ == "__main__":
    main()
