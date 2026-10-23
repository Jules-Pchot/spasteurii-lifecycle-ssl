"""New whole Stationary frames at 1080p for labeling, with no box (labeled from scratch by hand).

Why: the 640 x 640 tiles at full ×100 resolution show a small field (~18 % of the frame width), mostly background,
and are hard to label. A whole frame scaled to 1080 px high (3584 x 2746 -> 1410 x 1080, factor 0.393) shows the
full field and brings the bacteria to ~7 px wide, the Expo ×40 scale. The tiles already labeled stay as they are.

- Frames: N_FRAMES spread evenly over the video, skipping every frame already in the Label Studio project (or listed
  in a previous batch) by less than USED_MARGIN, and every frame within VAL_MARGIN of a fixed validation frame
  (neighbors would leak into the score). All of them go to train.
- Image: brightness-normalized frame (as the tiles), reduced by area averaging (PIL BOX filter = mean over the
  covered source pixels, same as OpenCV INTER_AREA), then DISPLAY_RANGE mapped to 0-255.

Output (OUTPUT_DIR): frames1080/*.png, tasks_frames1080.json (to import into the existing project),
frames_frames1080.csv. Nothing is overwritten.

Run with: python Make_Frame_Tasks_1080p.py
"""

import csv
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

import Make_Labeling_Tasks as base

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LABEL_STUDIO_DB = Path.home() / "Library/Application Support/label-studio/label_studio.sqlite3"
PROJECT_TITLE = "BacteriaSSL Focus"
VIDEO = ("Stationary", base.stationary)
N_FRAMES = 40              # new frames spread over the video
HEIGHT = 1080              # output frame height (pixels); the width keeps the aspect ratio
VAL_FRAMES = [2626, 6014]  # fixed validation frames of Train_YOLO.ipynb: new frames stay away from them
VAL_MARGIN = 100           # frames
USED_MARGIN = 50           # frames: new frames stay away from the used ones too (neighbors look almost the same)
BATCH = "frames1080"       # output folder, tasks file and `batch` field of the tasks
# ---------------------------------------------------------------------------


def used_frames(name: str) -> set[int]:
    """Frames of the video already in the Label Studio project or in the frame lists of previous batches."""
    with sqlite3.connect(f"file:{LABEL_STUDIO_DB}?mode=ro", uri=True) as db:
        rows = db.execute("select t.data from task t join project p on p.id = t.project_id where p.title = ?",
                          (PROJECT_TITLE,)).fetchall()
    used = {json.loads(data)["frame"] for (data,) in rows if json.loads(data)["video"] == name}
    for listing in base.OUTPUT_DIR.glob("frames*.csv"):
        with open(listing) as file:
            used |= {int(row["frame"]) for row in csv.DictReader(file) if row["video"] == name}
    return used


def new_frames(module, used: set[int]) -> np.ndarray:
    """N_FRAMES frames spread over the video, away from the used and validation frames."""
    n = module.frame_count(module.VIDEO_PATH)
    candidates = np.array([i for i in range(module.FRAME_START, n)
                           if all(abs(i - u) > USED_MARGIN for u in used)
                           and all(abs(i - v) > VAL_MARGIN for v in VAL_FRAMES)])
    return candidates[np.linspace(0, len(candidates) - 1, N_FRAMES + 2).astype(int)[1:-1]]  # skip both ends


def main() -> None:
    name, module = VIDEO
    out_dir = base.OUTPUT_DIR / BATCH
    tasks_path, frames_path = base.OUTPUT_DIR / f"tasks_{BATCH}.json", base.OUTPUT_DIR / f"frames_{BATCH}.csv"
    for path in (out_dir, tasks_path, frames_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists: change BATCH (nothing is overwritten)")
    frames = new_frames(module, used_frames(name))
    out_dir.mkdir()

    tasks, frame_rows = [], []
    for i in frames:
        frame = module.read_frames(module.VIDEO_PATH, np.array([i]))[0]
        height, width = frame.shape
        out_width = round(width * HEIGHT / height)
        small = Image.fromarray(module.normalize(frame).astype(np.float32), mode="F").resize(
            (out_width, HEIGHT), Image.Resampling.BOX)
        file_name = f"{name}_f{i:05d}.png"
        Image.fromarray(base.to_png(np.asarray(small))).save(out_dir / file_name)
        tasks.append({"data": {"image": "/data/local-files/?d=" + quote(f"{BATCH}/{file_name}"), "video": name,
                               "frame": int(i), "batch": BATCH}})
        frame_rows.append([name, int(i), f"{np.median(frame):.0f}"])
        print(f"{name} frame {i}: median brightness {np.median(frame):.0f} -> {out_width} x {HEIGHT}")
    tasks_path.write_text(json.dumps(tasks))
    with open(frames_path, "w", newline="") as file:
        csv.writer(file).writerows([["video", "frame", "median_brightness"], *frame_rows])
    print(f"{len(tasks)} frames -> {tasks_path}")


if __name__ == "__main__":
    main()
