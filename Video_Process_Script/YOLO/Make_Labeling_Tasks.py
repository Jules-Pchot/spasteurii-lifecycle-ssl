"""Prepare the manual YOLO labeling in Label Studio: tiles of both videos + pre-drawn boxes to correct.

For each video, N_FRAMES frames spread evenly over the video (both lamp levels are covered) are cut into
TILE x TILE tiles at full resolution (YOLO resizes its input to 640 px: a whole Stationary frame would shrink the
bacteria from 18 px to ~3 px). The tile shown for labeling is the brightness-normalized frame (% of the frame median
brightness, DISPLAY_RANGE mapped to 0-255), the same image the model will see later.

Pre-drawn boxes come from the v2 shapes (DoG sign regions split by watershed), imported from Phase_Extraction_v2
with each video's own settings:
  - dark shape  -> in_focus, bright shape -> out_of_focus
  - only shapes with DoG peak / noise std >= PREDRAW_MIN_SNR (noise shapes end around 10 on the shape histogram)
  - rings skipped: the DoG puts an opposite-sign ring around every object, so a shape touching an opposite-sign
    shape with a stronger DoG peak is that object's ring, not an object
  - a box cut by the tile edge is kept if at least EDGE_MIN_INSIDE of it is inside the tile
These boxes are only a starting point: every box is checked, corrected, removed or added by hand.

Output (OUTPUT_DIR): tiles/*.png, tasks.json (Label Studio tasks with the boxes as predictions),
label_config.xml (labeling interface), frames.csv (frames used and their brightness).

Run with: python Make_Labeling_Tasks.py
"""

import csv
import json
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Phase_Extraction_v2"))
import Napari_Expo_Process_v2 as expo
import Napari_Stationary_Process_v2 as stationary

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/Volumes/Extreme SSD/YOLO_Labeling")  # also the Label Studio local-files document root
VIDEOS = [("Expo", expo, 40),             # (name, phase module with its settings, N_FRAMES)
          ("Stationary", stationary, 8)]  # ~240 tiles each: Expo 6 tiles / frame, Stationary 30 tiles / frame
TILE = 640                 # tile side (pixels), YOLO input size
PREDRAW_MIN_SNR = 10.0     # pre-drawn boxes: shapes with DoG peak / noise std below this are not drawn
EDGE_MIN_INSIDE = 0.5      # pre-drawn boxes: share of the box that must be inside the tile
DISPLAY_RANGE = (50, 150)  # % of the median brightness mapped to 0-255 in the tiles (bacteria are within +-50 %)
CLASSES = {-1: "in_focus", 1: "out_of_focus"}  # shape sign -> class (bacteria of interest are dark)
# ---------------------------------------------------------------------------

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" brightnessControl="true" contrastControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="in_focus" background="#3987e5"/>
    <Label value="out_of_focus" background="#d95926"/>
  </RectangleLabels>
</View>
"""


def tile_origins(size: int) -> list[int]:
    """Start positions of full tiles covering `size` pixels; the last one is aligned on the border."""
    starts = list(range(0, max(size - TILE, 0) + 1, TILE))
    if starts[-1] + TILE < size:
        starts.append(size - TILE)
    return starts


def to_png(normalized: np.ndarray) -> np.ndarray:
    """Normalized frame (% of median brightness, background ~100) -> uint8, DISPLAY_RANGE mapped to 0-255."""
    low, high = DISPLAY_RANGE
    return np.clip(np.rint((normalized - low) * (255 / (high - low))), 0, 255).astype(np.uint8)


def ring_shapes(labels: np.ndarray, sign: np.ndarray, snr: np.ndarray) -> np.ndarray:
    """Boolean per label: the shape touches an opposite-sign shape with a stronger DoG peak (it is a ring)."""
    strongest = np.zeros(len(sign))  # strongest opposite-sign neighbor of every shape
    for a, b in [(labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])]:
        touch = (a != b) & (a > 0) & (b > 0)
        a, b = a[touch], b[touch]
        opposite = sign[a] != sign[b]
        a, b = a[opposite], b[opposite]
        np.maximum.at(strongest, a, snr[b])
        np.maximum.at(strongest, b, snr[a])
    return strongest > snr


def predrawn_boxes(labels: np.ndarray, sign: np.ndarray, shapes: dict) -> list[tuple]:
    """Shapes worth pre-drawing -> [(row0, col0, row1, col1, class name)] in frame coordinates."""
    snr = shapes["dog_snr"]
    keep = (snr >= PREDRAW_MIN_SNR) & ~ring_shapes(labels, sign, snr)
    keep[0] = False
    boxes = []
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is not None and keep[label]:
            boxes.append((box[0].start, box[1].start, box[0].stop, box[1].stop, CLASSES[int(sign[label])]))
    return boxes


def tile_predictions(boxes: list[tuple], row: int, col: int) -> list[dict]:
    """Boxes of the frame -> Label Studio rectangle results for the tile at (row, col), in % of the tile."""
    results = []
    for r0, c0, r1, c1, name in boxes:
        tr0, tc0 = max(r0, row), max(c0, col)
        tr1, tc1 = min(r1, row + TILE), min(c1, col + TILE)
        if tr1 <= tr0 or tc1 <= tc0 or (tr1 - tr0) * (tc1 - tc0) < EDGE_MIN_INSIDE * (r1 - r0) * (c1 - c0):
            continue
        results.append({"from_name": "label", "to_name": "image", "type": "rectanglelabels",
                        "original_width": TILE, "original_height": TILE, "image_rotation": 0,
                        "value": {"x": 100 * (tc0 - col) / TILE, "y": 100 * (tr0 - row) / TILE,
                                  "width": 100 * (tc1 - tc0) / TILE, "height": 100 * (tr1 - tr0) / TILE,
                                  "rotation": 0, "rectanglelabels": [name]}})
    return results


def process_video(name: str, module, n_frames: int, tasks: list, frame_rows: list) -> None:
    """Background, frames, shapes, tiles and pre-drawn boxes of one video (settings from its phase module)."""
    n = module.frame_count(module.VIDEO_PATH)
    background = module.masked_mean_background(
        module.read_frames(module.VIDEO_PATH, np.linspace(module.FRAME_START, n - 1, module.BG_SAMPLE_FRAMES).astype(int)))
    for i in np.linspace(module.FRAME_START, n - 1, n_frames).astype(int):
        frame = module.read_frames(module.VIDEO_PATH, np.array([i]))[0]
        normalized = module.normalize(frame)
        difference = normalized - background
        dog = module.dog_filter(difference)
        labels, sign = module.dog_regions(dog)
        boxes = predrawn_boxes(labels, sign, module.measure_shapes(labels, sign, difference, dog))
        image = to_png(normalized)
        n_boxes = 0
        for row in tile_origins(image.shape[0]):
            for col in tile_origins(image.shape[1]):
                tile_name = f"{name}_f{i:05d}_r{row:04d}_c{col:04d}.png"
                Image.fromarray(image[row:row + TILE, col:col + TILE]).save(OUTPUT_DIR / "tiles" / tile_name)
                result = tile_predictions(boxes, row, col)
                n_boxes += len(result)
                tasks.append({"data": {"image": "/data/local-files/?d=" + quote(f"tiles/{tile_name}"),
                                       "video": name, "frame": int(i), "row": row, "col": col},
                              "predictions": [{"model_version": "v2_dog_shapes", "result": result}]})
        frame_rows.append([name, int(i), f"{np.median(frame):.0f}", len(boxes)])
        print(f"{name} frame {i}: median brightness {np.median(frame):.0f}, {len(boxes)} pre-drawn boxes "
              f"({n_boxes} in tiles)")


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"{OUTPUT_DIR} already exists: rename it or change OUTPUT_DIR (nothing is overwritten)")
    (OUTPUT_DIR / "tiles").mkdir(parents=True)
    tasks, frame_rows = [], []
    for name, module, n_frames in VIDEOS:
        process_video(name, module, n_frames, tasks, frame_rows)
    (OUTPUT_DIR / "tasks.json").write_text(json.dumps(tasks))
    (OUTPUT_DIR / "label_config.xml").write_text(LABEL_CONFIG)
    with open(OUTPUT_DIR / "frames.csv", "w", newline="") as file:
        csv.writer(file).writerows([["video", "frame", "median_brightness", "predrawn_boxes"], *frame_rows])
    print(f"{len(tasks)} tiles, {sum(len(t['predictions'][0]['result']) for t in tasks)} pre-drawn boxes -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
