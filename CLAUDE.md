# BacteriaSSL — Project context

## What this project is
A computer-vision / machine-learning research internship project.
Goal: model the life cycle of *Sporosarcina pasteurii* (a non-pathogenic soil bacterium,
widely studied for biocementation / microbially induced calcite precipitation) from
microscopy image sequences (snapshots, time-lapse), **without manual annotations**.

A self-supervised model will learn a representation of morphological states in a
structured latent space, so that transitions between biological stages can be
identified automatically.

**Scope:** image processing, segmentation, self-supervised representation learning,
data analysis and visualization (Python, NumPy / SciPy, scikit-image, napari, PyTorch, ultralytics).
The work is purely computational on existing microscopy videos — no wet-lab protocols.

## Current state
- **Pipeline v1 (validated, kept as is):** temporal-median background removal →
  round-object removal (min-area rectangle) → hollow / out-of-focus object removal
  (hole ratio + two-threshold area ratio) → black-on-white mask video.
  Scripts: `Video_Process_Script/Phase_Extraction/` — outputs: `Processing_Results/<Phase>/` (mp4).
- **Pipeline v2 (in progress, main work):** `Video_Process_Script/Phase_Extraction_v2/`
  (`Napari_Expo_Process_v2.py`, `Napari_Stationary_Process_v2.py`, same code, only CONFIG differs).
  - No threshold in the output: intensity is kept and measured (mean / std per shape).
  - Background = masked mean (validated), DoG filter (validated, band-pass rejected). Next:
    per-shape measures, TIFF output (int16 + zlib), then a YOLO model (ultralytics, classes
    `in_focus` / `out_of_focus`, no phase labels).
  - Image processing with NumPy / SciPy; OpenCV only decodes the mp4.
  - Bimodal intensity histograms are shown in a napari dock panel.
- Source videos (v2): external drive `/Volumes/Extreme SSD/Video_test/` (`Expo.mp4`, `Stationary.mp4`).
  Too big for RAM: the background is estimated from frames sampled over the video.
- Magnifications: Expo ×40 throughout; Stationary ×40 then ×100 (objective change at frames
  360–367) — only the ×100 part is processed (`FRAME_START = 368`). Measured widths: ~7 px (×40),
  ~18 px (×100). Absolute µm/px unknown (no calibration).
- Folder names: no spaces, same phase spelling everywhere (`Expo`, `Stationary`, `Death`).
- The project is a git repository, but data and results are not all tracked: never overwrite
  or rename a file without checking first that the destination does not exist.

## Way of working
- One step at a time. For each step, propose several solutions; the user picks,
  tests visually and validates before moving on.
- Remove comparison/test code from the main script once a method is chosen.

## Research journal (Obsidian)
- Vault folder: `/Users/juwul/Documents/Obsidian/Claude Obsidian/BacteriaSSL/`
- At the end of each work session, write a note in `BacteriaSSL/Sessions/`
  named `YYYY-MM-DD-topic.md` with these sections:
  goal · methodology · options considered · hypotheses tested · what worked ·
  what failed (and why) · decisions made · **notions** (short definitions of the concepts
  used or explained during the session) · open questions / next steps.
- Keep the method logs in `BacteriaSSL/Methods/` up to date during the session: every option
  explored (chosen, rejected, dropped, proposed), its status, the reason and the numbers.
  Rejected methods must stay traceable there even after their code is removed: they are
  needed for the report and the presentations.
- Keep the reasoning behind each decision, not just the outcome.
- Add a `[[Sessions/YYYY-MM-DD-topic]]` link to `BacteriaSSL/Index.md`.
