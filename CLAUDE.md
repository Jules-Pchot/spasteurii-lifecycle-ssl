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
  - The preprocessing never classifies the growth stage: each video is one stage, both scripts run
    the same code and differ only by CONFIG. Bacteria of interest are dark; bright shapes are
    out-of-focus bacteria (or DoG rings), flagged, not removed.
  - Per-frame brightness normalization (% of frame median, lamp jumps) + masked mean background
    (validated), DoG filter (validated, band-pass rejected), shapes = DoG sign regions split
    by a masked watershed, mean / std on frame - background (validated). Whole-video TIFF output:
    postponed by the user (2026-10-20). Moving average (step 5) not needed for now (normalization
    leaves a ±0.5 % offset). Then a YOLO model (ultralytics, classes `in_focus` / `out_of_focus`).
  - YOLO labels are drawn **by hand** by the user (decision 2026-10-20), focus classes only. This is
    the one manual annotation of the project: the growth stage is never annotated.
  - Image processing with NumPy / SciPy; OpenCV only decodes the mp4.
  - Bimodal intensity histograms are shown in a napari dock panel.
- **YOLO focus detector: chapter closed (2026-10-23).** `Video_Process_Script/YOLO/`, run `v2_yolo11n`
  (Expo in_focus mAP50 0.50). Published as code + metrics only: no weights, no microscopy images.
- **Motion track (2026-10-26 →):** `Video_Process_Script/Motion_Tracking/`. Concept notebooks (marimo,
  `marimo run`, code hidden) on a synthetic simulator in µm / s with values from `Research Report.pdf`
  (no kinematic data published for S. pasteurii: proxies only). Next step: optical flow analysis.
- **Optic_Flow track (2026-11-09 →):** `Video_Process_Script/Optic_Flow/`, on `TIFFIle/Expo3_1080p_gray.tif` only
  (made by `video_to_tiff.py`). 01 stabilizer and 02 rolling background committed; segments 0–386 / 415–962
  processed separately. 03 spam DIC dropped (too few bacteria; `03_Local_DIC.py` kept, uncommitted).
  03 trackpy detection (`03_Trackpy.py`) in test: the user picks the mass threshold in napari; next = `tp.link`.
  04 scale space (`04_Scale_Space.py`, 2026-11-10, in test): in / out of focus from per-pixel DoG bands
  (octave vs half octave, isotropic), reference = segment 1 frame 273 (global 688); sharpening by band
  reweighting (napari sliders), heightmap = −sharpened colored by σ peak. Validation visual.
  The rolling mean of 02 erases many out-of-focus cells on Expo3 → second video `TIFFIle/0003_678-780s_1080p_gray.tif`
  (0003.mp4 11:18–13:00, 292 frames, `video_to_tiff.py --start/--end`), step 01 skipped (user), 02 done
  (one segment 0–291); 02 and 04 pick the video with `VIDEO` in CONFIG. 04 run on 0003 frame 207 (user's pick);
  grids extended to σ 1–32 (2026-11-11, user), output names carry the σ range.
  Analysis notebook `04_Scale_Space_Analysis.ipynb` (kernel `internship`, `%gui qt`): steps 0–11 drive the napari
  viewers of `build_viewers()`, read the user's class Shapes layers (`in_focus` / `out_of_focus` / `ambigu`), no interpretation;
  a "Concepts et transformations" cell (tag `concepts`) under each cell. Steps 9 LoG blobs and 10 multi-scale Hessian
  (half octave 1–32, 2026-11-11, in test, outputs `..._log.tif` / `_hessian_maps.tif` / `_log_blobs.csv`). Clumps: later.
- **Shape simulator (2026-11-12 →):** `Video_Process_Script/Shape_Simulator/shape_sim.py` (v2 validated, commit
  `0794947`). One S. pasteurii lineage per thumbnail (224 px, 0.06 µm/px), circle-chain shapes, brightfield rendering;
  thumbnail i of every phase (Lag / Expo / End_Expo / Stationary) = same lineage i (growth curve). All parameters in the
  `PARAMETERS` block at the top. Dataset `Processing_Results/Shape_Simulator/v2/` (1000 lineages). Encoder / clustering /
  decoder are done by the user in the PhenoME notebook, not here.
- **New acquisitions (protocol):** `Data_Collection/SPasteurii_Data_Collection_Process.tex` (user's text,
  approved changes only). ×40, 1080p, 30 fps (≥ 17), 10 s clips, 96-well plates, 12 cultures per medium.
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
- Commit to git after each step the user validates (user rule, 2026-10-19): the removed comparison
  code then stays recoverable. Do not commit steps that are still in test.

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
