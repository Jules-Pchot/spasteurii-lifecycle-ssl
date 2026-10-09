---
title: Microscopy video processing — isolating in-focus bacteria
date: 2026-10-09
tags:
  - internship
  - microscopy
  - image-processing
  - napari
  - opencv
status: pipeline validated, full-video processing in place
---

# Microscopy video processing — isolating in-focus bacteria

> [!abstract] In one sentence
> A Python script ([[Napari_Stationary_Process]]) removes the background of a microscopy video, removes round objects and out-of-focus bacteria, then outputs a video of **black bacteria on a white background**.

## 1. Goal and way of working

**Goal:** see only the bacteria that are in the right focal plane. This means removing:

1. the **background**;
2. **round objects**;
3. bacteria that are **out of the focal plane** (large, or hollow with fine edges, as opposed to solid dark sticks).

The result is turned into black and white so that only the bacteria stand out.

**Working rule set at the start:**

- Go **one step at a time**.
- For each step, a **list of solutions** is proposed, you pick several, they are tested visually, and you **validate** (or not) before moving on.
- My role: write the code you ask for and propose image-processing solutions.

## 2. Final pipeline

```
video → [1] median background removed → [2] round objects removed (min-area rectangle)
      → [3a] hollow objects removed (hole ratio)
      → [3b] out-of-focus bacteria removed (two-threshold area ratio)
      → [4] plain mask → black-on-white video (.mp4)
```

| Step | Chosen method | Status |
|---|---|---|
| 1. Background | Temporal median of each pixel over all frames | validated |
| 2. Round objects | Min-area rectangle: elongation (+ fill) | validated |
| 3a. Hollow objects | Hole ratio | validated |
| 3b. Out of focus (blurry hollow bacteria) | Two-threshold area ratio | validated |
| 4. Output | Plain mask, then inverted to black on white | validated |

## 3. Decision history, step by step

### Step 1 — Remove the background
- **Choice:** median computed **for each pixel over all frames** of the video. Bacteria move, so they are not in the median.
- The median (the background) is subtracted from every frame, then re-centered on gray level 128. The background becomes mid-gray, and objects are darker or lighter than that gray.
- **Interpretation to note:** "the median calculated on every frame" was understood as a **temporal** median. The alternative was a spatial median blur on each frame; it was not used.
- **Limit:** a bacterium that stays still for most of the video is absorbed into the background.
- **Result:** validated by you ("Step 1 succeeded").

### Step 2 — Remove round objects
Options proposed: aspect ratio, eccentricity, circularity, solidity, **min-area rectangle (elongation + fill)**, **opening with a disk**, distance transform, Hough.

- **Tested:** E (min-area rectangle) and F (opening with a disk).
- **Chosen: E.** You judged the result "very good".
- **F dropped.** Limits I had flagged: it separates by width only, so a thick rod is treated as round; and a thin hollow ring disappears in the opening and passes for a rod.
- **Technical detail:** the contour runs through pixel centers, so 1 pixel is added to each side of the rectangle to recover the real width.

### Step 3 — Remove bacteria outside the focal plane
Options proposed: width, contrast, dark-core fill, edge sharpness, Laplacian variance, **band-pass**, **hole ratio**, erosion test.

- **Tested:** band-pass (difference of Gaussians) and hole ratio.
- **Synthetic test:** the band-pass catches **wide** objects (retention 0.19 for a wide bar), the hole ratio catches **hollow** objects (0.31 for a hollow rectangle).
- **Chosen: hole ratio** (object area divided by the area inside its outer contour). The band-pass code was then removed from the script.
- **Limit found later:** the hole ratio only sees holes **surrounded by a closed outline**. See the hollow-bacteria problem (section 4).

### Step 4 — Black-and-white output
Four drawing methods were implemented for comparison: **plain mask**, dilation, hysteresis growth, closing + hole filling.

- **Chosen: the plain mask.** The dilation, hysteresis and closing code was then removed from the main script.
- The question "white mask on black, or mask applied to the grayscale image?" was **dropped** at your request.
- **Final inversion requested:** the result video is now **black on a white background**.

## 4. The hollow out-of-focus bacteria problem

> [!bug] Diagnosis (you traced it to the temporary mask)
> A hollow out-of-focus bacterium shows **two thin dark edges** with a lighter interior. The temporary mask keeps the edges and loses the interior, so the bacterium is split into **two elongated objects**. Each one passes the elongation test **and** the hole ratio, because the space between the two edges is not surrounded by a closed outline. Result: several small "bacteria" appear in the final image.

Five options tested in [[Treshold_Test]] (formerly `Hollow_Bacteria_Test.py`):

| # | Method | Idea | Limit |
|---|---|---|---|
| 1 | Closing, then gap ratio | Close the mask to merge the two edges, then measure real pixels divided by closed area | Also removes two solid sticks lying close together; the disk size has to match the gap between the edges |
| **2** | **Two-threshold area ratio** | Area at the strict threshold divided by area at a lower threshold: the blurry interior only appears at the low threshold | Needs a second threshold; noise can link objects at the low threshold |
| 3 | Stronger blur | Merge the edges before thresholding, then a width limit | Thickens every bacterium and can merge neighbors |
| 4 | Parallel-pair grouping | Two close, parallel, facing objects are the two edges of the same bacterium | Wrongly groups two solid sticks side by side |
| 5 | Double-dip profile | Intensity across the width: one minimum for a solid stick, two for a hollow bacterium | The most complex |

- **Synthetic test:** options 1, 4 and 5 remove the hollow bacterium with the default settings, but also remove two solid sticks side by side. Options 2 and 3 only reacted when conditions suited them (interior blurry enough for option 2, edges close enough for option 3). The default settings are estimates to adjust on your video.
- **Choice: option 2 (two thresholds)**, after visual comparison on your real image.
- **Observation on your video:** on the test frame, the hole ratio removed nothing (168 objects kept out of 168). This is consistent with the diagnosis: it cannot see separated edges. The two-threshold test fills that gap.

## 5. Files

> [!info] Current locations (after your reorganization of October 9)
> - `Video_Process_Script/Phase Extraction/Napari_Stationary_Process.py`
> - `Video_Process_Script/Treshold_Test.py`
> - `Napari_Focus_Test.py` (project root)
> - Source videos are in `Videosplit/`.

### [[Napari_Stationary_Process]] — main script
*Former name: `Bacteria_Video_Process.py`, renamed at your request.*

**Role:** runs the whole pipeline and produces the final video. Launch: `python Napari_Stationary_Process.py`.

**How it works:**
- `main()` loads the video, computes the background, subtracts it, processes frame `TEST_FRAME`, then (if `PROCESS_VIDEO`) processes **all** frames, saves the `.mp4` and opens napari.
- napari first shows the debug layers of the test frame (temporary mask, scores, plain mask), then the **"RESULT video (all frames)"** layer on top.
- Per-frame messages are hidden while the video is processed; only `processing frame i/N` is printed.

**Main functions:**

| Function | Role |
|---|---|
| `load_video`, `median_background`, `subtract_background` | Step 1 (the median is computed in blocks of rows to save memory) |
| `deviation_image`, `segmentation_threshold`, `label_objects` | Temporary mask: distance from gray 128, Otsu threshold, labeling |
| `rectangle_measures`, `rectangle_rods` | Step 2: elongation and fill of the min-area rectangle |
| `hole_ratio`, `keep_above` | Step 3a: hole ratio and score-based filtering |
| `two_threshold_ratio` | Step 3b: two-threshold area ratio |
| `process_frame` | Full pipeline on one frame; returns every intermediate result |
| `process_video`, `video_fps`, `save_video` | Whole-video processing (black on white) and `.mp4` export |

**Parameters (current values):**

| Parameter | Value | Effect |
|---|---|---|
| `GRAY_LEVEL` | 128 | gray given to the background after subtraction |
| `TEST_FRAME` | 14 | frame used for tuning |
| `SEG_BLUR_SIGMA` | 1 | blur before thresholding |
| `SEG_THRESHOLD` | `None` | automatic Otsu threshold, **recomputed on every frame** |
| `SEG_MIN_AREA` | 30 | smaller objects are ignored (pixels) |
| `RECT_MIN_ELONGATION` | 3.0 | long side ÷ short side of the rectangle; below it, the object is round and removed |
| `RECT_MIN_FILL` | 0.0 | fill criterion, disabled |
| `HOLE_RATIO_MIN` | 0.9 | below it, the hollow object is removed |
| `LOOSE_THRESHOLD_RATIO` | 0.5 | low threshold = 0.5 × strict threshold |
| `TWO_THRESHOLD_MIN_RATIO` | 0.6 | strict area ÷ loose area; below it, the object is removed |
| `PROCESS_VIDEO` | `True` | `False` = test frame only (fast, for tuning) |
| `OUTPUT_PATH` | `None` | `None` = `Results/` folder next to the script |

**Test on your video** (`Stationary2_split_152.mp4`, 29 frames of 3584 × 2746): full processing in about 12 seconds, a 29-frame `.mp4` saved. This check was done with napari stubbed out and only shows pixel counts: **I never saw the visual result**, you are the one who validates it.

> [!warning] Default output path
> The default `Results/` folder is **next to the script**, so now `Video_Process_Script/Phase Extraction/Results/`. The video already produced is in `Results/` at the project root (created before the reorganization). You created `Processing Results/Stationnary/`: if that is the intended destination, set `OUTPUT_PATH`.

### [[Treshold_Test]] — hollow-bacteria test bench
*Former name: `Hollow_Bacteria_Test.py`. The docstring and the launch line still carry the old name.*

**Role:** reuses the median background and the validated steps of the main script on **one frame**, then saves one PNG image per option in `Hollow_Bacteria_Test_Output/` (created next to the script, in `Video_Process_Script/`).

**Image contents:**
- `00_background_removed`, `01_temporary_mask`, `02_baseline_current_pipeline`;
- then `03` to `07` for options 1 to 5.
- White = kept; **red = removed by the option compared to the baseline** (`SHOW_REMOVED_IN_RED`).
- Each option only **removes** objects from the baseline result, so the outline of the plain mask is unchanged.

**Default settings (estimates, to adjust):** `CLOSE_GAP_RADIUS` 6, `GAP_RATIO_MIN` 0.8, `LOOSE_THRESHOLD_RATIO` 0.5, `TWO_THRESHOLD_MIN_RATIO` 0.6, `BLUR_SIGMA` 3.0, `BLUR_MAX_WIDTH` 12, `PAIR_MAX_ANGLE` 10°, `PAIR_MAX_DISTANCE` 25, `PAIR_MIN_OVERLAP` 0.5, `DIP_PEAK_MIN` 0.5, `DIP_VALLEY_MAX` 0.7.

**Status:** concluded. Option 2 is built into the main script. The file serves as a record and could be archived.

**Fix made on October 9:** it now adds `Phase Extraction/` to `sys.path` to import the main script, and has its own `disk()` function since the main script no longer has one.

### [[Napari_Focus_Test]] — sharp vs. blurry test (project root)
**Role:** tests ways to separate in-focus bacteria from out-of-focus ones (band-pass, edge sharpness, Laplacian variance, erosion).

**Fix made on October 9:** the import now goes through `sys.path`, and the script uses `median_background` + `subtract_background` from the main script. The old calls `combined_background` and `gamma_transform` no longer exist anywhere in the project, so **the "gamma" step is gone**: the script now works on the plain background-removed frame (mid-gray background). The layer is renamed "background removed". Results may differ slightly from what the old version showed.

### Other project files
I did not write or modify these files. Descriptions come from their docstrings, without a full re-read.

- `Video_Process_Script/BG_Stat_Medians.py`: "Expo" pipeline applied to a Stationary split with two median backgrounds compared in napari (formerly `Napari_Stationary_Medians.py`).
- `Video_Process_Script/Phase Extraction/Napari_Expo_Process.py`: median background + removal of round objects by eccentricity, black-on-white output.
- `Video_Process_Script/downscale_video.py`, `Videosplit/split_video.py`: video utilities.
- Existing notes and projects: `README.md`, `contexte_projet_microscopy_ai.md`, `Base/` (notebook and `model.pt`).

## 6. Incidents and points of attention

> [!danger] To check
> - **Rename done without checking first.** I renamed `Bacteria_Video_Process.py` to `Napari_Stationary_Process.py` with `mv` without checking that no file with that name already existed (`mv` overwrites without warning). Hints that the name was free: the file did not exist at the start of the session, and your old script seems to have become `BG_Stat_Medians.py`.
> - **A possible loss.** `combined_background` and `gamma_transform`, which `Napari_Focus_Test.py` used to call, are **not in any file of the project** (not in `BG_Stat_Medians.py` either). They were in the old `Napari_Stationary_Process.py`. I cannot tell whether that file was deleted or edited before the session, or overwritten by my `mv`. The project is not a git repository, so there is no history to recover from. **Check whether you have a backup (Time Machine, etc.) or another copy of that older version.**

**Fixed on October 9:**
- **Broken imports.** `Treshold_Test.py` and `Napari_Focus_Test.py` imported `Video_Process_Script.Napari_Stationary_Process`, which does not exist (the module is in `Phase Extraction/`, and a folder name with a space cannot be imported as a package). Both now add the right folder to `sys.path`. Checked by running all three scripts from another working directory with napari stubbed out: `Treshold_Test` on a synthetic frame, `Napari_Focus_Test` and `Napari_Stationary_Process` on the test frame of your real video.
- **Comparison code.** The dilation, hysteresis growth and closing + hole filling methods, their parameters and their napari layers were removed from the main script. The only remaining step-4 layer is "4 plain mask".
- **Leftovers.** The `Hollow_Bacteria_Test_Output/` folder at the project root (8 PNGs of October 5, regenerable by running `Treshold_Test.py`) and the old `__pycache__` folder were deleted.

**Remaining remarks:**
- **Possible flicker.** The Otsu threshold is recomputed on every frame. On the first 5 frames, the white pixel counts were 89,292, 74,632, 93,303, 107,354 then 180,399 (before the inversion). It may be real, or due to the threshold; if the bacteria blink when playing the video, the threshold can be computed **once** for the whole video.
- **Motionless bacteria.** They are absorbed by the temporal median (step 1 limit).
- **Duplicated code.** The two-threshold test exists in both `Treshold_Test.py` and the main script.
- **Settings to refine on the real video.** Notably `TWO_THRESHOLD_MIN_RATIO` (0.6), `LOOSE_THRESHOLD_RATIO` (0.5) and `RECT_MIN_ELONGATION` (3.0).

## 7. Quick glossary

- **Otsu:** automatic method that picks a threshold separating two populations of pixels.
- **Temporal median:** for each pixel, the median value over all frames; keeps what does not move.
- **Min-area rectangle:** smallest oriented rectangle containing the object; its elongation (long side ÷ short side) tells a rod from a round object.
- **Hole ratio:** object area ÷ area inside its outer contour (1 = solid).
- **Two-threshold ratio:** area at the strict threshold ÷ area at the low threshold; low when the object has a blurry interior.
- **Out of focus:** blurry object, large or hollow with fine edges, as opposed to a solid, dark, sharp rod.
