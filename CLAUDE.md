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
data analysis and visualization (Python, OpenCV, scikit-image, napari, PyTorch).
The work is purely computational on existing microscopy videos — no wet-lab protocols.

## Current state
- Video preprocessing pipeline validated: temporal-median background removal →
  round-object removal (min-area rectangle) → hollow / out-of-focus object removal
  (hole ratio + two-threshold area ratio) → black-on-white mask video.
- Main script: `Video_Process_Script/Phase Extraction/Napari_Stationary_Process.py`
- Source videos: `Videosplit/`
- The project is **not** a git repository: never overwrite or rename a file without
  checking first that the destination does not exist.

## Way of working
- One step at a time. For each step, propose several solutions; the user picks,
  tests visually and validates before moving on.
- Remove comparison/test code from the main script once a method is chosen.

## Research journal (Obsidian)
- Vault folder: `/Users/juwul/Documents/Obsidian/Claude Obsidian/BacteriaSSL/`
- At the end of each work session, write a note in `BacteriaSSL/Sessions/`
  named `YYYY-MM-DD-topic.md` with these sections:
  goal · methodology · options considered · hypotheses tested · what worked ·
  what failed (and why) · decisions made · open questions / next steps.
- Keep the reasoning behind each decision, not just the outcome.
- Add a `[[Sessions/YYYY-MM-DD-topic]]` link to `BacteriaSSL/Index.md`.
