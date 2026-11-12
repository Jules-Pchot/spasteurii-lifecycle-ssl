This repository contains the code developed during a research internship at the EPFL Center for Imaging (Biomedical Imaging Group), under the supervision of Dr. Daniel Sage.

The project explores self-supervised deep learning methods to model the life cycle of Sporosarcina pasteurii — a bacterium used in biocementation (MICP) — from microscopy image sequences (snapshots and time-lapse), without manual annotations.

A structured latent space is learned from morphological features of the bacteria across growth stages (lag, log, stationary, decline). Temporal ordering constraints are applied to ensure the latent trajectory reflects the biological progression of the life cycle.

The work is purely computational, on existing microscopy videos. Microscopy images and trained weights are not published in this repository: only code and metrics are.

Stack: Python · NumPy / SciPy · scikit-image · napari · trackpy · ultralytics (YOLO11) · PyTorch · DINOv2 · marimo

Planned for the representation learning: SimCLR · Masked Autoencoder · UMAP · t-SNE

## Progress

| Stage | Status | Where |
|---|---|---|
| Video preprocessing v1: background removal, shape filters, binary masks, batch run over every video of a phase | done | `Video_Process_Script/Phase_Extraction/`, `Video_Process_Script/Process_All_Phases.py` |
| Video preprocessing v2: brightness normalization, masked-mean background, DoG, threshold-free shapes | done | `Video_Process_Script/Phase_Extraction_v2/` |
| In-focus / out-of-focus cell detector (YOLO11): in-focus mAP50 0.50 on the exponential phase | done | [`Video_Process_Script/YOLO/`](Video_Process_Script/YOLO/README.md) |
| Motion track, concepts: synthetic motility simulator, detection, association, Kalman filter, trajectories and MSD, outliers | done | [`Video_Process_Script/Motion_Tracking/`](Video_Process_Script/Motion_Tracking/README.md) |
| Optic flow track, steps 1–2: video to grayscale TIFF (time-range clips), translation stabilizer, rolling masked-mean background | done | `Video_Process_Script/video_to_tiff.py`, `Video_Process_Script/Optic_Flow/` |
| Optic flow track, step 3: bacteria detection with trackpy | in test | `Video_Process_Script/Optic_Flow/03_Trackpy.py` |
| Optic flow track, step 4: in-focus / out-of-focus from scale space (DoG bands, LoG blobs, multi-scale Hessian), band-reweighting sharpening, heightmap, with an analysis notebook | in test | `Video_Process_Script/Optic_Flow/04_Scale_Space.py`, `04_Scale_Space_Analysis.ipynb` |
| Shape simulator: S. pasteurii lineages as circle chains, rendered as brightfield thumbnails, same lineage followed across the growth phases | done | `Video_Process_Script/Shape_Simulator/` |
| Self-supervised representation of morphological states (PhenoMe notebook, DINOv2 encoder, on the simulated dataset first) | in progress | `Video_Process_Script/Shape_Simulator/01_interactive_quickstart.ipynb` |
| Acquisition protocol for new videos (×40, 1080p, 30 fps, 10 s clips, 96-well plates) | written | `SPasteurii Data collection Process.pdf` |

## Notes

- Source videos are too large for git and are kept on external storage; scripts read their paths from the `CONFIG` block at the top of each file.
- The simulator parameters (cell lengths, phase windows, probabilities) are illustrative choices from the literature, not measurements.
- No absolute calibration (µm per pixel) is available for the source videos yet.
