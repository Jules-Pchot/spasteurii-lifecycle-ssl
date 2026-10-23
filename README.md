This repository contains the code developed during a research internship at the EPFL Center for Imaging (Biomedical Imaging Group), under the supervision of Dr. Daniel Sage.

The project explores self-supervised deep learning methods to model the life cycle of Sporosarcina pasteurii — a bacterium used in biocementation (MICP) — from microscopy image sequences (snapshots and time-lapse), without manual annotations.

A structured latent space is learned from morphological features of the bacteria across growth stages (lag, log, stationary, decline). Temporal ordering constraints are applied to ensure the latent trajectory reflects the biological progression of the life cycle.

Stack : Python · PyTorch · SimCLR · Masked Autoencoder · UMAP · t-SNE

## Progress

| Stage | Status | Where |
|---|---|---|
| Video preprocessing v1: background removal, shape filters, binary masks | done | `Video_Process_Script/Phase_Extraction/` |
| Video preprocessing v2: brightness normalization, masked-mean background, DoG, threshold-free shapes | done | `Video_Process_Script/Phase_Extraction_v2/` |
| In-focus / out-of-focus cell detector (YOLO11): in-focus mAP50 0.50 on the exponential phase | done | [`Video_Process_Script/YOLO/`](Video_Process_Script/YOLO/README.md) |
| Self-supervised representation of morphological states | next | — |
