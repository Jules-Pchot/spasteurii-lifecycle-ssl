his repository contains the code developed during a research internship at the EPFL Center for Imaging (Biomedical Imaging Group), under the supervision of Dr. Daniel Sage.

The project explores self-supervised deep learning methods to model the life cycle of Sporosarcina pasteurii — a bacterium used in biocementation (MICP) — from microscopy image sequences (snapshots and time-lapse), without manual annotations.

A structured latent space is learned from morphological features of the bacteria across growth stages (lag, log, stationary, decline). Temporal ordering constraints are applied to ensure the latent trajectory reflects the biological progression of the life cycle.

Stack : Python · PyTorch · SimCLR · Masked Autoencoder · UMAP · t-SNE
