# peri_tactile

This repository contains the scripts used to perform the analyses and generate the figures contained in the following manscript: 

[*Salih, R. H. F., Cobar, L. F., Pauzin, F. P., Zoccolan, D., & Nigro, M. J. (2025). Tactile responses in the mouse perirhinal cortex show invariance to physical features of the stimulus. bioRxiv, 2025-08.*](https://www.biorxiv.org/content/10.1101/2025.08.15.670508v1)

Virtually all scripts depend on a common data container, `Pop_data`, which aggregates:

- Single-unit spike trains grouped by mouse → probe → brain region
- Stimulation metadata for every trial
- Speed traces, when available
- A full per-unit metadata and quality-metrics table

The data are built by a preprocessing script that loads subject-level data, merges quality metrics, assigns anatomical regions, computes trial-level variability metrics, and serializes the results into a standard format. We provide a summary of the structure below (A more in-depth explanation of the preprocessing pipeline can be found in the folder `dataset_generation`). Note, however, that the scripts can be easily modified to not use this container.

The other folders contain the scripts used for analysis and plotting, along with a `README.md` file stating the figures they were used to generate. 
