# peri_tactile

This repository contains the scripts used to perform the analyses and generate the figures contained in the following manscript: 

[*Salih, R. H. F., Cobar, L. F., Pauzin, F. P., Zoccolan, D., & Nigro, M. J. (2025). Tactile responses in the mouse perirhinal cortex show invariance to physical features of the stimulus. bioRxiv, 2025-08.*](https://www.biorxiv.org/content/10.1101/2025.08.15.670508v1)

Virtually all scripts depend on a common data container, `Pop_data`, which aggregates:

- Single-unit spike trains grouped by mouse → probe → brain region
- Stimulation metadata for every trial
- Speed traces, when available
- A full per-unit metadata and quality-metrics table

The data are built by a preprocessing script that loads subject-level data, merges quality metrics, assigns anatomical regions, computes trial-level variability metrics, and serializes the results into a standard format. We provide a summary of the structure below (A more in-depth explanation of the preprocessing pipeline can be found in the folder `dataset_generation`). Note, however, that the scripts can be easily modified to not use this container.

## Pop_data Class Structure

After preprocessing, an instance of `Pop_data` contains four main attributes:

- pop_sp — nested dictionary of spike trains
- pop_stims — per-mouse stimulation dataframes
- df — full metadata and quality metrics table
- spd — per-mouse speed traces (not used in the manuscript)

## 1. Spike Dictionary (pop_data.pop_sp)

A nested dictionary with the structure:

mouse_id → probe → region → unit_id → spike_times_array

- mouse_id: integer ID
- probe: "BAR" or "PER"
- region: anatomical region inferred from probe location sheets
- unit_id: cluster ID stored as a string
- spike_times_array: 1-D NumPy array of spike times (seconds)

Noise units are removed before building this structure.

## 2. Stimulation Metadata (pop_data.pop_stims)

Each mouse has a stimulation dataframe containing:

- pulse_t: pulse time in milliseconds
- type_: stimulation type index (this is simply an integer that denotes which stimulation type the data corresponds to; ipsilateral, contralateral, or bilateral)
- exp_block: experimental block ID (0 and 1 to separate between audio-only and audio-tactile trial blocks)
- up: up/down deflection indicator (as the stimulus moves in a square-wave pattern)
- trial_idx: integer specifying the trial number chronologically

## 3. Unit Metadata Table (pop_data.df)

A concatenated pandas DataFrame containing one row per unit.

Key columns include:

- mouse_id
- cluster
- probe
- region
- subregion
- class (FS or RS)
- cluster_group (“MUA” or “good”)
- trial_cv
- good_bool (QC flag; see below)

Region name normalization:

- AUD → AUDv, AUDd, or AUDp  
- Tea/TEa → TeA

Spike-shape rules:

- Fast-spiking; FS = duration < 0.4 ms
- Regular-spiking; RS = duration ≥ 0.4 ms

## 4. Speed Traces (pop_data.spd) (not used for manuscript)

A dictionary:

mouse_id → speed_trace_array or None

If no speed file exists for a mouse, the value is None.

## Access Examples

```python
# Spikes from a region
pop_data.pop_sp[mouse_id]["BAR"]["CA1"]

# All PERI units
pop_data.df[pop_data.df.region == "PERI"]

# Stimulation table
pop_data.pop_stims[mouse_id]

# Speed trace
pop_data.spd[mouse_id]
```

## QC Logic

A unit is marked good_bool = True if:

- trial_cv ≤ 1
- presence_ratio ≥ 0.9
- spike count ≥ 500

QC flags annotate units but do not remove them.
