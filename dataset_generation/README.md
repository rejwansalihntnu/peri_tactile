# Dataset Generation Pipeline

The two scripts contained in this folder perform data ingestion (`combine_all_data.py`) and data on the periods of running for each mouse (`generate_running_bounds.py`). The latter also plots the running speed and superimposes blocks marking the periods of stimulation. Below, we explain the format of the raw data, and the structure of the custom data container (defined in `dataset_class_module/data_struct`), used in the former script to generate the combined dataset containing all the relevant data.

## 1. Constituting Data Organisation

The experimental data acquired in a given session is stored in a folder named by the corresponding mouse ID number. Within this folder are two `.mat` files containing:

- Speed data (two columns, one for speed and another for the time). Path and naming convention: `{mouse_dir}/{mouse_id}_Speed_Data.mat`
- Stimulation data, organised with the columns `['pulse_t','type_','exp_block','up','trial_idx']`, with a row for each movement of the stimulus. Path and naming convention: `{mouse_dir}/{mouse_id}_StimPulses_All.mat`

In addition, there are two folders `BAR/PER` which contain data from the two neuropixels probes whose primary targets were the Barrel cortex and Perirhinal cortex. All relevant files have a prefix `{probe_dir}/{mouse_id}_{probe_name}`, where `{probe_dir}` is either `PER` or `BAR`. Specifically, the folder contains:

- Quality metrics spreadsheet (suffix: `_metrics_test.xlsx`) with quality metrics listed as columns for each neuron (rows) extracted from kilosort.
- A probe location spreadsheet (suffix: `_Probe_Location.xlsx`) containing the columns `region	subregion	min_id	max_id`, which list the layers and electrode indexes for each region, e.g.

<img width="522" height="175" alt="image" style="display: block; margin: 0 auto;" src="https://github.com/user-attachments/assets/fbd4fa3f-ec9f-412e-9c49-0f1f3c349dce" />










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
- up: up/down deflection indicator (0 and 1 for the two directions; the stimulus moves in a square-wave pattern)
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
