# Dataset Generation Pipeline

The two scripts contained in this folder perform data ingestion (`combine_all_data.py`) and data on the periods of running for each mouse (`generate_running_bounds.py`). The latter also plots the running speed and superimposes blocks marking the periods of stimulation. Below, we explain the format of the raw data, and the structure of the custom data container (defined in `dataset_class_module/data_struct`), used in the former script to generate the combined dataset containing all the relevant data.

## 1. Constituting Data Organisation

The experimental data acquired in a given session is stored in a folder named by the corresponding mouse ID number. Within this folder are two `.mat` files containing:

- Speed data (two columns, one for speed and another for the time). Path and naming convention: `{mouse_dir}/{mouse_id}_Speed_Data.mat`
- Stimulation data, organised with the columns `['pulse_t','type_','exp_block','up','trial_idx']`, with a row for each movement of the stimulus. Path and naming convention: `{mouse_dir}/{mouse_id}_StimPulses_All.mat`

In addition, there are two folders `BAR/PER` which contain data from the two neuropixels probes whose primary targets were the Barrel cortex and Perirhinal cortex. All relevant files have a prefix `{probe_dir}/{mouse_id}_{probe_name}`, where `{probe_dir}` is either `PER` or `BAR`. Specifically, the folder contains:

- Quality metrics spreadsheet (suffix: `_metrics_test.xlsx`) with quality metrics listed as columns for each neuron (rows) extracted from kilosort. These metrics were all obtained through SpikeInterface.
- A probe location spreadsheet (suffix: `_Probe_Location.xlsx`) containing the columns `region	subregion	min_id	max_id`, which list the layers and electrode indexes for each region, e.g.

<img width="522" height="175" alt="image" src="https://github.com/user-attachments/assets/fbd4fa3f-ec9f-412e-9c49-0f1f3c349dce" />

- A dataset of spike times (suffix: `_Spikes.mat`), which is effectively the `rez.mat` file output by kilosort after spike-sorting.

## 2. The `Pop_data` class

This custom-made class centralises the loading and storage of all session data across all sessions. It consists of the following attributes:

- pop_sp — nested dictionary of spike trains
- pop_stims — per-mouse stimulation dataframes
- df — full metadata and quality metrics table
- spd — per-mouse speed traces (not used in the manuscript)

### 2.1 Spike Dictionary (`Pop_data.pop_sp`)

A nested dictionary with the structure:

mouse_id → probe → region → unit_id → spike_times_array

- mouse_id: integer ID
- probe: "BAR" or "PER"
- region: anatomical region inferred from probe location sheets
- unit_id: cluster ID stored as a string
- spike_times_array: 1-D NumPy array of spike times (seconds)

Noise units are removed before building this structure.

### 2.2 Stimulation Metadata (`pop_data.pop_stims`)

Each mouse has a stimulation dataframe containing:

- pulse_t: pulse time in milliseconds
- type_: stimulation type index (this is simply an integer that denotes which stimulation type the data corresponds to; ipsilateral, contralateral, or bilateral)
- exp_block: experimental block ID (0 and 1 to separate between audio-only and audio-tactile trial blocks)
- up: up/down deflection indicator (0 and 1 for the two directions; the stimulus moves in a square-wave pattern)
- trial_idx: integer specifying the trial number chronologically

### 2.3 Unit Metadata Table (`pop_data.df`)

A concatenated pandas DataFrame of quality metrics, containing one row per unit.

Key columns include:

- mouse_id
- cluster
- probe
- region
- subregion
- class (Regular or fast-spiking; see below)
- cluster_group (Multi-unit activity “MUA” or single-unit activity “good”)
- trial_cv
- good_bool (Quality control flag; see below)

Spike-shape rules:

- Fast-spiking; FS = duration < 0.4 ms
- Regular-spiking; RS = duration ≥ 0.4 ms

Quality control condition:

A unit is marked good_bool = True if:

- Trial-based coefficient of variation, trial_cv ≤ 1
- presence_ratio ≥ 0.9
- Whole-session spike count ≥ 500

QC flags annotate units but do not remove them.

### 2.4 Speed Traces (`pop_data.spd`)

A dictionary:

mouse_id → speed_trace_array or None

If no speed file exists for a mouse, the value is None.

## 3. Dataset Ingestion and Access Logic

The `combine_all_data.py` script will collect the raw data outlined above and transform them into the format outlined above for the `Pop_data` class. It will also initalise an instance of the `Pop_data` class and assign the transformed data to the four attributes of the class. The combined data and the four individual components will be individually saved. This allows one to load the entire dataset directly, or through the use of the `Pop_data` class:

```python
pop_data = Pop_data(data_dir="/path/to/data")
```

After loading, the data components can be accessed through the class attributes, e.g.

```python
# Load the dataset either through 
# Spikes from a region
pop_data.pop_sp[mouse_id]["BAR"]["CA1"]

# All PERI units
pop_data.df[pop_data.df.region == "PERI"]

# Stimulation table
pop_data.pop_stims[mouse_id]

# Speed trace
pop_data.spd[mouse_id]
```
