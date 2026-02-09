# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 14:42:32 2025

@author: rejwanfs
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")   # adjust if needed
import matplotlib

from config import full_dataset_dir
from supporting_functions import get_t_data, unit_smoothed_psth, \
    normalize_psth, id_fn, select_trials_from_stim_df, \
        compute_population_correlation

plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 12})
matplotlib.use('cairo')
# figtype = 'png'

#### PLOTTING PARAMETERS ####
plt.rcParams["figure.figsize"] = [19.2 ,  9.83]
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["figure.titlesize"] = 18
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams[''] = 
plt.rcParams.update({
    # 'xtick.labelsize': 12,   # X and Y tick label size
    # 'ytick.labelsize': 12,
    # 'xtick.color': 'blue',   # X and Y tick label color
    # 'ytick.color': 'blue',
    # 'xtick.direction': 'inout', # X and Y tick direction
    # 'ytick.direction': 'inout',
    'xtick.major.size': 8,   # X and Y tick major size
    'ytick.major.size': 8,
    'xtick.major.width': 1.5,  # X and Y tick major width
    'ytick.major.width': 1.5,
    'xtick.minor.size': 4,   # X and Y tick minor size
    'ytick.minor.size': 4,
    'xtick.minor.width': 1,  # X and Y tick minor width
    'ytick.minor.width': 1,
})
# plt.xticks(fontsize = 25)
# plt.yticks(fontsize = 25)
plt.rc('xtick',labelsize=30)
plt.rc('ytick',labelsize=30)
plt.rcParams["axes.labelsize"] = 25
plt.rc('legend',fontsize=10)
plt.rcParams["legend.markerscale"] = 1.5
plt.rcParams['lines.markersize']= 7.5
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams["font.size"] = 20 
##############################

# Set tick mark thickness and length
plt.rcParams['xtick.major.size'] = 8  # Major tick length
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 6  # Minor tick length
plt.rcParams['ytick.minor.size'] = 6

plt.rcParams['xtick.major.width'] = 3  # Major tick thickness
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.width'] = 2  # Minor tick thickness
plt.rcParams['ytick.minor.width'] = 2

# Set global savefig parameters
plt.rcParams["savefig.transparent"] = True  # Transparent background
plt.rcParams["savefig.bbox"] = "tight"  # Remove extra whitespace

plt.ioff()

# ---------------------- PARAMETERS (edit) ----------------------
pulse_times = np.arange(0, 5, 0.5)
pulse_colors = np.tile(['black', 'dimgray'], 10)
# ----------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------
bin_width = 0.05          # 50 ms
pre_stim = 3.5
post_stim = 2.5
stim_onset = 0.0

plotting_bounds = [-1, 6] # time in s for what to plot

zscore_baseline = True     # z-score using only baseline
average_trials = True      # OR: match trial numbers per neuron, see below
equalize_trials = True # equalize so 6 trials are chosen randomly
# Mouse / pooling choices
mouse_ids = None              # None => pool all mice in combined_data (recommended)
region_groups = [['PERI','ECT'], ['AUDp','AUDd','AUDv'], 
                 ['AUDp'], ['AUDd'], ['AUDv'], ['TeA']]  # list-of-lists; will loop over groups
cluster_group = 'good'
good_only = True

need_control = True # do not change, obviously

spike_count_based = False # use spike counts and not firing rates

manual_protocol_choice = True
f = 1 # frequency
n_cycles = 5 # num cycles < which is 0.5*number of deflections >
tot_stim_win_dur = 5.0

# time / PSTH
t_win = (-pre_stim, 5.0+post_stim)
dt = bin_width
pad = int(100*(0.001/dt))
sigma_ms = 0 # in ms
sigma = sigma_ms/(dt*1000)  # smoothing

# comparison choices
# Define which stim selector corresponds to each condition. If you know stim.type_ codes:
# For example audio-only types = [1] and audiotactile types = [2]  (replace with your values)
audio_only_types = None      # None -> use exp_block==0 (control) as audio-only; else list of ints
audiotactile_types = None    # None -> use exp_block==1 (all types) as A+T

normalise = False # keep this false, it doesn't affect outcome

# For side comparison: specify lists of stim.type_ that correspond to ipsi/contra.
ipsi_types = [2]
contra_types = [1]

# Bootstrapping / subsampling
n_boot_sub = 1000          # number of subsamples per neuron to estimate metric distribution
n_select = 30           # if not None, subsample n_select trials per cond (<= min available)
w_replace = True # sample trials with replacement
max_trials = 50 # number of trials to sample w/w/o replacement
rng_seed = 42

# How to aggregate: "trialavg" (default) uses trial-averaged PSTHs with subsampling,
# "singletrial" uses pairwise single-trial comparisons then averages within neuron
mode = "trialavg"  # or "singletrial"

suffix = ""
if spike_count_based:
    suffix = ", spike_count_based"
if zscore_baseline:
    suffix += ", base_z_scored"
if not average_trials:
    suffix += ", trial-based"
if equalize_trials:
    suffix += ", trial-equalized"

out_dir = f"./original/dt={int(dt*1000)}ms, sigma={sigma_ms}ms, t_win={t_win}{suffix}"
os.makedirs(out_dir, exist_ok=True)

# ---------- load combined data ----------
print("Loading combined_data.pkl ...")
pop_data = pd.read_pickle(f"{full_dataset_dir}/combined_data.pkl")
sp = pop_data.pop_sp
stims_all = pop_data.pop_stims
df_all = pop_data.df
del pop_data
gc.collect()

stim_meta = pd.read_excel(
    io=f"{full_dataset_dir}/stimulation_metadata.xlsx")

# group the mouse ids corresponding to same protocol
grouped_stim_meta = stim_meta.groupby(
    ['frequency', 'num_cycles', 'total_duration']).agg({
    'mouse_id': list,
    'baseline_duration': list
}).reset_index()

if manual_protocol_choice == True:
    grouped_stim_meta = grouped_stim_meta[
        (grouped_stim_meta.frequency == f) &
        (grouped_stim_meta.num_cycles == n_cycles) &
        (grouped_stim_meta.total_duration == tot_stim_win_dur)]
# get mouse ids from grouped stim meta
mouse_ids = grouped_stim_meta.mouse_id.values[0]

# --- NEW: optionally remove mice that do not have valid control stimulations ---
if need_control:
    # Read the stim_meta DataFrame (already loaded above). Look up each mouse_id
    # and keep only those with valid_control_stimulations == 'Yes' (case-insensitive).
    # The stim_meta may contain multiple rows per mouse (different protocols) — so
    # we check any row for that mouse that indicates valid control.
    valid_mask = stim_meta['valid_control_stimulations'].astype(str).str.lower() == 'yes'
    mice_with_valid_control = stim_meta.loc[valid_mask, 'mouse_id'].unique().tolist()

    # Intersect with the current mouse_ids
    prior_mouse_ids = list(mouse_ids) if isinstance(mouse_ids, (list, np.ndarray)) else mouse_ids
    filtered_mouse_ids = [m for m in prior_mouse_ids if m in mice_with_valid_control]

    if len(filtered_mouse_ids) == 0:
        raise RuntimeError("No mice remain after filtering for valid control stimulations. "
                           "Check 'valid_control_stimulations' column in stimulation_metadata.xlsx.")
    
    # 108025 has invalid control stims but can stil be salvaged
    if np.isin(108025, prior_mouse_ids):
        filtered_mouse_ids.append(108025)
    mouse_ids = filtered_mouse_ids

    # Optionally inform user
    print(f"[prepare_all_data] need_control=True -> keeping {len(mouse_ids)} mouse(s) with valid control stimulations.")
    
    # keep only the mice that have valid control stimulation data
    df_all = df_all[np.isin(df_all.mouse_id, mouse_ids)]

if normalise == True:
    norm_fn = normalize_psth
else:
    norm_fn = id_fn


# loop region groups
for region_group in region_groups:
    print("Processing region group:", region_group)

    # filter units
    df_region = df_all[df_all.region.isin(region_group)]
    if good_only:
        df_region = df_region[df_region.good_bool == True]
    df_region = df_region[df_region.cluster_group == cluster_group]
    df_region = df_region.reset_index(drop=True)
    if df_region.shape[0] == 0:
        print("No units found, skipping.")
        continue
    
    # time axis
    t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win = get_t_data(t_win, dt, pad)
    stim_mask = (t_mids >= 0) & (t_mids <= 5.0)   # same stim window as your other code

    # build psth arrays per neuron per mouse: we will store a list of (n_trials, ntbins) arrays
    all_neuron_psth_Aonly = []   # audio-only (control)
    all_neuron_psth_At  = []     # audio+tactile (experimental)
    all_neuron_meta = []         # (mouse_id, probe, region, cluster, unit_idx_in_df_region)
    all_mouse_meta = []          # (mouse_id, ntrials audio, ntrials audiotactile)

    # If user provided lists for audio-only / audiotactile mapping use those, else default:
    # default: audio-only -> exp_block 0; A+T -> exp_block 1
    use_audio_only_expblock = (audio_only_types is None and audiotactile_types is None)

    # iterate units: collect trials (pooled across mice)
    for i, row in df_region.reset_index(drop=True).iterrows():
        mouse_id = row.mouse_id
        probe = row.probe
        region = row.region
        cluster = row.cluster
        unit_sp = sp[mouse_id][probe][region][str(cluster)]
        
        if mouse_id == 108025:
            sub_stim_types = [4,1,2] # 4 is the unknown type due to video error
        else:
            sub_stim_types = [1,2]

        # choose trials for audio-only and A+T depending on parameters
        stims_mouse = stims_all[mouse_id]
        if use_audio_only_expblock:
            trials_Aonly = select_trials_from_stim_df(stims_mouse, exp_block=0, type_list=sub_stim_types)
            trials_At    = select_trials_from_stim_df(stims_mouse, exp_block=1, type_list=sub_stim_types)
        else:
            trials_Aonly = select_trials_from_stim_df(stims_mouse, exp_block=None, type_list=audio_only_types)
            trials_At    = select_trials_from_stim_df(stims_mouse, exp_block=None, type_list=audiotactile_types)

        # build PSTHs: unit_smoothed_psth expects trial_mat (n_trials x 2: [pulse_t, trial_idx])
        trials_Aonly_mat = np.c_[trials_Aonly, np.arange(len(trials_Aonly))]
        trials_At_mat = np.c_[trials_At, np.arange(len(trials_At))]

        if len(trials_Aonly) > 0:
            psth_Aonly = unit_smoothed_psth(unit_sp, trials_Aonly_mat, "stim", sigma, pad, t_edges, pad_t_edges, pad_t_win, dt, spike_count_based)
        else:
            psth_Aonly = np.empty((0, ntbins))

        if len(trials_At) > 0:
            psth_At = unit_smoothed_psth(unit_sp, trials_At_mat, "stim", sigma, pad, t_edges, pad_t_edges, pad_t_win, dt, spike_count_based)
        else:
            psth_At = np.empty((0, ntbins))

        all_neuron_psth_Aonly.append(norm_fn(psth_Aonly)) # (n_trials, ntbins)
        all_neuron_psth_At.append(norm_fn(psth_At))

        all_neuron_meta.append((mouse_id, probe, region, cluster, i))
        # this will contain alot of duplicates, which we will remove later
        all_mouse_meta.append((mouse_id, len(trials_Aonly), len(trials_At)))
    
    all_mouse_meta = list(set(all_mouse_meta))
    # ------------------------------ BOOTSTRAP OVER NEURONS ------------------------------
    n_neurons_total = len(all_neuron_psth_Aonly)
    rng = np.random.default_rng(rng_seed)
    
    if n_select is None:
        n_select = min(30, n_neurons_total)   # fallback default
    
    boot_corr = np.zeros((n_boot_sub, len(t_mids)))
    
    for b in range(n_boot_sub):
        # choose N neurons without replacement
        idx = rng.choice(n_neurons_total, size=n_select, replace=False)
    
        # subset PSTH lists
        subset_Aonly = [all_neuron_psth_Aonly[i] for i in idx]
        subset_At    = [all_neuron_psth_At[i]    for i in idx]
        
        # subset neuron_meta the same
        subset_neuron_meta = [all_neuron_meta[i] for i in idx]
    
        # compute correlation trace for this bootstrap
        boot_corr[b,:] = compute_population_correlation(
            subset_Aonly, subset_At, subset_neuron_meta, all_mouse_meta, t_mids,
            average_trials, zscore_baseline, equalize_trials, max_trials, 
            w_replace
        )
    
    # compute mean and std across bootstraps
    mean_corr = np.nanmean(boot_corr, axis=0)
    std_corr  = np.nanstd(boot_corr, axis=0)
    lower = mean_corr - std_corr
    upper = mean_corr + std_corr
    
    
    # ------------------------------ BOOTSTRAP CONFIDENCE INTERVALS ------------------------------
    mean_corr = np.nanmean(boot_corr, axis=0)
    lower_ci  = np.percentile(boot_corr, 2.5, axis=0)
    upper_ci  = np.percentile(boot_corr, 97.5, axis=0)
    
    # subset data for plotting
    plot_mask = (t_mids >= plotting_bounds[0]) & (t_mids <= plotting_bounds[1])
    
    plot_corr = mean_corr[plot_mask]
    t_plot = t_mids[plot_mask]
    plot_low_ci = lower_ci[plot_mask]
    plot_high_ci = upper_ci[plot_mask]
    ci_sig_mask = plot_low_ci > 0
    
    # ------------------------------ PLOT ------------------------------
    h = 8
    fig, ax = plt.subplots(figsize=(2.15*h, h))
    
    ax.plot(t_plot, plot_corr, color='black', label='Mean correlation')
    ax.fill_between(t_plot, plot_low_ci, plot_high_ci, alpha=0.3, color='gray')
    
    # ------------------------------ SIGNIFICANCE BARS ------------------------------
    from matplotlib.transforms import blended_transform_factory
    
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_frac = 0.93  # vertical location of bars in axes coordinates
    half_step = (t_mids[1] - t_mids[0]) / 2
    
    # Draw FDR-corrected significance bars (red)
    for t, sig in enumerate(ci_sig_mask):
        if sig:
            ax.hlines(
                y=y_frac,
                xmin=t_plot[t] - half_step,
                xmax=t_plot[t] + half_step,
                color='red',
                linewidth=4,
                transform=trans
            )

    # ------------------------------ PULSE LINES ------------------------------
    for t, c in zip(pulse_times, pulse_colors):
        ax.axvline(t, color=c, linestyle=(0,(3,3)), linewidth=3)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    
    plt.savefig(os.path.join(out_dir,
                             f"{n_boot_sub}_bootstrapped_{max_trials}_trials_pop_corr{'_'.join(region_group)}.png"),
                dpi=200)
    plt.savefig(os.path.join(out_dir,
                             f"{n_boot_sub}_bootstrapped_{max_trials}_trials_pop_corr{'_'.join(region_group)}.svg"))
    plt.close()



