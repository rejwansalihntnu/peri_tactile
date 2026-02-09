# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:49:26 2024

@author: rejwan
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

import pandas as pd
import os

import gc

# Function to format y-axis as percentages
def to_percentage_no_symbol(y, position):
    return f"{y * 100:.3g}"

# as prepare_all_data but with spike counts instead of firing rate
def prepare_all_data_spike_counts(full_dataset_dir, manual_protocol_choice, f,
                                  n_cycles, tot_stim_win_dur, stim_names, 
                                  stim_types, t_win, dt, pad, good_only, 
                                  cluster_group, region_groups, sigma, 
                                  stim_type_cols, kmeans_dir):
    
    # open stim metadata and use it to group similar protocols, or decide 
    stim_meta = pd.read_excel(
        io=f"{full_dataset_dir}/stimulation_metadata.xlsx")
    
    # load population data
    if os.getcwd()[0] == 'C':
        pop_data = pd.read_pickle("./../combined_data.pkl")
    else:
        pop_data = pd.read_pickle(f"{full_dataset_dir}/combined_data.pkl")
    # pop_data = pd.read_pickle("../combined_data.pkl")
    
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
    
    # take out the constituting data
    sp = pop_data.pop_sp
    stims = pop_data.pop_stims
    df = pop_data.df
    
    # take out data per mouse to feed into data for separation
    stim_dict = partition_stims(mouse_ids, stims, stim_names)
    
    # get num stims in each condition and mouse for bootstrapping later
    mouse_num_stims = {}
    for mouse_id in mouse_ids:
        iter_dat = mouse_num_stims[mouse_id] = []
        
        for type_ in stim_types:
            type_name = stim_names[type_]
            num_type = np.shape(stim_dict[mouse_id][type_name])[0]
            iter_dat.append(num_type)
        # go through, add key as mouse id, with two-len arrays of [0] = con
        #num trials and [1] = ips num trials
    
    # generate bin edges and midpoints
    t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win = \
        get_t_data(t_win, dt, pad)
        
    # filter out units based on criteria you chose
    if good_only == True:
        df = df[df.good_bool == True]
    # choose unit type i.e. single units or MUA
    df = df[df.cluster_group == cluster_group]
    
    # get all units in regions of interest
    mice_idxs = np.isin(df.mouse_id, mouse_ids)
    group_idxs = np.isin(df.region, np.concatenate(region_groups))
    group_idxs = (mice_idxs & group_idxs)
    # limit df to this while resetting index
    df_valid = df.loc[df.index[group_idxs],:].copy().reset_index()
    tot_units = len(df_valid)
    
    
    
    # to improve sampling, I'm changing this to pick shuffle idxs so each 
    #bootstrap also randomises the trials in each group
    psth_dict = {}
    for type_ in stim_types:
        type_name = stim_names[type_]
        psth_dict[type_name] = np.empty(tot_units, dtype=object)
        
        for i, row in df_valid.iterrows():
            
            # get neuron's metadata (see top comment for why indexing from 1)
            mouse_id, cluster, probe, region = row.iloc[1:5]
            
            # extract neuron's spikes and associated stims
            unit_sp = sp[mouse_id][probe][region][str(cluster)]
            iter_stims = stim_dict[mouse_id][type_name]
            unit_psth = unit_smoothed_psth(unit_sp, iter_stims, type_name, sigma, 
                                           pad, t_edges, pad_t_edges, pad_t_win,
                                           dt)
            # convert to spike counts
            unit_psth *= dt
            psth_dict[type_name][i] = unit_psth.astype(int)
            
    # remove unused large data to free up RAM for multiprocessing
    del sp, pop_data
    gc.collect()
    
    # create label
    ## make stim window have relevant label
    stim_idxs = (t_mids < 5) & (t_mids > 0)
    type_names = list(psth_dict.keys())
    
    # load k means data <we just need the labels>
    kmeans = np.load(f"{kmeans_dir}/k_means_output.npy", allow_pickle=True)
    # now perform svm classification on these two stim types
    print("plt.title/np.save needs fixing if more tha one cluster group")
    
    return df_valid, psth_dict, kmeans, ntbins, mouse_ids, mouse_num_stims, \
        type_names, t_mids, stim_idxs


def prepare_all_data(full_dataset_dir, manual_protocol_choice, f, n_cycles,
                     tot_stim_win_dur, stim_names, stim_types, t_win, dt, pad,
                     good_only, cluster_group, region_groups, sigma, 
                     stim_type_cols, kmeans_dir):
    
    # open stim metadata and use it to group similar protocols, or decide 
    stim_meta = pd.read_excel(
        io=f"{full_dataset_dir}/stimulation_metadata.xlsx")
    
    # load population data
    if os.getcwd()[0] == 'C':
        pop_data = pd.read_pickle("./../combined_data.pkl")
    else:
        pop_data = pd.read_pickle(f"{full_dataset_dir}/combined_data.pkl")
    
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
    
    # take out the constituting data
    sp = pop_data.pop_sp
    stims = pop_data.pop_stims
    df = pop_data.df
    
    # take out data per mouse to feed into data for separation
    stim_dict = partition_stims(mouse_ids, stims, stim_names)
    
    # get num stims in each condition and mouse for bootstrapping later
    mouse_num_stims = {}
    for mouse_id in mouse_ids:
        iter_dat = mouse_num_stims[mouse_id] = []
        
        for type_ in stim_types:
            type_name = stim_names[type_]
            num_type = np.shape(stim_dict[mouse_id][type_name])[0]
            iter_dat.append(num_type)
        # go through, add key as mouse id, with two-len arrays of [0] = con
        #num trials and [1] = ips num trials
    
    # generate bin edges and midpoints
    t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win = \
        get_t_data(t_win, dt, pad)
        
    # filter out units based on criteria you chose
    if good_only == True:
        df = df[df.good_bool == True]
    # choose unit type i.e. single units or MUA
    df = df[df.cluster_group == cluster_group]
    
    # get all units in regions of interest
    mice_idxs = np.isin(df.mouse_id, mouse_ids)
    group_idxs = np.isin(df.region, np.concatenate(region_groups))
    group_idxs = (mice_idxs & group_idxs)
    # limit df to this while resetting index
    df_valid = df.loc[df.index[group_idxs],:].copy().reset_index()
    tot_units = len(df_valid)
    
    
    
    # to improve sampling, I'm changing this to pick shuffle idxs so each 
    #bootstrap also randomises the trials in each group
    psth_dict = {}
    for type_ in stim_types:
        type_name = stim_names[type_]
        psth_dict[type_name] = np.empty(tot_units, dtype=object)
        
        for i, row in df_valid.iterrows():
            
            # get neuron's metadata (see top comment for why indexing from 1)
            mouse_id, cluster, probe, region = row.iloc[1:5]
            
            # extract neuron's spikes and associated stims
            unit_sp = sp[mouse_id][probe][region][str(cluster)]
            iter_stims = stim_dict[mouse_id][type_name]
            unit_psth = unit_smoothed_psth(unit_sp, iter_stims, type_name, sigma, 
                                           pad, t_edges, pad_t_edges, pad_t_win,
                                           dt)
            psth_dict[type_name][i] = unit_psth
            
    # remove unused large data to free up RAM for multiprocessing
    del sp, pop_data
    gc.collect()
    
    # create label
    ## make stim window have relevant label
    stim_idxs = (t_mids < 5) & (t_mids > 0)
    type_names = list(psth_dict.keys())
    
    # load k means data <we just need the labels>
    kmeans = np.load(f"{kmeans_dir}/k_means_output.npy", allow_pickle=True)
    # now perform svm classification on these two stim types
    print("plt.title/np.save needs fixing if more tha one cluster group")
    
    return df_valid, psth_dict, kmeans, ntbins, mouse_ids, mouse_num_stims, \
        type_names, t_mids, stim_idxs


def prepare_all_data_blockwise(full_dataset_dir, manual_protocol_choice, f, 
                               n_cycles, tot_stim_win_dur, stim_names, 
                               stim_types, t_win, dt, pad, good_only, 
                               cluster_group, region_groups, sigma, 
                               stim_type_cols, exp_block=1, 
                               need_control=False):
    
    # open stim metadata and use it to group similar protocols, or decide 
    stim_meta = pd.read_excel(
        io=f"{full_dataset_dir}/stimulation_metadata.xlsx")
    
    # load population data
    if os.getcwd()[0] == 'C':
        pop_data = pd.read_pickle("./../combined_data.pkl")
    else:
        pop_data = pd.read_pickle(f"{full_dataset_dir}/combined_data.pkl")
    # pop_data = pd.read_pickle("../combined_data.pkl")
    
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
    
        mouse_ids = filtered_mouse_ids
    
        # Optionally inform user
        print(f"[prepare_all_data] need_control=True -> keeping {len(mouse_ids)} mouse(s) with valid control stimulations.")
    # --- end NEW block ---
    
    # take out the constituting data
    sp = pop_data.pop_sp
    stims = pop_data.pop_stims
    df = pop_data.df
    
    # take out data per mouse to feed into data for separation
    stim_dict = partition_stims(mouse_ids, stims, stim_names, exp_block, n_cycles)
    
    # get num stims in each condition and mouse for bootstrapping later
    mouse_num_stims = {}
    # track min stims per condition and mouse
    min_num_stims = np.inf
    for mouse_id in mouse_ids:
        iter_dat = mouse_num_stims[mouse_id] = []
        
        for type_ in stim_types:
            type_name = stim_names[type_]
            num_type = np.shape(stim_dict[mouse_id][type_name])[0]
            iter_dat.append(num_type)
            min_num_stims = np.min([min_num_stims, num_type])
        # go through, add key as mouse id, with two-len arrays of [0] = con
        #num trials and [1] = ips num trials
    
    # repeat procedure for opposite exp block to get global min num stims
    stim_dict_opposite = partition_stims(mouse_ids, stims, stim_names, 
                                         (exp_block+1)%2, n_cycles)
    
    for mouse_id in mouse_ids:
        for type_ in stim_types:
            type_name = stim_names[type_]
            num_type = np.shape(stim_dict_opposite[mouse_id][type_name])[0]
            min_num_stims = np.min([min_num_stims, num_type])
    
    
    # generate bin edges and midpoints
    t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win = \
        get_t_data(t_win, dt, pad)
        
    # filter out units based on criteria you chose
    if good_only == True:
        df = df[df.good_bool == True]
    # choose unit type i.e. single units or MUA
    df = df[df.cluster_group == cluster_group]
    
    # get all units in regions of interest
    mice_idxs = np.isin(df.mouse_id, mouse_ids)
    group_idxs = np.isin(df.region, np.concatenate(region_groups))
    group_idxs = (mice_idxs & group_idxs)
    # limit df to this while resetting index
    df_valid = df.loc[df.index[group_idxs],:].copy().reset_index()
    tot_units = len(df_valid)
    
    
    
    # to improve sampling, I'm changing this to pick shuffle idxs so each 
    #bootstrap also randomises the trials in each group
    psth_dict = {}
    for type_ in stim_types:
        type_name = stim_names[type_]
        psth_dict[type_name] = np.empty(tot_units, dtype=object)
        
        for i, row in df_valid.iterrows():
            
            # get neuron's metadata (see top comment for why indexing from 1)
            mouse_id, cluster, probe, region = row.iloc[1:5]
            
            # extract neuron's spikes and associated stims
            unit_sp = sp[mouse_id][probe][region][str(cluster)]
            iter_stims = stim_dict[mouse_id][type_name]
            unit_psth = unit_smoothed_psth(unit_sp, iter_stims, type_name, sigma, 
                                           pad, t_edges, pad_t_edges, pad_t_win,
                                           dt)
            psth_dict[type_name][i] = unit_psth
            
    # remove unused large data to free up RAM for multiprocessing
    del sp, pop_data
    gc.collect()
    
    # create label
    ## make stim window have relevant label
    stim_idxs = (t_mids < 5) & (t_mids > 0)
    type_names = list(psth_dict.keys())
    
    return df_valid, psth_dict, ntbins, mouse_ids, mouse_num_stims, \
        type_names, t_mids, stim_idxs, int(min_num_stims)

def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X.T).T
    return Xz

def partition_stims(mouse_ids, pop_stims, stim_names, exp_block=1, n_cycles=5):
    """
    Returns dict of stim times for each mouse and stim type.
    exp_block: 0 = control, 1 = experimental
    """
    n_deflect = int(2*n_cycles)
    mouse_stim_dict = {}
    for mouse_id in mouse_ids:
        mouse_stims = pop_stims[mouse_id]
        # filter by block
        mouse_stims = mouse_stims[mouse_stims.exp_block == exp_block]
        # take only the pulse times and stim id
        mouse_stims = mouse_stims[['pulse_t','type_','trial_idx']].values
        # take first stim times for baseline windows and trial durations
        first_stims = mouse_stims[::n_deflect,:]
        mouse_stim_dict[mouse_id] = {}
        for stim_idx, stim_name in stim_names.items():
            valid_rows = first_stims[:,1] == stim_idx
            valid_data = first_stims[valid_rows][:,(0,2)]
            mouse_stim_dict[mouse_id][stim_name] = valid_data
    return mouse_stim_dict

def get_t_data(t_win, dt, pad):
    
    t_edges = dt * np.arange(t_win[0]/dt, t_win[1]/dt + 1, 1)
    t_mids = 0.5 * (t_edges[1:] + t_edges[:-1])
    ntbins = len(t_mids)
    pad_t_edges = dt * np.arange(t_win[0]/dt - pad, t_win[1]/dt + 1 + pad, 1)
    pad_t_mids = 0.5 * (pad_t_edges[1:] + pad_t_edges[:-1])
    pad_t_win = [pad_t_edges[0], pad_t_edges[-1]]
    
    return t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win



def psth(aligned_sp, t_edges, num_stims, dt=0.001):
    
    bin_counts, _ = np.histogram(aligned_sp, t_edges, density = False) 
    psth = bin_counts / (num_stims * dt)
    
    return psth


# z score according to the baseline
def base_z_score_data(data, baseline_start, baseline_end, dt=0.001):
    """
    Z-score the data based on the specified baseline period.

    Parameters:
    data: np.ndarray
        The firing rate data for a neuron.
    baseline_start: int
        The starting index of the baseline period. In ms
    baseline_end: int
        The ending index of the baseline period. In ms.

    Returns:
    np.ndarray
        The z-scored data.
    """
    base_idx_0 = int(baseline_start*(0.001/dt))
    base_idx_1 = int(baseline_end*(0.001/dt))
    
    baseline_mean = np.mean(data[base_idx_0:base_idx_1])
    baseline_std = np.std(data[base_idx_0:base_idx_1])
    if baseline_std == 0:
        z_scored_data = np.zeros(np.shape(data))
    else:
        z_scored_data = (data - baseline_mean) / baseline_std
    return z_scored_data


def align_sp_to_stim(unit_sp, stim_t, t_win, num_stims):
    
    # searchsorted and combine all spike times to single array then hist that
    start_idxs = np.searchsorted(unit_sp, stim_t + t_win[0], 'left')
    end_idxs = np.searchsorted(unit_sp, stim_t + t_win[1], 'right')
    aligned_sp = [unit_sp[start_idxs[i]:end_idxs[i]] - stim_t[i] 
                  for i in range(num_stims)]
    # remove this line if you want to separate by trial
    # aligned_sp = np.concatenate(aligned_sp)
    
    return aligned_sp


    
# same as psth but for each trial separately
def trialwise_binned_counts(aligned_sp, t_edges, dt=0.001):
    
    num_trials = len(aligned_sp)
    binned_counts = np.zeros((num_trials, len(t_edges)-1))
    
    for t in range(num_trials):
        binned_counts[t], _ = np.histogram(aligned_sp[t], t_edges, 
                                           density=False) 
    
    return binned_counts

def unit_smoothed_psth(unit_sp, iter_stims, type_name, sigma, pad, t_edges, 
                       pad_t_edges, pad_t_win, dt=0.001):
    # note that the pop_df has reset_index() applied, so old index will be 
    #first column!
    
    stim_t = iter_stims[:,0]
    num_stims = len(stim_t)
    
    # bin spikes <with extra padding forsmoothing>
    aligned_unit_sp = align_sp_to_stim(unit_sp, stim_t, pad_t_win, 
                                       num_stims)
    binned_unit_sp = trialwise_binned_counts(aligned_unit_sp, pad_t_edges)
    binned_unit_sp = np.vstack(binned_unit_sp) / dt
    
    # smooth if desired
    if sigma != 0:
        smoothed_sp = gaussian_filter1d(binned_unit_sp, sigma=sigma)
    else:
        smoothed_sp = binned_unit_sp
        
    # in any case, drop the padding
    smoothed_sp = smoothed_sp[:, pad:-pad]
    
    # # double check that we did the padding and slicing off padding correct
    assert np.array_equal(pad_t_edges[pad:-pad], t_edges)
    
        
    return smoothed_sp

