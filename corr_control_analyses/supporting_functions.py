# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 14:40:38 2025

@author: rejwanfs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr



# ----------------------------------------------------------
# INPUTS expected:
# psth_Aonly_list: list of (n_trials, n_bins) arrays, one per neuron
# psth_At_list:    same structure
#
# Time axis: t_mids (n_bins,)
# ----------------------------------------------------------

def select_trials_from_stim_df(stim_df, exp_block=None, type_list=None, 
                               n_deflect=5):
    """
    Returns trial start times (first pulse per trial) matching selection rules.
    If type_list is None, returns all trials in exp_block (or all if exp_block is None).
    """
    sub = stim_df if exp_block is None else stim_df[stim_df.exp_block == exp_block]
    if type_list is not None:
        sub = sub[sub.type_.isin(type_list)]
    # first pulse per trial: dataset has trains of 10 pulses at indices 0,10,20,... as in your old code
    firsts = sub.iloc[::n_deflect*2]
    return firsts.pulse_t.values

def align_sp_to_stim(unit_sp, stim_t, t_win, num_stims):
    
    # searchsorted and combine all spike times to single array then hist that
    start_idxs = np.searchsorted(unit_sp, stim_t + t_win[0], 'left')
    end_idxs = np.searchsorted(unit_sp, stim_t + t_win[1], 'right')
    aligned_sp = [unit_sp[start_idxs[i]:end_idxs[i]] - stim_t[i] 
                  for i in range(num_stims)]
    # remove this line if you want to separate by trial
    # aligned_sp = np.concatenate(aligned_sp)
    
    return aligned_sp

def get_t_data(t_win, dt, pad):
    
    t_edges = dt * np.arange(t_win[0]/dt, t_win[1]/dt + 1, 1)
    t_mids = 0.5 * (t_edges[1:] + t_edges[:-1])
    ntbins = len(t_mids)
    pad_t_edges = dt * np.arange(t_win[0]/dt - pad, t_win[1]/dt + 1 + pad, 1)
    pad_t_mids = 0.5 * (pad_t_edges[1:] + pad_t_edges[:-1])
    pad_t_win = [pad_t_edges[0], pad_t_edges[-1]]
    
    return t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win

# same as psth but for each trial separately
def trialwise_binned_counts(aligned_sp, t_edges, dt=0.001):
    
    num_trials = len(aligned_sp)
    binned_counts = np.zeros((num_trials, len(t_edges)-1))
    
    for t in range(num_trials):
        binned_counts[t], _ = np.histogram(aligned_sp[t], t_edges, 
                                           density=False) 
    
    return binned_counts

def unit_smoothed_psth(unit_sp, iter_stims, type_name, sigma, pad, t_edges, 
                       pad_t_edges, pad_t_win, dt=0.001, counts_only=False):
    # note that the pop_df has reset_index() applied, so old index will be 
    #first column!
    
    stim_t = iter_stims[:,0]
    num_stims = len(stim_t)
    
    # bin spikes <with extra padding forsmoothing>
    aligned_unit_sp = align_sp_to_stim(unit_sp, stim_t, pad_t_win, 
                                       num_stims)
    binned_unit_sp = trialwise_binned_counts(aligned_unit_sp, pad_t_edges)
    binned_unit_sp = np.vstack(binned_unit_sp) / dt

    if counts_only:
        binned_unit_sp *= dt
    
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

def normalize_psth(psth, eps=1e-9):
    """
    Normalize each trial's PSTH to zero mean and unit variance.
    psth: (n_trials, ntbins)
    returns normalized psth of same shape
    """
    if psth.size == 0:
        return psth
    mu = psth.mean(axis=1, keepdims=True)
    sd = psth.std(axis=1, keepdims=True) + eps
    return (psth - mu) / sd


def id_fn(psth):
    return psth

def match_trials(psthA, psthB):
    """Match trial counts by random subsampling (robust)."""
    n = min(psthA.shape[0], psthB.shape[0])
    if n == 0:
        return None, None
    idxA = np.random.choice(psthA.shape[0], size=n, replace=False)
    idxB = np.random.choice(psthB.shape[0], size=n, replace=False)
    return psthA[idxA], psthB[idxB]

def baseline_zscore(psth, t_mids, pre_stim=5.0):
    """Z-score each neuron using only baseline bins."""
    baseline_mask = (t_mids < 0)
    if np.sum(baseline_mask) == 0:
        return psth
    base = psth[:, baseline_mask]
    # z-score per neuron across baseline *bins*, then apply to full PSTH
    mean = np.mean(base, axis=1, keepdims=True)
    std = np.std(base, axis=1, keepdims=True) + 1e-9
    return (psth - mean) / std

def compute_population_correlation(psth_Aonly_list, psth_At_list, 
                                   all_neuron_meta, all_mouse_meta, t_mids,
                                   average_trials, zscore_baseline, 
                                   equalize_trials, max_trials=6, 
                                   w_replace=False, shuffled=False):
    """
    Returns: r_time (n_bins,) where each entry is 
    corr(pop_vector_Aonly, pop_vector_At) for that time bin.
    """
    popA = []
    popB = []
    
    rng = np.random.default_rng(42)
    
    random_idxs_arr = np.empty((len(all_mouse_meta),3), dtype=object) # mouse_id, idxs_aud, idxs_audtactile
    
    if equalize_trials:
        # loop through mice and pick random indices
        for i, mouse_meta in enumerate(all_mouse_meta):
            rand_aud_trials = rng.choice(mouse_meta[1], size=max_trials, 
                                      replace=w_replace)
            if shuffled == True: # add together, we will stack psths first
                audtac_idxs = np.array(mouse_meta[2]) + len(mouse_meta[1])
                rand_audtac_trials = rng.choice(audtac_idxs, size=max_trials, 
                                                replace=w_replace) 
                rand_all_trials = np.vstack([rand_aud_trials, rand_audtac_trials])
                shuff_all_trials = rng.choice(rand_all_trials, 
                                              len(rand_all_trials), 
                                              replace=False)
                rand_aud_trials = shuff_all_trials[:max_trials]
                rand_audtac_trials = shuff_all_trials[max_trials:]
            else:
                rand_audtac_trials = rng.choice(mouse_meta[2], size=max_trials, 
                                                replace=w_replace) 
            random_idxs_arr[i] = [mouse_meta[0], rand_aud_trials, rand_audtac_trials]
        
    for psth_A, psth_B, meta in zip(psth_Aonly_list, psth_At_list, all_neuron_meta):

        # skip neurons with zero trials
        if psth_A.shape[0] == 0 or psth_B.shape[0] == 0:
            continue
        

        if average_trials:
            if equalize_trials:
                unit_mouse_id = meta[0]
                idxs_row = np.where(random_idxs_arr[:,0] == unit_mouse_id)[0][0]
                sub_trial_idxs = random_idxs_arr[idxs_row,1:]
                if shuffled == True:
                    stacked_psth = np.vstack([psth_A, psth_B])
                    vA = stacked_psth[sub_trial_idxs[0],:].mean(axis=0)
                    vB = stacked_psth[sub_trial_idxs[1],:].mean(axis=0)
                else:
                    vA = psth_A[sub_trial_idxs[0],:].mean(axis=0)
                    vB = psth_B[sub_trial_idxs[1],:].mean(axis=0)
            else:
                vA = psth_A.mean(axis=0)       # (n_bins,)
                vB = psth_B.mean(axis=0)
        else:
            A, B = match_trials(psth_A, psth_B)
            if A is None:
                continue
            vA = A.mean(axis=0)
            vB = B.mean(axis=0)

        if zscore_baseline:
            vA = baseline_zscore(vA[None,:], t_mids)[0]
            vB = baseline_zscore(vB[None,:], t_mids)[0]

        popA.append(vA)
        popB.append(vB)

    if len(popA) == 0:
        raise ValueError("No neurons contained valid trials!")

    popA = np.vstack(popA)  # neurons × bins
    popB = np.vstack(popB)

    # Compute correlation per time bin
    r_time = np.zeros(len(t_mids))
    for i in range(len(t_mids)):
        r_time[i] = pearsonr(popA[:, i], popB[:, i])[0]

    return r_time

# ----------------------------------------------------------
# Example usage:
# r_time = compute_population_correlation(all_neuron_psth_Aonly,
#                                         all_neuron_psth_At,
#                                         t_mids)
#
# ----------------------------------------------------------

def plot_correlation(t_mids, r_time, pulse_times, pulse_colors, label="Aonly vs A+T"):
    plt.figure(figsize=(10,4))
    plt.plot(t_mids, r_time)
    for p in range(10):
        plt.axvline(pulse_times[p], color=pulse_colors[p],
                    linestyle=(0,(3,3)), linewidth=3)    
    plt.xlabel("Time (s)")
    plt.ylabel("Population correlation")
    plt.title(f"Time-resolved correlation: {label}")
    plt.tight_layout()
