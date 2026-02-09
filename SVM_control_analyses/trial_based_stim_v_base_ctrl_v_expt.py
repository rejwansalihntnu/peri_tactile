# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 08:51:09 2025

@author: rejwanfs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.append("..")
from config import full_dataset_dir
import svm_functions_label_shuffling as sf
import random
from tqdm.auto import tqdm
import pandas as pd
import multiprocessing as mp
import gc


#### PLOTTING PARAMETERS ####
plt.rcParams["figure.figsize"] = [19.2 ,  9.83]
##############################

########### GLOBAL VARIABLES #################
#change only if you know what you are doing
stim_type_cols = {1: 'red', 2:'blue', 3:'green'}
stim_cols = {1:'r', 2:'b', 3:'g'}
# stim_names = {1:'Contralateral', 2:'Ipsilateral', 3:'Bilateral', 4:'unknown'}
block_names = {0:'Control', 1:'Experimental'}
##############################################

######### MODIFIABLE VARIABLES ############
need_control = True # whether you will or wont remove mice without control data
good_only = True
cluster_group = 'good'
t_win = [-4,7.5]
dt = 0.05 #in seconds
# for smoothing, add padding to t_edges
sigma = 0
pad = int(100*(0.001/dt)) # in units of dt
# region_groups = [['PERI', 'ECT']]
# region_groups = [['SSp-bfd']]
region_groups = [['AUDp']]
# region_groups = [['AUDd']]
# region_groups = [['AUDv']]
# region_groups = [['SSs']]
# region_groups = [['TeA']]
N = 30 # number of neurons in each bootstrap sample
nboot = 100 # how many resamples to do
btf = 0.5 # what fraction of trials to sample in each fold 
idx_replace = False # whether to sample trials with replacement or not
C = 0.0001 # regularisation parameter for SVM
# set the grp_cvs param to None if you don't want to split 
grp_cvs = 2 # no. groups to split trials by for CV'd train/test MORE THAN 2 FOLDS NOT SUPPORTED!!!!!
# fix btf if idx replace is false
if idx_replace == False:
    btf = 1/grp_cvs
win_size_ms = 50
window_size = int(win_size_ms * (0.001/dt)) # win size in units of dt
step_ms = 50 # step size between windows in ms
step_size = int(step_ms * (0.001/dt)) # step size in units of dt
n_shuff = 0 # number of times to repeat whole analysis, but shuffling labels
shuff_nm = 'trialwise' # 'cyclic' or 'random' or 'trialwise'
n_cpus = 6
z_scored=True
# manual protocol choice if desired
manual_protocol_choice = True
f = 1 # frequency
n_cycles = 5 # num cycles < which is 0.5*number of deflections >
tot_stim_win_dur = 5.0
 

max_fold_trials = 5 # since 10 is the min number of trials in any 
#mouse-condition combination.
n_pseudotrials = 30 # number of pseudotrials to generate
fold_replace = True # whether to sample  from fold's trials with replacement
# if not replacing keep in mind you only have 42 trials in one of the mice so
#21 is the max amount
###########################################



def shuffle_labels(y_labels):
    rng = np.random.default_rng()
    shuff_y_labels = rng.permutation(y_labels)  
    
    return shuff_y_labels

def shuffle_labels_cyclic(labels):

    N = len(labels)
    shift = np.random.randint(1, N)  # Random shift amount between 1 and N-1
    return np.roll(labels, shift)

# this performs the shuffling without breaking within-trial temporal structure
#so we can assess whether activity is significantly different between trial 
#types, rather than caring about 
def shuffle_labels_trialwise(y_labels, window_size, n_pseudotrials):
    y_trial_labs = y_labels.reshape((2*n_pseudotrials, window_size))
    rng = np.random.default_rng()

    shuffled = rng.permutation(y_trial_labs).flatten() 
   
    return shuffled


def unshuffled_labels(y_labels):
    return y_labels


def z_score_train_apply_test(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
    
def identity_fn(x_train, x_test):
    return x_train, x_test

def stim_base_mean_acc(args):
    (
        boot_idx, avg_psth, df_mid_list, mouse_num_stims, block_names,
        mouse_ids, nboot, N, grp_cvs, clf, window_size, step_size, t_mids,
        z_func, ntbins, stim_t_mids, stim_idxs, num_windows, num_reg_units,
        n_pseudotrials, y_labs, y_labs_shuff
    ) = args

    iter_decoding_performance = np.zeros((2, 2, num_windows, grp_cvs))

    # Stimulus-locked data
    stim_psth_data = avg_psth[:, :, :, stim_idxs, :] # (condition, fold, units, time, pseudotrials)
    
    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx   = start_idx + window_size
        
        # One start index per pseudotrial
        base_starts = np.random.randint(int(3000*(0.001/dt)), size=n_pseudotrials)
        
        stim_window_data = stim_psth_data[:, :, :, start_idx:end_idx, :] # (condition, fold, units, window_size, pseudotrials)

        for train_cond in (0, 1):
            for grp in range(grp_cvs):

                train_folds = np.arange(grp_cvs) != grp

                # --------------------------
                #       TRAINING DATA
                # --------------------------
                # Stimulus
                Xts = stim_window_data[train_cond, train_folds, :, :, :] # (fold, units, window_size, pseudotrials)
                Xts = Xts.squeeze(0).reshape(N, -1).T # remove folds, then reshape
                
                Xtb = avg_psth[train_cond, train_folds, :, base_starts, np.arange(n_pseudotrials)] # this is in shape (n_pseudotrials, N)

                X_train = np.vstack([Xts, Xtb])
                y_train = y_labs_shuff  # matches X_train length by construction

                # --------------------------
                #       TEST DATA
                # --------------------------
                for test_cond in (0, 1):

                    Xvs = stim_window_data[test_cond, grp, :, :, :]
                    Xvs = Xvs.transpose(2, 1, 0).reshape(-1, N)

                    Xvb = avg_psth[test_cond, grp, :, base_starts, np.arange(n_pseudotrials)]
                    
                    X_test = np.vstack([Xvs, Xvb])
                    y_test = y_labs

                    # --------------------------
                    #    Z-scoring + decoding
                    # --------------------------
                    X_train_z, X_test_z = z_func(X_train, X_test)
                    clf.fit(X_train_z, y_train)
                    score = clf.score(X_test_z, y_test)

                    iter_decoding_performance[train_cond, test_cond, window_idx, grp] = score

    return iter_decoding_performance, boot_idx




def gen_fully_independent_pseudotrial(
        group_psth, grp_cvs, N, ntbins, mouse_ids, mouse_num_stims, 
        num_reg_units, df_mid_list, z_func, n_pseudotrials):
    # 2 stim types, 2 halves even/odd trials, num sampled units , 
    #num time bins
    avg_psth = np.empty((2, grp_cvs, N, ntbins, n_pseudotrials))
    
    # generate trial idxs to average over
    shuff_stim_idxs = {}
    for mouse_id in mouse_ids:
        mouse_stim_counts = mouse_num_stims[mouse_id]
        # num stims x num cv folds x fraction of trials to sample
        # latter is what will affect similarity and middle will affect
        #the precision of the <however biased> latter
        iter_stim_idxs = [np.random.choice(range(mouse_stim_counts[0]), 
                          int(mouse_stim_counts[0]*grp_cvs*btf),
                          replace=idx_replace),
                          np.random.choice(
                          range(mouse_stim_counts[1]), 
                          int(mouse_stim_counts[1]*grp_cvs*btf),
                          replace=idx_replace)]
        shuff_stim_idxs[mouse_id] = iter_stim_idxs
            
        
        
    # Iterate over stimulus types (e.g., type 1 and type 2)
    # pick N random units first
    boot_units = random.sample(range(num_reg_units), N)
    # randomly split trials into grp_cvs number of groups. Average and 
    #zscore each group
    for u, unit_idx in enumerate(boot_units):
        mouse_id = df_mid_list[unit_idx]
        # split trial types and even odd odd
        for trial_type in range(2):
            unit_stim_type_psth = group_psth[unit_idx,trial_type]
            shuff_stim_type_idxs = shuff_stim_idxs[mouse_id][trial_type]
            for grp in range(grp_cvs):
                grp_type_idxs = shuff_stim_type_idxs[grp::grp_cvs]
                grp_unit_trial_samples = np.random.choice(grp_type_idxs,
                                                          n_pseudotrials,
                                                          replace=fold_replace)
                grp_unit_psth = unit_stim_type_psth[grp_unit_trial_samples]
                avg_psth[trial_type, grp, u, :, :] = grp_unit_psth.T
    
    return avg_psth



def stim_base_shuff_unshuff(args):
    (boot_idx, group_psth, df_mid_list, mouse_num_stims, block_names, mouse_ids, 
     nboot, N, grp_cvs, clf, window_size, step_size, t_mids, ntbins, z_func,
     stim_t_mids, stim_idxs, num_windows, num_reg_units, n_pseudotrials, 
     y_labs, all_shuff_y_labs) = args
    
    
    boot_psth = gen_fully_independent_pseudotrial(
        group_psth, grp_cvs, N, ntbins, mouse_ids, mouse_num_stims, 
        num_reg_units, df_mid_list, z_func, n_pseudotrials)
    
    unshuff_args = (boot_idx, boot_psth, df_mid_list, mouse_num_stims, 
                    block_names, mouse_ids, nboot, N, grp_cvs, clf, window_size, 
                    step_size, t_mids, z_func, ntbins, stim_t_mids, stim_idxs, 
                    num_windows, num_reg_units, n_pseudotrials, y_labs, y_labs)

    unshuff_dec_perf, _ = stim_base_mean_acc(unshuff_args)
    
    shuf_boot_dec_perf = np.zeros((2, 2, num_windows, grp_cvs, n_shuff))
    
    for shuff_idx in range(len(all_shuff_y_labs)):
        iter_args = (
            boot_idx, boot_psth, df_mid_list, mouse_num_stims, block_names, 
            mouse_ids, nboot, N, grp_cvs, clf, window_size, step_size, 
            t_mids, z_func, ntbins, stim_t_mids, stim_idxs, num_windows, 
            num_reg_units, n_pseudotrials, y_labs, all_shuff_y_labs[shuff_idx])
        shuf_boot_dec_perf[:, :, :, :, shuff_idx], _ = stim_base_mean_acc(iter_args)
        
    return unshuff_dec_perf, shuf_boot_dec_perf, boot_idx
    

def partition_stims(mouse_ids, pop_stims, block_names, n_cycles=5):
    """
    Returns dict of stim times for each mouse and stim type.
    exp_block: 0 = control, 1 = experimental
    """
    n_deflect = int(2*n_cycles)
    mouse_stim_dict = {}
    for mouse_id in mouse_ids:
        mouse_stim_dict[mouse_id] = {}
        mouse_stims = pop_stims[mouse_id]
        for block_idx, block_name in block_names.items():
            # filter by block
            mouse_stim_block = mouse_stims[mouse_stims.exp_block == block_idx]
            # take only the pulse times and stim id
            mouse_stim_block = mouse_stim_block[['pulse_t','type_','trial_idx']].values
            # take first stim times for baseline windows and trial durations
            first_stims = mouse_stim_block[::n_deflect,:]
            valid_data = first_stims[:,(0,2)]
            mouse_stim_dict[mouse_id][block_name] = valid_data
    return mouse_stim_dict



def prepare_all_data_together(full_dataset_dir, manual_protocol_choice, f, 
                              n_cycles, tot_stim_win_dur, block_names, 
                              t_win, dt, pad, good_only, 
                              cluster_group, region_groups, sigma, 
                              stim_type_cols, need_control=False):
    
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
    # --- end NEW block ---
    
    # take out the constituting data
    sp = pop_data.pop_sp
    stims = pop_data.pop_stims
    df = pop_data.df
    
    # take out data per mouse to feed into data for separation
    stim_dict = partition_stims(mouse_ids, stims, block_names, n_cycles)
    
    # get num stims in each condition and mouse for bootstrapping later
    mouse_num_stims = {}
    # track min stims per condition and mouse
    min_num_stims = np.inf
    for mouse_id in mouse_ids:
        iter_dat = mouse_num_stims[mouse_id] = []
        for block_idx, block_name in block_names.items():
            num_type = np.shape(stim_dict[mouse_id][block_name])[0]
            iter_dat.append(num_type)
            min_num_stims = np.min([min_num_stims, num_type])
        # go through, add key as mouse id, with two-len arrays of [0] = con
        #num trials and [1] = ips num trials
    
   
    
    
    # generate bin edges and midpoints
    t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win = \
        sf.get_t_data(t_win, dt, pad)
        
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
    for block_idx, block_name in block_names.items():
        
        psth_dict[block_name] = np.empty(tot_units, dtype=object)
        
        for i, row in df_valid.iterrows():
            
            # get neuron's metadata (see top comment for why indexing from 1)
            mouse_id, cluster, probe, region = row.iloc[1:5]
            
            # extract neuron's spikes and associated stims
            unit_sp = sp[mouse_id][probe][region][str(cluster)]
            iter_stims = stim_dict[mouse_id][block_name]
            unit_psth = sf.unit_smoothed_psth(unit_sp, iter_stims, block_name, sigma, 
                                           pad, t_edges, pad_t_edges, pad_t_win,
                                           dt)
            psth_dict[block_name][i] = unit_psth
            
    # remove unused large data to free up RAM for multiprocessing
    del sp, pop_data
    gc.collect()
    
    # create label
    ## make stim window have relevant label
    stim_idxs = (t_mids < 5) & (t_mids > 0)
    type_names = list(psth_dict.keys())
    
    
    return df_valid, psth_dict, ntbins, mouse_ids, mouse_num_stims, \
        type_names, t_mids, stim_idxs, int(min_num_stims)





if __name__ == "__main__":
    
    # put here so it doesn't print multiple times <we're using multiprocessing>
    if idx_replace == False:
        print("idx_replace is set to False, changing btf to acommodate this by"
              + f"changing from {btf} to {1/grp_cvs}")
    
    dt_ms = int(dt*1000)
    print(f"\nPARAMETERS: \nRegion: {region_groups} \ndt={dt_ms}ms\nC={C}\nWindow size: {win_size_ms}ms ({window_size} bins)\nStep size: {step_ms}ms ({step_size} bins)\nno. shuffles: {n_shuff}\nCV folds: {grp_cvs}\nno. pseudotrials: {n_pseudotrials}\n")
    
    # create save directories
    zsuffix = ""
    if z_scored == True:
        zsuffix += "_z_scored"
        z_func = z_score_train_apply_test
    else:
        z_func = identity_fn
    
    # create save directories
    if shuff_nm == 'cyclic':
        shuff_fn = shuffle_labels_cyclic
    elif shuff_nm == 'random':
        shuff_fn = shuffle_labels
    elif shuff_nm == 'trialwise':
        shuff_fn = shuffle_labels_trialwise
    base_dir = f"./svm_stim_v_base_{shuff_nm}_shuffling{zsuffix}/" + \
        f"trial_based, dt={dt}s, sigma={sigma}/{region_groups}"
    os.makedirs(f"{base_dir}", exist_ok=True)
    
    # load and process data
    df_valid, psth_dict, ntbins, mouse_ids, mouse_num_stims, \
        type_names, t_mids, stim_idxs, min_trials = \
            prepare_all_data_together(
            full_dataset_dir, manual_protocol_choice, f, n_cycles,
            tot_stim_win_dur, block_names, t_win, dt, pad,
            good_only, cluster_group, region_groups, sigma, stim_type_cols, 
            need_control=need_control)
            
    #%%
        
    
    for r, regions in enumerate(region_groups):
    
        """this few lines are all you need to do everything else"""
        # get region group units
        reg_idxs = np.isin(df_valid.region, regions)
        group_psth = [psth_dict[block_names[i]][reg_idxs] for i in block_names.keys()]
        # stack the psths together
        reg_df_valid = df_valid[reg_idxs]
        group_psth = np.vstack([group_psth[i] for i in range(2)]).T
        """****************************************************"""
        
        
        
        # average over all time bins and z score
        num_reg_units = np.shape(group_psth)[0]
        
        df_mid_list = reg_df_valid.mouse_id.values
        
        # Create an SVM model
        clf = LinearSVC(C=C, max_iter=5000, dual=False) # 5K overkill?

        
        stim_t_mids = np.arange(win_size_ms/2, 5000 - (win_size_ms-step_ms)/2, 
                                step_ms) / 1000
        if int((5 - stim_t_mids[-1])*1000) < win_size_ms/2:
            stim_t_mids = stim_t_mids[:-1]
        
        num_windows = len(stim_t_mids)
        
        
        # stim v stim UNSHUFFLED
        # Prepare storage for decoding performance over time
        unshuffled_decoding_performance = np.zeros(
            (2, 2, num_windows, grp_cvs, nboot), dtype=float)

      
        n_samples_per_class = n_pseudotrials * window_size  # e.g. 30 * 30 = 900
        y_labs = np.hstack([
            np.ones(n_samples_per_class),         # class 1
            np.full(n_samples_per_class, 2)       # class 2
        ])

        
       

        
        
        # Prepare storage for decoding performance over time
        full_shuff_dec_perf = np.zeros(
            (2, 2, num_windows, grp_cvs, n_shuff, nboot), dtype=float)
        
        # prepare all shuffled y labels
        all_shuff_y_labs = [
            shuff_fn(y_labs, window_size, n_pseudotrials)
            if shuff_nm == "trialwise"
            else shuff_fn(y_labs)
            for _ in range(n_shuff)
        ]
        
        # Arguments: each process gets a non-overlapping index along axis 0
        args_list = [(boot_idx, group_psth, df_mid_list, 
                      mouse_num_stims, block_names, mouse_ids, nboot, N, 
                      grp_cvs, clf, window_size, step_size, t_mids, ntbins,
                      z_func, stim_t_mids, stim_idxs, num_windows, 
                      num_reg_units, n_pseudotrials, y_labs, all_shuff_y_labs)
                     for boot_idx in range(nboot)]
        
        
        # Process in parallel
        if n_cpus > 1:
            pool = mp.Pool(n_cpus) # 8 vs 4 was ~16 vs 19m for 10 shuffles...
            with tqdm(total=nboot, position=0, leave=True, ncols=80) as pbar:
                for res, shuff_res, boot_idx in pool.imap_unordered(
                        stim_base_shuff_unshuff, args_list):
                    # Insert result in the correct index
                    unshuffled_decoding_performance[..., boot_idx] = res 
                    full_shuff_dec_perf[..., boot_idx] = shuff_res 
                    pbar.update()
            pool.close()
            pool.join()
        else:
            with tqdm(total=nboot, position=0, leave=True, ncols=80) as pbar:
                for args in args_list:
                    res, shuff_res, boot_idx = stim_base_shuff_unshuff(args)
            
                    # Insert result in the correct index
                    unshuffled_decoding_performance[..., boot_idx] = res
                    full_shuff_dec_perf[..., boot_idx] = shuff_res
            
                    pbar.update()

        
        
        # save unshuffled data
        suffix = f"{nboot}-boot_{grp_cvs}-fold_{sigma}ms_{N}_units" + \
            f"_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"

        # Save decoding performance data for future analysis
        np.save(f"{base_dir}/performance_data_{regions}_{suffix}.npy", 
                unshuffled_decoding_performance)
        
        if n_shuff > 0:
            
            # average and add to shuffled_decoding_performance array
            shuffled_decoding_performance = \
                np.mean(full_shuff_dec_perf, axis=(3,5))
                
            
            # save shuffled data
            suffix = f"{nboot}-boot_{n_shuff}-shuff_{grp_cvs}-fold_{sigma}ms_" + \
                f"{N}_units_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"
            
            # Save decoding performance data for future analysis
            np.save(f"{base_dir}/shuffled_performance_data_{regions}_{suffix}.npy", 
                    shuffled_decoding_performance)
    
        import trial_based_stim_v_base_ctrl_v_expt_PLOTTING as plot_svb
        plot_svb.main()
