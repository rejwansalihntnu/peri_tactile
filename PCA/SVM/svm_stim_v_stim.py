# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:04:54 2025

@author: rejwanfs
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import os
import sys
sys.path.append("..")
from config import full_dataset_dir, stim_names
import svm_functions_label_shuffling as sf
from collections import Counter
import random
from tqdm.auto import tqdm

import multiprocessing as mp


#### PLOTTING PARAMETERS ####
plt.rcParams["figure.figsize"] = [19.2 ,  9.83]
##############################

########### GLOBAL VARIABLES #################
#change only if you know what you are doing
stim_type_cols = {1: 'red', 2:'blue', 3:'green'}
stim_cols = {1:'r', 2:'b', 3:'g'}
##############################################

######### MODIFIABLE VARIABLES ############
stim_types = [1,2]
good_only = True
cluster_group = 'good'
t_win = [-4,7.5]
dt = 0.005 #in seconds
# for smoothing, add padding to t_edges
sigma = 0
pad = int(100*(0.001/dt)) # in units of dt
# region_groups = [['PERI', 'ECT']]
region_groups = [['SSp-bfd']]
# region_groups = [['AUDp']]
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
grp_cvs = 2 # no. groups to split trials by for CV'd train/test
# fix btf if idx replace is false
if idx_replace == False:
    btf = 1/grp_cvs
win_size_ms = 30 # window size in ms
window_size = int(win_size_ms * (0.001/dt)) # win size in units of dt
step_ms = 10 # step size between windows in ms
step_size = int(step_ms * (0.001/dt)) # step size in units of dt

n_shuff = 1000 # number of times to repeat whole analysis, but shuffling labels
shuff_nm = 'random' # 'random'
n_cpus = 6
z_scored=False
# manual protocol choice if desired
manual_protocol_choice = True
f = 1 # frequency
n_cycles = 5 # num cycles < which is 0.5*number of deflections >
tot_stim_win_dur = 5.0
###########################################

############# VARIABLES FOR K-MEANS-BASED SUBSAMPLING #################
# define base save dir <load in this case>
kmeans_t_win = [-1.0, 6.0]
kmeans_sigma = 20 # in ms
quality = 'ALL units'
if good_only == True:
    quality = 'HQ units'

cluster_group_nm = cluster_group
if cluster_group == 'good':
    cluster_group_nm = 'Single'
base_dir = f"{f}Hz, {n_cycles} cycles, {tot_stim_win_dur}s/"
base_dir += f"win_relative = {kmeans_t_win}, {cluster_group_nm} units, "
base_dir += f"{quality}"

n_reg_K = [4,4,3,4,3,3,4] # how many clusters for each region (same order as region
#groups)
npcs = 3
kmeans_reg_grps = [['PERI', 'ECT'], ['SSp-bfd'], ['SSs'], ['AUDp'], ['AUDd'], 
                   ['AUDv'], ['TeA']]
kmeans_dir = f"./../K-means/{base_dir}/sigma = {kmeans_sigma}ms"
kmeans_dir += f"/stim_types={stim_types}_{npcs}-PCs"
r_kmeans = {f"{kmeans_reg_grps[0]}": 0, f"{kmeans_reg_grps[1]}": 1,
            f"{kmeans_reg_grps[2]}": 2, f"{kmeans_reg_grps[3]}": 3,
            f"{kmeans_reg_grps[4]}": 4, f"{kmeans_reg_grps[5]}": 5,
            f"{kmeans_reg_grps[6]}": 6}
"""lazy way of getting the varialbe (adjust if PER transient subpop needed)"""

k_means_group = np.arange(1, n_reg_K[r_kmeans[f"{region_groups[0]}"]]+1, 1)
# k_means_group = np.array([3,4]) # which clusters to use for region_groups reg.

if f"{k_means_group}" == f"{np.array([2,3])}":
    print("These clusters are for the old k-means clustering!! See plots!")
    raise SystemExit
#######################################################################


def shuffle_labels(y_labels):
    rng = np.random.default_rng()
    shuff_y_labels = rng.permutation(y_labels)  
    
    return shuff_y_labels

def unshuffled_labels(y_labels):
    return y_labels

def stim_stim_mean_acc(args):
    """
    Generates average accuracy for all 4 combinations
    """
    boot_idx, avg_psth, df_mid_list, mouse_num_stims, type_names, \
    mouse_ids, nboot, N, grp_cvs, clf, window_size, step_size, t_mids, \
    ntbins, stim_t_mids, stim_idxs, num_windows, num_reg_units, \
    y_labs = args
    

    iter_decoding_performance = np.zeros((num_windows, grp_cvs)) 
    # stim v stim so remove baseline
    stim_psth_data = avg_psth[:,:,:,stim_idxs] # stim only part 
    # Iterate over each time window
    
    
    
    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        # Prepare X and y for this window
        stim_window_data = stim_psth_data[:,:,:,start_idx:end_idx]  
        
        for grp in range(grp_cvs):
            
            X_train_both = stim_window_data[:,np.arange(grp_cvs)!=grp,:,:]
        
            # Combine stimulus data
            # window_size data in index grp will be testing
            X_con = np.hstack(X_train_both[0])
            X_ipsi = np.hstack(X_train_both[1])
            
            # Combine both stimulus and baseline data
            X_train = np.hstack([X_con, X_ipsi])
            
            y_train = np.repeat(y_labs, grp_cvs-1) #-1 because LOO CV method
            
            # Make test data using odd trials
            X_test_both = stim_window_data[:,grp,:,:]
            X_test = np.hstack([X_test_both[0], X_test_both[1]])
            
            y_test = y_labs
            
            # Perform SVM training and testing
            clf.fit(X_train.T, y_train)
            score = clf.score(X_test.T, y_test)  # Accuracy score
            iter_decoding_performance[window_idx, grp] = score
         
    
    return iter_decoding_performance, boot_idx 


def shuff_stim_stim_mean_acc(args):
    
    boot_idx, avg_psth, df_mid_list, mouse_num_stims, type_names, \
    mouse_ids, nboot, N, grp_cvs, clf, window_size, step_size, t_mids, \
    ntbins, stim_t_mids, stim_idxs, num_windows, num_reg_units, \
    all_shuff_y_labs = args
    
    shuf_boot_dec_perf = np.zeros((num_windows, grp_cvs, n_shuff))
    
    for shuff_idx in range(len(all_shuff_y_labs)):
        iter_args = (
            boot_idx, avg_psth, df_mid_list, mouse_num_stims, type_names, 
            mouse_ids, nboot, N, grp_cvs, clf, window_size, step_size, 
            t_mids, ntbins, stim_t_mids, stim_idxs, num_windows, 
            num_reg_units, all_shuff_y_labs[shuff_idx])
        
        shuf_boot_dec_perf[:, :, shuff_idx], _ = stim_stim_mean_acc(iter_args)
        
    return shuf_boot_dec_perf, boot_idx


def get_boot_avg_psths(nboot, group_psth, grp_cvs, N, ntbins, mouse_ids, 
                       mouse_num_stims, num_reg_units, df_mid_list, z_func):
    
    all_boot_avg_psth = np.empty((nboot, 2, grp_cvs, N, ntbins))
    for boot in range(nboot):
        # 2 stim types, 2 halves even/odd trials, num sampled units , 
        #num time bins
        avg_psth = all_boot_avg_psth[boot]
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
            shuff_trial_idxs = shuff_stim_idxs[mouse_id]
            # split trial types and even odd odd
            for trial_type in range(2):
                unit_stim_type_psth = group_psth[unit_idx,trial_type]
                shuff_unit_psth = unit_stim_type_psth[
                    shuff_trial_idxs[trial_type]]
                for grp in range(grp_cvs):
                    # skip param ensures we get N trials instead of 
                    #N/grp_cvs - makes data more similar, so this should
                    #be tuned to make sure data is not too similar
                    grp_psth = shuff_unit_psth[grp::grp_cvs,:]
                    reg_unit_psth = np.mean(grp_psth, axis=0)
                    avg_psth[trial_type, grp, u, :] = z_func(reg_unit_psth, 
                                                             0, 3000)
    
    return all_boot_avg_psth
  
    
def identity_fn(data, base_start, base_end):
    return data



if __name__ == "__main__":
    
    plt.ioff()
    
    # put here so it doesn't print multiple times <we're using multiprocessing>
    if idx_replace == False:
        print("idx_replace is set to False, changing btf to acommodate this by"
              + f"changing from {btf} to {1/grp_cvs}")
    
    # create save directories
    zsuffix = ""
    if z_scored == True:
        zsuffix += "_z_scored"
        # set z_scoring as the function for getting avg psths
        z_func = sf.base_z_score_data
    else:
        z_func = identity_fn
    
    # create save directories
    if shuff_nm == 'random':
        shuff_fn = shuffle_labels
    base_dir = f"./svm_stim_v_stim_{shuff_nm}_shuffling{zsuffix}/cluster_specific, dt={dt}s, sigma={sigma}/" + \
        f"{region_groups}_clusters_{k_means_group}"
    
    os.makedirs(f"{base_dir}", exist_ok=True)
    
    # load and process data
    df_valid, psth_dict, kmeans, ntbins, mouse_ids, mouse_num_stims, \
        type_names, t_mids, stim_idxs = sf.prepare_all_data(
            full_dataset_dir, manual_protocol_choice, f, n_cycles,
            tot_stim_win_dur, stim_names, stim_types, t_win, dt, pad,
            good_only, cluster_group, region_groups, sigma, stim_type_cols, 
            kmeans_dir)
        
    for r, regions in enumerate(region_groups):
    
        """this few lines are all you need to do everything else"""
        # get region group units
        reg_idxs = np.isin(df_valid.region, regions)
        group_psth = [psth_dict[stim_names[i]][reg_idxs] for i in stim_types]
        # stack the psths together
        reg_df_valid = df_valid[reg_idxs]
        group_psth = np.vstack([group_psth[i] for i in range(2)]).T
        """****************************************************"""
        
        
        """THIS IS TO ADD K MEANS DATA TO DF TO SPLIT BY CLUSTER(S)"""
        reg_numK = n_reg_K[r_kmeans[f"{regions}"]]
        reg_kmeans = kmeans[r_kmeans[f"{regions}"], reg_numK-2]
        
        reg_labels = reg_kmeans.labels_
        
        counter_dict = Counter(reg_labels)
        unique_labels = list(counter_dict.keys())
        lab_sorter = np.argsort(unique_labels)
        
        label_counts = list(Counter(reg_labels).values())
        label_counts = np.array(label_counts)[lab_sorter]
        
        
        K_col_nm = f"K_idx_({reg_numK})"
        reg_df_valid.insert(np.shape(reg_df_valid)[1], K_col_nm, 
                            reg_labels)
        """*********************************************************"""
        
        # split by clusters you want
        k_grp_idxs = np.isin(reg_df_valid[K_col_nm], k_means_group-1)
        reg_df_valid = reg_df_valid[k_grp_idxs]
        group_psth = group_psth[k_grp_idxs]
        
        # average over all time bins and z score
        num_reg_units = np.shape(group_psth)[0]
        
        df_mid_list = reg_df_valid.mouse_id.values
        
        # Create an SVM model
        clf = svm.SVC(C=C, kernel='linear', max_iter=1000)
        
        stim_t_mids = np.arange(win_size_ms/2, 5000 - (win_size_ms-step_ms)/2, 
                                step_ms) / 1000
        if int((5 - stim_t_mids[-1])*1000) < win_size_ms/2:
            stim_t_mids = stim_t_mids[:-1]
        
        num_windows = len(stim_t_mids)
        
        
        #%% stim v stim UNSHUFFLED
        # Prepare storage for decoding performance over time
        unshuffled_decoding_performance = np.zeros((num_windows, grp_cvs, 
                                                    nboot))
        
        # generate all bootstrapped avg psth data
        all_avg_psths = get_boot_avg_psths(nboot, group_psth, grp_cvs, N, 
                                           ntbins, mouse_ids, 
                                           mouse_num_stims, num_reg_units, 
                                           df_mid_list, z_func)
        
        # generate shuffled labels to use for all bootstrapped pop psths
        y_con = np.ones(window_size)  # 1 for con
        y_ipsi = np.full(window_size, 2)    # 2 for ipsi
        y_labs = np.hstack([y_con, y_ipsi])
        
        pool = mp.Pool(n_cpus) 

        # Arguments: each process gets a non-overlapping index along axis 0
        args_list = [(boot_idx, all_avg_psths[boot_idx], df_mid_list, 
                      mouse_num_stims, type_names, mouse_ids, nboot, N, 
                      grp_cvs, clf, window_size, step_size, t_mids, ntbins,
                      stim_t_mids, stim_idxs, num_windows, num_reg_units,
                      y_labs)
                     for boot_idx in range(nboot)]
        # Process in parallel
        with tqdm(total=nboot, position=0, leave=True, ncols=80) as pbar:
            for result, boot_idx in pool.imap_unordered(stim_stim_mean_acc, args_list):
                # Insert result in the correct index
                unshuffled_decoding_performance[..., boot_idx] = result 
                pbar.update()
        pool.close()
        pool.join()
        
        
        # save unshuffled data
        suffix = f"{nboot}-boot_{grp_cvs}-fold_{sigma}ms_{N}_units" + \
            f"_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"

        # Save decoding performance data for future analysis
        np.save(f"{base_dir}/performance_data_{regions}_{suffix}.npy", 
                unshuffled_decoding_performance)
        
        #%% repeat for shuffled variant
        if n_shuff == 0:
            break
            
        # Prepare storage for decoding performance over time
        full_shuff_dec_perf = np.zeros((num_windows, grp_cvs, n_shuff, nboot))
        
        # prepare all shuffled y labels
        all_shuff_y_labs = [shuff_fn(y_labs) for i in range(n_shuff)]
        
        # Arguments: each process gets a non-overlapping index along axis 0
        args_list = [(boot_idx, all_avg_psths[boot_idx], df_mid_list, 
                      mouse_num_stims, type_names, mouse_ids, nboot, N, 
                      grp_cvs, clf, window_size, step_size, t_mids, ntbins,
                      stim_t_mids, stim_idxs, num_windows, num_reg_units,
                      all_shuff_y_labs)
                     for boot_idx in range(nboot)]
        
        # use half the available cores
        pool = mp.Pool(n_cpus) # 8 vs 4 was ~16 vs 19m for 10 shuffles...
        
        # Process in parallel
        with tqdm(total=nboot, position=0, leave=True, ncols=80) as pbar:
            for result, boot_idx in pool.imap_unordered(
                    shuff_stim_stim_mean_acc, args_list):
                # Insert result in the correct index
                full_shuff_dec_perf[:, :, :, boot_idx] = result 
                pbar.update()
        
        pool.close()
        pool.join()
      
            
        # average and add to shuffled_decoding_performance array
        shuffled_decoding_performance = \
            np.mean(full_shuff_dec_perf, axis=(1,3))
            
        
        # save shuffled data
        suffix = f"{nboot}-boot_{n_shuff}-shuff_{grp_cvs}-fold_{sigma}ms_" + \
            f"{N}_units_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"
        
        # Save decoding performance data for future analysis
        np.save(f"{base_dir}/shuffled_performance_data_{regions}_{suffix}.npy", 
                shuffled_decoding_performance)
