# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:56:04 2025

@author: rejwanfs

Generates the figures for stim base decoding
"""



import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from config import stim_names
import svm_functions as sf
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.transforms import blended_transform_factory

# CAIRO FOR ILLUSTRATOR
import matplotlib.font_manager as fm
import matplotlib
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 12})
font = fm.FontProperties(family = 'arial')
matplotlib.use('cairo') # comment out when testing plots
figtype = 'pdf'
        
#### PLOTTING PARAMETERS ####
plt.rcParams["figure.figsize"] = [19.2 ,  9.83]
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["figure.titlesize"] = 18
plt.rcParams['axes.linewidth'] = 5
plt.rcParams.update({
    'xtick.major.size': 12,   # X and Y tick major size
    'ytick.major.size': 12,
    'xtick.major.width': 3,  # X and Y tick major width
    'ytick.major.width': 3,
    'xtick.minor.size': 4,   # X and Y tick minor size
    'ytick.minor.size': 4,
    'xtick.minor.width': 1,  # X and Y tick minor width
    'ytick.minor.width': 1,
})
plt.rc('xtick',labelsize=30)
plt.rc('ytick',labelsize=30)
plt.rcParams["axes.labelsize"] = 30
plt.rc('legend',fontsize=30)
plt.rcParams["legend.markerscale"] = 20
plt.rcParams['lines.markersize']= 20
plt.rcParams['lines.linewidth'] = 3
plt.rcParams["font.size"] = 20 
##############################

# Set tick mark thickness and length
plt.rcParams['xtick.major.size'] = 15  # Major tick length
plt.rcParams['ytick.major.size'] = 15
plt.rcParams['xtick.minor.size'] = 15  # Minor tick length
plt.rcParams['ytick.minor.size'] = 15

plt.rcParams['xtick.major.width'] = 4  # Major tick thickness
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['xtick.minor.width'] = 3  # Minor tick thickness
plt.rcParams['ytick.minor.width'] = 3

# Set global savefig parameters
plt.rcParams["savefig.transparent"] = True  # Transparent background
plt.rcParams["savefig.bbox"] = "tight"  # Remove extra whitespace

########### GLOBAL VARIABLES #################
#change only if you know what you are doing
stim_type_cols = {1: 'red', 2:'blue', 3:'green'}
stim_cols = {1:'r', 2:'b', 3:'g'}
pulse_times = np.arange(0, 5, 0.5) # in seconds
pulse_colors = np.tile(['black', 'dimgray'], 10)
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
region_groups = [['PERI', 'ECT']]
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
"""these two below for testing max iter"""
z_scored=False
# manual protocol choice if desired
manual_protocol_choice = True
f = 1 # frequency
n_cycles = 5 # num cycles < which is 0.5*number of deflections >
tot_stim_win_dur = 5.0

plot_shuff = False
n_shuff = 1000 # number of times to repeat whole analysis, but shuffling labels
shuff_nm = 'random' # 'cyclic' or 'random'

ci_perc = 95

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
z_scored = False
zsuffix = ""
if z_scored == True:
    zsuffix += "_z_scored"
#######################################################################

#%% load accuracy data and set save directories <STIM V BASE SPECIFIC>

# set load directories
base_dir = f"./svm_stim_v_base_{shuff_nm}_shuffling{zsuffix}/cluster_specific, dt={dt}s, sigma={sigma}/" + \
    f"{region_groups}_clusters_{k_means_group}"
# base_dir_cc = f"{base_dir}_CC"
suffix = f"{nboot}-boot_{grp_cvs}-fold_{sigma}ms_{N}_units" + \
    f"_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"

    
# set save directory
save_dir = f"./svm_stim_v_base_{shuff_nm}_shuffling{zsuffix}/cluster_specific_postprocessing, dt={dt}s, sigma={sigma}/" + \
    f"{region_groups}_clusters_{k_means_group}"
os.makedirs(f"{save_dir}", exist_ok=True)

# set file names for saving uniquely
cc_diff_fname = f"cc_diff_histogram_{region_groups[0]}_{suffix}"
if plot_shuff == False:
    fname_2x2 = f"2x2_acc_plots_{region_groups[0]}_{suffix}"
else:
    shuff_suffix = f"{nboot}-boot_{n_shuff}-shuff_{grp_cvs}-fold_{sigma}ms_" + \
        f"{N}_units_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"
    # load empirical chance level shuffled accuracy data
    shuff_file_name = f"shuffled_performance_data_{region_groups[0]}_{shuff_suffix}"
    # shuff_decoding_performance = np.zeros((2, 2, num_windows, n_shuff))
    shuff_decoding_performance = np.load(f"{base_dir}/{shuff_file_name}.npy")
    fname_2x2 = f"2x2_acc_plots_{region_groups[0]}_{shuff_suffix}"

# load population decoding accuracy data
file_name = f"performance_data_{region_groups[0]}_{suffix}.npy"
decoding_performance = np.load(f"{base_dir}/{file_name}")

# infer relevant variables
ntypes, _, ntbins, _, _ = np.shape(decoding_performance)
sub_stim_names = [stim_names[type_] for type_ in stim_types]

# generate time points for plotting
stim_t_mids = np.arange(win_size_ms/2, 5000 - (win_size_ms-step_ms)/2, 
                        step_ms) / 1000
if int((5 - stim_t_mids[-1])*1000) < win_size_ms/2:
    stim_t_mids = stim_t_mids[:-1]
    
    
#%% 1x2 matrix of ipsips concon and ipscon conips along with differences as 
#histogram on right of plot
# Plot 1x2: Left = ipsilateral; Right = contralateral (cross-condition)
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18,6))
plt.subplots_adjust(wspace=0.15)
for i, stim_name in enumerate(sub_stim_names):
    # diagonal part
    # avg over 2 folds
    stim_acc = np.mean(decoding_performance[i,0], axis=1)
    mean_acc = np.mean(stim_acc, axis=1)
    # sem_acc = ss.sem(stim_acc, axis=1)
    sdev_acc = np.std(stim_acc, axis=1)
    ax[i].plot(stim_t_mids, mean_acc, label=f'{stim_name}', 
                 color=stim_cols[i+1])#, alpha=0.6)
    ax[i].fill_between(stim_t_mids, mean_acc - sdev_acc, mean_acc + sdev_acc, 
                         color=stim_cols[i+1], alpha=0.2)
    
    
    
    # chance level of diagonal part
    if plot_shuff == True:
        # MAX T APPROACH
        stim_chance_acc = shuff_decoding_performance[i,0]
        mean_chance_acc = np.mean(stim_chance_acc, axis=1)
        # get the deviation from the means
        deviation_arr = mean_chance_acc[:,None] - stim_chance_acc 
        max_dev_arr = np.max(deviation_arr, axis=0)
        min_dev_arr = np.min(deviation_arr, axis=0)
        upper_bound = np.percentile(max_dev_arr, ci_perc + (100-ci_perc)/2)
        lower_bound = np.percentile(min_dev_arr, (100-ci_perc)/2)
        upper_bounds_arr = mean_chance_acc + upper_bound
        lower_bounds_arr = mean_chance_acc + lower_bound
        # calculate significance bounds 
        # Identify significant time points (above corrected upper bound)
        sig_mask = mean_acc > upper_bounds_arr  # boolean array
        
        # Calculate the percentage of significant time bins for this region
        sig_percentage = np.sum(sig_mask) / len(sig_mask) * 100
     
        half_step_sec = (step_ms*1.1 / 1000) / 2
    
        # Create a transform that maps x in data coords, y in axes coords
        trans = blended_transform_factory(ax[i].transData, ax[i].transAxes)
        
        # Set fixed vertical position just above the top of the plot
        _, y_max = ax[i].get_ylim()
        max_acc = np.max(mean_acc)
        y_frac = 0.97 
        y_sig_ax = y_frac 
        
        for t, sig in enumerate(sig_mask):
            if sig:
                ax[i].hlines(y=y_sig_ax,
                          xmin=stim_t_mids[t] - half_step_sec,
                          xmax=stim_t_mids[t] + half_step_sec,
                          color=stim_cols[i+1], linewidth=6, transform=trans,
                          clip_on=False)  # allow it outside the axis
    # Apply percentage formatting to y-axis
    ax[i].yaxis.set_major_formatter(FuncFormatter(sf.to_percentage_no_symbol))
    # add lines to mark piezo movements
    [ax[i].axvline(pulse_times[p], color=pulse_colors[p], 
                      linestyle=(0, (3, 3)), linewidth=3) for p in range(10)]
    
    
    # off-diagonal part - cross conditional training and testing
    cc_test_type = stim_names[(i+1)%2+1]
    # Off-diagonal part - cross conditional training and testing
    stim_acc_cc = np.mean(decoding_performance[i,1], axis=1)
    mean_acc_cc = np.mean(stim_acc_cc, axis=1)
    sdev_acc_cc = np.std(stim_acc_cc, axis=1)
    ax[(i+1) % 2].plot(stim_t_mids, mean_acc_cc, label=f'{cc_test_type}', 
                           color='black')#, alpha=0.6)
    ax[(i+1) % 2].fill_between(stim_t_mids, mean_acc_cc - sdev_acc_cc, 
                                  mean_acc_cc + sdev_acc_cc, 
                                  color='black', alpha=0.2)
    
    # Chance level of off-diagonal part
    if plot_shuff == True:
        # MAX T APPROACH
        stim_chance_acc_cc = shuff_decoding_performance[i,1]
        mean_chance_acc_cc = np.mean(stim_chance_acc_cc, axis=1)
        # get the deviation from the means
        deviation_arr = mean_chance_acc_cc[:,None] - stim_chance_acc_cc 
        max_dev_arr = np.max(deviation_arr, axis=0)
        min_dev_arr = np.min(deviation_arr, axis=0)
        upper_bound = np.percentile(max_dev_arr, ci_perc + (100-ci_perc)/2)
        lower_bound = np.percentile(min_dev_arr, (100-ci_perc)/2)
        upper_bounds_arr = mean_chance_acc_cc + upper_bound
        lower_bounds_arr = mean_chance_acc_cc + lower_bound
        sig_mask = mean_acc_cc > upper_bounds_arr  # boolean array
        
        # Calculate the percentage of significant time bins for this region
        sig_percentage = np.sum(sig_mask) / len(sig_mask) * 100
     
        half_step_sec = (step_ms*1.1 / 1000) / 2
    
        # Create a transform that maps x in data coords, y in axes coords
        trans = blended_transform_factory(ax[(i+1) % 2].transData, 
                                          ax[(i+1) % 2].transAxes)
        
        # Set fixed vertical position just above the top of the plot
        _, y_max = ax[(i+1) % 2].get_ylim()
        max_acc = np.max(np.concatenate([mean_acc, mean_acc_cc]))
        y_frac = 0.93 
        y_sig_ax = y_frac 
        
        for t, sig in enumerate(sig_mask):
            if sig:
                ax[(i+1) % 2].hlines(y=y_sig_ax,
                          xmin=stim_t_mids[t] - half_step_sec,
                          xmax=stim_t_mids[t] + half_step_sec,
                          color='black', linewidth=6, transform=trans,
                          clip_on=False)  # allow it outside the axis
        
    [ax[j].axhline(.50, color='k', linestyle='dashed') for j in range(2)]
    
    # Calculate the difference for the current stimulus type
    mean_diff = np.mean(decoding_performance[(i+1)%2, 1], axis=(1, 2)) -\
                        np.mean(decoding_performance[i, 0], axis=(1, 2)) 
                        # Reverse indexing for cross-condition comparison

    # Calculate the average accuracy difference for this stimulus type
    avg_diff = np.mean(mean_diff) * 100  # Convert to percentage
    
    ax[i].text(
        0.95, 0.8,
        rf"$\mu={avg_diff:.1f}\%$",
        transform=ax[i].transAxes, fontsize=18, ha='right',
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=1)
    )

# Label the left edges of the subplots
ax[0].set_ylabel("Accuracy (%)")

# Label the bottom edges of the subplots
ax[0].set_xlabel("Time (s)")
ax[1].set_xlabel("Time (s)")

# Force consistent x-axis ticks and limits on all subplots
for i in range(2):
    ax[i].set_xticks([0,1,2,3,4,5], [0,1,2,3,4,5])
 
fig.tight_layout()

# save figure
plt.savefig(f"{save_dir}/{fname_2x2}_BARS_SURROGATE.{figtype}", transparent=True)
plt.close()



#%% decoding accuracy differences between within- and cross-condition

import numpy as np
from scipy.stats import permutation_test

def paired_permutation_test(data1, data2, statistic='mean', 
                            alternative='two-sided', seed=None, 
                            n_resamples=10000):
    """
    Performs a paired permutation test.

    Args:
        data1: The first set of paired data.
        data2: The second set of paired data.
        statistic: The statistic to compute (e.g., 'mean', 'median').
        alternative: The alternative hypothesis ('two-sided', 'less', 'greater').
        seed: Random seed for reproducibility.
        n_resamples: Number of permutations.

    Returns:
        The p-value.
    """
    def _statistic(x, y):
        if statistic == 'mean':
            return np.mean(x - y)
        elif statistic == 'median':
            return np.median(x - y)
        else:
            raise ValueError("Invalid statistic. Choose 'mean' or 'median'.")

    # Ensure data is a NumPy array and handle potential errors
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    if len(data1) != len(data2):
        raise ValueError("Input arrays must have the same length")

    # Perform the permutation test using SciPy
    result = permutation_test(
        (data1, data2),
        _statistic,
        permutation_type='samples',
        alternative=alternative,
        random_state=seed,
        n_resamples=n_resamples
    )
    return result.pvalue

# Loop over each stimulus type to generate individual figures
# Create a new figure with GridSpec layout for the 70% / 30% column widths
fig = plt.figure(figsize=(10*1.1, 6*1.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[0.75, 0.25])

# First subplot (decoding differences over time)
ax_diff = fig.add_subplot(gs[0])

# Second subplot (KDE), sharing y-axis with the first
ax_kde = fig.add_subplot(gs[1], sharey=ax_diff)

# Loop over each stimulus type to plot both in the same figure
for i, stim_name in enumerate(sub_stim_names):
    # get name of stim type being tested 
    cc_test_type = stim_names[(i+1)%2+1] 
    
    # calculate difference over fold-averaged accuracies, for each bootstrap
    diff_arr = np.mean(decoding_performance[(i+1)%2, 1], axis=1) - \
                np.mean(decoding_performance[i, 0], axis=1)
    mean_diff = np.mean(diff_arr, axis=1)
    sdev_diff = np.std(diff_arr, axis=1)

    # Calculate the average accuracy difference
    avg_diff = np.mean(mean_diff)

    # Plot time series of decoding differences
    ax_diff.plot(stim_t_mids, mean_diff, label=f'{cc_test_type}', 
                 color=stim_cols[i + 1])
    
    # plot std deviation also
    ax_diff.fill_between(stim_t_mids, mean_diff - sdev_diff,
                         mean_diff + sdev_diff, color=stim_cols[i + 1], 
                         alpha=0.2)
    
    # === SURROGATE DISTRIBUTION VIA PERMUTATION ===
    n_surrogates = 1000
    surrogate_means = np.zeros(n_surrogates)
    
    curve_a = np.mean(decoding_performance[(i+1)%2, 1], axis=(1, 2))
    curve_b = np.mean(decoding_performance[i, 0], axis=(1, 2))
    
    for s in range(n_surrogates):
        surrogate_diff = np.zeros(ntbins)
        for t in range(ntbins):
            if np.random.rand() < 0.5:
                surrogate_diff[t] = curve_a[t] - curve_b[t]
            else:
                surrogate_diff[t] = curve_b[t] - curve_a[t]
        surrogate_means[s] = np.mean(surrogate_diff)  # convert to %
        
    # Dashed line for real mean
    ax_kde.axhline(avg_diff, linestyle='--', color=stim_cols[i + 1], linewidth=2)
    
    # Two-sided test with continuity correction
    count = np.sum(np.abs(surrogate_means - np.mean(surrogate_means)) >= np.abs(avg_diff - np.mean(surrogate_means)))

    p_value = (count + 1) / (n_surrogates + 1)
    print(f"P-value for testing on {stim_name}: {p_value:.4f}")
    
    # KDE Line Plot for the distribution of differences
    sns.kdeplot(y=mean_diff, ax=ax_kde, color=stim_cols[i + 1], 
                fill=False, bw_adjust=0.25, label=f'{stim_name}')

    

# === Final plot styling ===
# Horizontal reference line at 0
ax_diff.axhline(0, color='k', linewidth=2.5)
# Pulse time vertical lines
for p in range(10):
    ax_diff.axvline(pulse_times[p], color=pulse_colors[p], 
                    linestyle=(0, (5, 10)), linewidth=2.5)

# Format axes
ax_diff.yaxis.set_major_formatter(FuncFormatter(sf.to_percentage_no_symbol))
ax_diff.set_xlabel("Time (s)")
ax_diff.set_ylabel(r"$\Delta$ accuracy (%)")

# KDE subplot cleanup
ax_kde.spines['top'].set_visible(False)
ax_kde.spines['right'].set_visible(False)
ax_kde.spines['bottom'].set_visible(False)
ax_kde.yaxis.set_major_formatter(FuncFormatter(sf.to_percentage_no_symbol))
ax_kde.yaxis.set_ticks_position('left')
ax_kde.set_xticks([])
ax_kde.set_xlabel("")
plt.setp(ax_kde.get_yticklabels(), visible=False)

# Align layout
ax_diff.set_xticks([0,1,2,3,4,5], [0,1,2,3,4,5])
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(wspace=0.0)

plt.close(fig)



