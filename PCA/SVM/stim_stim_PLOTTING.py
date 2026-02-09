# -*- coding: utf-8 -*-
"""
Created on Fri May 23 07:55:41 2025

@author: rejwanfs
"""



import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
import svm_functions_label_shuffling as sf
from matplotlib.lines import Line2D
import scipy.stats as ss
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
plt.rcParams['axes.linewidth'] = 3
plt.rcParams.update({
    'xtick.major.size': 8,   # X and Y tick major size
    'ytick.major.size': 8,
    'xtick.major.width': 1.5,  # X and Y tick major width
    'ytick.major.width': 1.5,
    'xtick.minor.size': 4,   # X and Y tick minor size
    'ytick.minor.size': 4,
    'xtick.minor.width': 1,  # X and Y tick minor width
    'ytick.minor.width': 1,
})
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
z_scored=False
# manual protocol choice if desired
manual_protocol_choice = True
f = 1 # frequency
n_cycles = 5 # num cycles < which is 0.5*number of deflections >
tot_stim_win_dur = 5.0

plot_shuff = True
n_shuff = 1000 # number of times to repeat whole analysis, but shuffling labels
shuff_nm = 'random' # 'cyclic' or 'random'

multiplot = True
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
kmeans_reg_grps = [['TeA'], ['SSp-bfd'], ['SSs'], ['AUDp'], ['AUDd'], ['AUDv'],
                   ['PERI', 'ECT']]
kmeans_dir = f"./../K-means/{base_dir}/sigma = {kmeans_sigma}ms"
kmeans_dir += f"/stim_types={stim_types}_{npcs}-PCs"
r_kmeans = {f"{kmeans_reg_grps[0]}": 0, f"{kmeans_reg_grps[1]}": 1,
            f"{kmeans_reg_grps[2]}": 2, f"{kmeans_reg_grps[3]}": 3,
            f"{kmeans_reg_grps[4]}": 4, f"{kmeans_reg_grps[5]}": 5,
            f"{kmeans_reg_grps[6]}": 6}
"""lazy way of getting the varialbe (adjust if PER transient subpop needed)"""
k_means_group = np.arange(1, n_reg_K[r_kmeans[f"{region_groups[0]}"]]+1, 1)
# k_means_group = np.array([1,2,3,4]) # which clusters to use for region_groups reg.
z_scored = False
zsuffix = ""
if z_scored == True:
    zsuffix += "_z_scored"
#######################################################################


print("currently this code only takes the data used in the plots of the FENS "
      + "poster I made. See fens_idxs.")
print("HANDLING 0 SDEV IN Z SCORE BY SETTING FR TO 0 " +
      "- THIS IS IMPORTANT AND SHOULD BE TAKEN INTO ACCOUNT")
print("ADD BARREL CORTEX TOO")

# plt.ioff()

#%% plot stim v stim accuracies

# generate time points for plotting
stim_t_mids = np.arange(win_size_ms/2, 5000 - (win_size_ms-step_ms)/2, 
                        step_ms) / 1000
if int((5 - stim_t_mids[-1])*1000) < win_size_ms/2:
    stim_t_mids = stim_t_mids[:-1]

# define file name suffixes for loading
suffix = f"{nboot}-boot_{grp_cvs}-fold_{sigma}ms_{N}_units" + \
    f"_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"


svs_load_dir = f"./svm_stim_v_stim_{shuff_nm}_shuffling{zsuffix}/cluster_specific, dt={dt}s, sigma={sigma}/" + \
    f"{region_groups}_clusters_{k_means_group}"

# set save directory
svs_save_dir = f"./svm_stim_v_stim_{shuff_nm}_shuffling{zsuffix}/cluster_specific_postprocessing, dt={dt}s, sigma={sigma}/" + \
    f"{region_groups}_clusters_{k_means_group}"
os.makedirs(f"{svs_save_dir}", exist_ok=True)

# set file names for saving uniquely
if plot_shuff == False:
    fname_svs = f"stim_v_stim_{region_groups[0]}_{suffix}"
else:
    shuff_suffix = f"{nboot}-boot_{n_shuff}-shuff_{grp_cvs}-fold_{sigma}ms_" + \
        f"{N}_units_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"
    shuf_fname = f"shuffled_performance_data_{region_groups[0]}_{shuff_suffix}.npy"
    shuff_decoding_performance = np.load(f"{svs_load_dir}/{shuf_fname}")
    fname_svs = f"stim_v_stim_{region_groups[0]}_{shuff_suffix}"
    
    
# load population decoding accuracy data
file_name = f"performance_data_{region_groups[0]}_{suffix}.npy"
decoding_performance = np.load(f"{svs_load_dir}/{file_name}")

# infer relevant variables
ntbins, _, _ = np.shape(decoding_performance)


# Create a new figure with GridSpec layout for the 70% / 30% column widths
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[0.7, 0.3])
ax_diff = fig.add_subplot(gs[0])
ax_kde = fig.add_subplot(gs[1])


ax_diff.axhline(0.5, color='k', label='Chance')
# Apply percentage formatting to y-axis
ax_diff.yaxis.set_major_formatter(FuncFormatter(sf.to_percentage_no_symbol))
# add lines to mark piezo movements
[ax_diff.axvline(pulse_times[p], color=pulse_colors[p], 
                  linestyle=(0, (5, 10)), linewidth=2.5) 
 for p in range(10)]
# plot mean and sem of decoding performance
stim_flat_acc = decoding_performance.reshape(ntbins,grp_cvs*nboot)
mean_acc = np.mean(stim_flat_acc, axis=1)
sem_acc = ss.sem(stim_flat_acc, axis=1)
ax_diff.plot(stim_t_mids, mean_acc, color='black')
ax_diff.fill_between(stim_t_mids, mean_acc - sem_acc, mean_acc + sem_acc, 
                     color='black', alpha=0.2)

# repeat for empirical distn shuffled
if plot_shuff == True:
    # get the deviation from the means
    # MAX T APPROACH
    shuff_mean = np.mean(shuff_decoding_performance, axis=1)
    # get the deviation from the means
    deviation_arr = shuff_mean[:,None] - shuff_decoding_performance 
    max_dev_arr = np.max(deviation_arr, axis=0)
    min_dev_arr = np.min(deviation_arr, axis=0)
    upper_bound = np.percentile(max_dev_arr, ci_perc + (100-ci_perc)/2)
    lower_bound = np.percentile(min_dev_arr, (100-ci_perc)/2)
    upper_bounds_arr = shuff_mean + upper_bound
    lower_bounds_arr = shuff_mean + lower_bound
    
    ax_diff.plot(stim_t_mids, shuff_mean, label='Shuffled', 
                 color='grey')
    ax_diff.fill_between(stim_t_mids, lower_bounds_arr, upper_bounds_arr, 
                         color='grey', alpha=0.2)   
    
ax_diff.set_xlabel('Time (s)')
ax_diff.set_ylabel("Accuracy (%)")


# make distribution based on this
# add solid line indicating 50%
ax_kde.axhline(0.5, color='black', linewidth=2.5)

# KDE Line Plot for the distribution of differences across time for the current 
#stimulus type
sns.kdeplot(y=mean_acc, ax=ax_kde, color='black', fill=False, 
            linewidth=2, bw_adjust=0.25)  # Line-only KDE

# get mean accuracy as percentage
avg_acc = np.mean(mean_acc) * 100

# Add average difference as text in the plot area
ax_kde.text(0.3, 0.95, rf"$\mu = {avg_acc:.2f}$%", 
            transform=ax_kde.transAxes, ha='left', va='top',
            fontsize=20, color='k')


# Customize the KDE subplot aesthetics
ax_kde.spines['top'].set_visible(False)
ax_kde.spines['right'].set_visible(False)
ax_kde.spines['bottom'].set_visible(False)  # Only keep y-axis spine
ax_kde.yaxis.set_major_formatter(FuncFormatter(sf.to_percentage_no_symbol))  # Set y-axis to percentage
ax_kde.yaxis.set_ticks_position('left')  # Keep only y-axis ticks
ax_kde.set_xticks([])  # No x-axis ticks
ax_kde.set_ylabel("Accuracy (%)")




fig.suptitle('SVM Decoding Accuracy: Contralateral vs Ipsilateral' + 
              f"\n{region_groups[0]} | Clusters: {k_means_group} |\n" + 
              f"{nboot} bootstraps | {grp_cvs}-fold CV | σ = {sigma}ms" + 
              f" | N = {N} units | {win_size_ms}ms chunks " + 
              f"({step_ms}ms step) | C = {C} | " + 
              f"sample fraction: {int(btf*100)}%")
fig.tight_layout()
plt.savefig(f"{svs_save_dir}/{fname_svs}.{figtype}", transparent=True)
plt.close()





#%% PLOTTING MULTIPLE REGIONS DECODING ACCURACIES TOGETHER
if multiplot == True:
    
    # ws1 PER
    reg_keep_bool = [False, True, False, False, False, False, True]
    # tea s2
    # reg_keep_bool = [True, False, True, False, False, False, False]
    # PER ws1 ws2 TEa
    # reg_keep_bool = [True, True, True, False, False, False, True]
    # AUD REGIONS
    # reg_keep_bool = [False, False, False, True, True, True, False]
    # all
    # reg_keep_bool = [True, True, True, True, True, True, True]

    regions = [grp for grp, keep in zip(kmeans_reg_grps, reg_keep_bool) if keep]
    reg_cols = {str(kmeans_reg_grps[0]): 'maroon',
                str(kmeans_reg_grps[1]): 'darkorange',
                str(kmeans_reg_grps[2]): 'purple',
                str(kmeans_reg_grps[3]): 'olive',
                str(kmeans_reg_grps[4]): 'teal',
                str(kmeans_reg_grps[5]): 'limegreen',
                str(kmeans_reg_grps[6]): 'black'}
    sub_n_reg_k = np.array(n_reg_K)[reg_keep_bool]
    reg_plot_names = ['TEa', 'wS1', 'S2', 'AUDp', 'AUDd', 'AUDv', 'PER']
    sub_reg_plot_names = np.array(reg_plot_names)[reg_keep_bool]
    
    # define file name suffixes for loading
    suffix = f"{nboot}-boot_{grp_cvs}-fold_{sigma}ms_{N}_units" + \
        f"_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"
        
    shuff_suffix = f"{nboot}-boot_{n_shuff}-shuff_{grp_cvs}-fold_{sigma}ms_" + \
        f"{N}_units_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"
    
    # set save directory
    svs_save_dir = f"./svm_stim_v_stim_{shuff_nm}_shuffling{zsuffix}/" + \
        f"cluster_specific_postprocessing, dt={dt}s, sigma={sigma}/" + \
        f"{regions}_clusters_ALL"
    os.makedirs(f"{svs_save_dir}", exist_ok=True)
    
    # Define function for loading decoding accuracy data
    def load_decoding_data(region, suffix, load_dir, shuffled=False):
        if shuffled == True: prefix = "shuffled_"
        else: prefix = ""
        file_name = f"{prefix}performance_data_{region}_{suffix}.npy"
        return np.load(f"{load_dir}/{file_name}")
    
    # Function to process accuracy data
    def process_accuracy_data(decoding_performance):
        stim_flat_acc = decoding_performance.reshape(ntbins, grp_cvs * nboot)
        mean_acc = np.mean(stim_flat_acc, axis=1)
        sem_acc = ss.sem(stim_flat_acc, axis=1)
        return mean_acc, sem_acc
    
    # Function to process accuracy data
    def process_accuracy_data_sdev(decoding_performance):
        # average over 2 folds
        stim_acc = np.mean(decoding_performance, axis=1)
        mean_acc = np.mean(stim_acc, axis=1)
        sdev_acc = np.std(stim_acc, axis=1)
        return mean_acc, sdev_acc
    
    # define two load directories
    svs_load_base = f"./svm_stim_v_stim_{shuff_nm}_shuffling{zsuffix}/" + \
        f"cluster_specific, dt={dt}s, sigma={sigma}" 
    
    # Create a new figure with GridSpec layout for 70% / 30% column widths
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.7, 0.3])
    ax_diff = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])  # Change name from ax_kde to ax_bar for bar plot
    
   
    
    ax_diff.axhline(0.5, color='k', linestyle="--")
    # Apply percentage formatting to y-axis
    ax_diff.yaxis.set_major_formatter(FuncFormatter(sf.to_percentage_no_symbol))
    # add lines to mark piezo movements
    [ax_diff.axvline(pulse_times[p], color=pulse_colors[p], 
                      linestyle=(0, (5, 10)), linewidth=2.5) 
     for p in range(10)]
    
    # generate time points for plotting
    stim_t_mids = np.arange(win_size_ms/2, 5000 - (win_size_ms-step_ms)/2, 
                            step_ms) / 1000
    if int((5 - stim_t_mids[-1])*1000) < win_size_ms/2:
        stim_t_mids = stim_t_mids[:-1]
        
        
    
    # Customize the KDE subplot aesthetics
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_xticks([])  # No x-axis ticks
    ax_bar.set_ylabel("Percentage sig. bins")    
    
    # same for dec acc plot
    ax_diff.spines['top'].set_visible(False)
    ax_diff.spines['right'].set_visible(False)
    
    legend_handles = []
    max_acc = 0
    for r, iter_regs in enumerate(regions):
        
        reg_str = str(iter_regs)
        
        k_means_group = np.arange(1, sub_n_reg_k[r]+1, 1)
        svs_load_name = f"{svs_load_base}/[{iter_regs}]_clusters_{k_means_group}"
    
        # Load decoding accuracy data for both regions
        decoding_performance = load_decoding_data(iter_regs, suffix, svs_load_name)
        
        # Infer relevant variables
        ntbins, _, _ = np.shape(decoding_performance)
        
        # Compute means and SEMs <now sdev> for both regions
        mean_acc, sdev_acc = process_accuracy_data_sdev(decoding_performance)
        
        # update maxy based on this
        max_acc = np.max([max_acc, max(mean_acc)])
        
        # Plot mean and SEM of decoding performance for both regions
        ax_diff.plot(stim_t_mids, mean_acc, color=reg_cols[reg_str], )
        ax_diff.fill_between(stim_t_mids, mean_acc - sdev_acc, 
                             mean_acc + sdev_acc, 
                             color=reg_cols[reg_str], alpha=0.2)
        ax_diff.set_xlabel('Time (s)')
        ax_diff.set_ylabel('Accuracy (%)')
    
        
        # Create custom legend handles with thicker lines
        legend_handles.append(Line2D([0], [0], color=reg_cols[reg_str], lw=4, 
                                     label=sub_reg_plot_names[r]))
    
    
    # derive optimal position to put signif bars
    _, y_max = ax_diff.get_ylim()
    y_frac = max_acc/y_max
    num_regs = len(regions)
    sig_percentages = []  # To store the percentage of significant time bins for each region
    for r, iter_regs in enumerate(regions):
        
        reg_str = str(iter_regs)
        
        k_means_group = np.arange(1, sub_n_reg_k[r]+1, 1)
        svs_load_name = f"{svs_load_base}/[{iter_regs}]_clusters_{k_means_group}"
        
        # Load decoding accuracy data for both regions
        decoding_performance = load_decoding_data(iter_regs, suffix, svs_load_name)
        
        # Infer relevant variables
        ntbins, _, _ = np.shape(decoding_performance)
        
        # Compute means and SEMs for both regions
        mean_acc, sdev_acc = process_accuracy_data_sdev(decoding_performance)
    
        # load shuffled decoding accracy to get CI intevals
        shuff_decoding_performance = load_decoding_data(iter_regs, shuff_suffix, svs_load_name, shuffled=True)
        
        
        # MAX T APPROACH
        shuff_mean = np.mean(shuff_decoding_performance, axis=1)
        # get the deviation from the means
        deviation_arr = shuff_mean[:,None] - shuff_decoding_performance 
        max_dev_arr = np.max(deviation_arr, axis=0)
        upper_bound = np.percentile(max_dev_arr, ci_perc + (100-ci_perc)/2)
        upper_bounds_arr = shuff_mean + upper_bound
        
        # Identify significant time points (above corrected upper bound)
        sig_mask = mean_acc > upper_bounds_arr  # boolean array
        
        # Calculate the percentage of significant time bins for this region
        sig_percentage = np.sum(sig_mask) / len(sig_mask) * 100
        sig_percentages.append(sig_percentage)  # Store the percentage
     
        half_step_sec = (step_ms*1.1 / 1000) / 2
    
        # Create a transform that maps x in data coords, y in axes coords
        trans = blended_transform_factory(ax_diff.transData, ax_diff.transAxes)
        
        # Set fixed vertical position just above the top of the plot
        y_sig_ax = y_frac + (num_regs + 1 - r) * 0.02  # space them out for each region if needed
        
        for t, sig in enumerate(sig_mask):
            if sig:
                ax_diff.hlines(y=y_sig_ax,
                               xmin=stim_t_mids[t] - half_step_sec,
                               xmax=stim_t_mids[t] + half_step_sec,
                               color=reg_cols[reg_str],
                               linewidth=4,
                               transform=trans,
                               clip_on=False)  # allow it outside the axis

    
    
    # Create the bar chart for the second subplot
    ax_bar.bar(range(num_regs), sig_percentages, color=[reg_cols[str(r)] for r in regions], alpha=0.7)
    
    # Add the legend for the bar chart
    ax_bar.set_xticks(range(num_regs), sub_reg_plot_names, rotation=45, ha='right')  # Rotate labels for clarity

    for reg, perc in zip(sub_reg_plot_names, sig_percentages):
        print(f"{reg}: {perc:.4f}% sig. bins")
    
    
    # create legend
    ax_diff.legend(handles=legend_handles, loc='best', 
                   prop={'weight': 'bold', 'size': 12})
    
    
    # Super title
    fig.suptitle('SVM Decoding Accuracy: Contralateral vs Ipsilateral\n' + 
                  f"{nboot} bootstraps | {grp_cvs}-fold CV | \u03C3 = {sigma}ms" + 
                  f" | N = {N} units | {win_size_ms}ms chunks ({step_ms}ms step) | C = {C} | " + 
                  f"sample fraction: {int(btf*100)}%")
    
    fig.tight_layout()
    plt.savefig(f"{svs_save_dir}/CI_stim_v_stim_{suffix}.{figtype}", transparent=True)
    plt.close()

