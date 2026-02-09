# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:24:50 2025

@author: rejwanfs

This produces the regression plots in Figure 3. Make sure to run the other
PCA script with [1] and [2] in the stim types loop to make this data available
for analysis in this script
"""


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from config import  full_dataset_dir, stim_names

import matplotlib

matplotlib.use('Qt5Agg')


# CAIRO FOR ILLUSTRATOR
import matplotlib.font_manager as fm
# %matplotlib inline
import matplotlib
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 12})
font = fm.FontProperties(family = 'arial')
matplotlib.use('cairo') # comment out when testing plots
matplotlib.style.use('seaborn-v0_8-poster') # comment out when testing plots


plt.rcParams["figure.figsize"] = [19.2 ,  9.83]
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["figure.titlesize"] = 18
plt.rc('xtick',labelsize=25)
plt.rc('ytick',labelsize=25)
plt.rcParams["axes.labelsize"] = 25
plt.rc('legend',fontsize=20)
plt.rcParams["legend.markerscale"] = 5
plt.rcParams['lines.markersize']= 10
plt.rcParams['lines.linewidth'] = 5

# Set tick mark thickness and length
plt.rcParams['xtick.major.size'] = 12  # Major tick length
plt.rcParams['ytick.major.size'] = 12
plt.rcParams['xtick.minor.size'] = 8  # Minor tick length
plt.rcParams['ytick.minor.size'] = 8

plt.rcParams['xtick.major.width'] = 4  # Major tick thickness
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['xtick.minor.width'] = 3  # Minor tick thickness
plt.rcParams['ytick.minor.width'] = 3

# Set global savefig parameters
plt.rcParams["savefig.transparent"] = True  # Transparent background
plt.rcParams["savefig.bbox"] = "tight"  # Remove extra whitespace



if __name__ == "__main__":
    
    
    #%% set up relevant variables for loading the data
    dt = 0.001 #in seconds
    stim_types_arr = [[1],[2]] # array of stim_types values you used for 
    #comparisons
    # for smoothing, add padding to t_edges
    sigma = 20
    region_groups = [['PERI', 'ECT'], ['SSp-bfd'], ['SSs'], ['AUDp'], ['AUDd'], 
                     ['AUDv'], ['TeA']]
    pc_to_compare = 1 # which principal component to focus on
    pc_idx = pc_to_compare - 1 # convert above to python indexing format
    good_only = True # whether 3 QC passed required or not
    cluster_group = 'good' # single units (good) or MUA
    # manual experimental protocol choice if desired
    manual_protocol_choice = True
    f = 1
    n_cycles = 5
    tot_stim_win_dur = 5
    #########################################################################

    
    #%% load PCA data
    
    # open stim metadata and use it to group similar protocols, or decide 
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
        
    # define base save dir
    quality = 'ALL units'
    if good_only == True:
        quality = 'HQ units'
    
    cluster_group_nm = cluster_group
    if cluster_group == 'good':
        cluster_group_nm = 'Single'
        
    #%% loop through protcols and load PCA data
    for i, protocol in grouped_stim_meta.iterrows():
        stim_freq = protocol.frequency
        num_cycles = protocol.num_cycles
        n_deflect = 2*num_cycles
        # deflect frequency is double cycle frequency
        deflect_freq = 2*stim_freq
        stim_win_dur = protocol.total_duration
        
        # get two lists from current protocol
        base_durations = protocol.baseline_duration
        mouse_ids = protocol.mouse_id
        
        # create window based on lowest baseline duration
        min_dur = np.min(base_durations)
        # this keeps things simple and prevents overlap of baseline or trials
        window = [float(-0.5*min_dur), float(stim_win_dur + 0.5*min_dur)]
        
        print(f"Analysed mouse_ids: {mouse_ids}")
        
       
        protocol_dir = f"{stim_freq}Hz, {num_cycles} cycles, {stim_win_dur}s/"
        win_dir = \
            f"win_relative = {window}, {cluster_group_nm} units, {quality}"
            
        # save dir
        save_dir = rf"./PCA_output/{protocol_dir}" + \
            f"win_relative = {window}, {cluster_group_nm} units, " + \
            f"{quality}/sigma = {sigma}ms, comparison = {stim_types_arr}"
        os.makedirs(save_dir, exist_ok=True)
        
        # define base save dir
        pca_dict = {}
        for stim_types in stim_types_arr:
            load_stim_names = [stim_names[type_] for type_ in stim_types]

            load_dir = f"./PCA_output/{protocol_dir}/{win_dir}" + \
                f"/sigma = {sigma}ms, stim_types = {load_stim_names}"
            pca_stim_type_data = pd.read_pickle(f"{load_dir}/pca_data.pkl")
            pca_dict[f"{stim_types}"] = pca_stim_type_data
        
        
        
        # compare weights of pc1 across stim types for both regions
        fig, axs = plt.subplots(1,len(region_groups))
        for r, regions in enumerate(region_groups):
            
            pc1_weights = []
            for stim_types in stim_types_arr:
                try:
                    reg_stim_pca = pca_dict[f"{stim_types}"][f"{regions}"]
                except:
                    print(f"{stim_types} | {regions} NOT FOUND")
                    continue #move to next if no data
                else:
                    pc1_weights.append(reg_stim_pca.components_[pc_idx])
            
            # next region if nothing found
            if pc1_weights == []:
                continue
            
            # Extract x and y for linear regression
            x = pc1_weights[0]
            y = pc1_weights[1]
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            
            
            # Create regression line
            line = slope * np.array(x) + intercept
            
            # get sign of intercept 
            if np.sign(intercept) == -1:
                sign = " - " # minus sign will already be printed out
            else:
                sign = " + "
            # now make intercept positive so we put sign manually
            intercept = np.abs(intercept)
            
            # Plot scatter with regression line
            axs[r].scatter(x, y, alpha=0.5)
            axs[r].plot(x, line, color='black', 
                     label=f'Fit: y={slope:.2f}x{sign}{intercept:.2e}')
            axs[r].set_title(f"{regions} PC{pc_to_compare} weights distribution")
            axs[r].set_xlabel(f"{stim_names[stim_types_arr[0][0]]} weights")
            axs[r].set_ylabel(f"{stim_names[stim_types_arr[1][0]]} weights")
            
            # set border width
            plt.setp(axs[r].spines.values(), lw=4)
            
            
            # Add R value and p-value as text to the plot
            axs[r].text(0.03, 0.97, f'R: {r_value:.2f}\np-value: {p_value:.2e}', 
                     transform=axs[r].transAxes, fontsize=21, 
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                        alpha=0.8, color='white'))
            # Add legend
            axs[r].legend()
            
            # plot figure separately
            fig, ax = plt.subplots(1,1, figsize=(9.83*0.882,9.83))
            
            # Plot scatter with regression line
            ax.scatter(x, y, alpha=0.5)
            ax.plot(x, line, color='black', 
                     label=f'Fit: y={slope:.2f}x{sign}{intercept:.2e}')
            # ax.set_title(f"{regions}")
            ax.set_xlabel(f"{stim_names[stim_types_arr[0][0]]} PC{pc_to_compare} weights")
            ax.set_ylabel(f"{stim_names[stim_types_arr[1][0]]} PC{pc_to_compare} weights")
            
            # set border width
            plt.setp(ax.spines.values(), lw=4)
            
            # Add R value and p-value as text to the plot
            ax.text(0.03, 0.97, f'R: {r_value:.2f}\np-value: {p_value:.2e}', 
                     transform=plt.gca().transAxes, fontsize=21, 
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                        alpha=0.8, color='white'))
            plt.title(f"{regions}")
            # Add legend
            ax.legend()
            
            fig.savefig(f"{save_dir}/pc_{pc_to_compare}_{regions}.png")
            
            plt.close(fig)
            
            
        # save and close
        plt.savefig(f"{save_dir}/pc_{pc_to_compare}.png")
        plt.close('all')
        
   
                 