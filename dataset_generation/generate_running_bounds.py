# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd

#### PLOTTING PARAMETERS ####
plt.rcParams["figure.figsize"] = [19.2 ,  9.83]
plt.rcParams["figure.titlesize"] = 17

plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams["axes.labelsize"] = 12
plt.rcParams['lines.linewidth'] = 1

plt.rc('legend',fontsize=8)
#################################




def binary_classification(mouse_spd, mouse_stims, num_pulses, stim_dur, 
                          chosen_sig=100, min_span_ms=1000, v_thresh=5):

    save_dir = "./../full_dataset/speed_classification/"
    os.makedirs(f"{save_dir}", exist_ok=True)
    hist_dir = f"{save_dir}/spd_histograms"
    os.makedirs(f"{hist_dir}", exist_ok=True)
    
    stims = mouse_stims
    spd = mouse_spd
    # remove nan data (start and end of recording)
    spd_nonnan = spd[~np.isnan(spd[:,1]),:]
    spd_t = spd_nonnan[:,1]
    spd_val = spd_nonnan[:,0]
    bin_dt = 0.005
    #interpolate spd to be every 5 milliseconds (else we're oversampling data)
    t_mids = np.arange(np.round(spd_t[0],2), np.round(spd_t[-1],2)+bin_dt, bin_dt)
    interp_spd = np.interp(t_mids, spd_t, spd_val)

    """quick check that the interp is not completely off"""
    # plt.plot(t_mids, interp_spd, marker=None, linestyle='--')
    # plt.plot(spd_t, spd_val, marker=None, linestyle='--')
    """"""
    
    plt.plot(t_mids, interp_spd, marker=None, linewidth=1)
    plt.tight_layout()

    # convert to number of bins needed to satisfy across time window
    min_span_bins = np.ceil(min_span_ms / (bin_dt*1000)).astype(int)

    spd_smth = gaussian_filter1d(interp_spd, sigma=chosen_sig)
    plt.plot(t_mids, spd_smth, marker=None, label=fr"$\sigma=${chosen_sig*5}ms")
    plt.axhline(v_thresh, color='red')
    plt.legend()
    
    # add stim times
    # define stim info
    stim_t = stims.loc[::num_pulses,'pulse_t'].values
    stim_id = stims.loc[::num_pulses,'type_'].values
    stim_cols = {1:'r', 2:'b', 3:'g'} 
    for type_idx, type_name in stim_names.items():
        iter_stims = stim_t[np.where(stim_id == type_idx)[0]]
        [plt.axvspan(iter_stims[i], iter_stims[i]+stim_dur, 
                        ymin=0, ymax=0.04, 
                        label="_"*i + f"{type_name}", 
                        alpha=0.35, color=stim_cols[type_idx]) 
         for i in range(len(iter_stims))]

    spd_bool = spd_smth > v_thresh
    spd_binary = np.where(spd_smth > v_thresh)[0]
    if len(spd_binary) != 0:
        # get points where speeds drops back down <add 1 so we slice correctly>
        binary_bounds = np.where(np.diff(spd_binary) > min_span_bins)[0]
        # add 0 as that's where first idx where speed goes above threshold
        binary_bounds = np.insert(binary_bounds, 0, -1)
        binary_bounds = np.append(binary_bounds, len(spd_binary)-1)
       
        # now determine which of these is bigger than min_span_bins
        accepted_slices = []
        for i in range(len(binary_bounds)-1):
            idx0 = spd_binary[binary_bounds[i] + 1]
            idx1 = spd_binary[binary_bounds[i+1]]
            # make sure we grabbed the right slice
            slice_bool = spd_bool[idx0:idx1]
            # assert np.all(slice_bool) == True
            slice_num_bins = len(slice_bool)
            if slice_num_bins >= min_span_bins:
                accepted_slices.append([idx0, idx1])
    
        t_bounds_arr = np.empty((len(accepted_slices),2))
        for i, slice_ in enumerate(accepted_slices):
            t_bounds_arr[i] = t_mids[slice_[0]], t_mids[slice_[1]]
            plt.axvspan(t_bounds_arr[i,0], t_bounds_arr[i,1], color='purple', 
                        alpha=0.2, ymin=0.0, ymax=1)
            
        # save the running bounds array
        np.save(f"{save_dir}/{mouse_id}_running_bounds.npy", t_bounds_arr)
        
    # save the plot
    plt.savefig(f"{save_dir}/{mouse_id}_v-thresh={v_thresh}," +
                f"min_span_ms={min_span_ms},dt={bin_dt}," + 
                f"sig={int(chosen_sig*bin_dt*1000)}ms.png")
    plt.close()
    
    # Make a plot of the running speed distribution
    plt.hist(spd_val, bins=100)
    plt.savefig(f"{hist_dir}/{mouse_id}_speed_distribution.png")
    plt.close()
    
    plt.hist(interp_spd, bins=100)
    plt.savefig(f"{hist_dir}/{mouse_id}_speed_distribution_smoothed.png")
    plt.close()
    
    plt.hist(spd_val, bins=100); plt.yscale('log')
    plt.savefig(f"{hist_dir}/{mouse_id}_speed_distribution_log-scale.png")
    plt.close()
    
    plt.hist(spd_val[spd_val != 0], bins=100)
    plt.savefig(f"{hist_dir}/{mouse_id}_speed_distribution_non-zero.png")
    plt.close()



if __name__ == "__main__":
    
    plt.ioff() # this prevents the plots from showing up on your screen
    plt.close('all') # close all plots
    
    stim_names = {1:'Contralateral', 2:'Ipsilateral', 3:'Bilateral'}
    
    
    # load metadata file to extract stimulation parameters
    stim_meta = pd.read_excel(io="./../full_dataset/stimulation_metadata.xlsx")
    
    
    #load speed and stim data
    pop_spd = pd.read_pickle("./../full_dataset/pop_speed_dict.pkl")
    pop_stims = pd.read_pickle("./../full_dataset/pop_stims_dict.pkl")
    
    
    for _, row in stim_meta.iterrows():
        # extract relevant data from mouse metadata
        mouse_id = row.mouse_id
        stim_freq = row.frequency
        num_pulses = row.num_cycles
        stim_dur = row.total_duration
        base_dur = row.baseline_duration
        
        # double num_pulses to account for up and down deflections
        num_pulses *= 2
        # convert to int
        num_pulses = int(num_pulses)
        mouse_id = int(mouse_id)
        
        mouse_stims = pop_stims[mouse_id]
        mouse_spd = pop_spd[mouse_id]
        
        # if data not there, ignore and move on
        if mouse_spd is None:
            continue
        
        # old simulink and arduino traces are different so use 5 instead of 1
        v_thresh = 1
        if np.isin(mouse_id, [105090, 105092, 105810, 105811]) == True:
            v_thresh = 5
        
        binary_classification(mouse_spd, mouse_stims, num_pulses, stim_dur,
                              v_thresh=v_thresh)
