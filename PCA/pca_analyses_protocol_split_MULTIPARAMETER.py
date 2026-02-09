# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 09:52:30 2025

@author: rejwanfs

Used to generate the PCA figures (Figures 3 and 3-1), but also generates 
additional figures and animations of the 3D PCA trajectories
"""



import numpy as np
import os
import pickle
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt

from matplotlib import colors

from collections import Counter

import utility_functions as uf
from config import  full_dataset_dir, stim_names

# import the animation and the HTML module to create and render the animation
from matplotlib import animation 
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 1024 # in MB

from functools import partial

matplotlib.use('Qt5Agg')

plt.rcParams["figure.figsize"] = [19.2 ,  9.83]
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["figure.titlesize"] = 18
plt.rcParams['axes.linewidth'] = 1.5
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
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
plt.rcParams["axes.labelsize"] = 18
plt.rc('legend',fontsize=10)
plt.rcParams["legend.markerscale"] = 1.5
plt.rcParams['lines.markersize']= 7.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams["font.size"] = 20 
##############################


def split_expt_windows(onset_window, bin_width, stim_win_dur):
    
    onset_window_ms = (np.array(onset_window) * 1000).astype(int)
    bin_width_ms = int(bin_width * 1000)
    # change duration if ignoring some of the pulses
    stim_win_dur = 1000*stim_win_dur
    
    
    if onset_window_ms[0] >= 0:
        num_init_base_bins = 0
    else:
        num_init_base_bins = -onset_window_ms[0] // bin_width_ms
        
    if onset_window_ms[1] > stim_win_dur:
        num_stim_bins = stim_win_dur // bin_width_ms
        num_end_base_bins = (onset_window_ms[1] - stim_win_dur) // bin_width_ms
    else:
        num_stim_bins = onset_window_ms[1] // bin_width_ms
        num_end_base_bins = 0
        
    num_bins_arr = np.array([num_init_base_bins, num_stim_bins, 
                            num_end_base_bins]).astype(int)
    trial_size = int(np.diff(onset_window_ms)[0] // bin_width_ms)
    # make sure all bins accounted for
    assert np.sum(num_bins_arr) == trial_size
    
    return {'stim_win_dur': stim_win_dur, 'trial_size': trial_size, 
            'onset_window_ms': onset_window_ms, 'bin_width_ms': bin_width_ms, 
            'num_init_base_bins': num_init_base_bins,
            'num_end_base_bins': num_end_base_bins,
            'num_stim_bins': num_stim_bins}

        
        
def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X.T).T
    return Xz
    




def pop_avg_pca(reg_avgs, df_valid, init_bins, end_bins, pulse_idx, window,
                save_dir, smooth=True, sigma=3, offset=0, step=1, t_mids=None,
                pulse_times=None, pulse_colors=None, stim_types=None):
    # offset is if we want to ignore the first few pulses
    # valid_df is the combine_dataset.pkl file's dataframe, subsampled to only
    #inclue the units we care about
    
    stim_cols = {1:'r', 2:'b', 3:'g'}
    
    # utility function to clean up and label the axes
    def style_3d_ax(ax, exp_var):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xlabel(f'PC 1 ({exp_var[0]}%)')
        ax.set_ylabel(f'PC 2 ({exp_var[1]}%)')
        ax.set_zlabel(f'PC 3 ({exp_var[2]}%)')
        
    component_x = 0
    component_y = 1
    component_z = 2
    pcs_choice_3d = [component_x, component_y, component_z]
    sub_stim_names = [stim_names[type_] for type_ in stim_types]
    trial_size = np.shape(reg_avgs[sub_stim_names[0]])[1]
    num_init_base_bins = init_bins
    num_end_base_bins = end_bins
    
    
    # change these manually based on region group and mouse data outputs
    reg_groups_az_el_mapping = {"['PERI', 'ECT']": [-70, -92], 
                                "['SSp-bfd']": [74, 88]}
    az = np.zeros(len(region_groups))
    el = np.zeros(len(region_groups))
    for r, regions in enumerate(region_groups):
        try:
            az[r], el[r] = reg_groups_az_el_mapping[f"{regions}"]
        except:
            continue
    
    # animated 3d trajctories
    pca_outputs = {}
    for r, regions in enumerate(region_groups):
        reg_idxs = np.isin(df_valid.region, regions)
        num_units = np.sum(reg_idxs)
        print(f"regions: {regions}, N = {num_units}")
        
        # simple way to avoid rewriting to incorporate kmeans clusters per
        #region
        if num_units == 0:
            continue
        
        reg_data = [z_score(reg_avgs[name][reg_idxs]) for name in sub_stim_names]
        
        Xa = np.concatenate(reg_data, axis=1)
        
        
        reg_az = az[r]
        reg_el = el[r]

        # standardize and apply PCA
        pca = PCA()
        Xa_p = pca.fit_transform(Xa.T).T
        
        # add this pca to the dictionary
        pca_outputs[f"{regions}"] = pca
        
        # remove other components
        Xa_p = Xa_p[:3, :]
        
        
        # apply some smoothing to the trajectories
        if smooth == True:
            for c in range(Xa_p.shape[0]):
                Xa_p[c, :] =  gaussian_filter1d(Xa_p[c, :], sigma=sigma)
        
        reg_exp_var = np.round(
            pca.explained_variance_ratio_[:3] * 100, 1).astype(str)
        

        # plot and save exp variance and components
        fig, axs = plt.subplots(1,2)
        axs[0].plot(np.cumsum(pca.explained_variance_ratio_)*100, 
                    color = 'black', marker = '.')
        axs[0].set_xlabel("PC index")
        axs[0].set_ylabel("Cumualtive explained variance (%)")
        
        pc_weights = pca.components_
        v_min = np.min(pc_weights)
        v_max = np.max(pc_weights)
        divnorm=colors.TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)
        im = axs[1].imshow(pc_weights, aspect='auto', cmap='seismic', 
                            norm=divnorm)
        axs[1].set_xlabel("Unit index")
        axs[1].set_ylabel("PC index")
        cbar = plt.colorbar(im, ax=axs[1], fraction=0.02)
        cbar.set_label(label="PC weights", labelpad=-80, size=10)
        
        fig.suptitle(f"{regions}")
        
        plt.savefig(f"{save_dir}/{regions}_PCs_summary.png")
        plt.close()
        
        # plot and save bar plots of first three principal components
        bar_x = range(num_units)
        fig, axs = plt.subplots(1,3)
        for i in range(3):
            axs[i].bar(bar_x, pc_weights[i])
            axs[i].set_xlabel(f"PC {i+1} ({reg_exp_var[i]}%)")
            
        fig.suptitle(f"{regions} PC weights")
        
        plt.savefig(f"{save_dir}/{regions}_PC_weights.png")
        plt.close()
        
        
        # Create plot of single pcs across time
        fig, axs = plt.subplots(3,1)
        for pc in range(3):
            axs[pc].set_title(f"PC {pc+1} ({reg_exp_var[pc]}%)")
            for t, t_type in enumerate(stim_types):
                pc_stim_type_data = Xa_p[pc, t*trial_size:(t+1)*trial_size]
                axs[pc].plot(t_mids, pc_stim_type_data, 
                             color=stim_cols[t_type], 
                             label=f"{stim_names[t_type]}")
            [axs[pc].axvline(pulse_times[pulse], color = pulse_colors[pulse], 
                              linewidth=2, linestyle=(0, (1,1))) 
             for pulse in range(len(pulse_times))]
            axs[pc].axhline(0, linewidth=2, color='black')
        
        axs[0].legend()
        axs[2].set_xlabel("Time (s)")
            
        fig.suptitle(f"{regions} PC projections over time")
        
        plt.savefig(f"{save_dir}/{regions}_PCs_over_time.png")
        plt.close()
        
        
        # create the figure
        fig = plt.figure(figsize=[9, 9]); plt.close()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        
        
        def animate(i, az, el, exp_var):
            
            ax.clear() # clear up trajectories from previous iteration
            style_3d_ax(ax, exp_var)
            ax.view_init(elev=el, azim=az)
        
            for t, t_type in enumerate(stim_types):
            
                x = Xa_p[component_x, t * trial_size :(t+1) * trial_size][0:i]
                y = Xa_p[component_y, t * trial_size :(t+1) * trial_size][0:i]
                z = Xa_p[component_z, t * trial_size :(t+1) * trial_size][0:i]
                        
                stim_mask = ~np.logical_and(
                    np.arange(z.shape[0]) >= num_init_base_bins,
                    np.arange(z.shape[0]) < (trial_size-num_end_base_bins))
                
                z_stim = z.copy()
                z_stim[stim_mask] = np.nan
                z_prepost = z.copy()
                z_prepost[~stim_mask] = np.nan
                
                
                ax.plot(x, y, z_stim, color=stim_cols[t_type])
                ax.plot(x, y, z_prepost, color=stim_cols[t_type], ls=':')
            
            return []
        
        
        print(f"Animating regions: {regions}")
        anim = animation.FuncAnimation(fig, partial(animate, az=reg_az, 
                                                    el=reg_el, 
                                                    exp_var=reg_exp_var),
                                        frames=range(0, trial_size, step), 
                                        interval=50, blit=False)
        print("Animation complete. Saving")
        with open(f"{save_dir}/POPULATION_{regions}" + 
                  f"_3d_trial-averaged_PCs_{pcs_choice_3d}.html", "w") as file:
            file.write(anim.to_html5_video())
        print("save complete.")
        
        # set up a figure with two 3d subplots, so we can have two different views
        fig2 = plt.figure(figsize=[9, 4])
        ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
        axs = [ax1, ax2]
        
        for ax in axs:
            for t, t_type in enumerate(stim_types):
        
                # for every trial type, select the part of the component
                # which corresponds to that trial type:
                x = Xa_p[component_x, t * trial_size :(t+1) * trial_size]
                y = Xa_p[component_y, t * trial_size :(t+1) * trial_size]
                z = Xa_p[component_z, t * trial_size :(t+1) * trial_size]
                
                stim_mask = ~np.logical_and(
                    np.arange(z.shape[0]) > num_init_base_bins-1,
                    np.arange(z.shape[0]) < (trial_size-num_end_base_bins))
        
        
                # use the mask to plot stimulus and pre/post stimulus separately
                z_stim = z.copy()
                z_stim[stim_mask] = np.nan
                z_prepost = z.copy()
                z_prepost[~stim_mask] = np.nan
        
                ax.plot(x, y, z_stim, color=stim_cols[t_type])
                ax.plot(x, y, z_prepost, color=stim_cols[t_type], ls=':')
                
                # make the axes a bit cleaner
                style_3d_ax(ax, reg_exp_var)
                
        # specify the orientation of the 3d plot        
        ax1.view_init(elev=22, azim=30)
        ax2.view_init(elev=22, azim=110)
        plt.tight_layout()
        
        fig2.savefig(f"{save_dir}/POPULATION_{regions}_3d_trial-averaged_PCs_{pcs_choice_3d}_STATIC.png")
        fig2.savefig(f"{save_dir}/POPULATION_{regions}_3d_trial-averaged_PCs_{pcs_choice_3d}_STATIC.svg")
        # plt.close("fig2")
        
    # save pcas
    pca_outputs['metadata'] = df_valid
    pickle.dump(pca_outputs, open(f"{save_dir}/pca_data.pkl", "wb"))
        

def pca_protocol_split_analysis(region_groups, cluster_group, stim_types, 
                                sigma=20, npcs=3, good_only=True, speed_cutoff=False,
                                dt=0.001, pad=100, smooth=False, sigma_anim=10, step=10,
                                manual_protocol_choice=False, f=1, n_cycles=5, 
                                tot_stim_win_dur=5, k_means_subsample=False,
                                smoothing_kern='gaussian'):

    
    ############# VARIABLES FOR K-MEANS-BASED SUBSAMPLING #################
    n_reg_K = [4,4,3,4,3,3,4] # how many clusters for each region (same order as region
    #groups)
    kmeans_t_win = [-1.0, 6.0]
    kmeans_sigma = 20 # in ms
    cluster_group_nm = cluster_group
    quality = 'ALL units'
    if good_only == True:
        quality = 'HQ units'
    if cluster_group == 'good':
        cluster_group_nm = 'Single'
    base_dir = f"{f}Hz, {n_cycles} cycles, {tot_stim_win_dur}s/"
    base_dir += f"win_relative = {kmeans_t_win}, {cluster_group_nm} units, "
    base_dir += f"{quality}"

    #groups)
    npcs = 3
    kmeans_dir = f"./../K-means/{base_dir}/sigma = {kmeans_sigma}ms"
    kmeans_dir += f"/stim_types={stim_types}_{npcs}-PCs"
    #######################################################################
       
    
    # open stim metadata and use it to group similar protocols, or decide 
    stim_meta = pd.read_excel(
        io=f"{full_dataset_dir}/stimulation_metadata.xlsx")
    
    
    # filter out units based on criteria you chose
    df_subset = df[df.cluster_group == cluster_group]
    if good_only == True:
        df_subset = df_subset[df_subset.good_bool == True]
    
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
    
    #%% loop through protcols and run PCA analysis
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
        
        # get num base bins and end base bins based on t window
        res = split_expt_windows(window, dt, stim_win_dur)
        num_base = res['num_init_base_bins']
        num_end = res['num_end_base_bins']
        
        # create pulse times array
        pulse_times = np.arange(0, n_deflect, 1)/deflect_freq # in seconds
        # this is the protocol of 40hz with envelope of 1hz
        if np.isin(110581, mouse_ids) == True:
            raw_t_arr = np.arange(0,40, 1)/80
            pulse_times = np.concatenate([raw_t_arr + i for i in range(5)])
        pulse_colors = np.tile(['black', 'dimgray'], n_deflect)
        
        print(f"Analysed mouse_ids: {mouse_ids}")
        # remove mouse metadata that will not be analysed
        protocol_df = df_subset[np.isin(df_subset.mouse_id, mouse_ids)]
        # take out data per mouse to feed into data for separation
        stim_dict = uf.partition_stims(mouse_ids, stims, stim_names)
        # generate bin edges and midpoints
        t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win = \
            uf.get_t_data(window, dt, pad)
        
        
        # get all units in regions of interest
        group_idxs = np.isin(
            protocol_df.region, np.concatenate(region_groups))
        # limit df to this while resetting index
        df_valid = protocol_df.loc[
            protocol_df.index[group_idxs],:].copy().reset_index()
        tot_units = len(df_valid)
        # now get psths of these neurons in each of the stim types you selected
        psth_dict = {} # will contain psths for each stim type
        for type_ in stim_types:
            type_name = stim_names[type_]

            psth_dict[type_name] = uf.flexi_get_smoothed_psth(
                sp, df_valid, stim_dict, type_name, tot_units, sigma, pad, 
                t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win,
                smoothing_kern=smoothing_kern)
        
        
            
        # define base directory for saving/loading
        protocol_dir = f"{stim_freq}Hz, {num_cycles} cycles, {stim_win_dur}s/"
        win_dir = \
            f"win_relative = {window}, {cluster_group_nm} units, {quality}"
            
        save_stim_names = [stim_names[type_] for type_ in stim_types]
        save_dir = f"./PCA_output/{protocol_dir}/{win_dir}" + \
            f"/{smoothing_kern} smoothing, sigma = {sigma}ms, stim_types = {save_stim_names}"
        os.makedirs(save_dir, exist_ok=True)
        
        if (sigma != 0 and smooth == True):
            print("You already smoothed average responses, doesn't make sense" + 
                  " to smooth again! Cancelling pop_avg_pca computation.")
            exit(1)

        
        # run the analysis
        if k_means_subsample == False:
            
            pop_avg_pca(psth_dict, df_valid, num_base, num_end, window=window,
                        pulse_idx=0, save_dir=save_dir, smooth=smooth, 
                        sigma=sigma_anim, step=step, t_mids=t_mids, 
                        pulse_times=pulse_times, pulse_colors=pulse_colors, 
                        stim_types=stim_types)
        else:
            # load k means data
            kmeans = np.load(f"{kmeans_dir}/k_means_output.npy", allow_pickle=True)
            k_means_save_dir = f"{save_dir}/cluster-specific"
            os.makedirs(k_means_save_dir, exist_ok=True)
            for r, regions in enumerate(region_groups):
                # sub df_valid based on region
                reg_idxs = np.isin(df_valid.region, regions)
                sub_psth_dict = {}
                for type_ in stim_types:
                    st_name = stim_names[type_]
                    sub_psth_dict[st_name] = psth_dict[st_name][reg_idxs]
                
                
                
                reg_numK = n_reg_K[r]
                reg_kmeans = kmeans[r, reg_numK-2]
                
                reg_labels = reg_kmeans.labels_
                
                counter_dict = Counter(reg_labels)
                unique_labels = list(counter_dict.keys())
                lab_sorter = np.argsort(unique_labels)
                
                label_counts = list(Counter(reg_labels).values())
                label_counts = np.array(label_counts)[lab_sorter]
                
                
                K_col_nm = f"K_idx_({reg_numK})"
                reg_df_valid = df_valid[reg_idxs]
                reg_df_valid.insert(np.shape(reg_df_valid)[1], K_col_nm, 
                                    reg_labels)
                
                reg_dir = f"{k_means_save_dir}/{regions}, K = {reg_numK}"
                os.makedirs(reg_dir, exist_ok=True)
                reg_df_valid.to_pickle(f"{reg_dir}/reg_df.pkl")
                
                for k_lab in range(reg_numK):
                    k_lab_dir = f"{reg_dir}/k_lab = {k_lab}"
                    os.makedirs(k_lab_dir, exist_ok=True)
                    # sub psth data and df
                    lab_idxs = (reg_labels == k_lab)
                    lab_df = reg_df_valid[lab_idxs]
                    # make sure we've subsampled correctly
                    assert np.all(lab_df[K_col_nm] == k_lab)
                    lab_psth_dict = {}
                    for type_ in stim_types:
                        st_name = stim_names[type_]
                        lab_psth_dict[st_name] = sub_psth_dict[st_name][lab_idxs]
                    
                    # now plug into pca plotter
                    pop_avg_pca(lab_psth_dict, lab_df, num_base, num_end, 
                                window=window, pulse_idx=0, 
                                save_dir=k_lab_dir, smooth=smooth, 
                                sigma=sigma_anim, step=step, t_mids=t_mids, 
                                pulse_times=pulse_times, 
                                pulse_colors=pulse_colors, 
                                stim_types=stim_types)





if __name__ == "__main__":
    
    plt.ioff()
    
    #%% load and filter data
    #load population data
    pop_data = pd.read_pickle(f"{full_dataset_dir}/combined_data.pkl")
    # take out the constituting data
    sp = pop_data.pop_sp
    stims = pop_data.pop_stims
    df = pop_data.df
        
    ############### GENERAL PARAMETERS ################
    dt = 0.001 #in seconds
    smoothing_kern = 'causal_half_gaussian' # 'gaussian/causal_half_gaussian/exp'
    sigma = 20 # for smoothing PSTHs
    smooth = False # smooth pca trace for animated plots?
    sigma_anim = 10 # generating animations, how much to smooth
    step = 10 # points to skip in plotting - must be at least 1
    pad = 100 # in units of dt
    region_groups = [['PERI', 'ECT'], ['SSp-bfd'], ['TeA']]
    # region_groups = [['TeA'], ['AUDv']]
    speed_cutoff = False
    mod_type = None
    good_only = True # whether passing 3qc is required or not
    # cluster_group = 'good' # good=single units
    k_means_subsample = False # Whether to subsample neurons based on extracted
    #k means clusters
    
    
    # manual protocol choice if desired
    manual_protocol_choice = True
    f = 1 # frequency
    n_cycles = 5 # num cycles < which is 0.5*number of deflections >
    tot_stim_win_dur = 5.0
    
    for stim_types in [[1,2]]:
        # can also do [1] and [2] separately in above loop for generating 
        #regression of individual stimulation types
        for cluster_group in ['good']:
            # good = single units, MUA = MUA
            pca_protocol_split_analysis(
                region_groups, cluster_group, stim_types, sigma=sigma, 
                manual_protocol_choice=manual_protocol_choice, f=f, 
                n_cycles=n_cycles, tot_stim_win_dur=tot_stim_win_dur, 
                k_means_subsample=k_means_subsample, 
                smoothing_kern=smoothing_kern)
    
    
    
                    