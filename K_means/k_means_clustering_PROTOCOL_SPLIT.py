# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:55:19 2025

@author: rejwan
"""

import os
import sys
sys.path.append("..")
from config import full_dataset_dir, stim_names
import utility_functions as uf

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as ss
import pandas as pd
import supporting_functions as sf # in the same directory


#### PLOTTING PARAMETERS ####
plt.rcParams["figure.figsize"] = [19.2 ,  9.83]
plt.rcParams["figure.titlesize"] = 17

plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['lines.linewidth'] = 1

plt.rc('legend',fontsize=12)
#################################

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
            'num_end_base_bins': num_end_base_bins}






def compute_zpsth(pop_data, protocol_df, stim_dict, t_win, base_dir, sp, 
                  t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win,
                  dt=0.001, 
                  spd_cut=False, good_only=True, 
                  cluster_group='good', sigma=20, pad=100):
    
    # other codes use this data so not good to split
    stim_types = [1,2]
    
    tot_units = len(protocol_df)
    
    
    # now get psths of these neurons in each of the stim types you selected
    """this is so fast that it's not worth saving psth without smoothing"""
    z_psth_data = {'psth': {}, 'metadata': protocol_df}
    pop_psth = {'metadata': protocol_df}
    zpsth_dict = z_psth_data['psth']
    for type_ in stim_types:
        
        type_raw_psth = {}
        
        type_name = stim_names[type_]
        # get smoothed psth and mean baseline firing rate
        type_psth, type_base = sf.get_smoothed_psth(
            sp, protocol_df, stim_dict, type_name, tot_units, sigma, pad, t_edges, 
            t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win)
        
        # add to raw psth dict
        type_raw_psth['psth'] = type_psth
        type_raw_psth['baselines'] = type_base
        pop_psth[type_name] = type_raw_psth
        
        #subtract baseline
        base_sub_psth = type_psth - type_base[:,None]
        # get mean and stdve of this
        base_sub_mean = np.nanmean(base_sub_psth, axis = 1, 
                                    keepdims = True)
        base_sub_sdev = np.nanstd(base_sub_psth, axis = 1, 
                                    keepdims = True)
        
        zpsth_dict[type_name] = (base_sub_psth - base_sub_mean) / base_sub_sdev
        
    #save resulting data
    save_dir = \
        f"{base_dir}/sigma = {sigma}ms"
    os.makedirs(save_dir, exist_ok = True)
    
    pickle.dump(z_psth_data, open(f"{save_dir}/pop_z_psth.pkl", "wb"))
    
    pickle.dump(pop_psth, open(f"{base_dir}/pop_psth.pkl", "wb"))
        



def zpsth_pca(t_win, base_dir, region_groups, sigma=20, spd_cut=False):
    
    # other codes use this data so not good to split
    stim_types = [1,2]
    
    from sklearn.decomposition import PCA

    
    # load data
    load_dir = \
        f"{base_dir}/sigma = {sigma}ms"
    
    zpsth_data = pd.read_pickle(f"{load_dir}/pop_z_psth.pkl")
    df = zpsth_data['metadata']
    zpsth = zpsth_data['psth']
    
    pca_data = {}
    for regions in region_groups:
        
        reg_pca_data = pca_data[f"{regions}"] = {}
        reg_pca_transforms = reg_pca_data['pca'] = {}
        reg_pca_exp_var = reg_pca_data['exp_var'] = {}
        reg_pca_comps = reg_pca_data['components'] = {}
        
        reg_idxs = np.isin(df.region, regions)
        reg_pca_data['df_idxs'] = reg_idxs
        
        #compute PCA
        for type_ in stim_types:
            stim_name = stim_names[type_]
            reg_type_zpsth = zpsth[stim_name][reg_idxs]
            stim_reg_pca = PCA()
            pc_transform = stim_reg_pca.fit_transform(reg_type_zpsth).T
            reg_pca_transforms[stim_name] = pc_transform
            reg_pca_exp_var[stim_name] = stim_reg_pca.explained_variance_ratio_
            reg_pca_comps[stim_name] = stim_reg_pca.components_
            # plot explained var ratio
            plt.plot(np.cumsum(reg_pca_exp_var[stim_name]), color = 'black', 
                      marker = '.')
            plt.xlabel("PC index")
            plt.ylabel("Cumualtive explained variance (%)")
            plt.savefig(f"{load_dir}/" + 
                        f"{regions}_{stim_name}_explained_variance.png")
            plt.close()
    
    #save dictionary 
    pickle.dump(pca_data, open(f"{load_dir}/pca_data.pkl", "wb"))
            
        
        
        
        
        
        
def zpsth_kmeans(t_win, base_dir, sigma, n_pcs, region_groups, spd_cut=False, 
                 stim_types=[1,2,3]):
    #NEW: runs sillhouette analysis as well
    
    sub_names = [stim_names[type_] for type_ in stim_types]
    
    from sklearn.cluster import KMeans 
    # from sklearn.metrics import silhouette_samples, silhouette_score
    
    # laod data
    load_part = f"{base_dir}/sigma = {sigma}ms"
    load_name = f"{load_part}/pca_data.pkl"
    
    # define save dir for k means output and silhouette analysis
    save_dir = f"{load_part}/stim_types={stim_types}_{n_pcs}-PCs"
    sill_dir = f"{save_dir}/silhouette_analysis"
    os.makedirs(sill_dir, exist_ok = True)

    pca_data = pd.read_pickle(load_name)
    
    num_regions = len(region_groups)
    k_means_output = np.full((num_regions,9), np.nan, dtype = object)
    for r, regions in enumerate(region_groups):  
        reg_pca_data = pca_data[f"{regions}"]['pca']
        
        concat_data = np.concatenate([reg_pca_data[stim_name][0:n_pcs,:]
                                      for stim_name in sub_names], 0)
        
        n_units = len(concat_data[0])
        max_k = np.min([n_units-1, 11])
        # run k means many times
        for i, K in enumerate(range(2,max_k)):
            kmeans = KMeans(n_clusters = K, init='k-means++', max_iter = 1000,
                            n_init = 100)
            k_means_output[r,i] = kmeans.fit(concat_data.T)
            # compute sillhouette score while here
            if len(stim_types) > 1:
                fig = sf.silhouette_summary(concat_data, k_means_output[r,i],
                                            sub_names)
                fig.suptitle(f"{regions} | K = {K}", fontsize=14,
                             fontweight="bold")
                plt.savefig(f"{sill_dir}/{regions}_{K}_clusters.png")
                plt.close()
            

    
    # save k means output
    np.save(f"{save_dir}/k_means_output.npy", k_means_output)
    
    return k_means_output






def kmeans_plotter(t_win, base_dir, sigma, n_pcs, region_groups, t0_ms, t1_ms,
                   spd_cut=False, 
                   stim_types=[1,2,3], dt = 0.001, filetype='png'):
    
    
    stim_cols = {1:'r', 2:'b', 3:'g'}
    
    sub_stim_names = {i: stim_names[i] for i in stim_types}
    num_types = len(stim_types)
    
    
    from collections import Counter
    
    plt.ioff()
    
    t_edges = dt * np.arange(t_win[0] / dt, t_win[1] / dt + 1, 1)
    if (t_edges[-1] > t_win[1]) & \
        (len(t_edges) > int(np.diff(t_win)[0]/dt + 1)):
            t_edges = t_edges[:-1] # get rid of last element
    
    # do times in seconds
    t_plot = (t_edges[1:] + t_edges[:-1]) / 2
    t0_s = t0_ms / 1000
    t1_s = t1_ms / 1000
    
    # laod data
    load_part = f"{base_dir}/sigma = {sigma}ms"
    load_dir = f"{load_part}/stim_types={stim_types}_{n_pcs}-PCs"
    kmeans_output = np.load(f"{load_dir}/k_means_output.npy", 
                            allow_pickle = True)
    
    num_regions, num_K = np.shape(kmeans_output)
    
    raw_psth_data = pd.read_pickle(f"{base_dir}/pop_psth.pkl")
    
    
    # laod zpsth data
    zpsth_data = pd.read_pickle(f"{load_part}/pop_z_psth.pkl")
    
    # load dataframe with all the metadata of the region groups
    df = zpsth_data['metadata']
    # same for zpsth
    zpsth = zpsth_data['psth']
    
    for r, regions in enumerate(region_groups):
        
        reg_idxs = np.isin(df.region, regions)
        
        num_units = np.sum(reg_idxs)
        elbow_arr = np.full(num_K, np.nan)
        num_clusters = np.empty(num_K, dtype = int)
        labels = np.full(num_K, np.nan, dtype = object)
        
        
        import matplotlib.colors as colors
        
        # FOR EACH NUMBER OF CLUSTERS
        for k_idx in range(num_K):
            iter_kmeans = kmeans_output[r,k_idx]
            elbow_arr[k_idx] = iter_kmeans.inertia_
            labels[k_idx] = iter_kmeans.labels_
            counter_dict = Counter(labels[k_idx])
            unique_labels = list(counter_dict.keys())
            lab_sorter = np.argsort(unique_labels)
            num_clusters[k_idx] = len(unique_labels)
            
            # plot imshow for this kmeans iteration
            # iter_labels = np.tile(labels[k_idx], 3)
            
            label_counts = list(Counter(labels[k_idx]).values())
            label_counts = np.array(label_counts)[lab_sorter]
            
            # subtract 0.5 so it is placed correctly on the imshow plot
            hline_locs = np.cumsum(label_counts)[:-1] - 0.5
            
            
            plot_order = np.argsort(labels[k_idx])
            fig, axs = plt.subplots(num_types, sharex = True)
            if num_types == 1:
                axs = [axs] # else the axs is not an array hence not indexable
            plt.subplots_adjust(wspace=0, hspace = 0.05)
            # for stim_idx, stim_name in sub_stim_names.items():
            for i, stim_idx in enumerate(sub_stim_names.keys()):
                stim_name = sub_stim_names[stim_idx]
                iter_zpsth = zpsth[stim_name][reg_idxs]
                axs[i].imshow(
                    iter_zpsth[plot_order], aspect = 'auto',
                    extent = [t0_ms, t1_ms, num_units-0.5, -0.5],
                    interpolation='none') 
                axs[i].axvline(0, color = 'black', linewidth = 1)
                axs[i].axvline(5, color='black', linewidth=1)
                axs[i].yaxis.set_tick_params(labelleft=False)
                axs[i].set_ylabel(stim_name)
                axs[i].hlines(hline_locs, xmin=t0_ms, xmax = t1_ms, 
                                        linestyle = '--', linewidth = 0.7,
                                        color = 'white')
                
            fig.suptitle(rf"{regions} | $\sigma$ = {sigma}ms | " + 
                          f"K = {num_clusters[k_idx]}" +
                          f" | T = [{t0_ms}, {t1_ms}]ms")
            
            fig.text(0.05,0.5, "Neurons", rotation = 'vertical', size = 20,
                      va = 'center')
            fig.text(0.5,0.05, "Time (s)", size = 20,
                      ha = 'center')
            plt.savefig(f"{load_dir}/{regions}_ZPSTH_{num_clusters[k_idx]}" +
                        f"-CLUSTERS.{filetype}")
            plt.close('all')
            
            
            fig2, axs2 = plt.subplots(num_types+1,num_clusters[k_idx], 
                                      sharex = True) 
            plt.subplots_adjust(wspace=0.2, hspace = 0.0)
            for i, stim_idx in enumerate(sub_stim_names.keys()):
                stim_name = sub_stim_names[stim_idx]
                #stim_idx is 123
                type_raw_dict = raw_psth_data[stim_name]
                iter_mean_base = type_raw_dict['baselines'][reg_idxs]
                
                smoothed_raw = type_raw_dict['psth'][reg_idxs]
                smoothed_raw -= iter_mean_base[:,None]
                
                
                """NORMALISING TO ONE"""
                smoothed_raw = smoothed_raw / np.max(np.abs(smoothed_raw), 
                                                      axis=1, keepdims=True)
                
                for k_lab in range(num_clusters[k_idx]):
                    plot_bool = (labels[k_idx] == k_lab)
                    k_raw_data = smoothed_raw[plot_bool]
                    k_raw_mean = np.mean(k_raw_data, axis = 0)
                    k_raw_sem = ss.sem(k_raw_data, axis = 0)
                    k_units = sum(plot_bool) # replace w label_counts[k_lab]
                    divnorm = colors.TwoSlopeNorm(vmin=np.min(k_raw_data), 
                                        vmax=np.max(k_raw_data), vcenter=0)
                    axs2[i+1,k_lab].imshow(k_raw_data, aspect='auto',
                                    extent = [t0_s, t1_s, k_units-0.5, -0.5],
                                    interpolation='none',cmap = 'jet',
                                    norm=divnorm) 
                    axs2[i+1,k_lab].axvline(0, color='black', linewidth=1)
                    axs2[i+1,k_lab].axvline(5, color='black', 
                                                  linewidth=1)
                    axs2[i+1,k_lab].yaxis.set_tick_params(labelleft=False)
                    # remove tick marks too
                    axs2[i+1,k_lab].yaxis.set_ticks([])
                    if i == 1: # make sure we get good tick marks
                        axs2[i+1,k_lab].xaxis.set_ticks(np.arange(t0_s, t1_s+1, 1))
                    if k_lab == 0:
                        axs2[i+1,k_lab].set_ylabel(stim_name)
                    
                    axs2[0,k_lab].set_title(f"Cluster {k_lab+1} " + 
                                            f"({k_units}/{num_units})")
                    axs2[0,k_lab].plot(t_plot, k_raw_mean, 
                                        color = stim_cols[stim_idx], 
                                        label = stim_name)
                    axs2[0,k_lab].fill_between(t_plot, k_raw_mean - k_raw_sem,
                                                k_raw_mean + k_raw_sem, 
                                                color=stim_cols[stim_idx], 
                                                alpha=0.2)
                    axs2[0,k_lab].axvline(0, color='black', linewidth=1)
                    axs2[0,k_lab].axvline(5, color='black', linewidth=1)
                
                axs2[0,0].legend()
                axs2[0,0].set_ylabel("Norm. FR")



                
            fig2.suptitle(rf"{regions} | $\sigma$ = {sigma}ms | " + 
                          f"K = {num_clusters[k_idx]}" +
                          f" | T = [{t0_ms}, {t1_ms}]ms")
            
            fig2.text(0.5,0.05, "Time (s)", size = 20, ha = 'center')
            plt.savefig(f"{load_dir}/{regions}_SUMMARY_{num_clusters[k_idx]}" +
                        f"-CLUSTERS.{filetype}")
            plt.close('all')
            ##################################################################
            ##################################################################
            
        
        
        #now plot elbow plot
        plt.plot(num_clusters, elbow_arr, color = 'black', marker = '.')
        plt.xlabel("# Clusters")
        plt.ylabel("Error")
        plt.title(rf"{regions} | $\sigma$ = {sigma}ms" + 
                  f" | T = [{t0_ms}, {t1_ms}]ms")
        plt.savefig(f"{load_dir}/{regions}_ELBOW_PLOT.{filetype}")
        plt.close()
            
            



                



if __name__ == "__main__":
    

    
    
    plt.ioff()
    plt.close('all')
    
    ######### MODIFIABLE VARIABLES ############
    stim_types = [1,2]
    good_only = True
    cluster_group = 'good'
    dt = 0.001 #in seconds DO NOT CHANGE THIS
    # for smoothing, add padding to t_edges
    sigma = 20 #in ms <since we always bin using dt 0.001>
    pad = 100 # in units of dt
    region_groups = [['PERI', 'ECT'], ['SSp-bfd'], ['SSs'], ['AUDp'], ['AUDd'], 
                     ['AUDv'], ['TeA']]
    n_pcs = 3 #None gets all PCs
    spd_cut = False
    ###########################################
    
    # manual protocol choice if desired
    manual_protocol_choice = True
    f = 1 # frequency
    n_cycles = 5 # num cycles < which is 0.5*number of deflections >
    tot_stim_win_dur = 5
    
    # DECIDE WHICH FUNCTIONS TO COMPUTE
    funct_to_compute = [1,1,1,1,1]
    
    zpsth = funct_to_compute[0]
    pca = funct_to_compute[1]
    compute_kmeans = funct_to_compute[2]
    plot_kmeans = funct_to_compute[3]
    plot_kmeans_cairo = funct_to_compute[4]
    
    # open stim metadata and use it to group similar protocols, or decide 
    stim_meta = pd.read_excel(
        io=f"{full_dataset_dir}/stimulation_metadata.xlsx")
    
    
    
    #load population data
    pop_data = pd.read_pickle(f"{full_dataset_dir}/combined_data.pkl")
    
    # load neurons' stims and spiking activity
    sp = pop_data.pop_sp
    stims = pop_data.pop_stims
    df = pop_data.df
    
    
    
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
        
    
    #%% loop through protcols and run k-means analysis
    for i, protocol in grouped_stim_meta.iterrows():
        stim_freq = protocol.frequency
        num_cycles = protocol.num_cycles
        n_deflect = 2*num_cycles
        # deflect frequency is double cycle frequency
        deflect_freq = 2*stim_freq
        stim_win_dur = protocol.total_duration
        
        # get two lists from current protocol
        base_durations = protocol.baseline_duration
        mouse_ids = np.array(protocol.mouse_id)
        
        
        # create window based on lowest baseline duration
        min_dur = np.min(base_durations)
        # this keeps things simple and prevents overlap of baseline or trials
        window = [-0.2*min_dur, stim_win_dur + 0.2*min_dur]
        
        
        # get num base bins and end base bins based on t window
        res = split_expt_windows(window, dt, stim_win_dur)
        num_base = res['num_init_base_bins']
        num_end = res['num_end_base_bins']
        
        # create pulse times array
        pulse_times = np.arange(0, n_deflect, 1)/deflect_freq # in seconds
       
        pulse_colors = np.tile(['black', 'dimgray'], n_deflect)
        
        print(f"Analysed mouse_ids: {mouse_ids}")
        # remove mouse metadata that will not be analysed
        protocol_df = df_subset[
            np.isin(df_subset.mouse_id, mouse_ids)].reset_index(drop=True)
        # take out data per mouse to feed into data for separation
        stim_dict = uf.partition_stims(mouse_ids, stims, stim_names)

        # generate bin edges and midpoints
        t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win = \
            uf.get_t_data(window, dt, pad)
            
        # get all units in regions of interest
        group_idxs = np.isin(
            protocol_df.region, np.concatenate(region_groups))
        # limit df to this while resetting index
        protocol_df = protocol_df.loc[
            protocol_df.index[group_idxs],:].copy().reset_index(drop=True)
            
        t0_ms = int(window[0] * 1000)
        t1_ms = int(window[1] * 1000)
        
        
        # define base directory for saving/loading
        base_dir = f"{stim_freq}Hz, {num_cycles} cycles, {stim_win_dur}s/"
        base_dir += f"win_relative = {window}, {cluster_group_nm} units, "
        base_dir += f"{quality}"
        os.makedirs(base_dir, exist_ok=True)
    
      
    
        if zpsth == 1:
            compute_zpsth(pop_data, protocol_df, stim_dict, window, 
                          base_dir, sp, t_edges, t_mids, ntbins, 
                          pad_t_edges, pad_t_mids, pad_t_win, dt, 
                          spd_cut=spd_cut, 
                          good_only=good_only, cluster_group=cluster_group, 
                          sigma=sigma, pad=pad)
        if pca == 1:
            zpsth_pca(window, base_dir, region_groups, sigma, spd_cut)
        
        if compute_kmeans == 1:
            zpsth_kmeans(window, base_dir=base_dir, sigma=sigma, n_pcs=n_pcs, 
                         region_groups=region_groups, spd_cut=spd_cut, 
                         stim_types=stim_types)
        
        if plot_kmeans == 1:
            kmeans_plotter(window, base_dir=base_dir, sigma=sigma, n_pcs=n_pcs, 
                           region_groups=region_groups, t0_ms=t0_ms, 
                           t1_ms=t1_ms, stim_types=stim_types, spd_cut=spd_cut)
            
        if plot_kmeans_cairo == 1:
            # CAIRO FOR ILLUSTRATOR
            import matplotlib.font_manager as fm
            import matplotlib
            plt.rc('font', family='arial')
            plt.rcParams.update({'font.size': 12})
            font = fm.FontProperties(family = 'arial')
            matplotlib.use('cairo') # comment out when testing plots
            
            kmeans_plotter(window, base_dir=base_dir, sigma=sigma, n_pcs=n_pcs, 
                           region_groups=region_groups, t0_ms=t0_ms, 
                           t1_ms=t1_ms, stim_types=stim_types,
                           spd_cut=spd_cut, filetype='pdf')
    
