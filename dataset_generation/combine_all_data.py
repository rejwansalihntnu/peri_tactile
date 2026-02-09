# -*- coding: utf-8 -*-
"""
Draft of preprocessing to get entire dataset in one package
"""
import os
import sys
sys.path.append("..")
import numpy as np
import scipy
import pickle
import pandas as pd

import dataset_class_module.data_struct.init_data as ncd


def trial_cv(stim_firsts, unit_sp):
    """
    NB: this code makes sure that all trials considered are at least 
    20 seconds long <otherwise CV calculation becomes very noisy for short
    trial lengths>. Modify based on your own experimental protocol

    Parameters
    ----------
    stim_firsts : TYPE
        DESCRIPTION.
    unit_sp : TYPE
        DESCRIPTION.

    Returns
    -------
    CV : TYPE
        DESCRIPTION.

    """
    
    avg_duration = np.mean(np.diff(stim_firsts))
    
    # if trials are short, multiply the average duration so its at least 20s
    if avg_duration < 14:
        skip = np.ceil(20 / avg_duration).astype(int)
        avg_duration *= skip
    
        CV_tbins = stim_firsts[::skip]
        
    else:
        CV_tbins = np.append(stim_firsts,stim_firsts[-1]+np.mean(avg_duration))
    
    CV_counts, _ = np.histogram(unit_sp, CV_tbins, density = False)
    
    rates = CV_counts / np.diff(CV_tbins)
    CV = np.std(rates) / np.mean(rates)
    
    return CV


def create_stim_df(stim_data, num_pulses):
    
    stim_df_cols = ['pulse_t','type_','exp_block','up','trial_idx']
    num_stims = len(stim_data) // num_pulses
    stim_idx = np.repeat(range(num_stims), repeats=num_pulses)
    final_arr = np.concatenate([stim_data,stim_idx[:,None]], axis=1)
    stim_df = pd.DataFrame(data=final_arr, columns=stim_df_cols)
    
    return stim_df



#############################################################################


if __name__ == "__main__":
    
    # global variables
    unit_class_dict = {0: 'noise', 1: 'MUA', 2: 'good'}
    probe_names = ['BAR', 'PER']
    
    raw_data_dir = r"Z:/Neuropixel_Data/Before KS4"
    
    save_dir = f"{raw_data_dir}/full_dataset"
    os.makedirs(save_dir, exist_ok=True)
    
    
    
    #load instance of pop data class
    pop_data = ncd.Pop_data()
    ###################################################
    
    # define everything that will fill the combined dataset
    df_list = []
    pop_stims = {}
    pop_spd = {}
    pop_sp_dict = {}
    
    # load metadata file to extract stimulation parameters
    stim_meta = pd.read_excel(io=f"{save_dir}/stimulation_metadata.xlsx")
    
    for _, row in stim_meta.iterrows():
        # extract relevant data from mouse metadata
        mouse_id = row.mouse_id
        stim_freq = row.frequency
        num_cycles = row.num_cycles
        stim_dur = row.total_duration
        base_dur = row.baseline_duration
        
        # convert durations to milliseconds
        stim_dur *= 1000
        base_dur *= 1000
        
        # double num_cycles to account for up and down deflections
        num_pulses = num_cycles * 2
        
        # convert mouse_id and num_pulses to int
        mouse_id = int(mouse_id)
        num_pulses = int(num_pulses)
        
        mouse_dir = f"{raw_data_dir}/{mouse_id}"
        pop_sp_dict[mouse_id] = {}
        
        # load stims and spd if possible
        stim_file = f"{mouse_dir}/{mouse_id}_StimPulses_All.mat"
        stim_data = scipy.io.loadmat(stim_file, simplify_cells=True)
        stim_key = list(stim_data.keys())[-1]
        stim_data = stim_data[stim_key]
        mouse_stim_df = create_stim_df(stim_data, num_pulses)
        pop_stims[mouse_id] = mouse_stim_df
        
        # get first pulse times in each trial - will use it later to get CV
        stim_firsts = mouse_stim_df['pulse_t'].values[::num_pulses]
        
        
        try:
            spd_file = f"{mouse_dir}/{mouse_id}_Speed_Data.mat"
            spd_data = scipy.io.loadmat(spd_file, simplify_cells=True)
        except:
            pop_spd[mouse_id] = None
        else:
            pop_spd[mouse_id] = spd_data['Speed_Data']
        
        
        
        for probe_name in probe_names:
            
            probe_dir = f"{mouse_dir}/{probe_name}"
            
            print(f"Data: {mouse_id} | {probe_name} probe")
            
            # first check if it exists
            if os.path.exists(probe_dir) == False:
                print(f"{mouse_id}_{probe_name} does not exist!")
                continue
            
            fname_prefix = f"{probe_dir}/{mouse_id}_{probe_name}"
            # region information excel sheet
            qc_fname = f"{fname_prefix}_metrics_test.xlsx"
            df_quality = pd.read_excel(io=qc_fname)
            df_quality.drop(df_quality.columns[0], axis=1, inplace=True)  
            df_quality.drop(
                ['epoch_name_quality_metrics', 'epoch_name_waveform_metrics'], 
                axis=1, inplace=True)  # Drops remaining columns,
            
            df_quality.insert(0, 'mouse_id', mouse_id)
            df_quality.insert(2, 'probe', probe_name)
            df_quality.insert(3, 'region', '')
            df_quality.insert(4, 'subregion', '')
            df_quality.rename(columns={'cluster_id': 'cluster'}, inplace=True)
            df_quality.insert(5, 'class', '')
            df_quality.insert(6, 'cluster_group', '')
            df_quality.insert(7, 'trial_cv', np.nan)
            df_quality.insert(np.shape(df_quality)[1], 'good_bool', None)
            
            
            # load information about region
            reg_fname = f"{fname_prefix}_Probe_Location.xlsx"
            df_region = pd.read_excel(io=reg_fname)
            
            # load mat file with all spiking information
            spikes_fname = f"{fname_prefix}_Spikes.mat"
            sp_data = scipy.io.loadmat(spikes_fname, simplify_cells=True)
            sp_key = list(sp_data.keys())[-1]
            sp_data = sp_data[sp_key]
            sp = sp_data['st']
            sp_unit_ids = sp_data['clu']
            unit_ids = sp_data['cids']
            clu_grps = sp_data['cgs']
            
            # split units in terms of cluster groups, removing noise units
            non_noise_idxs = np.nonzero(clu_grps)
            sub_unit_ids = unit_ids[non_noise_idxs]
            non_noise_grps = clu_grps[non_noise_idxs]
            
            # remove noise units from dataframe too
            df_noise_idxs = np.where(
                ~np.isin(df_quality.cluster, sub_unit_ids))[0]
            df_quality.drop(labels=df_noise_idxs, axis=0, inplace=True)
            
            ######## rename AUD to AUDv/d/p and correct spelling of TeA #######
            # Filter for 'AUD' in region
            mask = df_quality['region'] == 'AUD'
            
            # Extract first character from 'subregion'
            first_chars = df_quality.loc[mask, 'subregion'].str[0]
            
            # Append first character to 'region'
            df_quality.loc[mask, 'region'] += first_chars
            
            # Remove first character from 'subregion'
            df_quality.loc[mask, 'subregion'] = \
                df_quality.loc[mask, 'subregion'].str[1:]
            
            # --- Step 2: Rename 'Tea' and 'TEa' to 'TeA' in 'region' ---
            df_quality['region'] = \
                df_quality['region'].replace(['Tea', 'TEa'], 'TeA')
            ##################################################################
            
            # add cluster groups to dataframe
            for grp in range(1,3):
                grp_units = sub_unit_ids[non_noise_grps == grp]
                df_locs = np.where(np.isin(df_quality.cluster, grp_units))[0]
                # you can change this so the df_locs uses df to convert to 
                #df-based index, then use loc with those indices and "cluster-
                #"_group" instead of a number, which is not flexible to changes
                assert df_quality.columns[6] == 'cluster_group'
                df_quality.iloc[df_locs, 6] = unit_class_dict[grp]
            
            # assign each cluster a region
            all_elect_ids = df_quality.peak_channel
            for _, region_info in df_region.iterrows():
                within_unit_ids = ((all_elect_ids >= region_info.min_id) & 
                                    (all_elect_ids <= region_info.max_id))
                df_quality.loc[within_unit_ids, 'region'] = region_info.region
                df_quality.loc[
                    within_unit_ids, 'subregion'] = region_info.subregion
            
            # put non-classified units as 'none' for region
            df_quality.loc[
                df_quality.region == '', ['region', 'subregion']] = 'none'
                
                
            # define the spiking class of each unit
            fs_units = df_quality.duration < 0.4
            rs_units = df_quality.duration >= 0.4
            df_quality.loc[fs_units, 'class'] = 'FS'
            df_quality.loc[rs_units, 'class'] = 'RS'
            
            # mouse_reg_sp_dict = sp_dict[probe_type][f'{mouse_id}'] = {}
            sp_dict = {reg: {} for reg in df_quality.region.unique()}
            for unit_id in sub_unit_ids:
                filter_mask = np.where(sp_unit_ids == unit_id)[0]
                
                if len(filter_mask) == 0:
                    print(f"unit {unit_id} not found/has no spikes!")
                    continue
                else:
                    # get electrode id of this unit
                    df_unit = df_quality[df_quality.cluster == unit_id]
                    if len(df_unit) == 0:
                        pass
                    df_idx = df_unit.index
                    unit_reg = df_unit.region.values[0]
                    unit_sp = sp[filter_mask]
                    sp_dict[unit_reg][str(unit_id)] = unit_sp
                    unit_cv = trial_cv(stim_firsts, unit_sp)
                    df_quality.loc[df_idx, 'trial_cv'] = unit_cv
                    
                    
                    # use CV, FR, and PR to calculate "good_bool" part
                    unit_pr = df_unit.presence_ratio.values[0]       
                    if (unit_cv <= 1) & (unit_pr >= 0.9) & \
                        (len(unit_sp) >= 500):
                            df_quality.loc[df_idx, 'good_bool'] = True
                    else:
                        df_quality.loc[df_idx, 'good_bool'] = False
            
            # add this spikes dictionary to the population dictionary
            pop_sp_dict[mouse_id][probe_name] = sp_dict
            
            # add quality metrics to list
            df_list.append(df_quality)
    
    # concatenate all dataframes
    concat_pop_df = pd.concat(df_list)
    concat_pop_df.reset_index(inplace=True, drop=True)
    
    # now assign everything to the pop_data class object
    pop_data.pop_sp = pop_sp_dict
    pop_data.pop_stims = pop_stims
    pop_data.df = concat_pop_df
    pop_data.spd = pop_spd
    # save this
    pickle.dump(pop_data, open(f"{save_dir}/combined_data.pkl", 'wb'))
    
    # also save them individually
    # sp dict
    pickle.dump(pop_sp_dict, open(f"{save_dir}/pop_spikes_dict.pkl", "wb"))
    
    # stim data
    pickle.dump(pop_stims, open(f"{save_dir}/pop_stims_dict.pkl", 'wb'))
    
    # quality metrics dataframe
    concat_pop_df.to_pickle(f"{save_dir}/pop_metadata.pkl")
    # and to excel file
    concat_pop_df.to_excel(f"{save_dir}/pop_metadata.xlsx")
    
    # speed data
    pickle.dump(pop_spd, open(f"{save_dir}/pop_speed_dict.pkl", "wb"))