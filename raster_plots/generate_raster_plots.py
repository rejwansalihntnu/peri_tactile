# -*- coding: utf-8 -*-

import os
from config import full_dataset_dir
import supporting_functions_speed_shading as sfss
import pandas as pd, matplotlib.pyplot as plt, matplotlib, matplotlib.font_manager as fm

#### PLOTTING PARAMETERS ####
font = fm.FontProperties(family='arial')
matplotlib.use('cairo')
matplotlib.style.use('seaborn-v0_8-poster')
plt.rcParams["figure.figsize"] = [19.2, 9.83]
plt.rcParams.update({
    'axes.titlesize': 18, 'figure.titlesize': 18, 'axes.linewidth': 1,
    'xtick.major.size': 12, 'ytick.major.size': 12,
    'xtick.major.width': 3, 'ytick.major.width': 3
})
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rcParams["axes.labelsize"] = 18
plt.rc('legend', fontsize=20)
plt.rcParams["font.size"] = 20
##############################

stim_type_cols = {1: 'red', 2:'blue', 3:'green'}

######### MODIFIABLE VARIABLES ############
stim_types = [1,2] # label for ipsilateral and contralateral trials, respectively
good_only = True # only pick neurons passing the three quality metrics of 
#trial-based CV < 1, Presence ratio > 0.9, no. spikes > 500.
cluster_group = 'good' # 'good' means single units, MUA is MUA
manual_protocol_choice = True
f = 1              # <—— Only 1 Hz
n_cycles = 5
tot_stim_win_dur = 5.0
regions = [['PERI', 'ECT'], ['SSp-bfd'], ['SSs'], ['AUDp'], ['AUDd'], ['AUDv'], ['TeA']]
##############################################

# Load an excel file with stimulation paradigm metadata for each mouse so we 
#can filter by the type of experiment to focus on.
stim_meta = pd.read_excel(f"{full_dataset_dir}/stimulation_metadata.xlsx")

# load population data
if os.getcwd()[0] == 'C':
    pop_data = pd.read_pickle("./../combined_data.pkl")
else:
    pop_data = pd.read_pickle(f"{full_dataset_dir}/combined_data.pkl")

# group same protocols
grouped_stim_meta = stim_meta.groupby(['frequency', 'num_cycles', 'total_duration']).agg({
    'mouse_id': list,
    'baseline_duration': list
}).reset_index()

# select chosen protocol/paradigm
grouped_stim_meta = grouped_stim_meta[
    (grouped_stim_meta.frequency == f) &
    (grouped_stim_meta.num_cycles == n_cycles) &
    (grouped_stim_meta.total_duration == tot_stim_win_dur)
]
mouse_ids = grouped_stim_meta.mouse_id.values[0]

# extract population data
df = pop_data.df

# filter cluster group and quality
df_valid = df[(df.good_bool == good_only) & (df.cluster_group == cluster_group)]

# ---- REGION FILTER + MERGE ----
region_map = {
    'PERI': 'PER', 'ECT': 'PER',
    'AUDp': 'AUD', 'AUDd': 'AUD', 'AUDv': 'AUD',
    'SSp-bfd': 'SSp-bfd', 'SSs': 'SSs', 'TeA': 'TeA'
}
df_valid = df_valid[df_valid.region.isin(region_map.keys())].copy()
df_valid['region_grouped'] = df_valid['region'].map(region_map)

plt.close('all')

# run the raster plot + mean activity trace figures for selected regions
"""
t_win is time relative to trial onset to consider <in seconds>
dt is bin width in seconds
pad is padding for smoothing in ms
sigma is width of gaussian kernel
t_win_frac is optional, allowing t_win to be a fraction of the total trial 
length, e.g. [-0.5, 0.5] would be [-5, 5]s for trials of 10s and 
[-7.5, 7.5]s for trials of 15s. Mainly there to keep plotting similar across
protocols"""
sfss.updated_single_unit_traces(
    pop_data, good_only, cluster_group,
    stim_types=[1,2],
    t_win_frac=None, t_win = [-2.5, 7.5],
    dt=0.001, pad=100, sigma=10,
    regions=region_map
)
