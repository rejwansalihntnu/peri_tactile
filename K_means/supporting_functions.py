# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:05:13 2023

@author: rejwanfs
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

# sillhouette analysis
def silhouette_summary(fr_mat, k_means, stim_names):
    
    # transpose so fr mat is Nsamples x Nfeatures
    fr_mat = fr_mat.T
    
    cluster_labels = k_means.labels_
    n_clusters = len(np.unique(cluster_labels))
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_ylim([0, len(fr_mat) + (n_clusters + 1) * 10])
    
    # since we stack pca of both types, pc1 ipsi (or whatever type) will be
    #half way in the index
    ipsi_pc1 = np.shape(fr_mat)[1] // 2

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(fr_mat, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(fr_mat, cluster_labels)
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        fr_mat[:, 0], fr_mat[:, ipsi_pc1], marker=".", s=30, lw=0, alpha=0.7, 
        c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = k_means.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, ipsi_pc1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[ipsi_pc1], marker="$%d$" % i, alpha=1, s=50, 
                    edgecolor="k")

    ax2.set_title("Euclidean distances in PC space")
    ax2.set_xlabel(f"PC 1 ({stim_names[0]})")
    ax2.set_ylabel(f"PC 1 ({stim_names[1]})")
    
    return fig


#three functions for getting smoothed psths
def align_sp_to_stim(unit_sp, stim_t, t_win, num_stims):
    
    # searchsorted and combine all spike times to single array then hist that
    start_idxs = np.searchsorted(unit_sp, stim_t + t_win[0], 'left')
    end_idxs = np.searchsorted(unit_sp, stim_t + t_win[1], 'right')
    aligned_sp = [unit_sp[start_idxs[i]:end_idxs[i]] - stim_t[i] 
                  for i in range(num_stims)]
    # remove this line if you want to separate by trial
    aligned_sp = np.concatenate(aligned_sp)
    
    return aligned_sp

def psth(aligned_sp, t_edges, num_stims, dt=0.001):
    
    bin_counts, _ = np.histogram(aligned_sp, t_edges, density = False) 
    psth = bin_counts / (num_stims * dt)
    
    return psth

def get_smoothed_psth(pop_sp, pop_df, mouse_stim_dict, type_name, 
                      num_reg_units, sigma, pad, t_edges, t_mids, ntbins, 
                      pad_t_edges, pad_t_mids, pad_t_win):
    # note that the pop_df has reset_index() applied, so old index will be 
    #first column! 
    """NO LONGER TRUE WE'RE DROPPING INDEX COLUMN"""
    
    pop_reg_sp = np.empty((num_reg_units, ntbins))
    pop_mean_base = np.empty(num_reg_units)
    for i, row in pop_df.iterrows():
        # get neuron's metadata (see above for why no longer indexing from 1)
        mouse_id, cluster, probe, region = row.iloc[:4]
        # print(row)
        
        # extract neuron's spikes and associated stims
        unit_sp = pop_sp[mouse_id][probe][region][str(cluster)]
        iter_stims = mouse_stim_dict[mouse_id][type_name]
        stim_t = iter_stims[:,0]
        num_stims = len(stim_t)
        
        # bin spikes <with extra padding forsmoothing>
        aligned_unit_sp = align_sp_to_stim(unit_sp, stim_t, pad_t_win, 
                                           num_stims)
        binned_unit_sp = psth(aligned_unit_sp, pad_t_edges, num_stims)
        # repeat for baseline calculations
        """modified from 5s-->3s as some have short baselines"""
        aligned_base_sp = align_sp_to_stim(unit_sp, stim_t-3, [0,3], num_stims)
        pop_mean_base[i] = psth(aligned_base_sp, [0,3], num_stims, dt=3)
        
        # smooth if desired
        if sigma != 0:
            smoothed_sp = gaussian_filter1d(binned_unit_sp, sigma=sigma)
        else:
            smoothed_sp = binned_unit_sp
            
        # in any case, drop the padding
        smoothed_sp = smoothed_sp[pad:-pad]
        
        # double check that we did the padding and slicing off padding correct
        assert np.array_equal(pad_t_edges[pad:-pad], t_edges)
        
        pop_reg_sp[i] = smoothed_sp
        
    return pop_reg_sp, pop_mean_base




    
    
if __name__ == "__main__":
    
    pass