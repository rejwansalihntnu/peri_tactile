# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 09:25:06 2025

@author: rejwanfs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
import matplotlib.gridspec as gridspec
import matplotlib
import sys
sys.path.append("..")
import svm_functions_label_shuffling as sf

plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 12})
matplotlib.use('cairo')
# figtype = 'png'

#### PLOTTING PARAMETERS ####
plt.rcParams["figure.figsize"] = [19.2 ,  9.83]
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["figure.titlesize"] = 18
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams[''] = 
plt.rcParams.update({
    # 'xtick.labelsize': 12,   # X and Y tick label size
    # 'ytick.labelsize': 12,
    # 'xtick.color': 'blue',   # X and Y tick label color
    # 'ytick.color': 'blue',
    # 'xtick.direction': 'inout', # X and Y tick direction
    # 'ytick.direction': 'inout',
    'xtick.major.size': 8,   # X and Y tick major size
    'ytick.major.size': 8,
    'xtick.major.width': 1.5,  # X and Y tick major width
    'ytick.major.width': 1.5,
    'xtick.minor.size': 4,   # X and Y tick minor size
    'ytick.minor.size': 4,
    'xtick.minor.width': 1,  # X and Y tick minor width
    'ytick.minor.width': 1,
})
# plt.xticks(fontsize = 25)
# plt.yticks(fontsize = 25)
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

# ---------- Parameters (must match training) ----------
dt = 0.05
sigma = 0
# region_groups = [['PERI', 'ECT']]
# region_groups = [['SSp-bfd']]
region_groups = [['AUDp']]
# region_groups  = [['AUDp', 'AUDd', 'AUDv']]            # primary auditory
N = 30
nboot = 100
n_shuff = 1000
grp_cvs = 2
C = 0.0001
win_size_ms = 50
step_ms = 50
z_scored = True
ci_perc = 95
btf = 0.5
stim_cols = {1:'r', 2:'b', 3:'g'}
pulse_times = np.arange(0, 5, 0.5)
pulse_colors = np.tile(['black', 'dimgray'], 10)
zsuffix = "_z_scored" if z_scored else ""
shuff_nm = 'trialwise'


def main():
    
    # ---------- File paths ----------
    base_dir = f"./svm_stim_v_base_{shuff_nm}_shuffling{zsuffix}/" + \
        f"trial_based, dt={dt}s, sigma={sigma}/{region_groups}"
    out_dir = f"./svm_stim_v_base_{shuff_nm}_shuffling{zsuffix}/" + \
        f"trial_based postprocessing, dt={dt}s, sigma={sigma}/{region_groups}"
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"{nboot}-boot_{grp_cvs}-fold_{sigma}ms_{N}_units_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"
    shuff_suffix = f"{nboot}-boot_{n_shuff}-shuff_{grp_cvs}-fold_{sigma}ms_{N}_units_{win_size_ms}ms_win_{step_ms}ms_step_C={C}_btf={btf}"
    
    fname_real  = f"performance_data_{region_groups[0]}_{suffix}.npy"
    fname_shuff = f"shuffled_performance_data_{region_groups[0]}_{shuff_suffix}.npy"
    
    real_path  = os.path.join(base_dir, fname_real)
    shuff_path = os.path.join(base_dir, fname_shuff)
    
    assert os.path.exists(real_path),  f"Missing {real_path}"
    assert os.path.exists(shuff_path), f"Missing {shuff_path}"
    
    decoding_perf       = np.load(real_path)        # (2,2,W,grp_cvs,B)
    decoding_perf_shuff = np.load(shuff_path)       # (2,2,W,S)
    
    _, _, num_windows, grp_cvs_from_file, nboot_from_file = decoding_perf.shape
    assert grp_cvs_from_file == grp_cvs, "grp_cvs mismatch with saved data"
    assert nboot_from_file == nboot,     "nboot mismatch with saved data"
    
    stim_t_mids = np.arange(win_size_ms/2, 5000-(win_size_ms-step_ms)/2, step_ms)/1000
    if int((5 - stim_t_mids[-1]) * 1000) < win_size_ms/2:
        stim_t_mids = stim_t_mids[:-1]
    
    # Convenience: fold-avg per bootstrap → shape (2,2,W,B)
    foldavg_perf = decoding_perf.mean(axis=3)
    # # shuffled fold-avg per bootstrap → shape (2,2,W,S,B)
    
    # ---------- Plot Real vs Shuffled Accuracy ----------
    fig, ax = plt.subplots(1, 3, figsize=(24,6), gridspec_kw={'width_ratios':[1,1,0.6]})
    plt.subplots_adjust(wspace=0.25)
    labels = ['CTRL', 'EXPT']
    sig_counts = np.zeros((2,2))
    total_bins = len(stim_t_mids)
    
    for train_cond in [0,1]:
        # diagonal (within-condition)
        within = foldavg_perf[train_cond, train_cond]        # (W,B)
        mean_acc = within.mean(axis=-1)
        sdev_acc = within.std(axis=-1)
    
        ax[train_cond].plot(stim_t_mids, mean_acc, label=f"{labels[train_cond]}→{labels[train_cond]}",
                            color=stim_cols[train_cond+1])
        ax[train_cond].fill_between(stim_t_mids, mean_acc-sdev_acc, mean_acc+sdev_acc,
                                    color=stim_cols[train_cond+1], alpha=0.2)
    
        # shuffled nulls
        shuff_within = decoding_perf_shuff[train_cond, train_cond]
        mean_shuff = shuff_within.mean(axis=-1)
        deviation = mean_shuff[:, None] - shuff_within
        max_dev = np.max(deviation, axis=0)
        upper_bound = np.percentile(max_dev, ci_perc)
        upper_arr = mean_shuff + upper_bound
        sig_mask_within = mean_acc > upper_arr
        sig_counts[train_cond, train_cond] = 100 * np.sum(sig_mask_within) / total_bins
    
        # significance bars for within-condition (colored)
        trans_within = blended_transform_factory(ax[train_cond].transData, ax[train_cond].transAxes)
        half_step_sec = (step_ms*1.1/1000)/2
        y_frac = 0.97
        for t, sig in enumerate(sig_mask_within):
            if sig:
                ax[train_cond].hlines(y=y_frac, xmin=stim_t_mids[t]-half_step_sec,
                                      xmax=stim_t_mids[t]+half_step_sec,
                                      color=stim_cols[train_cond+1], linewidth=6,
                                      transform=trans_within)
    
        # cross-condition
        test_cond = 1 - train_cond
        cross = foldavg_perf[train_cond, test_cond]
        mean_cross = cross.mean(axis=-1)
        sdev_cross = cross.std(axis=-1)
    
        ax[test_cond].plot(stim_t_mids, mean_cross, color='black',
                           label=f"{labels[train_cond]}→{labels[test_cond]}")
        ax[test_cond].fill_between(stim_t_mids, mean_cross - sdev_cross,
                                   mean_cross + sdev_cross, color='black', alpha=0.2)
    
        # shuffled nulls for cross
        shuff_cross = decoding_perf_shuff[train_cond, test_cond]
        mean_shuff_cross = shuff_cross.mean(axis=-1)
        deviation = mean_shuff_cross[:, None] - shuff_cross
        max_dev = np.max(deviation, axis=0)
        upper_bound = np.percentile(max_dev, ci_perc)
        upper_arr = mean_shuff_cross + upper_bound
        sig_mask_cross = mean_cross > upper_arr
        sig_counts[train_cond, test_cond] = 100 * np.sum(sig_mask_cross) / total_bins
    
        # --- significance bars for cross-conditional decoding (on correct axis) ---
        trans_cross = blended_transform_factory(ax[test_cond].transData, ax[test_cond].transAxes)
        y_frac = 0.93
        for t, sig in enumerate(sig_mask_cross):
            if sig:
                ax[test_cond].hlines(y=y_frac, xmin=stim_t_mids[t]-half_step_sec,
                                     xmax=stim_t_mids[t]+half_step_sec,
                                     color='black', linewidth=6, transform=trans_cross)
    
    
    # ---- NEW: third subplot with summary ----
    bar_labels = ['CTRL→CTRL','CTRL→EXPT','EXPT→CTRL','EXPT→EXPT']
    ax[2].bar(bar_labels, sig_counts.flatten(), color=['r','gray','gray','b'])
    ax[2].set_ylabel('% significant bins')
    plt.setp(ax[2].get_xticklabels(), rotation=45, ha='right')
    
    # formatting
    for i in range(2):
        ax[i].axhline(0.5, color='k', linestyle='dashed')
        ax[i].yaxis.set_major_formatter(FuncFormatter(sf.to_percentage_no_symbol))
        ax[i].set_xlabel("Time (s)")
        for p in range(10):
            ax[i].axvline(pulse_times[p], color=pulse_colors[p],
                          linestyle=(0,(3,3)), linewidth=3)
    ax[0].set_ylabel("Accuracy (%)")
    ax[0].legend()
    ax[1].legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, f"xcond_accuracy_{shuff_suffix}.png"),
                transparent=True)
    plt.close(fig)
    
    
    # -------------------------------------------------------------
    # NEW FIGURE: 4 decoding combos with shuffled max-T bounds
    # -------------------------------------------------------------
    fig2, ax2 = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    
    titles = [
        "CTRL → CTRL",
        "CTRL → EXPT",
        "EXPT → CTRL",
        "EXPT → EXPT"
    ]
    
    for train_cond in (0, 1):
        for test_cond in (0, 1):
    
            row = train_cond
            col = test_cond
            ax_here = ax2[row, col]
    
            # -------------------------
            # Real decoding curve
            # -------------------------
            real_data = foldavg_perf[train_cond, test_cond]   # (W,B)
            mean_real = real_data.mean(axis=-1)               # (W,)
    
            # -------------------------
            # Shuffled max-T null
            # decoding_perf_shuff: (2,2,W,S)
            # -------------------------
            shuff = decoding_perf_shuff[train_cond, test_cond]  # (W,S)
            mean_sh = shuff.mean(axis=-1)                       # (W,)
    
            # Deviation
            deviation = mean_sh[:, None] - shuff                # (W,S)
            max_dev   = np.max(deviation, axis=0)               # (S,)
            min_dev_arr = np.min(deviation, axis=0)
    
            # Max-T upper bound
            upper_bound = np.percentile(max_dev,
                                        ci_perc)# + (100 - ci_perc) / 2)
            lower_bound = np.percentile(min_dev_arr, (100 - ci_perc))# / 2.0)
    
            upper_arr = mean_sh + upper_bound
            lower_arr = mean_sh + lower_bound  # symmetric lower bound
    
            # ------------------------------------
            # Plotting
            # ------------------------------------
            if train_cond == test_cond:
                color_real = stim_cols[train_cond+1]   # CTRL=red, EXPT=blue
            else:
                color_real = "black"
    
            # Shuffled band (max-T interval)
            ax_here.fill_between(stim_t_mids, lower_arr, upper_arr,
                                 color="gray", alpha=0.25,
                                 label="Shuffled max-T interval")
    
            # Shuffled mean
            ax_here.plot(stim_t_mids, mean_sh, color="gray",
                         linestyle="--", linewidth=2,
                         label="Shuffled mean")
    
            # Real decoding mean
            ax_here.plot(stim_t_mids, mean_real, color=color_real,
                         linewidth=2.5, label="Real")
    
            # Reference + formatting
            ax_here.axhline(0.5, linestyle="dashed", color="k")
            for p in range(10):
                ax_here.axvline(pulse_times[p], color=pulse_colors[p],
                                linestyle=(0,(3,3)), linewidth=2)
    
            ax_here.set_title(titles[2*row + col])
            ax_here.yaxis.set_major_formatter(
                FuncFormatter(sf.to_percentage_no_symbol)
            )
    
            if row == 1:
                ax_here.set_xlabel("Time (s)")
            if col == 0:
                ax_here.set_ylabel("Accuracy (%)")
    
            # single legend
            if train_cond == 0 and test_cond == 0:
                ax_here.legend()
    
    fig2.tight_layout()
    plt.savefig(os.path.join(out_dir,
                f"four_combos_with_shuffled_maxT_{shuff_suffix}.png"),
                transparent=True)
    plt.close(fig2)
    
    
    
    # ---------- Difference & generalization gap ----------
    fig = plt.figure(figsize=(15,6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.6, 0.25, 0.25])
    ax_diff = fig.add_subplot(gs[0])
    ax_kde  = fig.add_subplot(gs[1], sharey=ax_diff)
    ax_gap  = fig.add_subplot(gs[2], sharey=ax_diff)
    
    labels = ['CTRL','EXPT']
    
    for train_cond in [0,1]:
        cross  = foldavg_perf[1-train_cond, train_cond]   # (W,B)
        within = foldavg_perf[train_cond, train_cond]     # (W,B)
        diff_arr  = cross - within
    
        mean_diff = diff_arr.mean(axis=-1)
        sdev_diff = diff_arr.std(axis=-1)
        avg_diff  = mean_diff.mean()
    
        ax_diff.plot(stim_t_mids, mean_diff, color=stim_cols[train_cond+1],
                     label=f"{labels[train_cond]} cross - within")
        ax_diff.fill_between(stim_t_mids, mean_diff - sdev_diff, mean_diff + sdev_diff,
                             color=stim_cols[train_cond+1], alpha=0.2)
    
        # surrogate permutation test (paired flips)
        n_surrogates=1000
        surrogate_means=np.zeros(n_surrogates)
        for s in range(n_surrogates):
            flip = (np.random.rand(*cross.shape) < 0.5)
            perm_a = np.where(flip, cross,  within)
            perm_b = np.where(flip, within, cross)
            surrogate_means[s] = np.mean(perm_a - perm_b)
        centered = surrogate_means - surrogate_means.mean()
        count = np.sum(np.abs(centered) >= np.abs(avg_diff - surrogate_means.mean()))
        p_value = (count + 1) / (n_surrogates + 1)
        print(f"P-value for {labels[train_cond]}: {p_value:.4f}")
    
        sns.kdeplot(y=mean_diff, ax=ax_kde, color=stim_cols[train_cond+1], fill=False)
        ax_kde.axhline(avg_diff, linestyle='--', color=stim_cols[train_cond+1], linewidth=2)
    
    # formatting for diff + KDE
    ax_diff.axhline(0, color='k', linewidth=2.5)
    for p in range(10):
        ax_diff.axvline(pulse_times[p], color=pulse_colors[p],
                        linestyle=(0,(5,10)), linewidth=2.5)
    ax_diff.yaxis.set_major_formatter(FuncFormatter(sf.to_percentage_no_symbol))
    ax_diff.set_xlabel("Time (s)")
    ax_diff.set_ylabel(r"$\Delta$ accuracy (%)")
    ax_diff.legend()
    
    for spine in ['top','right','bottom']:
        ax_kde.spines[spine].set_visible(False)
    ax_kde.yaxis.set_major_formatter(FuncFormatter(sf.to_percentage_no_symbol))
    plt.setp(ax_kde.get_yticklabels(), visible=False)
    
    # ---- separate subplot for generalization gap Δ ----
    ctrl_within = foldavg_perf[0,0].mean(axis=-1)
    ctrl_cross  = foldavg_perf[0,1].mean(axis=-1)
    expt_within = foldavg_perf[1,1].mean(axis=-1)
    expt_cross  = foldavg_perf[1,0].mean(axis=-1)
    
    delta_ctrl = ctrl_cross - ctrl_within
    delta_expt = expt_cross - expt_within
    
    ax_gap.plot(stim_t_mids, delta_ctrl, 'r', label='Δ CTRL (A→A+T - A→A)')
    ax_gap.plot(stim_t_mids, delta_expt, 'b', label='Δ EXPT (A+T→A - A+T→A+T)')
    ax_gap.axhline(0, color='k', linestyle='--')
    ax_gap.set_xlabel("Time (s)")
    ax_gap.set_ylabel("Generalization gap Δ")
    ax_gap.legend()
    for p in range(10):
        ax_gap.axvline(pulse_times[p], color=pulse_colors[p],
                       linestyle=(0,(5,10)), linewidth=2)
    
    # Layout and save
    fig.tight_layout(rect=[0,0,1,0.95])
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(os.path.join(out_dir, f"xcond_difference_and_gap_{shuff_suffix}.png"),
                transparent=True)
    plt.close(fig)
    
    print("Done plotting CTRL↔EXPT cross-conditional results with generalization gap subplot.")
    
    # ------------------ TWO-PANEL FIGURE --------------------
    h = 8
    fig, ax = plt.subplots(1, 2, figsize=(2.15*h, h),
                           gridspec_kw={'width_ratios': [9, 2]})
    accuracy_ax, bar_ax = ax
    
    labels = ['A', 'AT']
    within_sig_percent = []  # store % significant bins
    
    for cond in (0, 1):
    
        # ----------------------
        # REAL WITHIN-COND DATA
        # ----------------------
        data = foldavg_perf[cond, cond]     # shape (W, folds)
        mean_acc = data.mean(axis=-1)
        sdev_acc = data.std(axis=-1)
    
        # accuracy trace
        accuracy_ax.plot(stim_t_mids, mean_acc,
                         color=stim_cols[cond+1],
                         label=f"{labels[cond]}→{labels[cond]}")
        accuracy_ax.fill_between(stim_t_mids,
                                 mean_acc - sdev_acc,
                                 mean_acc + sdev_acc,
                                 alpha=0.2,
                                 color=stim_cols[cond+1])
    
        # -------------------------
        # SHUFFLED NULLS FOR CI
        # -------------------------
        null = decoding_perf_shuff[cond, cond]  # (W, shuff)
        mean_null = null.mean(axis=-1)
    
        deviation = mean_null[:, None] - null
        max_dev = np.max(deviation, axis=0)
        upper_bound = np.percentile(max_dev, ci_perc)
        upper_arr = mean_null + upper_bound
    
        sig_mask = mean_acc > upper_arr
        percent_sig = 100 * sig_mask.sum() / len(stim_t_mids)
        within_sig_percent.append(percent_sig)
        print(f"{labels[cond]}: {percent_sig:.2g}%")
    
        # -------------------------
        # SIGNIFICANCE BARS ABOVE
        # -------------------------
        trans = blended_transform_factory(accuracy_ax.transData,
                                          accuracy_ax.transAxes)
        half_width = (step_ms * 1.1 / 1000) / 2
        y_frac = 0.97 if cond == 0 else 0.93   # slight vertical offset
    
        for i, sig in enumerate(sig_mask):
            if sig:
                accuracy_ax.hlines(y=y_frac,
                                   xmin=stim_t_mids[i] - half_width,
                                   xmax=stim_t_mids[i] + half_width,
                                   color=stim_cols[cond+1],
                                   linewidth=5,
                                   transform=trans)
    
    # ----------------------
    # FORMAT ACCURACY PANEL
    # ----------------------
    accuracy_ax.axhline(0.5, linestyle='--', color='k')
    accuracy_ax.set_xlabel("Time (s)")
    accuracy_ax.set_ylabel("Accuracy (%)")
    accuracy_ax.yaxis.set_major_formatter(
        FuncFormatter(sf.to_percentage_no_symbol)
    )
    for p in range(10):
        accuracy_ax.axvline(pulse_times[p],
                            linestyle=(0,(3,3)),
                            color=pulse_colors[p],
                            linewidth=2)
    
    # ----------------------
    # BAR CHART OF % SIG BINS
    # ----------------------
    bar_ax.bar(labels, within_sig_percent,
               color=[stim_cols[1], stim_cols[2]], alpha=0.85)
    bar_ax.set_ylabel("Significant bins (%)")
    bar_ax.set_ylim(0, max(within_sig_percent)*1.1 + 1)
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "within_condition_significance.png"),
                transparent=True)
    plt.savefig(os.path.join(out_dir, "within_condition_significance.svg"),
                transparent=True)
    plt.close(fig)
    
if __name__ == "__main__":
    main()

