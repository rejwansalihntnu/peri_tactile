# -*- coding: utf-8 -*-

import os
from config import full_dataset_dir
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_mean_resp(trial_sp, t_edges, sigma):
    num_trials = len(trial_sp)
    concat_sp = np.concatenate(trial_sp)
    psth, _ = np.histogram(concat_sp, bins=t_edges, density=False)
    psth = psth / ((t_edges[1] - t_edges[0])*num_trials)
    sm_psth = gaussian_filter1d(psth, sigma=sigma, mode='reflect')
    return sm_psth


def get_trial_sp(unit_sp, trial_t, t0, t1):
    start_idx = np.searchsorted(unit_sp, trial_t + t0)
    end_idx = np.searchsorted(unit_sp, trial_t + t1)
    
    num_trials = len(trial_t)
    
    aligned_trial_sp = [unit_sp[start_idx[i]:end_idx[i]] - trial_t[i] 
                        for i in range(num_trials)]
    
    return aligned_trial_sp

def get_t_data(t_win, dt, pad, in_ms=False):
    
    t_edges = dt * np.arange(t_win[0]/dt, t_win[1]/dt + 1, 1)
    t_mids = 0.5 * (t_edges[1:] + t_edges[:-1])
    ntbins = len(t_mids)
    pad_t_edges = dt * np.arange(t_win[0]/dt - pad, t_win[1]/dt + 1 + pad, 1)
    pad_t_mids = 0.5 * (pad_t_edges[1:] + pad_t_edges[:-1])
    pad_t_win = np.array([pad_t_edges[0], pad_t_edges[-1]])
    
    if in_ms == True:
        t_edges *= 1000
        t_mids *= 1000
        pad_t_edges *= 1000
        pad_t_mids *= 1000
        pad_t_win *= 1000
    
    return t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win


def shade_running_epochs(ax, running_epochs_per_trial, trial_height=1,
                         color='plum', alpha=0.2):
    """
    Shade running epochs for each trial in the raster.

    running_epochs_per_trial: list of list of (start, stop) relative times
    trial_height: height in y for each trial
    """
    for trial_idx, trial_epochs in enumerate(running_epochs_per_trial):
        for start, stop in trial_epochs:
            ax.axvspan(start, stop,
                       ymin=(trial_idx) / (len(running_epochs_per_trial)),
                       ymax=(trial_idx + trial_height) / (len(running_epochs_per_trial)),
                       facecolor=color, alpha=alpha, zorder=0)   
            
def updated_single_unit_traces(pop_data, hq_units, cluster_group,
                               stim_types=[1, 2],
                               t_win_frac=[-0.1, 0.9],
                               t_win=None,
                               dt=0.001, pad=100, sigma=10, regions=None):
    """
    Split plot: LEFT=control, RIGHT=experimental.
    For mouse 108025, control period trials with type_=4 (unknown identity)
    are plotted in black (both raster + smoothed PSTH).
    """
    stim_type_cols = {1: 'red', 2: 'blue', 3: 'green', 4: 'black'}

    def simplify_region(region):
        if region in ('PERI', 'ECT'):
            return 'PER'
        if isinstance(region, str) and region.startswith('AUD'):
            return 'AUD'
        return region

    sp, stims, df = pop_data.pop_sp, pop_data.pop_stims, pop_data.df
    stim_meta = pd.read_excel(f"{full_dataset_dir}/stimulation_metadata.xlsx")

    df_valid = df[(df.good_bool == hq_units) & (df.cluster_group == cluster_group)]

    if isinstance(regions, dict):
        allowed = set(regions.keys())
        df_valid = df_valid[df_valid.region.isin(allowed)].copy()

    mouse_ids = df_valid.mouse_id.unique()

    def map_running_bounds_to_trials(running_bounds, trial_starts_ordered, t_win):
        if len(trial_starts_ordered) == 0 or running_bounds is None or len(running_bounds) == 0:
            return [[] for _ in range(len(trial_starts_ordered))]
        t0, t1 = t_win
        trial_starts = np.asarray(trial_starts_ordered, float)
        windows = np.column_stack([trial_starts + t0, trial_starts + t1])
        epochs = [[] for _ in range(len(trial_starts))]
        for run_start, run_stop in np.asarray(running_bounds, float):
            if run_stop <= windows[0, 0] or run_start >= windows[-1, 1]:
                continue
            for i, (ws, we) in enumerate(windows):
                if run_stop <= ws or run_start >= we:
                    continue
                seg_s, seg_e = max(run_start, ws), min(run_stop, we)
                if seg_e > seg_s:
                    epochs[i].append((seg_s - trial_starts[i], seg_e - trial_starts[i]))
        return epochs

    for mouse_id in mouse_ids:
        # if mouse_id != 108025:
        #     continue
        mouse_df = df_valid[df_valid.mouse_id == mouse_id]
        rb_path = f"{full_dataset_dir}/speed_classification/{mouse_id}_running_bounds.npy"
        mouse_running_bounds = np.load(rb_path) if os.path.exists(rb_path) else np.empty((0, 2))

        mouse_stim_meta = stim_meta.loc[(stim_meta.mouse_id == mouse_id) & (stim_meta.frequency == 1)]
        if mouse_stim_meta.empty:
            continue

        _, frequency, n_cycles, stim_dur, base_dur = mouse_stim_meta.iloc[0, :5]
        tot_trial_dur = stim_dur + base_dur

        if t_win_frac is not None:
            t_win_mode = f"t_win_fractions={t_win_frac}"
            t_win_eff = [tot_trial_dur * t_win_frac[0], tot_trial_dur * t_win_frac[1]]
        elif t_win is not None:
            t_win_mode = f"t_win={t_win}"
            t_win_eff = t_win
        else:
            raise ValueError("Either t_win_frac or t_win must be specified.")

        t_edges, t_mids, ntbins, pad_t_edges, pad_t_mids, pad_t_win = get_t_data(t_win_eff, dt, pad)
        n_cycles = int(n_cycles)
        trial_n_movements = int(n_cycles * 2)
        pulse_colors = np.tile(['black', 'dimgray'], n_cycles)

        mouse_stims = stims[mouse_id]

        # Handle exception for mouse 108025 (control trials with unknown type_=4)
        unknown_control = (mouse_id == 108025)

        # organize blocks
        block_trials = {}
        for block in [0, 1]:
            if block == 0 and unknown_control:
                stim_types_this_block = stim_types + [4]
            else:
                stim_types_this_block = stim_types

            bdf = mouse_stims[mouse_stims.exp_block == block]
            bdf = bdf[bdf.type_.isin(stim_types_this_block)]
            if len(bdf) == 0:
                block_trials[block] = None
                continue

            stim_times = bdf.iloc[:, 0].values
            usable = (len(stim_times) // trial_n_movements) * trial_n_movements
            stim_times, bdf = stim_times[:usable], bdf.iloc[:usable]
            stacked = stim_times.reshape((-1, trial_n_movements))
            aligned = stacked - stacked[:, 0][:, None]
            mean_stim_t = np.mean(aligned, axis=0)
            firsts = stacked[:, 0]
            trial_types = bdf.type_.iloc[::trial_n_movements].values
            by_type_starts = {t: firsts[trial_types == t] for t in stim_types_this_block}

            block_trials[block] = dict(mean_stim_t=mean_stim_t,
                                       trial_starts=firsts,
                                       type_split=by_type_starts)

        def count_trials(info):
            return sum(len(v) for v in info['type_split'].values()) if info else 0

        n_ctrl, n_exp = count_trials(block_trials.get(0)), count_trials(block_trials.get(1))
        ll_ctrl, ll_exp = (1 if n_ctrl < 200 else 3), (1 if n_exp < 200 else 3)

        for _, row in mouse_df.iterrows():
            _, unit_id, probe, region_orig, subregion, rsfs = row.iloc[:6]
            CV, FR, PR = row.iloc[7:10]
            duration = row.duration

            region_for_sp = region_orig
            try:
                unit_sp = sp[mouse_id][probe][region_for_sp][str(unit_id)]
            except KeyError:
                try:
                    unit_sp = sp[mouse_id][probe][region_for_sp][int(unit_id)]
                except KeyError:
                    print(f"⚠️ Skipping: mouse {mouse_id}, probe {probe}, region {region_for_sp}, unit {unit_id} not found.")
                    continue

            side_data = {}
            for block in [0, 1]:
                info = block_trials.get(block)
                if info is None:
                    side_data[block] = dict(trial_sp=[], trial_cols=[], trial_starts=[], mean_stim_t=None)
                    continue
                trial_sp, trial_cols, starts_ordered = [], [], []
                for t, starts in info['type_split'].items():
                    if starts.size == 0:
                        continue
                    t_sp = get_trial_sp(unit_sp, starts, pad_t_win[0], pad_t_win[1])
                    trial_sp.extend(t_sp)
                    trial_cols.extend([stim_type_cols[t]] * len(starts))
                    starts_ordered.extend(starts)
                side_data[block] = dict(trial_sp=trial_sp, trial_cols=trial_cols,
                                        trial_starts=np.array(starts_ordered),
                                        mean_stim_t=info['mean_stim_t'])

            running_ctrl = map_running_bounds_to_trials(mouse_running_bounds, side_data[0]['trial_starts'], t_win_eff)
            running_exp = map_running_bounds_to_trials(mouse_running_bounds, side_data[1]['trial_starts'], t_win_eff)

            # ---- layout ----
            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(2, 2, height_ratios=[9, 3], width_ratios=[1, 1], hspace=0.05, wspace=0.075)
            axr_c = fig.add_subplot(gs[0, 0])
            axp_c = fig.add_subplot(gs[1, 0], sharex=axr_c)
            axr_e = fig.add_subplot(gs[0, 1])
            axp_e = fig.add_subplot(gs[1, 1], sharex=axr_e)

            def draw_side(axr, axp, side_key, running_epochs, ll, title_suffix):
                info = block_trials.get(side_key)
                if info is None or len(side_data[side_key]['trial_sp']) == 0:
                    axr.text(0.5, 0.5, f"No {title_suffix.lower()} trials found",
                             ha='center', va='center', transform=axr.transAxes,
                             fontsize=14, color='gray')
                    axr.set_axis_off()
                    axp.set_axis_off()
                    return

                sdat = side_data[side_key]
                trial_sp, trial_cols, mean_stim_t = sdat['trial_sp'], sdat['trial_cols'], sdat['mean_stim_t']

                # running shading
                for i, ep in enumerate(running_epochs):
                    for s, e in ep:
                        if e <= s:
                            continue
                        axr.add_patch(Rectangle((s, i - 0.5), e - s, 1,
                                                facecolor='plum', alpha=0.2, edgecolor='none', zorder=0))

                axr.eventplot(trial_sp, colors=trial_cols, linewidths=1, linelengths=ll)
                axr.vlines(mean_stim_t, -0.5, len(trial_sp),
                           colors=pulse_colors, linewidth=2, linestyles=(0, (1, 1)))

                axr.set_xlim([t_mids[0], t_mids[-1]])
                axr.set_ylim([-0.5, max(len(trial_sp) - 0.5, 0.5)])
                for s in ['top', 'right', 'bottom']:
                    axr.spines[s].set_visible(False)
                axr.tick_params(axis='both', which='both', length=0)
                axr.set_title(title_suffix)

                # PSTH
                traces = []
                for t, starts_t in info['type_split'].items():
                    if starts_t.size == 0:
                        continue
                    tsp = get_trial_sp(unit_sp, starts_t, pad_t_win[0], pad_t_win[1])
                    sm = get_mean_resp(tsp, pad_t_edges, sigma)[pad:-pad]
                    traces.append((stim_type_cols[t], sm))
                if traces:
                    ymin, ymax = min(np.min(tr) for _, tr in traces), max(np.max(tr) for _, tr in traces)
                    if ymax <= ymin:
                        ymin -= 0.5; ymax += 0.5
                    for c, tr in traces:
                        axp.plot(t_mids, tr, color=c, linewidth=1, alpha=1)
                    axp.vlines(mean_stim_t, ymin, ymax, colors=pulse_colors,
                               linewidth=2, linestyles=(0, (1, 1)))
                    axp.set_ylim([ymin, ymax])
                for s in ['top', 'right']:
                    axp.spines[s].set_visible(False)
                axp.tick_params(axis='both', which='both', length=0)
                axp.set_xlabel('Time (s)')
                axp.set_ylabel('Firing rate (Hz)')
                """remove this line below if plots still look strange """
                axp.set_xlim([t_mids[0], t_mids[-1]])

            draw_side(axr_c, axp_c, 0, running_ctrl, ll_ctrl, "Control")
            draw_side(axr_e, axp_e, 1, running_exp, ll_exp, "Experimental")
            axr_c.set_ylabel('Trials')

            region_simple = regions.get(region_orig, simplify_region(region_orig)) if isinstance(regions, dict) \
                            else simplify_region(region_orig)

            plt.suptitle(
                f"ID:{mouse_id} | cluster {unit_id} | {region_simple} | "
                f"FR:{FR:.2f}Hz | CV:{CV:.2f} | PR:{PR:.2f} | dur:{duration:.2f}ms"
            )

            save_root = f"./Single units, HQ units/{t_win_mode}"
            save_dir = f"{save_root}/{region_simple}"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{mouse_id}_{unit_id}.png")
            plt.close('all')
