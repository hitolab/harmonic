import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator

# Add Harmonicity root to path so Lib can be imported as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Lib.harmonicity as hl
import Lib.plot_utils as pu

# ============================================================
# 0) Command Line Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Experimental Data Noise Analysis")
parser.add_argument('-f', '--force', action='store_true', help="Force recalculation (bypass caches)")
args = parser.parse_args()
force_recalc = args.force

# ============================================================
# 1) Configuration
# ============================================================
dataset_info = [
    {
        'name': 'psbA1:luxAB',
        'path': '../Data/Cyano/Cyano_Kondo.csv',
        'title': 'psbA1:luxAB (Cyanobacteria)',
        'ylim_raw': [-0.4, 0.4]
    },
    {
        'name': 'SCN',
        'path': '../Data/MouseSCN/SCN.csv',
        'title': 'SCN (Mouse)',
        'ylim_raw': [-0.5, 0.5]
    }
]

sigma_list = ['raw', 0, 0.05, 0.1, 0.15, 0.2]
css_max = 0.2
css_ticks = np.arange(0, 0.21, 0.05)
points_per_cycle = 48
dt_norm = 1.0 / points_per_cycle

script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 2) Loop Over Datasets
# ============================================================
for d_info in dataset_info:
    base_name = d_info['name']
    title_prefix = d_info['title']
    ylim_raw = d_info['ylim_raw']

    print(f"\n--- Processing: {base_name} ---")
    data_path = os.path.join(script_dir, d_info['path'])
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found: {data_path}")
        continue

    # Load and Detrend (data is already resampled to 48 pts/cycle; column 0 is sample index)
    df = pd.read_csv(data_path)
    t_raw = df.iloc[:, 0].values
    x_raw = df.iloc[:, 1].values
    valid = np.isfinite(t_raw) & np.isfinite(x_raw)
    t_raw, x_raw = t_raw[valid], x_raw[valid]
    t_raw, unique_indices = np.unique(t_raw, return_index=True)
    x_raw = x_raw[unique_indices]

    # Linear Detrend
    p = np.polyfit(t_raw, x_raw, 1)
    x_detrend = x_raw - np.polyval(p, t_raw)

    x_coarse = x_detrend
    t_norm_full = np.arange(len(x_coarse)) * dt_norm
    
    # Identify middle 3 cycles from x_coarse
    peaks_c, _ = find_peaks(x_coarse, prominence=0.01)
    if len(peaks_c) < 4:
        # Fallback if not enough peaks: take most of it
        idx_3c_start_c = 0
        idx_3c_end_c = len(x_coarse)
    else:
        mid_c = (len(peaks_c) - 1) // 2
        idx_3c_start_c = peaks_c[mid_c - 1]
        idx_3c_end_c = peaks_c[mid_c + 2]

    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, len(sigma_list), height_ratios=[1, 1, 3])
    
    # Store limits for consistency across Row 1
    row1_lims = None

    # 2.3) Plot Sigma Loop
    for i, s_hat in enumerate(sigma_list):
        ax_x_pp = fig.add_subplot(gs[0, i])
        ax_x_tl = fig.add_subplot(gs[1, i])
        
        # --- Row 1 & 2 Cache ---
        row1_cache = f"{base_name}_Row1_PP_sigma{s_hat}"
        row2_cache = f"{base_name}_Row2_TL_sigma{s_hat}"
        
        df1 = hl.load_plot_data(row1_cache) if not force_recalc else None
        df2 = hl.load_plot_data(row2_cache) if not force_recalc else None
        
        if df1 is not None and df2 is not None:
            time_pp, x_pp, x_dot_pp, k_pp = df1['time'].values, df1['x'].values, df1['x_dot'].values, df1['curvature'].values
            time_tl, x_tl = df2['time'].values, df2['x'].values
        else:
            if s_hat == 'raw':
                x_3c = x_coarse[idx_3c_start_c:idx_3c_end_c]
                time_3c = t_norm_full[idx_3c_start_c:idx_3c_end_c] - t_norm_full[idx_3c_start_c]
                x_dot_3c = hl.central_diff(x_3c, dt_norm, order=1)
                k_3c = np.zeros_like(x_3c)
            else:
                sigma_norm = s_hat * points_per_cycle
                x_sm_all = hl.gauss_smooth(x_coarse, sigma_norm) if sigma_norm > 0 else x_coarse.copy()
                x_dot_all = hl.central_diff(x_sm_all, dt_norm, order=1)
                k_all = hl.compute_curvature(x_sm_all, x_dot_all, dt_norm)
                
                x_3c = x_sm_all[idx_3c_start_c:idx_3c_end_c]
                x_dot_3c = x_dot_all[idx_3c_start_c:idx_3c_end_c]
                k_3c = k_all[idx_3c_start_c:idx_3c_end_c]
                time_3c = t_norm_full[idx_3c_start_c:idx_3c_end_c] - t_norm_full[idx_3c_start_c]
            
            time_pp, x_pp, x_dot_pp, k_pp = time_3c, x_3c, x_dot_3c, k_3c
            time_tl, x_tl = time_3c, x_3c
            hl.save_plot_data({'time': time_pp, 'x': x_pp, 'x_dot': x_dot_pp, 'curvature': k_pp}, row1_cache)
            hl.save_plot_data({'time': time_tl, 'x': x_tl}, row2_cache)

        # Plot Row 1 & 2
        # Slice for Row 1 (1 cycle only)
        mask_1c = time_pp <= 1.0001
        x_1c = x_pp[mask_1c]
        x_dot_1c = x_dot_pp[mask_1c]

        if s_hat == 'raw':
            ax_x_pp.plot(x_1c, x_dot_1c, 'o', color='black', markersize=3, zorder=1)
            ax_x_pp.set_title(f"{title_prefix}\nX PP (Raw 48pts, 1cyc)")
            ax_x_tl.plot(time_tl, x_tl, 'o', color='black', markersize=3, zorder=1)
            ax_x_tl.set_title("X TL (Raw 48pts)")
            # Calculate Row 1 limits from Raw (1 cycle)
            row1_lims = pu.plot_axis(x_1c, x_dot_1c, 1.2, 1.2)
        else:
            ax_x_pp.plot(x_1c, x_dot_1c, color='black', lw=0.5, zorder=1)
            mp_pts, pm_pts = hl.get_inflection_points(time_pp, x_pp, x_dot_pp, k_pp)
            # Filter inflection points for 1 cycle
            if mp_pts.size > 0:
                mp_pts = mp_pts[mp_pts[:, 0] <= 1.0001]
                if mp_pts.size > 0: ax_x_pp.plot(mp_pts[:, 1], mp_pts[:, 2], 'o', color='cyan', markersize=3, zorder=2)
            if pm_pts.size > 0:
                pm_pts = pm_pts[pm_pts[:, 0] <= 1.0001]
                if pm_pts.size > 0: ax_x_pp.plot(pm_pts[:, 1], pm_pts[:, 2], 'o', color='magenta', markersize=3, zorder=2)
            ax_x_pp.set_title(f"X PP ($\hat{{\sigma}}={s_hat}$)")
            ax_x_tl.plot(time_tl, x_tl, color='black', lw=1)
            ax_x_tl.set_title(f"X TL ($\hat{{\sigma}}={s_hat}$)")

        if i == 0: ax_x_pp.set_ylabel("$\dot{x}$")
        if i == 0: ax_x_tl.set_ylabel("x")
        
        # Row 1 Styling
        if row1_lims:
            ax_x_pp.set_xlim(row1_lims[1], row1_lims[0])
            ax_x_pp.set_ylim(row1_lims[3], row1_lims[2])
        
        # Row 2 Styling
        ax_x_tl.set_xlim(0, 3) # Hardcoded 3 cycles for horizontal (no margin)
        ax_x_tl.set_xticks([0, 1])
        tl_lims = pu.plot_axis(time_tl, x_tl, 1.0, 1.2)
        ax_x_tl.set_ylim(tl_lims[3], tl_lims[2])

        for ax in [ax_x_pp, ax_x_tl]:
            ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
            ax.set_box_aspect(1)
        
    # 2.4) Row 3: CSS Map
    ax_x_css = fig.add_subplot(gs[2, :])
    row3_cache = f"{base_name}_Row3_CSS_Map"
    df3 = hl.load_plot_data(row3_cache) if not force_recalc else None
    
    if df3 is not None:
        T_pm_x = df3[df3['Direction'] == 'pm']['T_normalized'].values
        S_pm_x = df3[df3['Direction'] == 'pm']['Sigma_hat'].values
        T_mp_x = df3[df3['Direction'] == 'mp']['T_normalized'].values
        S_mp_x = df3[df3['Direction'] == 'mp']['Sigma_hat'].values
    else:
        T_pm_x, S_pm_x, T_mp_x, S_mp_x = [], [], [], []
        T_global_full = t_norm_full - t_norm_full[idx_3c_start_c]
        sigma_css_list = np.linspace(0.002, css_max, 100)
        for s_hat in sigma_css_list:
            sigma_norm = s_hat * points_per_cycle
            x_sm_all = hl.gauss_smooth(x_coarse, sigma_norm)
            x_dot_all = hl.central_diff(x_sm_all, dt_norm, order=1)
            k_val_all = hl.compute_curvature(x_sm_all, x_dot_all, dt_norm)
            
            x_3c = x_sm_all[idx_3c_start_c:idx_3c_end_c]
            x_dot_3c = x_dot_all[idx_3c_start_c:idx_3c_end_c]
            k_3c = k_val_all[idx_3c_start_c:idx_3c_end_c]
            T_sub = T_global_full[idx_3c_start_c:idx_3c_end_c]
            
            mp_pts, pm_pts = hl.get_inflection_points(T_sub, x_3c, x_dot_3c, k_3c)
            if mp_pts.size > 0: T_mp_x.extend(mp_pts[:, 0]); S_mp_x.extend([s_hat] * len(mp_pts))
            if pm_pts.size > 0: T_pm_x.extend(pm_pts[:, 0]); S_pm_x.extend([s_hat] * len(pm_pts))
        hl.save_plot_data({'T_normalized': list(T_pm_x) + list(T_mp_x), 'Sigma_hat': list(S_pm_x) + list(S_mp_x), 'Direction': ['pm']*len(T_pm_x) + ['mp']*len(T_mp_x)}, row3_cache)

    if len(T_pm_x) > 0: ax_x_css.plot(T_pm_x, S_pm_x, 'o', color='magenta', markersize=2, markeredgewidth=0)
    if len(T_mp_x) > 0: ax_x_css.plot(T_mp_x, S_mp_x, 'o', color='cyan', markersize=2, markeredgewidth=0)
    ax_x_css.set_xlim(0, 3); ax_x_css.set_xticks(np.arange(0, 3.1, 1))
    ax_x_css.set_ylim(0, css_max); ax_x_css.set_yticks([0, 0.1, 0.2])
    ax_x_css.set_box_aspect(1/2); ax_x_css.set_title(f"X CSS Map ({title_prefix})")
    ax_x_css.set_xlabel("Normalized Time ($t / T_{est}$)"); ax_x_css.set_ylabel("$\hat{\sigma}$")
    ax_x_css.spines['right'].set_visible(False); ax_x_css.spines['top'].set_visible(False)

    plt.tight_layout()
    fig_name = f"{base_name}_Noise_Analysis_Combined"
    pu.save_fig(fig, __file__, fig_name, eps_cmyk=True)
    plt.close(fig)
    print(f"Successfully saved: {fig_name}")
