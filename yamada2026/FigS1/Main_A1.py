import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Add Harmonicity root to path so Lib can be imported as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Lib.harmonicity as hl
import Lib.plot_utils as pu

# ============================================================
# 0) Command Line Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Models Noise Comparison Analysis")
parser.add_argument('-f', '--force', action='store_true', help="Force recalculation (bypass caches)")
args = parser.parse_args()
force_recalc = args.force

# ============================================================
# 1) Models Configuration
# ============================================================
# ... rest of the file ...
# Define the models and their corresponding parameters to run
models_to_run = [
    {
        'name': 'FN',
        'params': [3.0, 0.4, 0.6, 0.04, 1e-3], # a, b, c, noise_std, dt
        'title': 'FitzHugh-Nagumo Model',
        'sigma_list': ['raw', 0, 0.04, 0.08, 0.12, 0.16],
        'css_max': 0.2,
        'css_ticks': np.arange(0, 0.21, 0.05)
    },
    {
        'name': 'SL',
        'params': [4.0, 0.3, 0.4, 0.3, 1e-3], # A, B, W, noise_std, dt
        'title': 'Stuart-Landau Model',
        'sigma_list': ['raw', 0, 0.04, 0.08, 0.12, 0.16],
        'css_max': 0.2,
        'css_ticks': np.arange(0, 0.21, 0.05)
    }
]

td_sim = 48.0
num_models = len(models_to_run)

# ============================================================
# 2) Loop Over Models
# ============================================================
for m_idx, model_info in enumerate(models_to_run):
    model_name = model_info['name']
    params = model_info['params']
    dt_sim = params[-1]
    noise_std = params[-2]
    title_prefix = model_info['title']
    
    sigma_list = model_info['sigma_list']
    num_cols = len(sigma_list)
    
    # Create a separate figure for each model
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, num_cols, height_ratios=[1, 1, 3])
    
    # Store limits for consistency across Row 1
    row1_lims = None

    print(f"--- Running {model_name} ---")
    print(f"Parameters: {params}")
    
    # Define parameter names for this model for the CSV header
    if 'FN' in model_name:
        param_names_det   = ['a', 'b', 'c']
        param_names_stoch = ['a', 'b', 'c', 'noise_std', 'dtau', 'T']
    elif 'SL' in model_name:
        param_names_det   = ['A', 'B', 'W']
        param_names_stoch = ['A', 'B', 'W', 'noise_std', 'dtau', 'T']
    else:
        param_names_det   = [f'p{i}' for i in range(len(params) - 2)]
        param_names_stoch = param_names_det + ['noise_std', 'dtau', 'T']

    # 1) Noiseless preliminary run (base deterministic model) to estimate period accurately
    params_det = list(params[:-2])  # drop noise_std, dt -> [a, b, c]
    res_pre = hl.runge_iterate(model_name, dt_sim, 48.0, params_det, param_names=param_names_det, force_recalculate=force_recalc)
    time_pre = np.arange(len(res_pre[0])) * dt_sim
    T_est = hl.get_average_period(time_pre, res_pre[0], prominence=0.1)
    if np.isnan(T_est): T_est = 1.0 # fallback

    # 2) Main run: stochastic model (Models/{name}_stoch, normalized time, Euler-Maruyama),
    # fixed dτ = 1/100. Pure Euler-Maruyama (not RK4): the noise term must be sampled fresh
    # every model_func call, which only holds for a single-evaluation-per-step integrator.
    dtau = 1.0 / 100
    model_name_stoch = f"{model_name}_stoch"
    params_norm = list(params[:-1]) + [dtau, T_est]  # replace original dt with dtau, append T
    td_long_norm = 110.0  # 110 cycles in normalized time (τ-units)
    res_long = hl.euler_maruyama_iterate(model_name_stoch, dtau, td_long_norm, params_norm, param_names=param_names_stoch, force_recalculate=force_recalc)
    x_long = res_long[0]
    time_long = np.arange(len(x_long)) * dtau  # τ-time axis

    # 3) Extract cycles 100 to 106 using smoothed signal for robust peak detection
    x_long_sm = hl.gauss_smooth(x_long, sigma_samp=0.2 * 100)
    time_clean, _ = hl.extract_cycles(time_long, x_long_sm, skip_num=100, take_num=6)
    idx_start = np.searchsorted(time_long, time_clean[0])
    x_clean = x_long[idx_start:idx_start + len(time_clean)]

    peaks_all, _ = find_peaks(x_long_sm, prominence=0.1)
    num_cycles_total = len(peaks_all) - 1
    print(f"Simulation Finished. Total cycles: {num_cycles_total}")

    # ------------------------------------------------------------
    # 2.2) 50 points/cycle (subsample by 2 from the 100 pts/cycle norm run)
    # ------------------------------------------------------------
    points_per_cycle = 50
    x_coarse = x_clean[::2]
    t_coarse = time_clean[::2]

    dt_norm = 1.0 / points_per_cycle
    t_norm_full = np.arange(len(x_coarse)) * dt_norm

    # Identify middle 3 cycles from x_coarse (for extraction)
    peaks_c, _ = find_peaks(x_coarse, prominence=0.1)
    mid_c = (len(peaks_c) - 1) // 2
    idx_3c_start_c = peaks_c[mid_c - 1]
    idx_3c_end_c = peaks_c[mid_c + 2]
    
    # Construct parameter part for filename
    param_suffix = "_".join([f"{p:.6g}" for p in params_norm])
    metadata_base = {n: v for n, v in zip(param_names_stoch, params_norm)}
    
    # ------------------------------------------------------------
    # 2.3) Plot Phase Portraits and Time Series (Sigma Loop)
    # ------------------------------------------------------------
    for i, s_hat in enumerate(sigma_list):
        ax_x_pp = fig.add_subplot(gs[0, i])
        ax_x_tl = fig.add_subplot(gs[1, i])
        
        # --- Row 1 & 2 Cache Check ---
        row1_cache = f"{model_name}_Row1_PP_sigma{s_hat}_{param_suffix}_50pts"
        row2_cache = f"{model_name}_Row2_TL_sigma{s_hat}_{param_suffix}_50pts"
        
        df1 = hl.load_plot_data(row1_cache) if not force_recalc else None
        df2 = hl.load_plot_data(row2_cache) if not force_recalc else None
        
        # --- Data Preparation (if any cache missing) ---
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
            
            # Re-assign for plotting
            time_pp, x_pp, x_dot_pp, k_pp = time_3c, x_3c, x_dot_3c, k_3c
            time_tl, x_tl = time_3c, x_3c
            
            # Save both caches
            m1 = metadata_base.copy(); m1['sigma_hat'] = s_hat
            hl.save_plot_data({'time': time_pp, 'x': x_pp, 'x_dot': x_dot_pp, 'curvature': k_pp}, row1_cache, metadata=m1)
            hl.save_plot_data({'time': time_tl, 'x': x_tl}, row2_cache, metadata=m1)

        # Slice for Row 1 (1 cycle only)
        mask_1c = time_pp <= 1.0001
        x_1c = x_pp[mask_1c]
        x_dot_1c = x_dot_pp[mask_1c]
        
        # --- Plot Row 1 & 2 ---
        if s_hat == 'raw':
            ax_x_pp.plot(x_1c, x_dot_1c, 'o', color='black', markersize=3, zorder=1)
            ax_x_pp.set_title(f"{title_prefix}\nX PP (Raw)")
            ax_x_tl.plot(time_tl, x_tl, 'o', color='black', markersize=3, zorder=1)
            ax_x_tl.set_title("X TL (Raw)")
            row1_lims = pu.plot_axis(x_1c, x_dot_1c, 1.2, 1.2)
        else:
            ax_x_pp.plot(x_1c, x_dot_1c, color='black', lw=0.5, zorder=1)
            mp_pts, pm_pts = hl.get_inflection_points(time_pp, x_pp, x_dot_pp, k_pp)
            if mp_pts.size > 0:
                mp_pts = mp_pts[mp_pts[:, 0] <= 1.0001]
                if mp_pts.size > 0: ax_x_pp.plot(mp_pts[:, 1], mp_pts[:, 2], 'o', color='cyan', markersize=3, zorder=2)
            if pm_pts.size > 0:
                pm_pts = pm_pts[pm_pts[:, 0] <= 1.0001]
                if pm_pts.size > 0: ax_x_pp.plot(pm_pts[:, 1], pm_pts[:, 2], 'o', color='magenta', markersize=3, zorder=2)
            ax_x_pp.set_title(f"X PP ($\hat{{\sigma}}={s_hat}$)")
            ax_x_tl.plot(time_tl, x_tl, color='black', lw=1)
            ax_x_tl.set_title(f"X TL ($\hat{{\sigma}}={s_hat}$)")

        # Styling
        if i == 0: ax_x_pp.set_ylabel("$\dot{x}$")
        if i == 0: ax_x_tl.set_ylabel("x")
        
        # Row 1 Styling
        if row1_lims:
            ax_x_pp.set_xlim(row1_lims[1], row1_lims[0])
            ax_x_pp.set_ylim(row1_lims[3], row1_lims[2])
            
        # Row 2 Styling
        ax_x_tl.set_xlim(0, 3) # No margin horizontal
        ax_x_tl.set_xticks([0, 1])
        tl_lims = pu.plot_axis(time_tl, x_tl, 1.0, 1.2) # 20% vertical margin
        ax_x_tl.set_ylim(tl_lims[3], tl_lims[2])

        for ax in [ax_x_pp, ax_x_tl]:
            ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
            ax.set_box_aspect(1)
        ax_x_pp.tick_params(direction='out')
        
    # ------------------------------------------------------------
    # 2.4) Row 3: CSS Map
    # ------------------------------------------------------------
    ax_x_css = fig.add_subplot(gs[2, :])
    css_max = model_info['css_max']
    row3_cache = f"{model_name}_Row3_CSS_{param_suffix}_50pts"
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
        hl.save_plot_data({'T_normalized': list(T_pm_x) + list(T_mp_x), 'Sigma_hat': list(S_pm_x) + list(S_mp_x), 'Direction': ['pm']*len(T_pm_x) + ['mp']*len(T_mp_x)}, row3_cache, metadata=metadata_base)

    if len(T_pm_x) > 0: ax_x_css.plot(T_pm_x, S_pm_x, 'o', color='magenta', markersize=2, markeredgewidth=0, zorder=2)
    if len(T_mp_x) > 0: ax_x_css.plot(T_mp_x, S_mp_x, 'o', color='cyan', markersize=2, markeredgewidth=0, zorder=2)
    ax_x_css.set_xlim(0, 3); ax_x_css.set_xticks(np.arange(0, 3.1, 1))
    ax_x_css.set_ylim(0, css_max); ax_x_css.set_yticks([0, 0.1, 0.2])
    ax_x_css.set_box_aspect(1/2); ax_x_css.set_title(f"X CSS Map ({title_prefix})")
    ax_x_css.set_xlabel("Normalized Time ($t / T_{est}$)"); ax_x_css.set_ylabel("$\hat{\sigma}$")
    ax_x_css.spines['right'].set_visible(False); ax_x_css.spines['top'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Filename for saving the figure
    fig_name = f"{model_name}_Models_Noise_Comparison_{param_suffix}"
    pu.save_fig(fig, __file__, fig_name, eps_cmyk=True)
    plt.close(fig)
    print(f"Successfully saved: {fig_name}")
