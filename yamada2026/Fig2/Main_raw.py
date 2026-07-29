import sys
import os
import argparse
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

# Add Lib to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Lib.harmonicity as hl
import Lib.plot_utils as pu

datasets = [
    {'name': 'Cyano_Kondo', 'path': '../Data/Cyano/Cyano_Kondo.csv', 'sigma_est': 0.06},
    {'name': 'SCN', 'path': '../Data/MouseSCN/SCN.csv', 'sigma_est': 0.05}
]
sigma_hat_list = [0.1]
peak_prominence_coarse = 0.01
points_per_cycle = 48
dt_norm = 24.0 / points_per_cycle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig2 Experimental Data Analysis")
    parser.add_argument('-f', '--force', action='store_true', help="Force recalculation")
    args = parser.parse_args()
    force_recalc = args.force

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_results = []
    
    for ci, d_info in enumerate(datasets):
        base_name = d_info['name']
        data_path = os.path.join(script_dir, d_info['path'])
        sigma_est = d_info['sigma_est']
        
        print(f"Processing: {base_name}")
        
        # 1. Load Raw Data
        df = pd.read_csv(data_path)
        t = df.iloc[:, 0].values
        x = df.iloc[:, 1].values
        
        valid_idx = ~np.isnan(t) & ~np.isnan(x)
        t = t[valid_idx]
        x = x[valid_idx]
        
        t, unique_indices = np.unique(t, return_index=True)
        x = x[unique_indices]
        
        if "SCN" in base_name:
            x = x * 30000
            y_max_tick = 30000
        else:
            x = x * 10000
            y_max_tick = 8000
            
        t_raw_csv = (np.copy(t) - t[0]) * dt_norm
        x_raw_csv = np.copy(x)
        T_max_bound = t_raw_csv[-1]

        # Data is already uniformly sampled at 48 pts/cycle (integer indices)
        x_coarse = x.copy()
        t_norm_full = np.arange(len(x_coarse)) * dt_norm
        
        peaks_c, _ = scipy.signal.find_peaks(x_coarse, prominence=peak_prominence_coarse)
        if len(peaks_c) >= 2:
            idx_start = peaks_c[0]
            idx_end = peaks_c[-1]
        else:
            idx_start = 0
            idx_end = len(x_coarse) - 1

        results_sigma = []
        
        for sigma_hat in sigma_hat_list:
            cache_name = f"{base_name}_plot_data_sigma{sigma_hat}_t24_all"
            df_cached = hl.load_plot_data(cache_name) if not force_recalc else None
            
            if df_cached is not None:
                print(f"Loaded {cache_name} from cache.")
                time_sliced = df_cached['time'].values
                x_sliced = df_cached['x'].values
                x_dot_sliced = df_cached['x_dot'].values
                k_sliced = df_cached['curvature'].values
                mp_pts, pm_pts = hl.get_inflection_points(time_sliced, x_sliced, x_dot_sliced, k_sliced)
            else:
                sigma_norm = sigma_hat * points_per_cycle
                x_sm_all = hl.gauss_smooth(x_coarse, sigma_norm) if sigma_norm > 0 else np.copy(x_coarse)
                x_dot_all = hl.central_diff(x_sm_all, dt_norm, order=1)
                k_all = hl.compute_curvature(x_sm_all, x_dot_all, dt_norm)
                
                # Extract peak-to-peak cycles
                x_sliced = x_sm_all[idx_start:idx_end]
                x_dot_sliced = x_dot_all[idx_start:idx_end]
                k_sliced = k_all[idx_start:idx_end]
                time_sliced = t_norm_full[idx_start:idx_end]
                
                valid_mask = ~np.isnan(k_sliced) & ~np.isnan(x_sliced)
                time_sliced = time_sliced[valid_mask]
                x_sliced = x_sliced[valid_mask]
                x_dot_sliced = x_dot_sliced[valid_mask]
                k_sliced = k_sliced[valid_mask]
                
                mp_pts, pm_pts = hl.get_inflection_points(time_sliced, x_sliced, x_dot_sliced, k_sliced)
                hl.save_plot_data({'time': time_sliced, 'x': x_sliced, 'x_dot': x_dot_sliced, 'curvature': k_sliced}, cache_name)

            results_sigma.append({
                'T': time_sliced, 'R': x_sliced, 'D': x_dot_sliced, 'K': k_sliced,
                'T_full_max': t_norm_full[-1],
                'mp_pts': mp_pts, 'pm_pts': pm_pts,
                't_raw_csv': t_raw_csv, 'x_raw_csv': x_raw_csv,
                'T_max_bound': T_max_bound
            })
            
        all_results.append({
            'baseName': base_name,
            'Results': results_sigma,
            'y_max_tick': y_max_tick
        })
        
    print("Plotting grid...")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    
    num_datasets = len(datasets)
    num_sigma = len(sigma_hat_list)
    cols = num_datasets * num_sigma
    rows = 5
    subplot_width = 8
    subplot_height = subplot_width * (10.015 / 20)
    
    fig = plt.figure(figsize=(subplot_width * cols, subplot_height * rows))
    
    for ci in range(num_datasets):
        base_name = all_results[ci]['baseName']
        results = all_results[ci]['Results']
        y_max_tick = all_results[ci]['y_max_tick']
        
        for si in range(num_sigma):
            res = results[si]
            T = res['T']
            R = res['R']
            D = res['D']
            K = res['K']
            mp_pts = res['mp_pts']
            pm_pts = res['pm_pts']
            t_raw_csv = res['t_raw_csv']
            x_raw_csv = res['x_raw_csv']
            T_max_bound = res['T_max_bound']
            
            mask_pos = K > 0
            mask_neg = K < 0
            
            col_idx = ci * num_sigma + si
            
            # --- Row 1: Raw Data ---
            ax1 = fig.add_subplot(rows, cols, 0 * cols + col_idx + 1)
            ax1.plot(t_raw_csv, x_raw_csv, '.k', markersize=4, markeredgewidth=0)
            ax1.set_xlim([0, 144])
            ax1.set_xticks(np.arange(0, 145, 24))
            ax1.set_ylim([0, y_max_tick])
            ax1.set_yticks([0, y_max_tick])
            if col_idx == 0: ax1.set_ylabel("Raw")
            ax1.set_title(base_name)
            
            # --- Row 2: R (Peak to Peak) ---
            ax2 = fig.add_subplot(rows, cols, 1 * cols + col_idx + 1)
            R_min, R_max = R.min(), R.max()
            R_range = R_max - R_min
            R_norm = (R - R_min) / R_range
            D_norm = D / R_range
            
            transitions = mask_pos[:-1] != mask_pos[1:]
            mask_pos_ext = mask_pos.copy()
            mask_neg_ext = mask_neg.copy()
            mask_pos_ext[:-1] |= transitions
            mask_pos_ext[1:] |= transitions
            mask_neg_ext[:-1] |= transitions
            mask_neg_ext[1:] |= transitions
            
            R_norm_neg = np.where(mask_neg_ext, R_norm, np.nan)
            R_norm_pos = np.where(mask_pos_ext, R_norm, np.nan)
            ax2.plot(T, R_norm_neg, '.k', markersize=4, markeredgewidth=0)
            ax2.plot(T, R_norm_pos, '.k', markersize=4, markeredgewidth=0)
            
            ax2.set_xlim(0, 144)
            ax2.set_xticks(np.arange(0, 145, 24))
            _, _, ymax2, ymin2 = pu.plot_axis(T, R_norm, 1.0, 1.2)
            ax2.set_ylim(ymin2, ymax2)
            ax2.set_yticks([0, 1])
            if col_idx == 0: ax2.set_ylabel("X (norm)")
            
            # --- Row 3: Phase Portrait (all cycles) ---
            ax3 = fig.add_subplot(rows, cols, 2 * cols + col_idx + 1)
            
            D_norm_neg = np.where(mask_neg_ext, D_norm, np.nan)
            D_norm_pos = np.where(mask_pos_ext, D_norm, np.nan)
            ax3.plot(R_norm_neg, D_norm_neg, '.k', markersize=4, markeredgewidth=0, zorder=1)
            ax3.plot(R_norm_pos, D_norm_pos, '.k', markersize=4, markeredgewidth=0, zorder=1)

            if mp_pts.size > 0:
                mp_pts_norm = mp_pts.copy()
                mp_pts_norm[:, 1] = (mp_pts[:, 1] - R_min) / R_range
                mp_pts_norm[:, 2] = mp_pts[:, 2] / R_range
                ax3.plot(mp_pts_norm[:, 1], mp_pts_norm[:, 2], 'o', color='black', markersize=4, zorder=2)
            if pm_pts.size > 0:
                pm_pts_norm = pm_pts.copy()
                pm_pts_norm[:, 1] = (pm_pts[:, 1] - R_min) / R_range
                pm_pts_norm[:, 2] = pm_pts[:, 2] / R_range
                ax3.plot(pm_pts_norm[:, 1], pm_pts_norm[:, 2], 'o', color='black', markersize=4, zorder=2)
                
            x_ticks = [0, 1]
            xmax, xmin, _, _ = pu.plot_axis(x_ticks, [0, 1], 1.2, 1.2)
            ax3.set_xlim(xmin, xmax)
            ax3.set_ylim(-0.2, 0.2)
            ax3.set_xticks(x_ticks)
            ax3.set_yticks([-0.2, 0, 0.2])
            ax3.set_box_aspect(1)
            if col_idx == 0: ax3.set_ylabel("dX/dt (all)")
            
            # --- Row 4: K > 0 (Peak to Peak) ---
            ax4 = fig.add_subplot(rows, cols, 3 * cols + col_idx + 1)
            if np.any(mask_pos): ax4.semilogy(T[mask_pos], K[mask_pos], '.k', markersize=4, markeredgewidth=0)
            ax4.set_xlim(0, 144)
            ax4.set_xticks(np.arange(0, 145, 24))
            ax4.set_ylim(1e-6, 1e-2)
            ax4.set_yticks([1e-6, 1e-4, 1e-2])
            ax4.set_yticklabels(['10$^{-6}$', '10$^{-4}$', '10$^{-2}$'])
            ax4.minorticks_off()
            if col_idx == 0: ax4.set_ylabel("K > 0")
            
            # --- Row 5: K < 0 (Peak to Peak) ---
            ax5 = fig.add_subplot(rows, cols, 4 * cols + col_idx + 1)
            if np.any(mask_neg): ax5.semilogy(T[mask_neg], -K[mask_neg], '.k', markersize=4, markeredgewidth=0)
            ax5.set_xlim(0, 144)
            ax5.set_xticks(np.arange(0, 145, 24))
            ax5.set_ylim(1e-2, 1e-6)
            ax5.set_yticks([1e-2, 1e-4, 1e-6])
            ax5.set_yticklabels(['-10$^{-2}$', '-10$^{-4}$', '-10$^{-6}$'])
            ax5.minorticks_off()
            if col_idx == 0: ax5.set_ylabel("K < 0")
            
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.tick_params(direction='out')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if ax in [ax1, ax2]:
                    ax.set_box_aspect(1/3)
                if ax in [ax4, ax5]:
                    ax.set_box_aspect(16/74.5)
                
                
    plt.tight_layout()
    fig_name = 'Fig2_Experimental_TRDK'
    pu.save_fig(fig, __file__, fig_name)
    plt.close(fig)
