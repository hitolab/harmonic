import sys
import os
import argparse
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

# Ensure Lib is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import Lib.harmonicity as hl
import Lib.plot_utils as pu

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Sasai Analysis")
    parser.add_argument('-f', '--force', action='store_true', help="Force recalculation (bypass caches)")
    args = parser.parse_args()
    force_recalc = args.force

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Sasai', 'u30_3.csv'))
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    cache_name = "sasai_plot_data_6cyc_smooth_0_1"

    # Always load raw data for display
    print(f"Loading raw data from: {csv_path}")
    first_row = pd.read_csv(csv_path, sep=r'\s+', nrows=1).iloc[0]
    has_header = first_row.apply(lambda val: isinstance(val, str)).any()
    df = pd.read_csv(csv_path, sep=r'\s+', header=None if not has_header else 'infer')
    t_raw = df.iloc[:, 0].values[::10]
    x_raw = df.iloc[:, 1].values[::10]

    # Extract last 6 cycles from raw data (trough-to-trough) and save as CSV
    prom_raw = 0.1 * (np.max(x_raw) - np.min(x_raw))
    troughs_raw, _ = scipy.signal.find_peaks(-x_raw, prominence=prom_raw)
    if len(troughs_raw) >= 7:
        idx_s = troughs_raw[-7]
        idx_e = troughs_raw[-1]
        t_6cyc = t_raw[idx_s:idx_e + 1]
        x_6cyc = x_raw[idx_s:idx_e + 1]
    else:
        t_6cyc = t_raw.copy()
        x_6cyc = x_raw.copy()

    csv_6cyc_path = os.path.join(script_dir, 'sasai_6cycles_raw.csv')
    pd.DataFrame({'t': t_6cyc, 'x': x_6cyc}).to_csv(csv_6cyc_path, index=False, header=False)
    print(f"Saved 6-cycle raw data to: {csv_6cyc_path}")

    # Load from cache if possible
    df_cached = hl.load_plot_data(cache_name) if not force_recalc else None

    if df_cached is not None:
        print("Loaded processed data from cache.")
        T = df_cached['T'].values
        R = df_cached['R'].values
        D = df_cached['D'].values
        K = df_cached['K'].values
    else:
        t = t_6cyc.copy()
        x = x_6cyc.copy()

        dt = np.mean(np.diff(t))
        print(f"dt = {dt}")

        # Estimate period first to determine the sigma_samp
        prom = 0.1 * (np.max(x) - np.min(x))
        T_est_raw = hl.get_average_period(t, x, prominence=prom)
        sigma_samp = 0.1 * T_est_raw / dt
        print(f"Smoothing Sasai data with sigma_samp = {sigma_samp:.4f} (sigma_ratio = 0.1)")
        x_sm = hl.gauss_smooth(x, sigma_samp)
        
        # Calculate derivatives and curvature on smoothed data
        print("Calculating Derivatives & Curvature...")
        D_raw = hl.central_diff(x_sm, dt, order=1)
        K_raw = hl.compute_curvature(x_sm, D_raw, dt)
        
        # Align to handle boundary NaNs
        T, R_list, D_list, K_list = hl.align_and_trim(t, [x_sm], [D_raw], [K_raw])
        R = R_list[0]
        D = D_list[0]
        K = K_list[0]
        
        # Save to cache
        cache_dict = {'T': T, 'R': R, 'D': D, 'K': K}
        hl.save_plot_data(cache_dict, cache_name)

    # Calculate period from the full trough-to-trough processed data
    prom = 0.1 * (np.max(R) - np.min(R))
    T_est = hl.get_average_period(T, R, prominence=prom)
    print(f"Estimated Period for Sasai: {T_est:.4f}")

    # Slice processed data from first peak to last peak
    peaks_proc, _ = scipy.signal.find_peaks(R, prominence=prom)
    if len(peaks_proc) >= 2:
        T = T[peaks_proc[0]:peaks_proc[-1] + 1]
        R = R[peaks_proc[0]:peaks_proc[-1] + 1]
        D = D[peaks_proc[0]:peaks_proc[-1] + 1]
        K = K[peaks_proc[0]:peaks_proc[-1] + 1]

    # Use actual time (zeroed at start of trough-to-trough window)
    t_span = t_6cyc[-1] - t_6cyc[0]
    T_norm = T - t_6cyc[0]

    # Raw panel shows full trough-to-trough data
    x_raw_plot = x_6cyc
    t_raw_norm = t_6cyc - t_6cyc[0]

    mask_pos = (K > 0)
    mask_neg = (K < 0)
    
    # Expand masks to include boundary transition points to avoid gaps
    transitions = mask_pos[:-1] != mask_pos[1:]
    mask_pos_ext = mask_pos.copy()
    mask_neg_ext = mask_neg.copy()
    mask_pos_ext[:-1] |= transitions
    mask_pos_ext[1:] |= transitions
    mask_neg_ext[:-1] |= transitions
    mask_neg_ext[1:] |= transitions
    
    R_neg = np.where(mask_neg_ext, R, np.nan)
    R_pos = np.where(mask_pos_ext, R, np.nan)
    D_neg = np.where(mask_neg_ext, D, np.nan)
    D_pos = np.where(mask_pos_ext, D, np.nan)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Sasai (Smoothed 0.1)', fontweight='bold', fontsize=16)

    t_xticks = np.arange(0, t_span + 1, 24)

    # Panel 0: Raw Data
    ax0 = plt.subplot(5, 1, 1)
    ax0.plot(t_raw_norm, x_raw_plot, '-k', linewidth=0.8)
    ax0.set_xlim([0, t_span])
    ax0.set_xticks(t_xticks)
    ax0.set_ylim([0, 1])
    ax0.set_yticks([0, 0.5, 1])
    ax0.set_box_aspect(1/3)
    ax0.tick_params(direction='out')
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.set_xlabel('t', fontweight='normal')
    ax0.set_ylabel('x', fontweight='normal')

    # Panel 1: Time Course
    ax1 = plt.subplot(5, 1, 2)
    ax1.plot(T_norm, R_neg, '-k', linewidth=1.2)
    ax1.plot(T_norm, R_pos, '-c', linewidth=1.2)
    
    if len(T) > 0:
        ax1.set_xlim([0, t_span])
        ax1.set_xticks(t_xticks)
        ax1.set_ylim([0, 1])
        ax1.set_yticks([0, 0.5, 1])
    
    ax1.set_box_aspect(1/3)
    ax1.tick_params(direction='out')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel('t', fontweight='normal')
    ax1.set_ylabel('x', fontweight='normal')

    # Panel 2: Phase Space
    ax2 = plt.subplot(5, 1, 3)
    ax2.plot(R_neg, D_neg, '-k', linewidth=1.2)
    ax2.plot(R_pos, D_pos, '-c', linewidth=1.2)
    
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-0.1, 0.1])
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_yticks([-0.1, 0, 0.1])
    ax2.set_box_aspect(1)
    ax2.tick_params(direction='out')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel('x', fontweight='normal')
    ax2.set_ylabel('dx/dt', fontweight='normal')

    # Panel 3: Curvature+
    ax3 = plt.subplot(5, 1, 4)
    if np.any(mask_pos):
        ax3.semilogy(T_norm[mask_pos], K[mask_pos], '-c', linewidth=1.2)
    else:
        ax3.axhline(10**5, color='k', linestyle='--')
        
    if len(T) > 0:
        ax3.set_xlim([0, t_span])
        ax3.set_xticks(t_xticks)
        ax3.set_ylim([1e-2, 1e2])
        ax3.set_yticks([1e-2, 1, 1e2])
            
    ax3.minorticks_off()
    ax3.tick_params(direction='out')
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_box_aspect(1/3)
    ax3.set_xlabel('t', fontweight='normal')
    ax3.set_ylabel('$\kappa$', fontweight='normal')

    # Panel 4: Curvature-
    ax4 = plt.subplot(5, 1, 5)
    if np.any(mask_neg):
        ax4.semilogy(T_norm[mask_neg], np.abs(K[mask_neg]), '-k', linewidth=1.2)
    else:
        ax4.axhline(10**5, color='k', linestyle='--')
        
    if len(T) > 0:
        ax4.set_xlim([0, t_span])
        ax4.set_xticks(t_xticks)
        ax4.set_ylim([1e2, 1e-2]) # Inverted
        ax4.set_yticks([1e-2, 1, 1e2])
            
    ax4.minorticks_off()
    ax4.tick_params(direction='out')
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_box_aspect(1/3)
    ax4.set_xlabel('t', fontweight='normal')
    ax4.set_ylabel('$\kappa$', fontweight='normal')

    plt.tight_layout()
    
    # Save using standard flow
    pu.save_fig(fig, __file__, "Sasai")
    plt.close(fig)
