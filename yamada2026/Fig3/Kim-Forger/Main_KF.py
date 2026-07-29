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
    parser = argparse.ArgumentParser(description="Main Kim-Forger Analysis")
    parser.add_argument('-f', '--force', action='store_true', help="Force recalculation (bypass caches)")
    args = parser.parse_args()
    force_recalc = args.force

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Kim-Forger', 'KF_McPo.csv'))
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    cache_name = "kim_forger_plot_data_raw"
    
    # Load from cache if possible
    df_cached = hl.load_plot_data(cache_name) if not force_recalc else None
    
    if df_cached is not None:
        print("Loaded processed data from cache.")
        T = df_cached['T'].values
        R = df_cached['R'].values
        D = df_cached['D'].values
        K = df_cached['K'].values
    else:
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        t = df.iloc[:, 0].values
        x = df.iloc[:, 1].values
        
        # Drop valid pairs only
        valid_idx = ~np.isnan(t) & ~np.isnan(x)
        t = t[valid_idx]
        x = x[valid_idx]
        
        dt = np.mean(np.diff(t))
        print(f"dt estimated as: {dt:.4f}")

        # Calculate derivatives and curvature directly on raw data (no smoothing, no scaling)
        print("Calculating Derivatives & Curvature...")
        D_raw = hl.central_diff(x, dt, order=1)
        K_raw = hl.compute_curvature(x, D_raw, dt)
        
        # Align to handle boundary NaNs
        T, R_list, D_list, K_list = hl.align_and_trim(t, [x], [D_raw], [K_raw])
        R = R_list[0]
        D = D_list[0]
        K = K_list[0]
        
        # Save to cache
        cache_dict = {'T': T, 'R': R, 'D': D, 'K': K}
        hl.save_plot_data(cache_dict, cache_name)

    prom = 0.1 * (np.max(R) - np.min(R))
    T_est = hl.get_average_period(T, R, prominence=prom)
    print(f"Estimated Period for Kim-Forger: {T_est:.4f}")

    # Slice last 6 cycles (trough-to-trough)
    troughs, _ = scipy.signal.find_peaks(-R, prominence=prom)
    if len(troughs) >= 7:
        T = T[troughs[-7]:troughs[-1] + 1]
        R = R[troughs[-7]:troughs[-1] + 1]
        D = D[troughs[-7]:troughs[-1] + 1]
        K = K[troughs[-7]:troughs[-1] + 1]

    T_norm = T - T[0]
    t_xticks = np.arange(0, 145, 24)

    mask_pos = K > 0
    mask_neg = K < 0

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
    plt.rcParams['svg.fonttype'] = 'none'

    fig = plt.figure(figsize=(4, 12)) 
    fig.suptitle('Kim-Forger (Raw)', fontweight='bold', fontsize=16)

    # 1) Time Course
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(T_norm, R, '-k', linewidth=1.2)

    if len(T) > 0:
        ax1.set_xlim([0, 144])
        ax1.set_xticks(t_xticks)
        ax1.set_ylim([0, 3])
        ax1.set_yticks([0, 1, 2, 3])

    ax1.set_box_aspect(1/3)
    ax1.tick_params(direction='out')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel('t', fontweight='normal')
    ax1.set_ylabel('x', fontweight='normal')

    # 2) Phase Space
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(R_neg, D_neg, '-k', linewidth=1.2)
    ax2.plot(R_pos, D_pos, '-c', linewidth=1.2)

    ax2.set_xlim([0, 3])
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_yticks([-0.5, 0, 0.5])
    ax2.set_box_aspect(1)
    ax2.tick_params(direction='out')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel('x', fontweight='normal')
    ax2.set_ylabel('dx/dt', fontweight='normal')

    # 3) Curvature+
    ax3 = fig.add_subplot(4, 1, 3)
    if np.any(mask_pos):
        ax3.semilogy(T_norm[mask_pos], K[mask_pos], '-c', linewidth=1.2)
    else:
        ax3.semilogy([0, 144], [1e5, 1e5], 'k--')

    if len(T) > 0:
        ax3.set_xlim([0, 144])
        ax3.set_xticks(t_xticks)
        ax3.set_ylim([1e-2, 1e2])
        ax3.set_yticks([1e-2, 1, 1e2])

    ax3.tick_params(direction='out')
    ax3.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_box_aspect(1/3)
    ax3.set_xlabel('t', fontweight='normal')
    ax3.set_ylabel(r'$\kappa$', fontweight='normal')

    # 4) Curvature-
    ax4 = fig.add_subplot(4, 1, 4)
    if np.any(mask_neg):
        ax4.semilogy(T_norm[mask_neg], np.abs(K[mask_neg]), '-k', linewidth=1.2)
    else:
        ax4.axhline(1e5, color='k', linestyle='--')

    if len(T) > 0:
        ax4.set_xlim([0, 144])
        ax4.set_xticks(t_xticks)
        ax4.set_ylim([1e2, 1e-2])
        ax4.set_yticks([1e-2, 1, 1e2])

    ax4.tick_params(direction='out')
    ax4.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_box_aspect(1/3)
    ax4.set_xlabel('t', fontweight='normal')
    ax4.set_ylabel(r'$\kappa$', fontweight='normal')

    plt.tight_layout()
    
    # Save using standard flow
    pu.save_fig(fig, __file__, "KF_McPo")

    import os
    eps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img', "KF_McPo.eps")
    fig.savefig(eps_path, transparent=True, bbox_inches='tight', format='eps')
    plt.close(fig)
