import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Lib.plot_utils as pu
from Main_A3 import VARIABLES, power_labels, prepare_variable_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FigS3 Kim-Forger Analysis - all variables in one grid figure")
    parser.add_argument('-f', '--force', action='store_true', help="Force recalculation (bypass caches)")
    args = parser.parse_args()
    force_recalc = args.force

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['agg.path.chunksize'] = 10000

    rows = 4
    cols = len(VARIABLES)
    subplot_size = 2.2

    fig = plt.figure(figsize=(subplot_size * cols, subplot_size * rows))

    # --- Pass 1: load all variables ---
    all_data = {}
    for var_name in VARIABLES:
        print(f"=== Processing {var_name} ===")
        all_data[var_name] = prepare_variable_data(var_name, force_recalc)

    # Manually specified t-x (x value) ranges per variable.
    fixed_R_range = {
        'MnPo': (0, 12), 'McPo': (0, 4),
        'MnPt': (0, 26), 'McPt': (0, 8),
        'MnRo': (0, 10), 'McRo': (0, 4),
        'MnB': (20, 50), 'McB': (3, 11),
    }

    # --- Pass 2: plot, each variable self-centered on its own mean ---
    for col_idx, var_name in enumerate(VARIABLES):
        data = all_data[var_name]
        if data is None:
            continue

        T_norm, t_xticks = data['T_norm'], data['t_xticks']
        R, D, K = data['R'], data['D'], data['K']
        mask_pos, mask_neg = data['mask_pos'], data['mask_neg']
        R_neg, R_pos, D_neg, D_pos = data['R_neg'], data['R_pos'], data['D_neg'], data['D_pos']

        # --- Row 1: Time Course (colored by curvature sign) ---
        # Manually specified per-variable range (see fixed_R_range above).
        ax1 = fig.add_subplot(rows, cols, 0 * cols + col_idx + 1)
        ax1.plot(T_norm, R_neg, '-m', linewidth=1.0)
        ax1.plot(T_norm, R_pos, '-c', linewidth=1.0)
        if len(T_norm) > 0:
            ymin1, ymax1 = fixed_R_range[var_name]
            ax1.set_xlim([0, 144])
            ax1.set_xticks(t_xticks)
            ax1.set_ylim([ymin1, ymax1])
            ax1.set_yticks([ymin1, (ymin1 + ymax1) / 2, ymax1])
        ax1.set_box_aspect(1 / 2)
        ax1.set_title(var_name, fontsize=9)
        if col_idx == 0:
            ax1.set_ylabel('x')

        # --- Row 2: Phase Space ---
        # Horizontal axis: same range as Row 1. Vertical axis: centered on
        # the (ceiling-rounded, non-negative) mean of dx/dt, spread by the
        # same range width split evenly above/below that center.
        ax2 = fig.add_subplot(rows, cols, 1 * cols + col_idx + 1)
        ax2.plot(R_neg, D_neg, '-m', linewidth=1.0)
        ax2.plot(R_pos, D_pos, '-c', linewidth=1.0)
        valid2 = ~np.isnan(R) & ~np.isnan(D)
        if np.any(valid2):
            xmin2, xmax2 = fixed_R_range[var_name]
            width2 = xmax2 - xmin2
            Dcenter = np.ceil(np.nanmean(D[valid2]))
            ymin2 = Dcenter - width2 / 2
            ymax2 = Dcenter + width2 / 2
            ax2.set_xlim([xmin2, xmax2])
            ax2.set_ylim([ymin2, ymax2])
            ax2.set_xticks([xmin2, xmax2])
            ax2.set_yticks([ymin2, ymax2])
        ax2.set_box_aspect(1)
        if col_idx == 0:
            ax2.set_ylabel('dx/dt')

        # --- Row 3: K > 0 (fixed 1e-2 to 1e2) ---
        ax3 = fig.add_subplot(rows, cols, 2 * cols + col_idx + 1)
        if np.any(mask_pos):
            ax3.semilogy(T_norm[mask_pos], K[mask_pos], '-c', linewidth=1.0)
        if len(T_norm) > 0:
            ax3.set_xlim([0, 144])
            ax3.set_xticks(t_xticks)
            ax3.set_ylim([1e-2, 1e2])
            ax3.set_yticks([1e-2, 1, 1e2])
            ax3.set_yticklabels(power_labels([1e-2, 1, 1e2]))
        ax3.minorticks_off()
        ax3.set_box_aspect(1 / 2)
        if col_idx == 0:
            ax3.set_ylabel(r'$\kappa$ (K>0)')

        # --- Row 4: K < 0 (fixed 1e-2 to 1e2) ---
        ax4 = fig.add_subplot(rows, cols, 3 * cols + col_idx + 1)
        if np.any(mask_neg):
            ax4.semilogy(T_norm[mask_neg], np.abs(K[mask_neg]), '-m', linewidth=1.0)
        if len(T_norm) > 0:
            ax4.set_xlim([0, 144])
            ax4.set_xticks(t_xticks)
            ax4.set_ylim([1e-2, 1e2])
            ax4.set_yticks([1e-2, 1, 1e2])
            ax4.set_yticklabels(power_labels([1e-2, 1, 1e2], negative=True))
        ax4.minorticks_off()
        ax4.set_box_aspect(1 / 2)
        if col_idx == 0:
            ax4.set_ylabel(r'$\kappa$ (K<0)')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(direction='out', labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig_name = "KF_all_grid"
    pu.save_fig(fig, __file__, fig_name)
    plt.close(fig)
    print(f"Successfully saved: {fig_name}")
