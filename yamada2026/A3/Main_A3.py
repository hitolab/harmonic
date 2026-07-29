import sys
import os
import argparse
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

# Ensure Lib is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Lib.harmonicity as hl
import Lib.plot_utils as pu

# Kim-Forger nucleus (Mn) / cytoplasm (Mc) mRNA concentration variables.
# Per1, Per2, Cry1, Cry2, Bmal1, Npas2, Rev-erb
VARIABLES = [
    'MnPo', 'McPo', 'MnPt', 'McPt', 'MnRo', 'McRo', 'MnB', 'McB'
]


def power_labels(ticks, negative=False):
    labels = []
    for t in ticks:
        exp = int(round(np.log10(t)))
        sign = '-' if negative else ''
        labels.append(f'{sign}10$^{{{exp}}}$')
    return labels


def prepare_variable_data(var_name, force_recalc):
    """Load (or compute) T/R/D/K for one variable, slice to the last 6
    cycles, and return everything needed for plotting. Returns None if the
    source CSV is missing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, '..', 'Data', 'Kim-Forger', f'KF_{var_name}.csv'))

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return None

    cache_name = f"kim_forger_plot_data_{var_name}"

    df_cached = hl.load_plot_data(cache_name) if not force_recalc else None

    if df_cached is not None:
        print(f"Loaded processed data from cache ({var_name}).")
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
    print(f"Estimated Period for {var_name}: {T_est:.4f}")

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

    return {
        'T_norm': T_norm, 't_xticks': t_xticks,
        'R': R, 'D': D, 'K': K,
        'mask_pos': mask_pos, 'mask_neg': mask_neg,
        'R_neg': R_neg, 'R_pos': R_pos, 'D_neg': D_neg, 'D_pos': D_pos,
    }


def process_variable(var_name, force_recalc):
    data = prepare_variable_data(var_name, force_recalc)
    if data is None:
        return
    T_norm, t_xticks = data['T_norm'], data['t_xticks']
    R, D, K = data['R'], data['D'], data['K']
    mask_pos, mask_neg = data['mask_pos'], data['mask_neg']
    R_neg, R_pos, D_neg, D_pos = data['R_neg'], data['R_pos'], data['D_neg'], data['D_pos']
    T = T_norm  # len(T) checks below only need length

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['svg.fonttype'] = 'none'

    fig = plt.figure(figsize=(4, 12))
    fig.suptitle(f'Kim-Forger ({var_name})', fontweight='bold', fontsize=16)

    # 1) Time Course (y-range auto-scaled per variable, colored by curvature sign)
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(T_norm, R_neg, '-m', linewidth=1.2)
    ax1.plot(T_norm, R_pos, '-c', linewidth=1.2)

    if len(T) > 0:
        Rmax1, Rmin1 = np.nanmax(R), np.nanmin(R)
        Rmean1 = (Rmax1 + Rmin1) / 2
        Ramp1 = (Rmax1 - Rmin1) / 2
        ymax1 = Rmean1 + Ramp1 * 1.1
        ymin1 = Rmean1 - Ramp1 * 1.1
        ax1.set_xlim([0, 144])
        ax1.set_xticks(t_xticks)
        ax1.set_ylim([ymin1, ymax1])
        ax1.set_yticks([ymin1, Rmean1, ymax1])

    ax1.set_box_aspect(1 / 2)
    ax1.tick_params(direction='out')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel('t', fontweight='normal')
    ax1.set_ylabel('x', fontweight='normal')

    # 2) Phase Space (square, auto-scaled per variable)
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(R_neg, D_neg, '-m', linewidth=1.2)
    ax2.plot(R_pos, D_pos, '-c', linewidth=1.2)

    valid2 = ~np.isnan(R) & ~np.isnan(D)
    if np.any(valid2):
        xmax2, xmin2, ymax2, ymin2 = pu.plot_axis(R[valid2], D[valid2], 1.1, 1.1, 's')
        ax2.set_xlim([xmin2, xmax2])
        ax2.set_ylim([ymin2, ymax2])
        ax2.set_xticks([xmin2, (xmin2 + xmax2) / 2, xmax2])
        ax2.set_yticks([ymin2, (ymin2 + ymax2) / 2, ymax2])
    ax2.set_box_aspect(1)
    ax2.tick_params(direction='out')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel('x', fontweight='normal')
    ax2.set_ylabel('dx/dt', fontweight='normal')

    # 3) Curvature+ (log scale, fixed 1e-2 to 1e3)
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
        ax3.set_yticklabels(power_labels([1e-2, 1, 1e2]))

    ax3.tick_params(direction='out')
    ax3.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_box_aspect(1 / 2)
    ax3.set_xlabel('t', fontweight='normal')
    ax3.set_ylabel(r'$\kappa$', fontweight='normal')

    # 4) Curvature- (log scale, fixed 1e-2 to 1e3)
    ax4 = fig.add_subplot(4, 1, 4)
    if np.any(mask_neg):
        ax4.semilogy(T_norm[mask_neg], np.abs(K[mask_neg]), '-m', linewidth=1.2)
    else:
        ax4.axhline(1e5, color='k', linestyle='--')

    if len(T) > 0:
        ax4.set_xlim([0, 144])
        ax4.set_xticks(t_xticks)
        ax4.set_ylim([1e-2, 1e2])
        ax4.set_yticks([1e-2, 1, 1e2])
        ax4.set_yticklabels(power_labels([1e-2, 1, 1e2], negative=True))

    ax4.tick_params(direction='out')
    ax4.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_box_aspect(1 / 2)
    ax4.set_xlabel('t', fontweight='normal')
    ax4.set_ylabel(r'$\kappa$', fontweight='normal')

    plt.tight_layout()

    fig_name = f"KF_{var_name}"
    pu.save_fig(fig, __file__, fig_name)

    eps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img', f"{fig_name}.eps")
    fig.savefig(eps_path, transparent=True, bbox_inches='tight', format='eps')
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FigS3 Kim-Forger Analysis (all nucleus/cytoplasm mRNA variables)")
    parser.add_argument('-f', '--force', action='store_true', help="Force recalculation (bypass caches)")
    args = parser.parse_args()
    force_recalc = args.force

    for var_name in VARIABLES:
        print(f"=== Processing {var_name} ===")
        process_variable(var_name, force_recalc)
