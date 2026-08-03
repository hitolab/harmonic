import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FigS3 (for Main_S3's prepare_variable_data/power_labels) and Harmonicity root (for Lib)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import Lib.plot_utils as pu
from Main_S3 import power_labels, prepare_variable_data

# -----------------------------------------------------------------------------
# All 180 Kim-Forger DetailedModel state names (Data/Kim-Forger/DetailedModel.m,
# 'states' output), in model order -- conveniently already grouped in runs of
# 10 in the source itself.
# -----------------------------------------------------------------------------
ALL_STATES = [
    'GR', 'G', 'GrR', 'Gr', 'GcR', 'Gc', 'GBR', 'GB', 'GBRb', 'GBb',
    'MnPo', 'McPo', 'MnPt', 'McPt', 'MnRt', 'McRt', 'MnRev', 'McRev', 'MnRo', 'McRo',
    'MnB', 'McB', 'MnNp', 'McNp', 'B', 'Cl', 'BC', 'cyrev', 'revn', 'cyrevg',
    'revng', 'cyrevgp', 'revngp', 'cyrevp', 'revnp', 'gto', 'x00001', 'x00011', 'x00100', 'x00110',
    'x00200', 'x00210', 'x01000', 'x01010', 'x01011', 'x02000', 'x02010', 'x02011', 'x10000', 'x10100',
    'x20000', 'x20010', 'x20011', 'x20100', 'x20110', 'x20111', 'x21000', 'x21010', 'x21011', 'x21100',
    'x21110', 'x21111', 'x22000', 'x22010', 'x22011', 'x22100', 'x22110', 'x22111', 'x30000', 'x30100',
    'x30200', 'x30300', 'x40000', 'x40010', 'x40011', 'x40100', 'x40110', 'x40111', 'x40200', 'x40210',
    'x40211', 'x40300', 'x40310', 'x40311', 'x41000', 'x41010', 'x41011', 'x41100', 'x41110', 'x41111',
    'x41200', 'x41210', 'x41211', 'x41300', 'x41310', 'x41311', 'x42000', 'x42010', 'x42011', 'x42100',
    'x42110', 'x42111', 'x42200', 'x42210', 'x42211', 'x42300', 'x42310', 'x42311', 'x50000', 'x50010',
    'x50011', 'x50100', 'x50110', 'x50111', 'x50200', 'x50210', 'x50211', 'x50300', 'x50310', 'x50311',
    'x51000', 'x51010', 'x51011', 'x51100', 'x51110', 'x51111', 'x51200', 'x51210', 'x51211', 'x51300',
    'x51310', 'x51311', 'x52000', 'x52010', 'x52011', 'x52100', 'x52110', 'x52111', 'x52200', 'x52210',
    'x52211', 'x52300', 'x52310', 'x52311', 'x60000', 'x60010', 'x60011', 'x60100', 'x60110', 'x60111',
    'x60200', 'x60210', 'x60211', 'x60300', 'x60310', 'x60311', 'x61000', 'x61010', 'x61011', 'x61100',
    'x61110', 'x61111', 'x61200', 'x61210', 'x61211', 'x61300', 'x61310', 'x61311', 'x62000', 'x62010',
    'x62011', 'x62100', 'x62110', 'x62111', 'x62200', 'x62210', 'x62211', 'x62300', 'x62310', 'x62311',
]

CHUNK_SIZE = 10


def _resolve_file_stems(names):
    """Mirrors Data/Kim-Forger/main.m's case-insensitive collision handling:
    if two state names differ only by case (e.g. 'GR' vs 'Gr'), the exported
    CSV filename disambiguates with a '_caseN' suffix -- macOS's default
    APFS volume is case-insensitive, so two plain 'KF_GR.csv'/'KF_Gr.csv'
    would silently collide into the same file. Returns a dict: state name ->
    actual file stem (used for both the CSV filename and the plot_data
    cache key), identity for every non-colliding name."""
    groups = {}
    for n in names:
        groups.setdefault(n.lower(), []).append(n)
    stems = {}
    for members in groups.values():
        if len(members) == 1:
            stems[members[0]] = members[0]
        else:
            for rank, m in enumerate(members, start=1):
                stems[m] = f"{m}_case{rank}"
    return stems


FILE_STEMS = _resolve_file_stems(ALL_STATES)


def plot_chunk(chunk_idx, var_names, all_data):
    """One figure per chunk of CHUNK_SIZE variables: rows = t-x / phase space /
    K>0 / K<0 (same layout as Main_S3.py), columns = variables in this chunk.
    Missing-data variables (no source CSV yet) get a blank labeled panel."""
    cols = len(var_names)
    rows = 4
    fig = plt.figure(figsize=(2.2 * cols, 2.2 * rows))

    for col_idx, var_name in enumerate(var_names):
        data = all_data.get(var_name)

        ax1 = fig.add_subplot(rows, cols, 0 * cols + col_idx + 1)
        ax2 = fig.add_subplot(rows, cols, 1 * cols + col_idx + 1)
        ax3 = fig.add_subplot(rows, cols, 2 * cols + col_idx + 1)
        ax4 = fig.add_subplot(rows, cols, 3 * cols + col_idx + 1)

        if data is None:
            ax1.set_title(f"{var_name}\n(no data)", fontsize=8)
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xticks([]); ax.set_yticks([])
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            continue

        T_norm, t_xticks = data['T_norm'], data['t_xticks']
        R, D, K = data['R'], data['D'], data['K']
        mask_pos, mask_neg = data['mask_pos'], data['mask_neg']
        R_neg, R_pos, D_neg, D_pos = data['R_neg'], data['R_pos'], data['D_neg'], data['D_pos']

        # --- Row 1: t-x ---
        ax1.plot(T_norm, R_neg, '-m', linewidth=1.0)
        ax1.plot(T_norm, R_pos, '-c', linewidth=1.0)
        if len(T_norm) > 0:
            _, _, ymax1, ymin1 = pu.plot_axis(T_norm, R, 1.0, 1.2, 'fc')
            ax1.set_xlim([0, 144]); ax1.set_xticks(t_xticks)
            ax1.set_ylim([ymin1, ymax1])
        ax1.set_box_aspect(1 / 2)
        ax1.set_title(var_name, fontsize=8)
        if col_idx == 0:
            ax1.set_ylabel('x')

        # --- Row 2: phase space (x, dx/dt) ---
        ax2.plot(R_neg, D_neg, '-m', linewidth=1.0)
        ax2.plot(R_pos, D_pos, '-c', linewidth=1.0)
        valid2 = ~np.isnan(R) & ~np.isnan(D)
        if np.any(valid2):
            xmax2, xmin2, ymax2, ymin2 = pu.plot_axis(R[valid2], D[valid2], 1.2, 1.2, 's')
            ax2.set_xlim([xmin2, xmax2]); ax2.set_ylim([ymin2, ymax2])
        ax2.set_box_aspect(1)
        if col_idx == 0:
            ax2.set_ylabel('dx/dt')

        # --- Row 3: K > 0 ---
        if np.any(mask_pos):
            ax3.semilogy(T_norm[mask_pos], K[mask_pos], '-c', linewidth=1.0)
        if len(T_norm) > 0:
            ax3.set_xlim([0, 144]); ax3.set_xticks(t_xticks)
            ax3.set_ylim([1e-2, 1e2]); ax3.set_yticks([1e-2, 1, 1e2])
            ax3.set_yticklabels(power_labels([1e-2, 1, 1e2]))
        ax3.minorticks_off(); ax3.set_box_aspect(1 / 2)
        if col_idx == 0:
            ax3.set_ylabel(r'$\kappa$ (K>0)')

        # --- Row 4: K < 0 ---
        if np.any(mask_neg):
            ax4.semilogy(T_norm[mask_neg], np.abs(K[mask_neg]), '-m', linewidth=1.0)
        if len(T_norm) > 0:
            ax4.set_xlim([0, 144]); ax4.set_xticks(t_xticks)
            ax4.set_ylim([1e-2, 1e2]); ax4.set_yticks([1e-2, 1, 1e2])
            ax4.set_yticklabels(power_labels([1e-2, 1, 1e2], negative=True))
        ax4.minorticks_off(); ax4.set_box_aspect(1 / 2)
        if col_idx == 0:
            ax4.set_ylabel(r'$\kappa$ (K<0)')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(direction='out', labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig_name = f"Scan_{chunk_idx:02d}_{var_names[0]}-{var_names[-1]}"
    pu.save_fig(fig, __file__, fig_name)
    plt.close(fig)
    print(f"Saved: {fig_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan all 180 Kim-Forger DetailedModel variables for curvature inflection points"
    )
    parser.add_argument('-f', '--force', action='store_true', help="Force recalculation (bypass caches)")
    args = parser.parse_args()

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['agg.path.chunksize'] = 10000

    records = []
    all_data = {}
    for var_name in ALL_STATES:
        print(f"=== Processing {var_name} ===")
        data = prepare_variable_data(FILE_STEMS[var_name], args.force)
        all_data[var_name] = data
        if data is None:
            records.append({'variable': var_name, 'has_inflection': None})
            continue
        has_pos = bool(np.any(data['mask_pos']))
        has_neg = bool(np.any(data['mask_neg']))
        records.append({'variable': var_name, 'has_inflection': has_pos and has_neg})

    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.DataFrame(records, columns=['variable', 'has_inflection'])
    df.to_csv(os.path.join(script_dir, 'inflection_summary.csv'), index=False)
    print(f"\nSaved {os.path.join(script_dir, 'inflection_summary.csv')}")
    print(df['has_inflection'].value_counts(dropna=False))

    print("\nGenerating scan figures (10 variables per figure)...")
    for chunk_idx, start in enumerate(range(0, len(ALL_STATES), CHUNK_SIZE)):
        chunk_vars = ALL_STATES[start:start + CHUNK_SIZE]
        plot_chunk(chunk_idx, chunk_vars, all_data)


if __name__ == "__main__":
    main()
