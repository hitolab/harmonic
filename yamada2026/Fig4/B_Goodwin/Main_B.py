import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import Lib.harmonicity as hl
import Lib.plot_utils as pu

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
L_vals = [0.5]
N_EXP = 17
DT = 1e-3
VAR_NAMES = ['x', 'y', 'z']
MP_DPS = 32  # mpmath precision (decimal digits) for the high-precision Runge variant
model_name = "goodwin"

MP_TRAJ_COLUMNS = ['r_data_mp', 'd_data_mp', 'k_data_mp']
MP_SUMMARY_COLUMNS = ['K_min_abs_mp']


def goodwin_rhs_mp(state, params):
    """Same equations as Models/goodwin/model_goodwin.cpp, for runge_iterate_mp."""
    x, y, z = state
    lamb, n = params
    dx = 1 / (1 + z ** n) - lamb * x
    dy = x - lamb * y
    dz = y - lamb * z
    return (dx, dy, dz)


# -----------------------------------------------------------------------------
# Stage 1: state classification (B_states.csv)
# -----------------------------------------------------------------------------
def _load_existing_b_states(csv_path):
    """Returns the existing B_states.csv DataFrame if present and if it
    already covers exactly the current L_vals (at N_EXP); otherwise None."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    expected = {round(float(l), 2) for l in L_vals}
    have = {round(float(l), 2) for l in df['L']}
    if expected != have or (df['N'] != N_EXP).any():
        return None
    return df


def _var_cache_path(script_dir, var_name, l_val):
    cache_name = f"{model_name}_fig3B_{var_name}_L{l_val}_N{N_EXP}"
    return os.path.join(script_dir, 'plot_data', cache_name + '.csv')


def _cache_is_fresh(cache_path):
    """Cheap header-only check: does this Stage-2 cache already include the
    embedded summary columns (has_k_inversion), as opposed to an older
    float64-only cache?"""
    if not os.path.exists(cache_path):
        return False
    try:
        header = pd.read_csv(cache_path, comment='#', nrows=0)
    except Exception:
        return False
    return 'has_k_inversion' in header.columns


def _all_var_caches_fresh(states_df, script_dir):
    for row in states_df.itertuples():
        if row.state == 'limitcycle':
            for var_name in VAR_NAMES:
                if not _cache_is_fresh(_var_cache_path(script_dir, var_name, row.L)):
                    return False
    return True


# -----------------------------------------------------------------------------
# Stage 2: high-precision trajectory + curvature summary, per (var, L)
# -----------------------------------------------------------------------------
def _compute_summary(K_final_float, K_mp_trim):
    """has_k_inversion (does K change sign) and the min |K| -- kept as the
    original mpmath.mpf value (not re-derived from the float64 copy) so the
    single selected scalar retains full MP_DPS-digit precision."""
    valid_mask = ~np.isnan(K_final_float)
    if not np.any(valid_mask):
        return False, None
    has_pos = np.any(K_final_float[valid_mask] > 0)
    has_neg = np.any(K_final_float[valid_mask] < 0)
    has_k_inversion = bool(has_pos and has_neg)

    abs_arr = np.abs(K_final_float)
    abs_arr[~valid_mask] = np.inf
    min_idx = int(np.argmin(abs_arr))
    k_at_min = K_mp_trim[min_idx]
    K_min_abs_mp = abs(k_at_min) if k_at_min is not None else None
    return has_k_inversion, K_min_abs_mp


def compute_condition_data(l_val, R_raw_d, force_recalc=False):
    """
    Stage 2 for one L value: produces/loads the unified high-precision
    (mpmath) trajectory cache for x, y, z simultaneously (they come out of
    the same RK4 integration), plus the has_k_inversion/K_min_abs summary
    for each variable -- computed exactly once (on a cache miss) and cached
    alongside the raw trajectory. On a cache hit nothing is recomputed.
    Returns a dict: var_name -> {'T_norm','r_data','d_data','k_data'
    (float64), 'has_k_inversion' (bool), 'K_min_abs' (float)}.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_paths = {v: _var_cache_path(script_dir, v, l_val) for v in VAR_NAMES}

    if not force_recalc and all(_cache_is_fresh(p) for p in cache_paths.values()):
        result = {}
        for var_name in VAR_NAMES:
            cache_name = f"{model_name}_fig3B_{var_name}_L{l_val}_N{N_EXP}"
            cached = hl.load_plot_data_mp(
                cache_name, mp_columns=MP_TRAJ_COLUMNS + MP_SUMMARY_COLUMNS, dps=MP_DPS
            )
            result[var_name] = {
                'T_norm': cached['T_norm'],
                'r_data': hl.mp_to_float_array(cached['r_data_mp']),
                'd_data': hl.mp_to_float_array(cached['d_data_mp']),
                'k_data': hl.mp_to_float_array(cached['k_data_mp']),
                'has_k_inversion': bool(cached['has_k_inversion'][0]),
                'K_min_abs': float(cached['K_min_abs_mp'][0]) if cached['K_min_abs_mp'][0] is not None else np.nan,
            }
        return result

    print(f"Running high-precision simulation for L={l_val}, N={N_EXP}...")
    params = [l_val, N_EXP]

    # 1) Re-verify/refine convergence at high precision (mpmath), starting
    #    from the already double-precision-converged checkpoint.
    n_steps = len(R_raw_d[0])
    checkpoint = [R_raw_d[0][0], R_raw_d[1][0], R_raw_d[2][0]]
    print(f"Re-verifying convergence at {MP_DPS}-digit precision...")
    traj_conv, state_code, period_mp = hl.runge_cycle_mp(
        goodwin_rhs_mp, checkpoint, DT, params=(l_val, N_EXP), dps=MP_DPS
    )
    refined_state = traj_conv[-1]

    # 2) Re-integrate the same span with the high-precision Runge variant.
    print(f"Re-integrating {n_steps} steps at {MP_DPS}-digit precision...")
    traj_mp = hl.runge_iterate_mp(
        goodwin_rhs_mp, refined_state, DT, n_steps - 1,
        params=(l_val, N_EXP), dps=MP_DPS
    )
    R_mp = [[state[vi] for state in traj_mp] for vi in range(3)]
    D_mp = [hl.central_diff_mp(R_mp[vi], DT, order=1, dps=MP_DPS) for vi in range(3)]
    K_mp = [hl.compute_curvature_mp(R_mp[vi], D_mp[vi], DT, dps=MP_DPS) for vi in range(3)]

    R_raw = [hl.mp_to_float_array(r) for r in R_mp]
    D_raw = [hl.mp_to_float_array(d) for d in D_mp]
    K_raw = [hl.mp_to_float_array(k) for k in K_mp]

    T_raw = np.linspace(0, (len(R_raw[0]) - 1) * DT, len(R_raw[0]))
    T_final, R_final, D_final, K_final = hl.align_and_trim(T_raw, R_raw, D_raw, K_raw)

    # Recover the [start_trim, end_idx] sample-index window align_and_trim
    # used, so the same window can be applied to the parallel mpmath lists
    # without ever converting them to float first.
    start_trim = int(round(T_final[0] / DT))
    end_idx = int(round(T_final[-1] / DT))  # inclusive
    R_mp_trim = [r[start_trim:end_idx + 1] for r in R_mp]
    D_mp_trim = [d[start_trim:end_idx + 1] for d in D_mp]
    K_mp_trim = [k[start_trim:end_idx + 1] for k in K_mp]

    prom = 0.1 * (np.max(R_final[2]) - np.min(R_final[2]))
    peaks, _ = scipy.signal.find_peaks(R_final[2], prominence=prom)
    if len(peaks) >= 3:
        idx_start, idx_end = peaks[-3], peaks[-1]
    else:
        idx_start, idx_end = 0, len(T_final) - 1

    T_final = T_final[idx_start:idx_end + 1]
    R_final = [r[idx_start:idx_end + 1] for r in R_final]
    D_final = [d[idx_start:idx_end + 1] for d in D_final]
    K_final = [k[idx_start:idx_end + 1] for k in K_final]
    R_mp_trim = [r[idx_start:idx_end + 1] for r in R_mp_trim]
    D_mp_trim = [d[idx_start:idx_end + 1] for d in D_mp_trim]
    K_mp_trim = [k[idx_start:idx_end + 1] for k in K_mp_trim]
    T_norm_sim = np.linspace(0, 2, len(T_final))

    result = {}
    for var_name in VAR_NAMES:
        vi = VAR_NAMES.index(var_name)
        has_k_inversion, K_min_abs_mp = _compute_summary(K_final[vi], K_mp_trim[vi])

        cache_name = f"{model_name}_fig3B_{var_name}_L{l_val}_N{N_EXP}"
        hl.save_plot_data_mp(
            {
                'T_norm': T_norm_sim,
                'r_data_mp': R_mp_trim[vi], 'd_data_mp': D_mp_trim[vi], 'k_data_mp': K_mp_trim[vi],
                'has_k_inversion': [float(has_k_inversion)],
                'K_min_abs_mp': [K_min_abs_mp],
            },
            cache_name, mp_columns=MP_TRAJ_COLUMNS + MP_SUMMARY_COLUMNS, dps=MP_DPS
        )

        result[var_name] = {
            'T_norm': T_norm_sim,
            'r_data': R_final[vi], 'd_data': D_final[vi], 'k_data': K_final[vi],
            'has_k_inversion': has_k_inversion,
            'K_min_abs': float(K_min_abs_mp) if K_min_abs_mp is not None else np.nan,
        }
    return result


# -----------------------------------------------------------------------------
# Stage 3: plotting
# -----------------------------------------------------------------------------
def plot_variable(var_name, all_var_data):
    n_cols = len(L_vals)
    fig = plt.figure(figsize=(4 * n_cols, 12))

    for col_idx, l_val in enumerate(L_vals):
        data = all_var_data[var_name][l_val]
        T_norm = data['T_norm']
        r_data = data['r_data']
        d_data = data['d_data']
        k_data = data['k_data']

        mask1 = k_data < 0
        mask2 = k_data > 0
        transitions = mask2[:-1] != mask2[1:]
        mask2_ext = mask2.copy(); mask1_ext = mask1.copy()
        mask2_ext[:-1] |= transitions; mask2_ext[1:] |= transitions
        mask1_ext[:-1] |= transitions; mask1_ext[1:] |= transitions

        r_neg = np.where(mask1_ext, r_data, np.nan)
        r_pos = np.where(mask2_ext, r_data, np.nan)
        d_neg = np.where(mask1_ext, d_data, np.nan)
        d_pos = np.where(mask2_ext, d_data, np.nan)

        # --- Row 1: time series ---
        ax1 = plt.subplot(4, n_cols, col_idx + 1)
        ax1.plot(T_norm, r_neg, '-k', linewidth=1.2)
        ax1.plot(T_norm, r_pos, '-c', linewidth=1.2)
        ymin, ymax = 0, 3
        yt = [0, 3]
        ax1.set_xlim(0, 2)
        ax1.set_ylim(ymin, ymax)
        ax1.set_yticks(yt)
        ax1.set_xticks([0, 1, 2])
        ax1.set_box_aspect(1 / 2)
        ax1.set_title(f"L={l_val}")
        if col_idx == 0:
            ax1.set_ylabel(var_name)

        # --- Row 2: phase portrait ---
        ax2 = plt.subplot(4, n_cols, col_idx + 1 + n_cols)
        ax2.plot(r_neg, d_neg, '-k', linewidth=1.2)
        ax2.plot(r_pos, d_pos, '-c', linewidth=1.2)
        xmin, xmax = (-1, 1) if var_name == 'x' else (0, 2)
        ymin, ymax = -1, 1
        xt = [xmin, xmax]
        yt2 = [-1, 1]
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.set_xticks(xt)
        ax2.set_yticks(yt2)
        ax2.set_box_aspect(1)

        # --- Row 3/4 shared range: fixed at 10^-2 .. 10^2 (absolute value) ---
        k_plot = scipy.signal.medfilt(k_data, kernel_size=21)
        k_pos = np.where(k_plot > 0, k_plot, np.nan)
        k_neg = np.where(k_plot < 0, -k_plot, np.nan)
        k_exps = [-2, -1, 0, 1, 2]
        k_ylim = (10.0 ** k_exps[0], 10.0 ** k_exps[-1])
        k_yticks = [10.0 ** e for e in k_exps]

        # --- Row 3: K > 0 (cyan) ---
        ax3 = plt.subplot(4, n_cols, col_idx + 1 + 2 * n_cols)
        if np.any(mask2):
            ax3.semilogy(T_norm, k_pos, '-c')
        ax3.set_xlim(0, 2)
        ax3.set_ylim(*k_ylim)
        ax3.set_yticks(k_yticks)
        ax3.set_yticklabels([f'10$^{{{e}}}$' for e in k_exps])
        ax3.set_xticks([0, 1, 2])
        ax3.minorticks_off()
        ax3.set_box_aspect(1 / 2)

        # --- Row 4: K < 0 (black) ---
        ax4 = plt.subplot(4, n_cols, col_idx + 1 + 3 * n_cols)
        if np.any(mask1):
            ax4.semilogy(T_norm, k_neg, '-k')
        ax4.set_xlim(0, 2)
        ax4.set_ylim(*k_ylim)
        ax4.set_yticks(k_yticks)
        ax4.set_yticklabels([f'-10$^{{{e}}}$' for e in k_exps])
        ax4.invert_yaxis()
        ax4.set_xticks([0, 1, 2])
        ax4.minorticks_off()
        ax4.set_box_aspect(1 / 2)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(direction='out')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    pu.save_fig(fig, __file__, f"Fig4_B_{var_name}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fig4_B Goodwin Analysis (x, y, z)")
    parser.add_argument('-f', '--force', action='store_true', help="Force recalculation (bypass caches)")
    args = parser.parse_args()
    force_recalc = args.force

    script_dir = os.path.dirname(os.path.abspath(__file__))
    states_csv_path = os.path.join(script_dir, 'B_states.csv')

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['agg.path.chunksize'] = 10000

    all_var_data = {var_name: {} for var_name in VAR_NAMES}  # var_name -> {l_val -> data}

    existing_states = None if force_recalc else _load_existing_b_states(states_csv_path)

    if existing_states is not None and _all_var_caches_fresh(existing_states, script_dir):
        print("Stage 1/2 already complete for the current L_vals -- "
              "skipping computation, plotting only.")
        states_df = existing_states
        for row in states_df.itertuples():
            if row.state == 'limitcycle':
                cond_data = compute_condition_data(row.L, None)
                for var_name in VAR_NAMES:
                    all_var_data[var_name][row.L] = cond_data[var_name]
    else:
        records = []
        state_labels = {0: 'unfinished', 1: 'limitcycle', 2: 'fixed'}
        for l_val in L_vals:
            params = [l_val, N_EXP]
            # Fast double-precision RungeCycle: classifies the state AND
            # gives the checkpoint + step count the mp pipeline needs.
            R_raw_d, state_code, per_returned = hl.simulate(model_name, DT, params, 5, return_state=True)

            if len(R_raw_d) == 0 or len(R_raw_d[0]) == 0:
                state = 'unfinished'
                per = np.nan
                init_vals = [np.nan, np.nan, np.nan]
            else:
                state = state_labels.get(state_code, 'error')
                per = per_returned if state == 'limitcycle' else np.nan
                init_vals = [R_raw_d[0][-1], R_raw_d[1][-1], R_raw_d[2][-1]]

            records.append({
                'L': l_val, 'N': N_EXP, 'state': state, 'period': per,
                'init_x': init_vals[0], 'init_y': init_vals[1], 'init_z': init_vals[2],
            })

            if state == 'limitcycle':
                cond_data = compute_condition_data(l_val, R_raw_d, force_recalc=force_recalc)
                for var_name in VAR_NAMES:
                    all_var_data[var_name][l_val] = cond_data[var_name]
            else:
                print(f"Warning: L={l_val}, N={N_EXP} is not a limit cycle (state={state}) -- skipping.")

        states_df = pd.DataFrame(records, columns=['L', 'N', 'state', 'period', 'init_x', 'init_y', 'init_z'])
        states_df.to_csv(states_csv_path, index=False)
        print(f"\nSaved {states_csv_path}")

    print("Generating figures...")
    for var_name in VAR_NAMES:
        plot_variable(var_name, all_var_data)


if __name__ == "__main__":
    main()
