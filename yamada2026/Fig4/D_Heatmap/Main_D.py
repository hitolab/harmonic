import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.signal

# Ensure Lib and Models are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import Lib.harmonicity as hl
import Lib.plot_utils as pu
from Models.goodwin.model_goodwin_mp import goodwin_rhs_mp, lambda_boundary, boundary_expr, MP_DPS

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = 'goodwin'
DT = 1e-3
TARGET_CYCLES = 2      # analysis window: last 2 full cycles of the limit cycle
NUM_CYCLES_HP = TARGET_CYCLES + 2  # cycles to record in the shared high-precision
                                    # trajectory cache

Lmgn = 0.1
N_range = np.arange(5, 26, 2)    # 5, 7, ..., 23
L_range = np.arange(0.1, 1.2, Lmgn)  # 0.1, 0.2, ..., 1.0

model_name = MODEL_NAME

# Stage-2 cache columns (embedded once per condition, never recomputed on a
# cache hit): the curvature-derived summary metrics, in mpmath. The recorded
# trajectory itself is in period-normalized time (tau = t/T, see
# compute_condition_data) -- a cache missing 'period_post' predates that
# scheme and is treated as fully stale (trajectory and metrics both).
MP_SUMMARY_COLUMNS = ['kh_val_signed_mp', 'perimeter_scaled_mp']
MP_TRAJ_COLUMNS = ['X_mp', 'DotX_mp', 'K_mp']


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _axis_range_20pct(data):
    """Tight span of `data` padded by 10% on each side (20% total margin).
    Independent per subplot -- ranges are not meant to match across panels."""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return -1.0, 1.0
    dmin, dmax = np.min(data), np.max(data)
    span = dmax - dmin
    if span == 0:
        span = abs(dmax) if dmax != 0 else 1.0
    margin = span * 0.1
    return dmin - margin, dmax + margin


def get_n_cycle_indices(z, target_cycles):
    prominence = (np.nanmax(z) - np.nanmin(z)) * 0.1
    peaks, _ = scipy.signal.find_peaks(z, prominence=prominence)
    if len(peaks) >= target_cycles + 1:
        return peaks[0], peaks[target_cycles]
    if len(peaks) >= 2:
        return peaks[0], peaks[-1]
    return 0, len(z) - 1


# -----------------------------------------------------------------------------
# Stage 2: unified high-precision trajectory + curvature-derived summaries
# -----------------------------------------------------------------------------
def _compute_summary_metrics(X_mp, DotX_mp, K_mp):
    """
    Computes the curvature-derived summary metrics from one (already
    boundary-trimmed) high-precision X/dX/K trajectory:
      - has_k_inversion: does the (unscaled) curvature ever change sign?
      - kh_val / perimeter_scaled: perimeter-normalized minimum |curvature|
        over the TARGET_CYCLES window (used for the (N,L) heatmap). The
        legacy (10*pi)-normalized "KH_scaled" metric has been dropped.

    The perimeter-rescaled curvature is obtained by exact analytic scaling
    (K_scaled = K_mp * S) rather than by re-differentiating the rescaled
    (x/S, dx/dt/S) coordinates from scratch. Re-differentiating reintroduces
    fresh catastrophic-cancellation noise near near-zero curvature points --
    confirmed empirically on N=23, L=0.1: the re-differentiated values were
    three orders of magnitude noisier, with essentially random sign, compared
    to the analytically exact K_mp*S. Curvature scales linearly under a
    uniform coordinate rescale (kappa -> kappa*S for x -> x/S), so multiplying
    the already-clean K_mp by the scalar S is exact and introduces no new
    cancellation -- and is far cheaper (no extra central-difference/curvature
    pass over the whole trajectory).
    """
    import mpmath as mp

    K_arr = hl.mp_to_float_array(K_mp)
    has_k_inversion = bool(np.any(K_arr > 0) and np.any(K_arr < 0))
    # "Always negative": curvature never goes positive anywhere in the
    # (boundary-trimmed) trajectory -- i.e. not has_k_inversion AND there is
    # no positive value at all (as opposed to e.g. an all-NaN/degenerate case).
    valid_K = K_arr[~np.isnan(K_arr)]
    always_negative = bool(valid_K.size > 0 and not np.any(valid_K > 0))

    X_arr = hl.mp_to_float_array(X_mp)
    DotX_arr = hl.mp_to_float_array(DotX_mp)

    kh_val = None
    perimeter_postnorm = None
    perimeter_scaled = None
    if len(X_arr) > 1:
        p1, p2 = get_n_cycle_indices(X_arr, 1)
        X_cycle = X_arr[p1:p2]
        D_cycle = DotX_arr[p1:p2]
        diff_x = X_cycle[1:] - X_cycle[:-1]
        diff_dx = D_cycle[1:] - D_cycle[:-1]
        perimeter = np.sum(np.sqrt(diff_x ** 2 + diff_dx ** 2))
        # Perimeter of the (already tau-normalized, still space-raw) phase
        # curve -- the "Perimeter_PostNorm" diagnostic column.
        perimeter_postnorm = float(perimeter) if not np.isnan(perimeter) else None
        S = perimeter / 1.0 if perimeter > 1e-12 else 1.0
        if not np.isnan(S):
            S_mp = mp.mpf(repr(S))
            k_scaled_mp = [(v * S_mp) if v is not None else None for v in K_mp]

            p_start, p_end = get_n_cycle_indices(X_arr, TARGET_CYCLES)
            window = k_scaled_mp[p_start:p_end]
            valid2 = [v for v in window if v is not None]
            if valid2:
                # Signed value (not abs()): the entry with smallest magnitude,
                # keeping its original sign (always negative here, since the
                # curvature never changes sign -- see always_negative above).
                kh_val = min(valid2, key=abs)

            # Rescaled perimeter, for the record -- still a plain sum of
            # positive magnitudes (no cancellation risk), so float64 is fine.
            X_scaled2 = X_arr / S
            DotX_scaled2 = DotX_arr / S
            p1_s, p2_s = get_n_cycle_indices(X_scaled2, 1)
            X_cycle_s = X_scaled2[p1_s:p2_s]
            D_cycle_s = DotX_scaled2[p1_s:p2_s]
            diff_x_s = X_cycle_s[1:] - X_cycle_s[:-1]
            diff_dx_s = D_cycle_s[1:] - D_cycle_s[:-1]
            perim_s = np.sum(np.sqrt(diff_x_s ** 2 + diff_dx_s ** 2))
            perimeter_scaled = mp.mpf(repr(perim_s)) if not np.isnan(perim_s) else None

    return {
        'has_k_inversion': has_k_inversion,
        'always_negative': always_negative,
        'kh_val': kh_val,
        'perimeter_postnorm': perimeter_postnorm,
        'perimeter_scaled': perimeter_scaled,
    }


def compute_condition_data(n_val, l_val, R_raw, force_recalc=False):
    """
    Stage 2: produces/loads the unified high-precision (mpmath) trajectory
    cache for one (N, L) limitcycle condition, in PERIOD-NORMALIZED time
    (tau = t / T, T = the condition's measured limit-cycle period): the ODE
    right-hand side is scaled by T before integrating (dx/dtau = T * f(x)),
    so tau advances by exactly 1 per raw period without touching DT itself.
    X/dX/dtau/curvature-in-tau plus the curvature-derived summary metrics
    (has_k_inversion, kh_val, perimeter_scaled/postnorm/pre) are computed
    once and cached together. On a fresh cache hit nothing is recomputed,
    matching the "plot only if the data CSVs are already complete" execution
    model. A cache predating this scheme (missing 'period_post') means a
    different trajectory definition, not just stale metrics, so it's treated
    as a full cache miss -- everything below is recomputed from scratch.
    All curvature work is done in mpmath (MP_DPS digits) end to end; the
    raw arrays and the KH/perimeter_scaled summary scalars are stored on
    disk as full-precision decimal strings (via save_plot_data_mp/
    load_plot_data_mp). The remaining diagnostics (perimeter_pre,
    perimeter_postnorm, period_post, state_post) are plain float64 -- none
    of them feed the precision-critical KH pipeline, they're just recorded
    for the heatmap CSV.
    """
    import mpmath as mp

    cache_name = hl.hp_cache_name(MODEL_NAME, n_val, l_val)
    mp_cols = MP_TRAJ_COLUMNS + MP_SUMMARY_COLUMNS
    cached = hl.load_plot_data_mp(cache_name, mp_columns=mp_cols, dps=MP_DPS) if not force_recalc else None

    have_traj = (
        cached is not None and 'X_mp' in cached and len(cached['X_mp']) > 0
        and 'period_post' in cached
    )

    if have_traj:
        T_final = cached['T']
        X_mp, DotX_mp, K_mp = cached['X_mp'], cached['DotX_mp'], cached['K_mp']
        has_k_inversion = bool(cached['has_k_inversion'][0])
        always_negative = bool(cached['always_negative'][0])
        kh_val = cached['kh_val_signed_mp'][0]
        perimeter_scaled = cached['perimeter_scaled_mp'][0]
        perimeter_postnorm = float(cached['perimeter_postnorm'][0])
        perimeter_pre = float(cached['perimeter_pre'][0])
        period_post = float(cached['period_post'][0])
        state_post = int(cached['state_post'][0])
    else:
        # 1) Re-verify convergence at high precision from the already-converged
        #    double checkpoint, in the ORIGINAL (unscaled) time -- this both
        #    refines the on-cycle state and yields a high-precision period T.
        checkpoint = [R_raw[0][0], R_raw[1][0], R_raw[2][0]]
        traj_conv, state_code_pre, T_hp = hl.runge_cycle_mp(
            goodwin_rhs_mp, checkpoint, DT, params=(l_val, n_val), dps=MP_DPS
        )
        refined_state = traj_conv[-1]

        # Pre-normalization perimeter: cheap float64 diagnostic over the last
        # measured raw period of the (unscaled) convergence trajectory --
        # not part of the precision-critical KH pipeline, so float64 is fine.
        n_last = max(2, int(round(T_hp / DT))) if T_hp else len(traj_conv) - 1
        x_tail = np.array([float(s[0]) for s in traj_conv[-(n_last + 1):]])
        dot_tail = np.gradient(x_tail, DT)
        perimeter_pre = float(np.sum(np.hypot(np.diff(x_tail), np.diff(dot_tail))))

        # 2) Scale the RHS by the measured period T so tau = t/T advances
        #    exactly 1 per raw period: dx/dtau = T * f(x). DT is left
        #    untouched -- scaling the equation instead of the step size
        #    avoids per-condition DT tuning.
        T_hp_mp = mp.mpf(repr(T_hp))

        def scaled_rhs(state, params):
            d = goodwin_rhs_mp(state, params)
            return tuple(T_hp_mp * v for v in d)

        # 3) Re-run convergence detection under the scaled RHS from the
        #    refined state -- confirms the state classification survives the
        #    rescaling (should stay limitcycle, but this checks rather than
        #    assumes it) and gives a refined on-cycle state plus the
        #    realized tau-period (nominally ~1).
        traj_conv2, state_post, period_post_raw = hl.runge_cycle_mp(
            scaled_rhs, refined_state, DT, params=(l_val, n_val), dps=MP_DPS
        )
        period_post = period_post_raw if period_post_raw is not None else float('nan')
        refined_state2 = traj_conv2[-1]

        # 4) Integrate NUM_CYCLES_HP tau-cycles forward -- this is the
        #    trajectory that gets cached/plotted/analyzed. Fixed step count
        #    regardless of (N, L), since one tau-cycle is always length 1.
        n_steps = int(round(NUM_CYCLES_HP / DT))
        traj_mp = hl.runge_iterate_mp(
            scaled_rhs, refined_state2, DT, n_steps,
            params=(l_val, n_val), dps=MP_DPS
        )
        x_mp = [s[0] for s in traj_mp]
        dx_mp = hl.central_diff_mp(x_mp, DT, order=1, dps=MP_DPS)
        kx_mp = hl.compute_curvature_mp(x_mp, dx_mp, DT, dps=MP_DPS)

        # Only used to locate the valid (non-boundary) window via the
        # None -> NaN convention; the mpmath lists themselves are then sliced
        # by that same window so they never actually pass through float64.
        K_arr = hl.mp_to_float_array(kx_mp)
        T_raw = np.linspace(0, (len(K_arr) - 1) * DT, len(K_arr))

        valid_idx = np.where(~np.isnan(K_arr))[0]
        i0, i1 = valid_idx[0], valid_idx[-1] + 1
        T_final = T_raw[i0:i1]
        X_mp = x_mp[i0:i1]
        DotX_mp = dx_mp[i0:i1]
        K_mp = kx_mp[i0:i1]

        metrics = _compute_summary_metrics(X_mp, DotX_mp, K_mp)
        has_k_inversion = metrics['has_k_inversion']
        always_negative = metrics['always_negative']
        kh_val = metrics['kh_val']
        perimeter_postnorm = metrics['perimeter_postnorm']
        perimeter_scaled = metrics['perimeter_scaled']

        hl.save_plot_data_mp(
            {
                'T': T_final, 'X_mp': X_mp, 'DotX_mp': DotX_mp, 'K_mp': K_mp,
                'has_k_inversion': [float(has_k_inversion)],
                'always_negative': [float(always_negative)],
                'kh_val_signed_mp': [kh_val],
                'perimeter_scaled_mp': [perimeter_scaled],
                'perimeter_postnorm': [perimeter_postnorm],
                'perimeter_pre': [perimeter_pre],
                'period_post': [period_post],
                'state_post': [float(state_post)],
            },
            cache_name, mp_columns=mp_cols, dps=MP_DPS
        )

    return {
        'T': T_final, 'X_mp': X_mp, 'DotX_mp': DotX_mp, 'K_mp': K_mp,
        'perimeter_postnorm': perimeter_postnorm,
        'perimeter_pre': perimeter_pre,
        'period_post': period_post,
        'state_post': state_post,
        'has_k_inversion': has_k_inversion,
        'always_negative': always_negative,
        'kh_val': kh_val, 'perimeter_scaled': perimeter_scaled,
    }


# -----------------------------------------------------------------------------
# Stage 3: plotting
# -----------------------------------------------------------------------------
def plot_debug_figures(all_hp_data):
    """
    Debug plots: for each L value, one figure with a column per N value,
    showing t-x, x-dx/dt (phase portrait), and t-K -- all on the high-
    precision X data. Each panel is scaled independently to its own data
    with a 20% margin (no shared/fixed ranges across panels).
    """
    for l_val in sorted({l for (n, l) in all_hp_data.keys()}):
        n_vals_here = sorted({n for (n, l) in all_hp_data.keys() if l == l_val})
        if not n_vals_here:
            continue

        cols = len(n_vals_here)
        fig = plt.figure(figsize=(2.2 * cols, 2.2 * 3))

        for col_idx, n_val in enumerate(n_vals_here):
            data = all_hp_data.get((n_val, l_val))

            # --- Row 1: t-x ---
            ax1 = fig.add_subplot(3, cols, col_idx + 1)
            if data is not None:
                T, X = data['T'], data['R']
                ax1.plot(T, X, '-k', linewidth=1.0)
                xmin, xmax = _axis_range_20pct(T)
                ymin, ymax = _axis_range_20pct(X)
                ax1.set_xlim(xmin, xmax)
                ax1.set_ylim(ymin, ymax)
            ax1.set_title(f"N={n_val}", fontsize=8)
            ax1.tick_params(labelsize=6)
            if col_idx == 0:
                ax1.set_ylabel('x')

            # --- Row 2: x-dx/dt (phase portrait) ---
            ax2 = fig.add_subplot(3, cols, cols + col_idx + 1)
            if data is not None:
                X, DotX = data['R'], data['D']
                ax2.plot(X, DotX, '-k', linewidth=1.0)
                xmin, xmax = _axis_range_20pct(X)
                ymin, ymax = _axis_range_20pct(DotX)
                ax2.set_xlim(xmin, xmax)
                ax2.set_ylim(ymin, ymax)
            ax2.set_box_aspect(1)
            ax2.tick_params(labelsize=6)
            if col_idx == 0:
                ax2.set_ylabel('dx/dt')

            # --- Row 3: t-K ---
            ax3 = fig.add_subplot(3, cols, 2 * cols + col_idx + 1)
            if data is not None:
                T, K = data['T'], data['K']
                ax3.plot(T, K, '-k', linewidth=1.0)
                xmin, xmax = _axis_range_20pct(T)
                ymin, ymax = _axis_range_20pct(K)
                ax3.set_xlim(xmin, xmax)
                ax3.set_ylim(ymin, ymax)
            ax3.tick_params(labelsize=6)
            if col_idx == 0:
                ax3.set_ylabel(r'$\kappa$')

            for ax in [ax1, ax2, ax3]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.tight_layout()
        fig_name = f"Debug_X_L{l_val}"
        pu.save_fig(fig, __file__, fig_name)
        plt.close(fig)
        print(f"Saved debug figure: {fig_name}")


def _write_heatmap_csv(states_df, summary_grid, heatmap_csv_path):
    """
    Assembles Fig4_heatmap.csv directly from the Stage-2 summary metrics
    (no extra simulation/curvature work -- it's a cheap aggregation over
    already-cached scalars) and keeps KH/log10_KH/Perimeter_Scaled as
    MP_DPS-digit decimal strings. Conversion to float64 happens only later,
    when plot_heatmap() re-reads this CSV right before plotting.

    KH is signed (not abs()) -- here always negative, since the curvature
    never changes sign over the sampled (N, L) grid (see always_negative).
    log10_KH is therefore defined as -log10(|KH|) rather than log10(KH)
    (undefined for KH<0): it has the same magnitude as the old abs()-based
    log10_KH, just sign-flipped. plot_heatmap() mirrors vmin/vmax and the
    colormap direction to match, so the actual plotted colors are unchanged.

    Perimeter_PreNorm/Perimeter_PostNorm/Period_PostNorm/State_PostNorm are
    plain-float64 diagnostics for the period-normalization scheme (see
    compute_condition_data): perimeter before vs. after tau-normalization
    (both before the spatial S-scaling that produces Perimeter_Scaled), the
    realized tau-period (nominally ~1), and the state classification
    (1=limitcycle, 2=fixed, 0=not converged) re-checked under the
    period-scaled RHS.
    """
    import mpmath as mp

    N_vals = np.sort(states_df['N'].unique())
    L_vals = np.sort(states_df['L'].unique())

    Xaxi, Yaxi = [], []
    KH_str, log10_KH_str, P_scaled_str, AlwaysNeg_col = [], [], [], []
    P_pre_col, P_postnorm_col, Period_post_col, State_post_col = [], [], [], []
    for n in N_vals:
        for l in L_vals:
            Xaxi.append(l)
            Yaxi.append(n)
            entry = summary_grid.get((int(n), round(float(l), 2)))
            kh_val = entry['kh_val'] if entry else None
            p_scaled = entry['perimeter_scaled'] if entry else None
            always_negative = entry['always_negative'] if entry else None
            if kh_val is not None:
                KH_str.append(mp.nstr(kh_val, MP_DPS))
                abs_kh = abs(kh_val)
                log10_KH_str.append(mp.nstr(-mp.log10(abs_kh) if abs_kh > 0 else mp.inf, MP_DPS))
            else:
                KH_str.append('')
                log10_KH_str.append('')
            P_scaled_str.append(mp.nstr(p_scaled, MP_DPS) if p_scaled is not None else '')
            AlwaysNeg_col.append(always_negative if always_negative is not None else '')
            P_pre_col.append(entry['perimeter_pre'] if entry else '')
            P_postnorm_col.append(entry['perimeter_postnorm'] if entry else '')
            Period_post_col.append(entry['period_post'] if entry else '')
            State_post_col.append(entry['state_post'] if entry else '')

    df_heatmap = pd.DataFrame({
        'N': Yaxi, 'L': Xaxi,
        'KH': KH_str, 'log10_KH': log10_KH_str, 'Perimeter_Scaled': P_scaled_str,
        'Always_Negative': AlwaysNeg_col,
        'Perimeter_PreNorm': P_pre_col, 'Perimeter_PostNorm': P_postnorm_col,
        'Period_PostNorm': Period_post_col, 'State_PostNorm': State_post_col,
    })
    df_heatmap[df_heatmap['KH'] != ''].to_csv(heatmap_csv_path, index=False)
    print(f"Saved {heatmap_csv_path}")


def plot_heatmap(heatmap_csv_path):
    if not os.path.exists(heatmap_csv_path):
        print(f"Warning: {heatmap_csv_path} not found, skipping heatmap figure.")
        return

    # This is the first point where values are rounded to 16-digit double
    # (loaded right before plotting).
    df_plot = pd.read_csv(heatmap_csv_path)

    # log10_KH is now -log10(|KH|) (KH is signed, see _write_heatmap_csv), so
    # it's the old abs()-based log10_KH with the sign flipped. It can be
    # exactly +inf when KH itself computed to exactly 0 (a genuine degenerate
    # point where both the velocity and acceleration vanish -- not a bug,
    # just -log10(0)). matplotlib treats +-inf as invalid/masked data and
    # silently drops those points instead of clamping them via
    # cmap.set_over/set_under like an ordinary out-of-range value, so they'd
    # otherwise vanish from the scatter entirely. Clamp to a value safely
    # outside [vmin, vmax] first so they render as the extreme color instead
    # of disappearing.
    color_vals = np.nan_to_num(df_plot['log10_KH'].to_numpy(), posinf=12.0, neginf=-2.0)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Custom colormap from cyan [0,1,1] to magenta [1,0,1] -- reversed from
    # the old magenta->cyan direction to match log10_KH's sign flip, so a
    # given physical (N, L) point still renders in the exact same color as
    # before (small |K| -> magenta, large |K| -> cyan).
    cmap_custom = LinearSegmentedColormap.from_list("cyan_magenta", ['#00FFFF', '#FF00FF'], N=200)
    # Values outside vmin/vmax are clamped to the extreme colors.
    cmap_custom.set_over(cmap_custom(1.0))
    cmap_custom.set_under(cmap_custom(0.0))

    sc = ax.scatter(df_plot['L'], df_plot['N'], s=200, c=color_vals, cmap=cmap_custom, vmin=-1, vmax=10, zorder=2)

    # Analytic Hopf-bifurcation boundary curve (n/(n-8))*(8/(n-8))^(1/n)*lambda^3=1,
    # valid for n > 8 -- see Models.goodwin.model_goodwin_mp.lambda_boundary.
    n_curve = np.linspace(8.0 + 1e-3, 26, 300)
    lam_curve = lambda_boundary(n_curve)
    ax.plot(lam_curve, n_curve, color='black', linestyle='-', linewidth=2, zorder=3)

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks(np.arange(5, 26, 2))
    ax.set_xlim([0.05, 1.05])
    ax.set_ylim([6, 26])

    cbar = plt.colorbar(sc, ax=ax, ticks=[-1, 0, 5, 10])

    plt.tight_layout()

    # PDF + PNG only, RGB (no CMYK conversion, no EPS).
    pu.save_fig(fig, __file__, "Fig4_BP")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Precheck: is everything already computed for the current sweep grid?
# -----------------------------------------------------------------------------
def _load_existing_states_csv(csv_path):
    """Returns the existing Fig4_states.csv DataFrame if present and if it
    already covers exactly the current (N,L) sweep grid; otherwise None."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    expected = {(int(n), round(float(l), 2)) for n in N_range for l in L_range}
    have = {(int(r.N), round(float(r.L), 2)) for r in df.itertuples()}
    if expected != have:
        return None
    return df


def _cache_is_fresh(cache_path):
    """Cheap header-only check: does this Stage-2 cache already carry the
    period-normalized-time trajectory format (as opposed to an older cache
    predating that scheme, which means a different trajectory definition,
    not just stale metrics -- see compute_condition_data)? Avoids loading
    the full trajectory just to check freshness."""
    try:
        header = pd.read_csv(cache_path, comment='#', nrows=0)
    except Exception:
        return False
    required = {
        'has_k_inversion', 'always_negative', 'kh_val_signed_mp',
        'perimeter_postnorm', 'perimeter_pre', 'period_post', 'state_post',
    }
    return required.issubset(set(header.columns))


def _all_data_gathered(df_states, script_dir):
    """Given an existing states DataFrame covering the full sweep grid,
    check whether every limitcycle row also has a fresh (summary-metrics
    included) Stage-2 cache already on disk."""
    for row in df_states.itertuples():
        if row.state == 'limitcycle':
            cache_name = hl.hp_cache_name(MODEL_NAME, int(row.N), round(float(row.L), 2))
            cache_file = cache_name if cache_name.endswith('.csv') else cache_name + '.csv'
            cache_path = os.path.join(script_dir, 'plot_data', cache_file)
            if not (os.path.exists(cache_path) and _cache_is_fresh(cache_path)):
                return False
    return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fig4 Goodwin: (N,L) state sweep + high-precision curvature + heatmap (unified A+BC pipeline)"
    )
    parser.add_argument('-f', '--force', action='store_true',
                        help="Force recalculation of every condition, bypassing all caches")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    states_csv_path = os.path.join(script_dir, 'Fig4_states.csv')
    heatmap_csv_path = os.path.join(script_dir, 'Fig4_heatmap.csv')

    columns = ['N', 'L', 'init_x', 'init_y', 'init_z', 'period', 'state']
    all_hp_data = {}    # (N, L) -> {'T','R','D','K'} (float64), for the debug plots
    summary_grid = {}   # (N, L) -> {'kh_val','perimeter_scaled'} (mpmath/None), for the heatmap

    existing_states = None if args.force else _load_existing_states_csv(states_csv_path)

    if existing_states is not None and _all_data_gathered(existing_states, script_dir):
        print("Stage 1/2 already complete for the current sweep grid -- "
              "skipping computation, plotting only.")
        states_df = existing_states
        for row in states_df.itertuples():
            if row.state == 'limitcycle':
                n_val, l_val = int(row.N), round(float(row.L), 2)
                data = compute_condition_data(n_val, l_val, None)
                all_hp_data[(n_val, l_val)] = {
                    'T': data['T'],
                    'R': hl.mp_to_float_array(data['X_mp']),
                    'D': hl.mp_to_float_array(data['DotX_mp']),
                    'K': hl.mp_to_float_array(data['K_mp']),
                }
                summary_grid[(n_val, l_val)] = {
                    'kh_val': data['kh_val'],
                    'perimeter_scaled': data['perimeter_scaled'],
                    'always_negative': data['always_negative'],
                    'perimeter_pre': data['perimeter_pre'],
                    'perimeter_postnorm': data['perimeter_postnorm'],
                    'period_post': data['period_post'],
                    'state_post': data['state_post'],
                }
    else:
        records = []
        total_iters = len(N_range) * len(L_range)
        print(f"Starting parameter sweep: {total_iters} total conditions...")

        iters = 0
        for n in N_range:
            for l in L_range:
                iters += 1
                if iters % 10 == 0:
                    print(f"Progress: {iters}/{total_iters} (N={n}, L={l:.2f})")

                n_val = int(n)
                l_val = round(l, 2)
                params = [l_val, n_val]

                try:
                    # NUM_CYCLES_HP (not just a short "search" window) so the
                    # trajectory this produces is already long enough for the
                    # Stage-2 high-precision cache below.
                    R_raw, state_code, per_returned = hl.simulate(model_name, DT, params, NUM_CYCLES_HP, return_state=True)

                    if len(R_raw) == 0 or len(R_raw[0]) == 0:
                        state = 'unfinished'
                        per = np.nan
                    elif state_code == 2:
                        state = 'fixed'
                        per = np.nan
                    elif state_code == 1:
                        state = 'limitcycle'
                        per = per_returned
                    elif state_code == 0:
                        state = 'unfinished'
                        per = np.nan
                    else:
                        state = 'error'
                        per = np.nan

                    init_vals = [R_raw[0][-1], R_raw[1][-1], R_raw[2][-1]] if len(R_raw) >= 3 else [np.nan, np.nan, np.nan]

                    if state == 'limitcycle':
                        data = compute_condition_data(n_val, l_val, R_raw, force_recalc=args.force)
                        all_hp_data[(n_val, l_val)] = {
                            'T': data['T'],
                            'R': hl.mp_to_float_array(data['X_mp']),
                            'D': hl.mp_to_float_array(data['DotX_mp']),
                            'K': hl.mp_to_float_array(data['K_mp']),
                        }
                        summary_grid[(n_val, l_val)] = {
                            'kh_val': data['kh_val'],
                            'perimeter_scaled': data['perimeter_scaled'],
                            'always_negative': data['always_negative'],
                            'perimeter_pre': data['perimeter_pre'],
                            'perimeter_postnorm': data['perimeter_postnorm'],
                            'period_post': data['period_post'],
                            'state_post': data['state_post'],
                        }

                except Exception as e:
                    state = 'error'
                    init_vals = [np.nan, np.nan, np.nan]
                    per = np.nan

                records.append({
                    'N': n_val, 'L': l_val,
                    'init_x': init_vals[0], 'init_y': init_vals[1], 'init_z': init_vals[2],
                    'period': per, 'state': state,
                })

        states_df = pd.DataFrame(records, columns=columns)
        states_df.to_csv(states_csv_path, index=False)
        print(f"\nSaved {states_csv_path}")

    print("Building heatmap CSV from cached curvature summaries...")
    _write_heatmap_csv(states_df, summary_grid, heatmap_csv_path)

    print("Generating debug figures...")
    plot_debug_figures(all_hp_data)

    print("Generating heatmap figure...")
    plot_heatmap(heatmap_csv_path)


if __name__ == '__main__':
    main()
