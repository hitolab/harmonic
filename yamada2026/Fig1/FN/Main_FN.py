import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Ensure Lib is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import Lib.harmonicity as hl
import Lib.plot_utils as pu

# Prevent overflow error when plotting many points
plt.rcParams['agg.path.chunksize'] = 10000

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
A = 5.0
B = 0.4
C = 0.6
PARAMS = [A, B, C]
param_names = ['A', 'B', 'C']

DT = 1e-3
model_name = 'FN'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fig1 FN Analysis")
    parser.add_argument('-f', '--force', action='store_true', help="Force recalculation (bypass caches)")
    args = parser.parse_args()
    force_recalc = args.force

    print(f"Running Simulation (Model: {model_name})...")
    
    # -----------------------------------------------------------------------------
    # Data Processing / Caching
    # -----------------------------------------------------------------------------
    param_str = "_".join([f"{p:.6g}" for p in PARAMS])
    cache_name = f"{model_name}_plot_data_{param_str}"
    
    df_cached = hl.load_plot_data(cache_name) if not force_recalc else None
    
    if df_cached is not None:
        print("Loaded processed data from cache.")
        V = 0
        while f"R{V}" in df_cached.columns:
            V += 1
            
        T_norm = df_cached['T_norm'].values
        R_final = [df_cached[f'R{i}'].values for i in range(V)]
        D_final = [df_cached[f'D{i}'].values for i in range(V)]
        K_final = [df_cached[f'K{i}'].values for i in range(V)]
        
        import scipy.signal
        prom = 0.1 * (np.max(R_final[0]) - np.min(R_final[0]))
        peaks, _ = scipy.signal.find_peaks(R_final[0], prominence=prom)
        if len(peaks) > 0:
            idx_1cycle = peaks[0]
        else:
            idx_1cycle = len(R_final[0]) // 2
    else:
        print("No cache found. Running simulations...")
        # Run to convergence (RungeCycle) and record cycles on the settled limit cycle
        res_long, state_code, period = hl.simulate(model_name, DT, PARAMS, num_cycles=4, return_state=True)
        state_labels = {0: 'unfinished', 1: 'limitcycle', 2: 'fixed'}
        print(f"RungeCycle state: {state_labels.get(state_code, 'error')}, period: {period:.4f}")
        if state_code != 1:
            print(f"Warning: model did not converge to a limit cycle (state_code={state_code}).")

        V = len(res_long)
        
        print("Calculating Derivatives...")
        D_raw = []
        R_vel = [] 
        for r in res_long:
            d = hl.central_diff(r, DT, order=1)
            D_raw.append(d)
            R_vel.append(r)
        
        print("Computing Curvature...")
        K = []
        for i in range(V):
            k_val = hl.compute_curvature(R_vel[i], D_raw[i], DT)
            K.append(k_val)
            
        # Global Trimming (Dynamic)
        T_raw = np.arange(len(res_long[0])) * DT
        T_final, R_final, D_final, K_final = hl.align_and_trim(T_raw, R_vel, D_raw, K)
        
        # Slice to exactly 2 cycles based on first variable peaks
        import scipy.signal
        prom = 0.1 * (np.max(R_final[0]) - np.min(R_final[0]))
        peaks, _ = scipy.signal.find_peaks(R_final[0], prominence=prom)
        if len(peaks) >= 3:
            idx_start = peaks[0]
            idx_end = peaks[2]
            idx_1cycle = peaks[1] - peaks[0]
        else:
            idx_start = 0
            idx_end = len(T_final) - 1
            idx_1cycle = len(T_final) // 2
            
        T_final = T_final[idx_start:idx_end+1]
        R_final = [r[idx_start:idx_end+1] for r in R_final]
        D_final = [d[idx_start:idx_end+1] for d in D_final]
        K_final = [k[idx_start:idx_end+1] for k in K_final]
        
        # Normalized Time 0 to 2 for plotting
        T_norm = np.linspace(0, 2, len(T_final))
        
        # Save to cache
        cache_dict = {'T_norm': T_norm}
        for i in range(V):
            cache_dict[f'R{i}'] = R_final[i]
            cache_dict[f'D{i}'] = D_final[i]
            cache_dict[f'K{i}'] = K_final[i]
            
        hl.save_plot_data(cache_dict, cache_name, metadata={n: v for n, v in zip(param_names, PARAMS)})

    # Indicator Calculation
    indicators = []
    for k_val in K_final:
        if len(k_val) == 0:
            indicators.append(np.nan)
            continue
        k_abs = np.abs(k_val)
        k_min_abs = np.min(k_abs)
        k_amp = np.max(k_val) - np.min(k_val)
        if k_amp != 0:
            indicators.append(1.0 * k_min_abs)
        else:
            indicators.append(np.nan)
            
    print(f"Indicators: {indicators}")
    
    print("Plotting...")
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig = plt.figure(figsize=(16, 12))
    
    for i in range(V):
        r_data = R_final[i]
        d_data = D_final[i]
        k_data = K_final[i]
        
        if len(r_data) == 0: continue

        mask1 = k_data < 0
        mask2 = k_data > 0
        
        # Expand masks to include boundary transition points to avoid gaps
        transitions = mask2[:-1] != mask2[1:]
        mask2_ext = mask2.copy()
        mask1_ext = mask1.copy()
        mask2_ext[:-1] |= transitions
        mask2_ext[1:] |= transitions
        mask1_ext[:-1] |= transitions
        mask1_ext[1:] |= transitions
        
        # --- Row 1: Raw Data (R vs T) ---
        ax1 = plt.subplot(4, V, i + 1)
        r_neg = np.where(mask1_ext, r_data, np.nan)
        r_pos = np.where(mask2_ext, r_data, np.nan)
        ax1.plot(T_norm, r_neg, '-m', linewidth=1.2)
        ax1.plot(T_norm, r_pos, '-c', linewidth=1.2)
            
        # Row 1: y -3 to 5, x 0 to 2 (same for both variables)
        ax1.set_xlim(0, 2)
        ax1.set_ylim(-3, 5)
        ax1.set_xticks([0, 1, 2])
        ax1.set_yticks([-3, 0, 5])
        ax1.set_box_aspect(5/8)
        
        # --- Row 2: Phase (D vs R) ---
        ax2 = plt.subplot(4, V, i + 1 + V)
        r_data_1c = r_data[:idx_1cycle+1]
        d_data_1c = d_data[:idx_1cycle+1]
        k_data_1c = k_data[:idx_1cycle+1]
        mask1_1c = k_data_1c < 0
        mask2_1c = k_data_1c > 0
        
        transitions_1c = mask2_1c[:-1] != mask2_1c[1:]
        mask2_1c_ext = mask2_1c.copy()
        mask1_1c_ext = mask1_1c.copy()
        mask2_1c_ext[:-1] |= transitions_1c
        mask2_1c_ext[1:] |= transitions_1c
        mask1_1c_ext[:-1] |= transitions_1c
        mask1_1c_ext[1:] |= transitions_1c
        
        r_neg_1c = np.where(mask1_1c_ext, r_data_1c, np.nan)
        d_neg_1c = np.where(mask1_1c_ext, d_data_1c, np.nan)
        r_pos_1c = np.where(mask2_1c_ext, r_data_1c, np.nan)
        d_pos_1c = np.where(mask2_1c_ext, d_data_1c, np.nan)
        
        ax2.plot(r_neg_1c, d_neg_1c, '-m', linewidth=1.2)
        ax2.plot(r_pos_1c, d_pos_1c, '-c', linewidth=1.2)
            
        # Row 2: first variable x -10 to 5, y -8 to 7 / second variable x -2 to 3, y -2 to 3
        if i == 0:
            ax2.set_xlim(-8, 8)
            ax2.set_ylim(-10, 6)
            ax2.set_xticks([-8, 0, 8])
            ax2.set_yticks([-10, 0, 6])
        else:
            ax2.set_xlim(-2, 3)
            ax2.set_ylim(-2, 3)
            ax2.set_xticks([-2, 0, 3])
            ax2.set_yticks([-2, 0, 3])
        ax2.set_box_aspect(1)
        
        # --- Row 3: K > 0 ---
        ax3 = plt.subplot(4, V, i + 1 + 2*V)
        if np.any(mask2):
            valid_T = T_norm[~np.isnan(k_data)]
            valid_K = k_data[~np.isnan(k_data)]
            ax3.semilogy(valid_T, valid_K, '-c')
            
        ax3.set_xlim(0, 2)
        ax3.set_ylim(1e-2, 1e2)
        ax3.set_xticks([0, 1, 2])
        ax3.set_yticks([1e-2, 1, 1e2])
        ax3.minorticks_off()
        ax3.set_box_aspect(1/2)
        
        # --- Row 4: K < 0 ---
        ax4 = plt.subplot(4, V, i + 1 + 3*V)
        if np.any(mask1):
            valid_T = T_norm[~np.isnan(k_data)]
            valid_K = k_data[~np.isnan(k_data)]
            ax4.semilogy(valid_T, -valid_K, '-m')
            
        ax4.set_xlim(0, 2)
        ax4.set_ylim(1e-2, 1e2)
        ax4.invert_yaxis()
        ax4.set_xticks([0, 1, 2])
        ax4.set_yticks([1e-2, 1, 1e2])
        ax4.set_yticklabels(['-10$^{-2}$', '-10$^{0}$', '-10$^{2}$'])
        ax4.minorticks_off()
        ax4.set_box_aspect(1/2)
        
        # Universal formatting
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(direction='out')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # Save using standard flow
    fig_name = f"{model_name}_{param_str}"
    pu.save_fig(fig, __file__, fig_name)
    plt.close(fig)
