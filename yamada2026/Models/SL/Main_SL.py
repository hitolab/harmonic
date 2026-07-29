import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure Lib is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import Lib.harmonicity as hl

# Prevent overflow error when plotting many points
plt.rcParams['agg.path.chunksize'] = 10000

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
A = 4.0
B = 0.3
W = 0.4
PARAMS = [A, B, W]

DT = 1e-4
NUM_CYCLES = 2

model_name = 'SL'

if __name__ == '__main__':
    print(f"Running Simulation (Model: {model_name})...")
    
    result = hl.simulate(model_name, DT, PARAMS, NUM_CYCLES)
    
    V = len(result)
    
    # Generate Time vector based on raw output
    T_raw = np.linspace(0, (len(result[0])-1)*DT, len(result[0]))
    
    # Calculate derivatives using central differencing
    print("Calculating Derivatives...")
    D_raw = []
    R_vel = [] 
    
    for r in result:
        d = hl.central_diff(r, DT, order=1)
        D_raw.append(d)
        R_vel.append(r)
    
    # Compute Curvature
    print("Computing Curvature...")
    K = []
    for i in range(V):
        k_val = hl.compute_curvature(R_vel[i], D_raw[i], DT)
        K.append(k_val)
        
    # Global Trimming (Dynamic)
    T_final, R_final, D_final, K_final = hl.align_and_trim(T_raw, R_vel, D_raw, K)
    
    # Normalized Time 0 to 2 for plotting
    T_norm = np.linspace(0, 2, len(T_final))

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
    import Lib.plot_utils as pu
    
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
        
        # --- Row 1: Raw Data (R vs T) ---
        ax1 = plt.subplot(4, V, i + 1)
        if np.any(mask1):
            ax1.plot(T_norm[mask1], r_data[mask1], '.m', markersize=2)
        if np.any(mask2):
            ax1.plot(T_norm[mask2], r_data[mask2], '.c', markersize=2)
            
        xmax, xmin, ymax, ymin = pu.plot_axis(T_norm, r_data, 1, 1, 'fc')
        ax1.set_xlim(0, 2)
        ax1.set_ylim([-3, 3])
        ax1.set_xticks([0, 1, 2])
        # To avoid setting tick at 2 repeatedly if not max, but matching Goodwin:
        # Goodwin does ax1.set_yticks([ymin, 2])
        ax1.set_yticks([-3, 3])
        ax1.set_box_aspect(9/16)
        
        # --- Row 2: Phase (D vs R) ---
        ax2 = plt.subplot(4, V, i + 1 + V)
        if np.any(mask1):
            ax2.plot(r_data[mask1], d_data[mask1], '.m', markersize=2)
        if np.any(mask2):
            ax2.plot(r_data[mask2], d_data[mask2], '.c', markersize=2)
            
        xmax, xmin, ymax, ymin = pu.plot_axis(r_data, d_data, 1.1, 1.1, 's')
        ax2.set_xlim([-3, 3])
        ax2.set_ylim([-3, 3])
        ax2.set_xticks([-3, 3])
        ax2.set_yticks([-3, 3])
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
        ax3.set_box_aspect(9/16)
        
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
        ax4.set_box_aspect(9/16)
        
        # Universal formatting
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(direction='out')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, 'img')
    os.makedirs(save_dir, exist_ok=True)
    
    param_str = "_".join(map(str, PARAMS))
    save_path = os.path.join(save_dir, f"{model_name}_{param_str}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
