import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Import custom library
try:
    import Lib.harmonicity as hl
except ImportError:
    # Handle running from different directories or if Lib is not a package
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    import Lib.harmonicity as hl

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LAMB = 0.3
N_EXP = 11
PARAMS = [LAMB, N_EXP]

DT = 1e-3
NUM_CYCLES = 2 

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    model_name = "goodwin"
    print(f"Running Simulation (Model: {model_name})...")
    
    # Simulate
    R_raw = hl.simulate(model_name, DT, PARAMS, NUM_CYCLES)
    
    # We can infer V from the output of simulate
    V = len(R_raw)
    
    # Generate Time vector
    T_raw = np.linspace(0, (len(R_raw[0])-1)*DT, len(R_raw[0]))

    # Calculate Velocity (1st Derivative)
    print("Calculating Derivatives...")
    D_raw = []
    R_vel = [] 
    
    for r in R_raw:
        # central_diff now returns same length array (with NaNs at boundaries)
        d = hl.central_diff(r, DT, order=1)
        D_raw.append(d)
        # R and T are already aligned with D (same length)
        R_vel.append(r)
    
    # Compute Curvature
    print("Computing Curvature...")
    K = []
    
    for i in range(V):
        # compute_curvature now returns same length array (with NaNs at boundaries)
        # input R and D are aligned
        k_val = hl.compute_curvature(R_vel[i], D_raw[i], DT)
        K.append(k_val)
        
    # -----------------------------------------------------------------------------
    # Global Trimming (Dynamic)
    # -----------------------------------------------------------------------------
    # We inspect all arrays to find the maximum NaN width at boundaries
    # This ensures all arrays (T, R, D, K) are perfectly aligned
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
    
    print("Indicators:", indicators)

    # -----------------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------------
    import Lib.plot_utils as pu
    
    print("Plotting...")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['agg.path.chunksize'] = 10000
    
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
        ax1.set_ylim(ymin, 2) # Mimicking var3_Temp.m behavior
        ax1.set_xticks([0, 1, 2])
        ax1.set_yticks([ymin, 2])
        ax1.set_box_aspect(9/16) # pbaspect([16 9 1]) equiv
        
        # --- Row 2: Phase (D vs R) ---
        ax2 = plt.subplot(4, V, i + 1 + V)
        if np.any(mask1):
            ax2.plot(r_data[mask1], d_data[mask1], '.m', markersize=2)
        if np.any(mask2):
            ax2.plot(r_data[mask2], d_data[mask2], '.c', markersize=2)
            
        xmax, xmin, ymax, ymin = pu.plot_axis(r_data, d_data, 1.1, 1.1, 's')
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.set_xticks([xmin, xmax])
        ax2.set_yticks([ymin, ymax])
        ax2.set_box_aspect(1) # pbaspect([1 1 1])
        
        # --- Row 3: K > 0 ---
        ax3 = plt.subplot(4, V, i + 1 + 2*V)
        if np.any(mask2):
            # Plot only where valid
            # In MATLAB semilogy with NaNs connects lines or breaks them depending on NaNs.
            # To plot connected lines over valid points, we index properly
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
            # Matplotlib semilogy only plots positive values. We flip the data and visually alter y-ticks.
            # (or use symlog, but semilogy with negative values is requested)
            ax4.semilogy(valid_T, -valid_K, '-m')
            
        ax4.set_xlim(0, 2)
        ax4.set_ylim(1e-2, 1e2)
        # To match Matlab's [-1e2, -1e-2] visual look on log scale, we invert the axis
        ax4.invert_yaxis()
        ax4.set_xticks([0, 1, 2])
        # Setting tick labels to represent the negative values mathematically
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
    filename = f'GW_{param_str}.png'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
