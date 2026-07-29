import os
import sys
import numpy as np
import plotly.graph_objects as go
from skimage import measure

# Add the repository root to sys.path so Lib can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Lib.harmonicity as hl

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Parameters ---
    lamb = 0.6  # placeholder: K_curv2.m's fnval uses lamb and n but never defines them
    n = 12      # Goodwin equation exponent n
    # The MATLAB code names the parameters a, b, c (Rossler), but fval has the form
    # of the Goodwin model, so we assume the Goodwin model is what is simulated.
    param = [lamb, n]
    dt = 1e-3
    target_cycles = 1
    model_name = 'goodwin'
    
    # Initial state (an arbitrary point that converges onto the limit cycle)
    init = [0.1, 0.1, 0.1]
    NUM_CYCLES = target_cycles + 3
    
    print("Running simulation...")
    R_raw = hl.simulate(model_name, dt, param, NUM_CYCLES, init=init)
    
    if len(R_raw) == 0:
        print("Simulation failed.")
        return
        
    D_raw = [hl.central_diff(r, dt, order=1) for r in R_raw]
    K = [hl.compute_curvature(R_raw[i], D_raw[i], dt) for i in range(len(R_raw))]
    
    # Trim to the last single cycle
    T_raw = np.linspace(0, (len(R_raw[0])-1)*dt, len(R_raw[0]))
    T_final, R_final, D_final, K_final = hl.align_and_trim(T_raw, R_raw, D_raw, K)
    
    X = R_final[0]
    Y = R_final[1]
    Z = R_final[2]
    K_Z = K_final[2]
    
    idx_p = K_Z > 0
    idx_m = K_Z < 0
    
    # === Interactive rendering with Plotly ===
    fig = go.Figure()
    
    # K > 0 (cyan), K < 0 (magenta)
    # Insert NaN so the segments are drawn as disconnected lines
    def create_segments(X, Y, Z, condition):
        x_seg, y_seg, z_seg = [], [], []
        for i in range(len(X)-1):
            if condition[i] and condition[i+1]:
                x_seg.extend([X[i], X[i+1], None])
                y_seg.extend([Y[i], Y[i+1], None])
                z_seg.extend([Z[i], Z[i+1], None])
        return x_seg, y_seg, z_seg

    x_c, y_c, z_c = create_segments(X, Y, Z, idx_p)
    x_m, y_m, z_m = create_segments(X, Y, Z, idx_m)

    fig.add_trace(go.Scatter3d(
        x=x_c, y=y_c, z=z_c,
        mode='lines',
        line=dict(color='cyan', width=4),
        name='K > 0',
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=x_m, y=y_m, z=z_m,
        mode='lines',
        line=dict(color='magenta', width=4),
        name='K < 0',
        hoverinfo='skip'
    ))
            
    # === Isosurface plot ===
    print("Rendering isosurface...")
    N_grid = 150  # mesh refined by about 1.5x (100 -> 150)
    
    lim_x = [0, 2]
    lim_y = [0, 2]
    lim_z = [0, 2]
    
    xrange = np.linspace(lim_x[0], lim_x[1], N_grid)
    yrange = np.linspace(lim_y[0], lim_y[1], N_grid)
    zrange = np.linspace(lim_z[0], lim_z[1], N_grid)
    x, y, z = np.meshgrid(xrange, yrange, zrange, indexing='ij')
    
    # Evaluate fval
    zn = z**n
    z_n_minus_1 = z**(n - 1)
    
    term1_base = n * (x - y * lamb) * (y - z * lamb)
    term2 = (-1.0 + x * (1.0 + zn) * lamb)**2
    
    fval = (z_n_minus_1 * term1_base) + term2
         
    # Extract the isosurface (fval = 0) with the marching cubes algorithm
    try:
        verts, faces, normals, values = measure.marching_cubes(
            fval, 0, spacing=(xrange[1]-xrange[0], yrange[1]-yrange[0], zrange[1]-zrange[0])
        )
        # verts are offsets on the grid, so convert them to actual coordinates
        verts[:, 0] += xrange[0]
        verts[:, 1] += yrange[0]
        verts[:, 2] += zrange[0]
        
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='gray',
            opacity=0.5,
            name='F(X,Y,Z)=0',
            flatshading=True,
            hoverinfo='skip'
        ))
    except Exception as e:
        print(f"Failed to extract the isosurface: {e}")

    # === Axes and appearance ===
    # Settings that remove the box (frame and background panels) entirely
    axis_config = dict(
        title='',
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
        showbackground=False,     # hide the background panels
        showaxeslabels=False,
        visible=False             # hide Plotly's default axes entirely
    )
    
    fig.update_layout(
        title=f'Trajectory + F(X,Y,Z)=0 (L={lamb}, N={n})',
        scene=dict(
            xaxis=dict(**axis_config, range=lim_x),
            yaxis=dict(**axis_config, range=lim_y),
            zaxis=dict(**axis_config, range=lim_z),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )
    
    # Draw the principal axes manually so they are thicker and clearer
    fig.add_trace(go.Scatter3d(x=[0, lim_x[1]], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, lim_y[1]], z=[0, 0], mode='lines', line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, lim_z[1]], mode='lines', line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'))
    
    # === Save and display ===
    save_dir = os.path.join(script_dir, 'img')
    os.makedirs(save_dir, exist_ok=True)
    
    html_path = os.path.join(save_dir, f'Fig1-GW_{lamb:.1f}_{n}_interactive.html')
    fig.write_html(html_path)
    print(f"Saved interactive HTML: {html_path}")

    # === data.csv referenced by Fig6.nb (in NotebookDirectory, columns t,x,y,z) ===
    import pandas as pd
    df_csv = pd.DataFrame({
        't': T_final,
        'x': X,
        'y': Y,
        'z': Z
    })
    nb_data_path = os.path.join(script_dir, 'data.csv')
    df_csv.to_csv(nb_data_path, index=False)
    print(f"Saved data.csv for Fig6.nb: {nb_data_path}")

    # Open in a browser
    fig.show()

if __name__ == "__main__":
    main()
    
