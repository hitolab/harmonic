# Codes for curvature-based waveform analysis

**arXiv 2026**  
Yamada Y, Kabata Y, Fukasaku R, Ito H  
A curvature-based criterion for harmonic circadian waveforms
[arXiv:xxxxx](https://www.nature.com/articles/s41598-025-04614-z)

## Code overview (yamada2026/)

Fig1
Main_FN.py (FN/) : Curvature analysis of the FitzHugh-Nagumo model limit cycle.
Main_SL.py (SL/) : Curvature analysis of the Stuart-Landau model limit cycle.

Fig2
Main_raw.py : Curvature analysis of raw experimental bioluminescence traces (Cyanobacteria, mouse SCN).

Fig3
Main_KF.py (Kim-Forger/) : Curvature analysis of the Kim-Forger detailed clock model trajectory.
Main_sasai.py (Sasai/) : Curvature analysis of the Sasai model trajectory.

Fig4
Main_B.py (B_Goodwin/) : Curvature (K) sign/time-series and phase-portrait panels of the Goodwin model across L values.
Main_C.py (C_Heatmap/) : Heatmap of curvature-based metrics over the Goodwin model's (N, lambda) parameter grid.

Fig5
Main_Fig5.py : Maps the initial-value regions where the piecewise-linear Goodwin limit cycle stays convex, per lambda, with the actual switching points overlaid.

Fig6
Main.py : 3D (Plotly) rendering of the Goodwin limit cycle surface/isosurface.

A1
Main_A1.py : Compares curvature/CSS metrics across smoothing bandwidths (sigma_hat) for FitzHugh-Nagumo and Stuart-Landau.

A2
Main_A2.py : Same sigma_hat comparison as A1, applied to experimental datasets (Cyanobacteria, mouse SCN).

A3
Main_A3.py : Curvature analysis of individual Kim-Forger model state variables (mRNA species).
Main_A3_grid.py : Grid figure combining all Main_A3 state-variable panels into one plot.
Main_scan.py (Scan/) : Curvature analysis scanned across all states of the Kim-Forger "detailed model".
  Note: the detailed model itself (Data/Kim-Forger/DetailedModel.m, 180 states) is not our own model -- it is taken as-is from the original Kim-Forger paper; we only ran the curvature analysis on its output.

Lib
harmonicity.py : Core simulation/integration and curvature (K) computation routines, plus plot-data caching.
plot_utils.py : Shared plotting helpers (axis scaling, figure saving, etc.).
analysis.py : Shared analysis helper routines.
backend.py : Simulation backend interface.

Models
Each model below has its own directory with a `model_*.cpp` defining the ODE right-hand side (compiled to `.dylib` and called from Lib/harmonicity.py). The `_stoch` and `_stoch_norm_euler` folders are NOT different models -- they are the same FN/SL equations with additive noise (Euler-Maruyama) bolted on, so the model itself is unchanged between FN vs. FN_stoch vs. FN_stoch_norm_euler (likewise for SL).

FN : FitzHugh-Nagumo oscillator, deterministic. Params: a, b, c.
SL : Stuart-Landau oscillator, deterministic. Params: a, b, w.
FN_stoch : FN with additive noise, integrated with RK4 (noise sampled once per step to stay consistent with Euler-Maruyama). Adds params noise_std, dt.
SL_stoch : SL with additive noise, same RK4/noise scheme as FN_stoch. Adds params noise_std, dt.
FN_stoch_norm_euler : FN with additive noise, rewritten in normalized time tau = t/T and integrated with plain Euler-Maruyama (one model_func call per step, so noise is fresh every call instead of frozen across RK4 sub-stages). Adds params noise_std, dtau, T.
SL_stoch_norm_euler : SL, same normalized-time/Euler-Maruyama treatment as FN_stoch_norm_euler.
goodwin : Goodwin oscillator (cyclic inhibition), deterministic. Params: lambda, n. Also has a high-precision mpmath reimplementation (model_goodwin_mp.py) used where double precision isn't enough (e.g. Fig4/C_Heatmap).
