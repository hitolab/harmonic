# Codes for curvature-based waveform analysis

**arXiv 2026**  
Yamada Y, Kabata Y, Fukasaku R, Ito H  
A curvature-based criterion for harmonic circadian waveforms
[arXiv:xxxxx](https://www.nature.com/articles/s41598-025-04614-z)

## Code overview (yamada2026/)

Fig1/FN/Main_FN.py: Curvature analysis of the FitzHugh-Nagumo model limit cycle.

Fig1/SL/Main_SL.py: Curvature analysis of the Stuart-Landau model limit cycle.

Fig2/Main_raw.py : Curvature analysis of raw experimental bioluminescence traces (Cyanobacteria, mouse SCN).

Fig3/Kim-Forger/Main_KF.py : Curvature analysis of the Kim-Forger detailed clock model trajectory.
Fig3/Sasai/Main_sasai.py : Curvature analysis of the Sasai model trajectory.

Fig4/B_Goodwin/Main_B.py : Curvature (K) sign/time-series and phase-portrait panels of the Goodwin model across L values.
Fig4/C_Heatmap/Main_C.py : Heatmap of curvature-based metrics over the Goodwin model's (N, lambda) parameter grid.

Fig5/Main_Fig5.py : Maps the initial-value regions where the piecewise-linear Goodwin limit cycle stays convex, per lambda, with the actual switching points overlaid.

Fig6/Main.py : Derives the Goodwin limit cycle (main purpose); also renders the isosurface via Plotly/marching-cubes as an auxiliary/supplementary check.
Fig6/Fig6.nb : Produces the actual 3D surface + limit cycle plot used in the figure.

FigS1/Main_S1.py : Compares curvature/CSS metrics across smoothing bandwidths (sigma_hat) for FitzHugh-Nagumo and Stuart-Landau.

FigS2/Main_S2.py : Same sigma_hat comparison as FigS1, applied to experimental datasets (Cyanobacteria, mouse SCN).

FigS3/Main_S3.py : Curvature analysis of individual Kim-Forger model state variables (mRNA species).
FigS3/Main_S3_grid.py : Grid figure combining all Main_S3 state-variable panels into one plot.
FigS3/Scan/Main_scan.py : Curvature analysis scanned across all states of the Kim-Forger "detailed model".
  Note: the detailed model itself (Data/Kim-Forger/DetailedModel.m, 180 states) is not our own model -- it is taken as-is from Kim JK, Forger DB. A mechanism for robust circadian temperature compensation via degradation rate regulation. Mol Syst Biol. 2012;8:630. doi:10.1038/msb.2012.62; we only ran the curvature analysis on its output.

Lib
harmonicity.py : Core simulation/integration and curvature (K) computation routines, plus plot-data caching.
plot_utils.py : Shared plotting helpers (axis scaling, figure saving, etc.).
analysis.py : Shared analysis helper routines.
backend.py : Simulation backend interface.

Models
Each model below has its own directory with a `model_*.cpp` defining the ODE right-hand side (compiled to `.dylib` and called from Lib/harmonicity.py). The `_stoch` folders are NOT different models -- they are the same FN/SL equations with additive noise (Euler-Maruyama, normalized time tau = t/T) bolted on, so the model itself is unchanged between FN vs. FN_stoch (likewise for SL).

FN : FitzHugh-Nagumo oscillator, deterministic. Params: a, b, c.
SL : Stuart-Landau oscillator, deterministic. Params: a, b, w.
FN_stoch : FN with additive noise, rewritten in normalized time tau = t/T and integrated with Euler-Maruyama . Adds params noise_std, dtau, T.
SL_stoch : SL, same normalized-time/Euler-Maruyama noise treatment as FN_stoch. Adds params noise_std, dtau, T.
goodwin : Goodwin oscillator (cyclic inhibition), deterministic. Params: lambda, n. Also has a high-precision mpmath reimplementation (model_goodwin_mp.py) used where double precision isn't enough (e.g. Fig4/C_Heatmap).
