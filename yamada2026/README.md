# Harmonicity — Analysis Code

Code that analyses the curvature of oscillatory systems in the phase plane and
generates the figures of the paper. The numerical integration core is written
in C++ and called from Python through `ctypes`.

## Directory layout

```
999_For_Public/
├── Lib/            Shared library (integration, differentiation, curvature, plotting)
├── Models/         ODE definitions of the models (C++)
├── Data/           Experimental and numerical data (input; not generated)
├── Fig1/ … Fig6/   Main-text figures
└── A1/ A2/ A3/     Appendix figures
```

Besides the scripts, each figure directory holds the following generated
content.

| Directory | Contents |
| :--- | :--- |
| `img/` | Output figures (pdf / png / eps) |
| `plot_data/` | Cache of the intermediate data used for plotting |
| `sim_data/` | Cache of simulation results |

## Caching and the `-f` option

Each script reads the cache in `plot_data/` if it exists and skips the
computation entirely. With the caches in place, repeated runs therefore
reproduce exactly the same figures.

Passing `-f` / `--force` ignores the caches and recomputes everything. Note,
however, that `simulate()` reads its initial state from
`Models/<model>/states_history.csv` and writes the final state back to the same
file afterwards, so **a forced recomputation shifts the initial state slightly
along the limit cycle on every run**. The difference is far too small to change
the figures (below 0.3% of the value range), but the numbers will not match the
previous run exactly.

Scripts that accept `-f`: `Fig1/FN`, `Fig1/SL`, `Fig2`, `Fig3/*`, `Fig4/*`,
`Fig5`, `A1`, `A2`, `A3/*`

---

## Lib/ — shared library

See `Lib/README_Lib.md` for the full function reference.

| File | Description |
| :--- | :--- |
| `harmonicity.py` | Facade re-exporting `backend` and `analysis`; this is what callers import |
| `backend.py` | Calls into the C++ library, auto-compiles models, Runge–Kutta integration, differentiation and curvature |
| `analysis.py` | Smoothing, period estimation, inflection-point extraction, cache I/O |
| `plot_utils.py` | Automatic axis ranges and figure output as pdf / png / eps |
| `harmonicity.cpp` | C++ implementation of integration and differentiation (source of `libharmonicity.dylib`) |
| `model_interface.h` | Interface the C++ side of each model must implement |
| `Makefile` | Builds `libharmonicity.dylib` |
| `README_Lib.md` | Arguments, return values and behaviour of every function in the files above |

## Models/ — model definitions

Each subdirectory corresponds to one model, with `model_*.cpp` supplying the
right-hand side of the ODE. `Lib/backend.py` selects a model by directory name
and compiles it automatically when needed.

| Directory | Description |
| :--- | :--- |
| `FN/` | FitzHugh–Nagumo model |
| `SL/` | Stuart–Landau model |
| `goodwin/` | Goodwin model |
| `FN_stoch/` `SL_stoch/` | Carry a noise term, but A1 calls them with `noise_std=0` and uses them for the deterministic preliminary run that estimates the period |
| `FN_stoch_norm_euler/` `SL_stoch_norm_euler/` | Stochastic version with time normalised by the estimated period; used for A1's main computation (Euler–Maruyama) |

| File | Description |
| :--- | :--- |
| `FN/Main_FN.py`, `SL/Main_SL.py`, `goodwin/Main.py` | Standalone sanity-check scripts for each model; plot the phase plane and time series |
| `goodwin/model_goodwin_mp.py` | Arbitrary-precision (mpmath) right-hand side of the Goodwin model and the analytic Hopf-bifurcation boundary |

## Data/ — input data

**Not generated, so recomputation never changes it.**

| Directory | Description |
| :--- | :--- |
| `Cyano/` | Measured circadian rhythms of cyanobacteria |
| `MouseSCN/` | Measured circadian rhythms of the mouse suprachiasmatic nucleus |
| `Sasai/` | Numerical solution of the Sasai model |
| `Kim-Forger/` | Numerical solution of the detailed Kim–Forger model (180 variables) and the MATLAB scripts that produced it |

---

## Fig1 … Fig6 — main-text figures

| Script | Description |
| :--- | :--- |
| `Fig1/FN/Main_FN.py` | Time series, phase plane and curvature of FitzHugh–Nagumo |
| `Fig1/SL/Main_SL.py` | Time series, phase plane and curvature of Stuart–Landau |
| `Fig2/Main_raw.py` | Time series, phase plane and curvature for the measured cyanobacteria and mouse SCN data |
| `Fig3/Kim-Forger/Main_KF.py` | Curvature analysis of the Kim–Forger model (variable McPo) |
| `Fig3/Sasai/Main_sasai.py` | Curvature analysis of the Sasai model |
| `Fig4/B_Goodwin/Main_B.py` | Time series, phase plane and curvature of each Goodwin variable x, y, z, computed in arbitrary precision |
| `Fig4/C_Heatmap/Main_C.py` | Sweeps the Goodwin model over the (n, λ) plane and draws a heatmap of the perimeter-normalised minimum curvature |
| `Fig5/Main_Fig5.py` | Convex region of the piecewise-linearised Goodwin model and the switching points of its limit cycle |
| `Fig6/Main.py` | 3D rendering of the Goodwin limit cycle and the zero-curvature surface; also writes `data.csv` for `Fig6.nb` |

| Other | Description |
| :--- | :--- |
| `Fig5/InitialAB_for thesis.nb` | Derivation of the convexity condition and the limit-cycle solution (Mathematica) |
| `Fig6/Fig6.nb` | Reads `data.csv` from the same directory and draws the 3D figure (Mathematica) |

## A1 … A3 — appendix figures

| Script | Description |
| :--- | :--- |
| `A1/Main_A1.py` | How the curvature of the stochastic FitzHugh–Nagumo and Stuart–Landau models changes with the smoothing width |
| `A2/Main_A2.py` | How the curvature of the measured data changes with the smoothing width |
| `A3/Main_A3.py` | Curvature analysis of each nuclear and cytoplasmic mRNA variable of Kim–Forger (one figure per variable) |
| `A3/Main_A3_grid.py` | The same results collected into a single grid figure |
| `A3/Scan/Main_scan.py` | Scans all 180 variables of the detailed Kim–Forger model for those whose curvature has an inflection point |
