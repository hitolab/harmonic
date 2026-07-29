"""Fig5 B: initial-value conditions under which the limit cycle is convex.

For each lambda, shade the set of initial values for which
Kn = xdot*zdot - ydot^2 never changes sign (i.e. the orbit is convex) within
each of the two linear modes of the piecewise-linear Goodwin system
(n -> infinity), and overlay the switching points of the actual limit cycle
as x marks.

The switching points come from the six equations of Appendix B (continuity,
periodicity and the two switching conditions on the analytic solution of each
mode). Four of them are linear in the entry coordinates and are eliminated in
closed form, leaving two equations in the dwell times (tA, tB) which are solved
by Newton's method. Those two are used exactly as they come out -- the
denominators are NOT cleared (cf. eq1Ori / eq2Ori in InitialAB_for thesis.nb).

Region condition (equivalent to: discriminant of Kn * 2exp(2*t*lambda) with
respect to t is negative):
    z <  1 :  x^2 - 2*x*L^2 + y^2*L^2 - 2*y + 2*L > 0
    z >= 1 :  x^2 - 2*x*L^2 + y^2*L^2             > 0
Both describe the exterior of an ellipse, so the panel is filled first and the
ellipse interior is then punched out in white. This keeps the boundary smooth
regardless of grid resolution.

lambda is swept continuously from 0.98 down to 0.02. For every lambda the
converged solution, its Newton residual and whether each switching point lies
inside the convex region are written to plot_data/, and the figure is built
from that file.
"""
import os
import sys
import argparse

import numpy as np
import sympy as sp
import mpmath as mp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.lines import Line2D
from scipy.optimize import root

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Lib.harmonicity as hl
import Lib.plot_utils as pu

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LAMBDAS = [0.1, 0.3, 0.5, 0.7, 0.9]              # panels shown in Fig.5B
LAM_HI, LAM_LO, LAM_STEP = 0.98, 0.02, 0.002     # continuous sweep

AX_MAX = 1.2
AX_TICKS = [0, 1]

FILL_A = '#F7DCC4'   # z < 1  region (pale peach)
FILL_B = '#CFE2DD'   # z >= 1 region (pale teal)
MARK_A = '#3F9A8F'   # x mark for z < 1  (teal)
MARK_B = '#E07B39'   # x mark for z >= 1 (orange)
PUNCH = 'white'      # ellipse interior (where the condition does not hold)

CACHE_NAME = 'Fig5B_lambda_sweep'

# Settings for the verification RK4. The measured error is dominated by the
# RK4 truncation error (dt^4): about 4e-13 with 2000 steps per mode. Raising
# the step count drives it down to a floor of 2e-16..4e-15, which is the
# accuracy of the float64 root itself. Arithmetic precision is not the
# limiting factor -- sweeping dps from 16 to 40 leaves the measured value
# unchanged -- so dps only needs to sit slightly above double so that rounding
# on the verification side does not contaminate the measurement.
VERIFY_STEPS = 2000
VERIFY_DPS = 20


# -----------------------------------------------------------------------------
# Stage 1: the two-equation system in (tA, tB)
# -----------------------------------------------------------------------------
def build_system():
    """Build the reduced two-variable system and the closed-form entry points.

    Unknowns of the full problem (xA0, yA0, xB0, yB0, tA, tB):
      (xA0, yA0) = entry point of mode A (z<1, production ON); the point where
                   the orbit crosses z=1 with zdot<0
      (xB0, yB0) = entry point of mode B (z>=1, production OFF); crossing z=1
                   with zdot>0
      tA, tB     = dwell time in each mode
    In both modes z starts at 1 and returns to 1.

    The four continuity/periodicity equations are linear in
    (xA0, yA0, xB0, yB0) and are eliminated in closed form. Substituting them
    into the two switching conditions leaves two equations in (tA, tB).

    Those two are kept exactly as `zA - 1` and `zB - 1`; the denominator
    2*exp(tA*L)*(exp((tA+tB)*L)-1)^2*L^3 is deliberately NOT cleared, matching
    eq1Ori / eq2Ori in InitialAB_for thesis.nb.

    Returns (f_F, f_J, f_pts), all callable as f(tA, tB, lam).
    """
    lam, tA, tB = sp.symbols('lam tA tB', positive=True)
    xA0, yA0, xB0, yB0 = sp.symbols('xA0 yA0 xB0 yB0')
    Ea = sp.exp(tA * lam)

    # Analytic solution of mode A (z<1, production ON) advanced by tA
    gAx = sp.exp(-tA*lam) * (-1 + Ea + xA0*lam) / lam
    gAy = sp.exp(-tA*lam) * (-1 + Ea - tA*lam + yA0*lam**2 + xA0*tA*lam**2) / lam**2
    gAz = sp.exp(-tA*lam) * (-2 + 2*Ea - 2*tA*lam - tA**2*lam**2
                             + 2*lam**3 + 2*yA0*tA*lam**3 + xA0*tA**2*lam**3) / (2*lam**3)
    # Analytic solution of mode B (z>=1, production OFF) advanced by tB
    gBx = xB0 * sp.exp(-tB*lam)
    gBy = sp.exp(-tB*lam) * (yB0 + xB0*tB)
    gBz = sp.Rational(1, 2) * sp.exp(-tB*lam) * (2 + 2*yB0*tB + xB0*tB**2)

    # Continuity A -> B and periodicity B -> A: linear in the four coordinates
    solved = sp.solve([sp.Eq(gAx, xB0), sp.Eq(gAy, yB0),
                       sp.Eq(gBx, xA0), sp.Eq(gBy, yA0)],
                      [xA0, yA0, xB0, yB0], dict=True)[0]

    F = sp.Matrix([sp.simplify((gAz - 1).subs(solved)),    # z = 1 at end of mode A
                   sp.simplify((gBz - 1).subs(solved))])   # z = 1 at end of mode B
    J = F.jacobian(sp.Matrix([tA, tB]))
    pts = [solved[v] for v in (xA0, yA0, xB0, yB0)]

    return (sp.lambdify((tA, tB, lam), F, 'numpy'),
            sp.lambdify((tA, tB, lam), J, 'numpy'),
            sp.lambdify((tA, tB, lam), pts, 'numpy'))


# -----------------------------------------------------------------------------
# Stage 2: geometry of the region (ellipse)
# -----------------------------------------------------------------------------
def region_ellipse(L, upper):
    """Return centre and semi-axes of the ellipse interior (where the
    condition fails).

    upper=True  -> z < 1  : centre (L^2, 1/L^2), a = |L^3-1|/L, b = |L^3-1|/L^2
    upper=False -> z >= 1 : centre (L^2, 0),     a = L^2,       b = L
    The two ellipses touch at the single point (L^2, L).
    """
    if upper:
        R = abs(L**3 - 1.0) / L
        return (L**2, 1.0 / L**2), R, R / L
    return (L**2, 0.0), L**2, L


def region_value(x, y, L, upper):
    """Positive where the region is shaded (expanded form, equivalent to the
    discriminant condition DKn < 0)."""
    v = x**2 - 2*x*L**2 + y**2*L**2
    return v - 2*y + 2*L if upper else v


def elliptic_radius(x, y, L, upper):
    """Normalised distance from the ellipse centre.

    r > 1 means the point lies outside the ellipse, i.e. inside the shaded
    convex region. Being dimensionless, r is comparable across lambda, unlike
    the raw value of region_value().
    """
    (cx, cy), a, b = region_ellipse(L, upper)
    return float(np.hypot((x - cx) / a, (y - cy) / b))


def verify_region(L, upper, n=400):
    """Check on a grid that "inside the ellipse" matches region_value <= 0."""
    (cx, cy), a, b = region_ellipse(L, upper)
    g = np.linspace(0, AX_MAX, n)
    X, Y = np.meshgrid(g, g)
    inside_ellipse = ((X - cx) / a)**2 + ((Y - cy) / b)**2 <= 1.0
    inside_ineq = region_value(X, Y, L, upper) <= 0.0
    # Points very close to the boundary can fall either way numerically, so
    # they are excluded from the comparison
    near = np.abs(np.sqrt(((X - cx) / a)**2 + ((Y - cy) / b)**2) - 1.0) < 1e-3
    return int(np.sum((inside_ellipse != inside_ineq) & ~near))


# -----------------------------------------------------------------------------
# Stage 3: continuous sweep over lambda
# -----------------------------------------------------------------------------
def sweep_lambda():
    """Track the limit cycle from LAM_HI down to LAM_LO, one row per lambda.

    tB grows rapidly as lambda decreases (tB ~ 75 at lambda = 0.1), so a fixed
    initial guess collapses onto the trivial solution tA = tB = 0. Each step
    therefore starts from the solution at the previous lambda.
    """
    f_F, f_J, f_pts = build_system()

    def attempt(L, guess):
        """Return the solution if it converged, otherwise None.

        The acceptance test only checks that the dwell times are positive; the
        system also admits the trivial solution tA = tB = 0, which satisfies
        every equation for any coordinates.

        method='hybr' calls MINPACK's hybrj (Powell's hybrid method), which
        switches between the Newton direction and steepest descent inside a
        trust region and reduces to Newton's method near the root. Convergence
        is left to hybr's default xtol = sqrt(eps) ~ 1.49e-8; since Newton
        converges quadratically near a simple root, the iterate is already
        accurate to about eps once the displacement has shrunk to sqrt(eps).
        """
        r = root(lambda v: np.asarray(f_F(v[0], v[1], L), float).ravel(), guess,
                 jac=lambda v: np.asarray(f_J(v[0], v[1], L), float), method='hybr')
        return r.x if (r.success and min(r.x) > 1e-3) else None

    lams = [round(x, 5) for x in np.arange(LAM_HI, LAM_LO - 1e-9, -LAM_STEP)]
    rows, guess, n_fail = [], np.array([3.5, 0.9]), 0
    for L in lams:
        s = attempt(L, guess)
        if s is None:
            n_fail += 1
            continue
        guess = s
        tA_v, tB_v = s
        xa, ya, xb, yb = f_pts(tA_v, tB_v, L)
        rA = elliptic_radius(xa, ya, L, upper=True)
        rB = elliptic_radius(xb, yb, L, upper=False)
        rows.append(dict(
            lam=L, tA=tA_v, tB=tB_v, period=tA_v + tB_v,
            xA0=xa, yA0=ya, xB0=xb, yB0=yb,
            residual=float(np.max(np.abs(np.asarray(f_F(tA_v, tB_v, L), float)))),
            r_A=rA, r_B=rB,
            inside_A=int(rA > 1.0), inside_B=int(rB > 1.0),
        ))
    return rows, n_fail


# -----------------------------------------------------------------------------
# Stage 4: independent verification by integrating the piecewise-linear system
# -----------------------------------------------------------------------------
def _rhs_on(state, params):
    """Right-hand side for z < 1 (production ON)."""
    x, y, z = state
    (lamb,) = params
    return (1 - lamb*x, x - lamb*y, y - lamb*z)


def _rhs_off(state, params):
    """Right-hand side for z >= 1 (production OFF)."""
    x, y, z = state
    (lamb,) = params
    return (-lamb*x, x - lamb*y, y - lamb*z)


def check_switching_points(row, n_steps=VERIFY_STEPS, dps=VERIFY_DPS, tol=1e-8):
    """Verify each of the two switching points with an independent one-way map.

    eps_A: distance between beta = (xB0, yB0, 1) as reported in the solution
           and the point reached by integrating the z<1 mode for a time tA
           from alpha = (xA0, yA0, 1)
    eps_B: distance between alpha and the point reached by integrating the
           z>=1 mode for a time tB from beta

    Integrating one full period and only checking the distance back to the
    starting point would replace the intermediate beta with the integrated
    value, so beta itself would never be verified -- and beta is exactly the x
    mark in the bottom row.

    The switching times are known, so a step size of tA/n_steps makes the final
    step land exactly on the switching point. Each mode is therefore a smooth
    linear system integrated over a fixed interval and RK4 attains its full
    order; handing the integrator a right-hand side that branches at z=1 would
    let one step straddle the switch and stall the error near 1e-3.

    Also checks that z stays on the correct side within each mode: the six
    equations are only a necessary condition, since a solution in which z
    crosses back partway through a mode is not a valid orbit.
    """
    lamb = mp.mpf(str(row['lam']))
    alpha = (mp.mpf(str(row['xA0'])), mp.mpf(str(row['yA0'])), mp.mpf(1))
    beta = (mp.mpf(str(row['xB0'])), mp.mpf(str(row['yB0'])), mp.mpf(1))
    dist = lambda p, q: float(mp.sqrt(sum((a - b)**2 for a, b in zip(p, q))))

    trA = hl.runge_iterate_mp(_rhs_on, alpha, mp.mpf(str(row['tA'])) / n_steps,
                              n_steps, params=(lamb,), dps=dps)
    trB = hl.runge_iterate_mp(_rhs_off, beta, mp.mpf(str(row['tB'])) / n_steps,
                              n_steps, params=(lamb,), dps=dps)
    eps_A, eps_B = dist(trA[-1], beta), dist(trB[-1], alpha)

    # The endpoints sit exactly at z = 1, so only interior points are inspected
    z_ok = (max(float(s[2]) for s in trA[1:-1]) < 1.0
            and min(float(s[2]) for s in trB[1:-1]) > 1.0)
    return (eps_A, eps_B,
            (min(float(s[2]) for s in trA), max(float(s[2]) for s in trB)),
            max(eps_A, eps_B) < tol and z_ok)


# -----------------------------------------------------------------------------
# Stage 5: plotting
# -----------------------------------------------------------------------------
def make_figure(rows):
    """Build Fig.5B from the swept rows, picking the panels listed in LAMBDAS."""
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    lam_all = np.array([r['lam'] for r in rows])
    panels = [rows[int(np.argmin(np.abs(lam_all - L)))] for L in LAMBDAS]

    fig, axes = plt.subplots(2, len(LAMBDAS), figsize=(9.6, 4.5))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.82, bottom=0.13,
                        wspace=0.18, hspace=0.10)

    for col, row_data in enumerate(panels):
        L = row_data['lam']
        for r, upper in enumerate((True, False)):
            ax = axes[r, col]
            fill = FILL_A if upper else FILL_B
            mark = MARK_A if upper else MARK_B
            px, py = ((row_data['xA0'], row_data['yA0']) if upper
                      else (row_data['xB0'], row_data['yB0']))

            # Fill the whole panel, then punch out the ellipse interior in
            # white, so the boundary stays analytically smooth
            ax.add_patch(Rectangle((0, 0), AX_MAX, AX_MAX,
                                   facecolor=fill, edgecolor='none', zorder=0))
            (cx, cy), a, b = region_ellipse(L, upper)
            ax.add_patch(Ellipse((cx, cy), 2*a, 2*b, facecolor=PUNCH,
                                 edgecolor='none', zorder=1))

            ax.plot(px, py, marker='x', color=mark, markersize=6.5,
                    markeredgewidth=1.8, linestyle='none', zorder=3)

            ax.set_xlim(0, AX_MAX)
            ax.set_ylim(0, AX_MAX)
            ax.set_xticks(AX_TICKS)
            ax.set_yticks(AX_TICKS)
            ax.set_box_aspect(1)
            ax.tick_params(direction='out', labelsize=9, length=3)
            for s in ax.spines.values():
                s.set_linewidth(0.8)
            if col != 0:
                ax.set_yticklabels([])
            if r != 1:
                ax.set_xticklabels([])

    # --- Row labels ---
    for r, lab in enumerate((r'$z < 1$', r'$z \geq 1$')):
        p = axes[r, 0].get_position()
        fig.text(p.x0 - 0.055, (p.y0 + p.y1) / 2, lab,
                 ha='right', va='center', fontsize=12)

    # --- Axis labels ---
    p_left, p_right = axes[1, 0].get_position(), axes[1, -1].get_position()
    fig.text((p_left.x0 + p_right.x1) / 2, 0.035, r'$x$',
             ha='center', va='center', fontsize=13)
    p_top, p_bot = axes[0, 0].get_position(), axes[1, 0].get_position()
    fig.text(p_left.x0 - 0.088, (p_bot.y0 + p_top.y1) / 2, r'$y$',
             ha='center', va='center', fontsize=13)

    # --- lambda header (rule + column labels) ---
    top = axes[0, 0].get_position().y1
    y_rule = top + 0.055
    x0, x1 = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
    fig.add_artist(Line2D([x0, x1], [y_rule, y_rule], transform=fig.transFigure,
                          color='black', linewidth=0.8))
    fig.text((x0 + x1) / 2, y_rule + 0.035, r'$\lambda$',
             ha='center', va='bottom', fontsize=13)
    for col, row_data in enumerate(panels):
        p = axes[0, col].get_position()
        fig.text((p.x0 + p.x1) / 2, y_rule - 0.045, f"{row_data['lam']:g}",
                 ha='center', va='bottom', fontsize=11)

    # --- Panel letter ---
    fig.text(0.012, 0.965, 'B', ha='left', va='top',
             fontsize=17, fontweight='bold')

    return fig, panels


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fig5 B: convex region + limit-cycle switching points, swept over lambda")
    parser.add_argument('-f', '--force', action='store_true',
                        help="Force recalculation (bypass cache)")
    args = parser.parse_args()

    cached = None if args.force else hl.load_plot_data(CACHE_NAME)
    if cached is not None:
        rows = cached.to_dict('records')
        n_fail = 0
    else:
        print(f"Sweeping lambda from {LAM_HI} down to {LAM_LO} in steps of {LAM_STEP}...")
        rows, n_fail = sweep_lambda()
        hl.save_plot_data({k: [r[k] for r in rows] for k in rows[0]}, CACHE_NAME)

    lam = np.array([r['lam'] for r in rows])
    res = np.array([r['residual'] for r in rows])
    rA = np.array([r['r_A'] for r in rows])
    rB = np.array([r['r_B'] for r in rows])
    inside = np.array([bool(r['inside_A']) and bool(r['inside_B']) for r in rows])

    print("\n=== lambda sweep ===")
    print(f"  covered            : {lam.min():.3f} .. {lam.max():.3f} "
          f"({len(rows)} points, {n_fail} steps did not converge)")
    print(f"  max Newton residual: {res.max():.2e}")
    print(f"  min r_A            : {rA.min():.6f}  at lambda = {lam[rA.argmin()]:.3f}")
    print(f"  min r_B            : {rB.min():.6f}  at lambda = {lam[rB.argmin()]:.3f}")
    print(f"  inside the region  : {inside.sum()} / {len(rows)}")
    if not inside.all():
        bad = lam[~inside]
        raise RuntimeError("switching point outside the convex region for lambda in "
                           f"[{bad.min():.3f}, {bad.max():.3f}]")

    fig, panels = make_figure(rows)

    print("\n=== panels of Fig.5B (verified against the piecewise-linear system) ===")
    print("lambda      tA          tB     alpha=(x,y) z<1 mark   beta=(x,y) z>=1 mark"
          "  eps_A    eps_B     r_A      r_B")
    for row in panels:
        eps_A, eps_B, (z_min_A, z_max_B), ok = check_switching_points(row)
        print(f"{row['lam']:5.2f} {row['tA']:10.6f} {row['tB']:11.6f} "
              f"({row['xA0']:8.6f},{row['yA0']:9.6f}) "
              f"({row['xB0']:9.6f},{row['yB0']:9.6f}) "
              f"{eps_A:8.1e} {eps_B:8.1e} {row['r_A']:8.4f} {row['r_B']:8.4f} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            raise RuntimeError(
                f"verification failed for lambda={row['lam']}: "
                f"eps_A={eps_A:.2e}, eps_B={eps_B:.2e}, "
                f"z range A/B = {z_min_A:.4f}/{z_max_B:.4f}")

    bad = sum(verify_region(r['lam'], u) for r in panels for u in (True, False))
    print(f"\nregion patch vs inequality: {bad} mismatching grid points (expected 0)")
    if bad:
        raise RuntimeError("ellipse patch does not match the inequality")

    pu.save_fig(fig, __file__, 'Fig5_B_convex_region')
    plt.close(fig)


if __name__ == "__main__":
    main()
