"""High-precision (mpmath) companion to model_goodwin.cpp, for use with
Lib.harmonicity.runge_iterate_mp, plus the analytic Hopf-bifurcation boundary
for this model."""

MP_DPS = 32  # mpmath precision (decimal digits) for the high-precision Runge variant


def goodwin_rhs_mp(state, params):
    """Same equations as Models/goodwin/model_goodwin.cpp, for runge_iterate_mp."""
    x, y, z = state
    lamb, n = params
    dx = 1 / (1 + z ** n) - lamb * x
    dy = x - lamb * y
    dz = y - lamb * z
    return (dx, dy, dz)


def lambda_boundary(n):
    """(n/(n-8)) * (8/(n-8))^(1/n) * lambda^3 = 1  ->  lambda(n), valid for n > 8.
    The analytic Hopf-bifurcation boundary in the (n, lambda) plane -- below
    this curve the Goodwin oscillator has a limit cycle, above it a fixed
    point (Griffith, 1968; the n > 8 requirement itself is independent of
    lambda)."""
    return ((n - 8.0) / n * ((n - 8.0) / 8.0) ** (1.0 / n)) ** (1.0 / 3.0)


def boundary_expr(n, lam):
    """(n/(n-8)) * (8/(n-8))^(1/n) * lambda^3 -- the raw LHS value (not set
    to 1), evaluated at an actual (n, lambda) grid point. Values above 1 sit
    on the fixed-point side of lambda_boundary(); values below 1 sit on the
    limit-cycle side. The formula's derivation assumes n > 8, so for n <= 8
    this is just a numerical curiosity, not a physically meaningful boundary.
    For odd n <= 8 (e.g. n=5,7), 8/(n-8) is negative; Python's ** would
    return a complex principal-branch value for a negative base with
    fractional exponent, so the real odd root is taken explicitly instead."""
    ratio = 8.0 / (n - 8.0)
    if ratio < 0:
        root = -((-ratio) ** (1.0 / n))
    else:
        root = ratio ** (1.0 / n)
    return (n / (n - 8.0)) * root * lam ** 3
