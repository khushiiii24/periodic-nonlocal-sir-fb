"""Revision: seasonality that varies with position (monsoon-type gradient).  The critical
half width is computed from the Floquet exponent of the linearised problem and compared with
the closed form h0* that Proposition 1 gives for position-independent forcing."""
from _c import *
from scipy.optimize import brentq
from sirfb.core import *
from sirfb.ext import *
p = DENGUE
hs = h0_star(p)
W = 2 * math.pi / p.T
def crit(beta_tx):
    f = lambda ell: floquet_lambda(p, ell, beta_tx, n=101, periods=5)
    return brentq(f, 0.8, 4.0, xtol=1e-4)
rows = []
# phase gradient: peak day phi(x) = phi0 + kappa x  (kappa in days per km)
for kappa in (0.0, 0.03, 0.3, 1.0, 3.0, 10.0):
    b = lambda t, x, k=kappa: p.beta_bar * (1 + p.eps * np.cos(W * (t - p.phi - k * x)))
    h = crit(b)
    rows.append(dict(kind="phase", value=kappa, h0_star=h, rel_change=h / hs - 1))
    print(rows[-1], flush=True)
# amplitude gradient: eps(x) = eps0 + eta x, clipped to [0, 0.99]
for eta in (0.05, 0.15, 0.3):
    b = lambda t, x, e=eta: p.beta_bar * (1 + np.clip(p.eps + e * x, 0, 0.99) * np.cos(W * (t - p.phi)))
    h = crit(b)
    rows.append(dict(kind="amplitude", value=eta, h0_star=h, rel_change=h / hs - 1))
    print(rows[-1], flush=True)
save("r06_spatial", dict(h0_star_formula=hs, rows=rows))
