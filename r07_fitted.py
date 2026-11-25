"""Revision: key results for the seasonal profile fitted to Delhi data (r03)."""
from _c import *
from sirfb.core import *
from sirfb.ext import *
fit = load("r03_fit")["fit"]
EPS, PHI = round(fit["eps"], 2), round(fit["phi"])
p = DENGUE.with_(eps=EPS, phi=float(PHI))
hs = h0_star(p)
def shifted(q, t_in):
    return q.with_(phi=(q.phi - t_in) % q.T)
dates = np.arange(0, 365, 365 / 24)
out = dict(eps=EPS, phi=PHI, h0_star=hs)
for lab, Imin, sl in (("deterministic", 1e-60, None), ("Imin=1e-6", 1e-6, 20.0)):
    row = []
    for tin in dates:
        m = mu_star_x(shifted(p, tin), 1.0, hs, amp=0.01, dx=0.05, dt=0.125, rtol=1e-3, hi=64.0,
                      vanish_tol=Imin, spread_len=sl, mu_cap=1e4)
        row.append(dict(t_in=float(tin), mu_star=m))
    ms = np.array([r["mu_star"] for r in row])
    out[lab] = dict(rows=row, max=ms.max(), min=ms.min(), ratio=ms.max() / ms.min(),
                    argmax=float(dates[ms.argmax()]), argmin=float(dates[ms.argmin()]))
    print(lab, EPS, PHI, np.round(ms, 3), ms.max() / ms.min(), flush=True)
a = periodic_semiwave(p, 2.0, L=60.0, dz=0.05, years=5)["cbar"]
b = periodic_semiwave(p, 2.0, L=60.0, dz=0.025, years=5)["cbar"]
out["cbar"] = b + (b - a) / 3
r01 = load("r01_semiwave")
c0 = [r for r in r01["mu_sweep"] if r["mu"] == 2][0]
c0 = c0["c0_dz025"] + (c0["c0_dz025"] - c0["c0_dz05"]) / 3
out["c0"] = c0
out["reduction_pct"] = 100 * (1 - out["cbar"] / c0)
print("speed", out["cbar"], c0, out["reduction_pct"])
save("r07_fitted", out)
