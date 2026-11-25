"""Revision: (a) convergence of mu* in dx, dt, cut-off and horizon; (b) minimum-density
(extinction) test; (c) mu* against arrival date for all three diseases at h0 = h0*/2."""
from _c import *
from sirfb.core import *
from sirfb.ext import *
p = DENGUE
T = p.T
AMP = 0.01
hs = h0_star(p)
def shifted(q, t_in):
    return q.with_(phi=(q.phi - t_in) % q.T)
DATES = {"1 Jan": 0.0, "15 Feb": 45.625, "17 Jul": 197.708}
out = {"AMP": AMP, "h0": 1.0, "dates": DATES}
# ---------------- (a) convergence ----------------
conv = []
for lab, tin in DATES.items():
    q = shifted(p, tin)
    for dx, dt, scheme in ((0.1, 0.125, "euler"), (0.05, 0.125, "euler"), (0.025, 0.125, "euler"),
                           (0.05, 0.25, "euler"), (0.05, 0.0625, "euler"), (0.05, 0.03125, "euler"),
                           (0.05, 0.125, "rk2"), (0.05, 0.0625, "rk2"), (0.05, 0.03125, "rk2")):
        t0 = time.time()
        m = mu_star_x(q, 1.0, hs, amp=AMP, dx=dx, dt=dt, scheme=scheme, rtol=5e-4, hi=256.0)
        conv.append(dict(date=lab, dx=dx, dt=dt, scheme=scheme, mu_star=m, sec=time.time() - t0))
        print(conv[-1], flush=True)
    # cut-off and horizon
    for vt, tmax in ((1e-40, 8 * T), (1e-80, 8 * T), (1e-60, 12 * T)):
        m = mu_star_x(q, 1.0, hs, amp=AMP, dx=0.05, dt=0.125, rtol=5e-4, hi=256.0,
                      vanish_tol=vt, t_max=tmax)
        conv.append(dict(date=lab, dx=0.05, dt=0.125, scheme="euler", vanish_tol=vt,
                         t_max_years=tmax / T, mu_star=m))
        print(conv[-1], flush=True)
out["convergence"] = conv
save("r02_mustar", out)
# ---------------- (b) minimum density ----------------
dates = np.arange(0, 365, 365 / 24)
md = {}
for Imin in (1e-60, 1e-8, 1e-6, 1e-4):
    row = []
    for tin in dates:
        m = mu_star_x(shifted(p, tin), 1.0, hs, amp=AMP, dx=0.05, dt=0.125, rtol=1e-3, hi=64.0,
                      vanish_tol=Imin, spread_len=20.0, mu_cap=1e4)
        row.append(dict(t_in=float(tin), mu_star=m))
    md[str(Imin)] = row
    print("Imin", Imin, [round(r["mu_star"], 3) for r in row], flush=True)
out["min_density"] = md
save("r02_mustar", out)
# ---------------- (c) three diseases, h0 = h0*/2 ----------------
dis = {}
for q0 in (DENGUE, INFLUENZA, CHOLERA):
    hq = h0_star(q0)
    h0 = hq / 2
    dx = h0 / 20
    row = []
    for tin in dates:
        m = mu_star_x(shifted(q0, tin), h0, hq, amp=AMP, dx=dx, dt=0.125, rtol=1e-3, hi=256.0)
        row.append(dict(t_in=float(tin), mu_star=m))
    m0 = mu_star_x(q0.with_(eps=0.0), h0, hq, amp=AMP, dx=dx, dt=0.125, rtol=1e-3, hi=256.0)
    ms = [r["mu_star"] for r in row]
    dis[q0.name] = dict(h0=h0, dx=dx, rows=row, no_season=m0, max=max(ms), min=min(ms),
                        ratio=max(ms) / min(ms),
                        argmax=float(dates[int(np.argmax(ms))]), argmin=float(dates[int(np.argmin(ms))]))
    print(q0.name, {k: v for k, v in dis[q0.name].items() if k != "rows"}, flush=True)
out["diseases_date"] = dis
save("r02_mustar", out)
