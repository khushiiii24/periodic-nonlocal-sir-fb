"""Solver verification: grid/time-step refinement, domain truncation, semi-wave check,
mean seasonal speed (Richardson-extrapolated) and the 'seasonal ratchet'."""
from _common import *
from sirfb.core import *
from sirfb.semiwave import semiwave_speed

p = DENGUE
out = {}
T = p.T
# (a) dx refinement at dt = 0.25 ; (b) dt refinement at dx = 0.25
out["dx_refinement"] = []
for dx in (2.0, 1.0, 0.5, 0.25, 0.125):
    r = simulate(p, 5.0, 2.0, 730, dx=dx, dt=0.25, amp=0.1)
    out["dx_refinement"].append(dict(dx=dx, h730=r.h, speed_yr2=r.speed_over(365, 730)))
out["dt_refinement"] = []
for dt in (0.5, 0.25, 0.125, 0.0625, 0.03125):
    r = simulate(p, 5.0, 2.0, 730, dx=0.25, dt=dt, amp=0.1)
    out["dt_refinement"].append(dict(dt=dt, h730=r.h, speed_yr2=r.speed_over(365, 730)))
# (c) domain truncation
out["domain"] = []
for L in (400.0, 800.0, 1600.0):
    r = simulate(p, 5.0, 2.0, 730, dx=0.5, dt=0.25, amp=0.1, L=L)
    out["domain"].append(dict(L=L, h730=r.h))
# (d) long-run mean speed, years 3-5, dt -> 0 (Richardson, first order)
mean = []
for dt in (0.25, 0.125, 0.0625, 0.03125):
    r = simulate(p, 5.0, 2.0, 5 * T, dx=0.25, dt=dt, amp=0.1)
    mean.append(dict(dt=dt, cbar=r.speed_over(3 * T, 5 * T), mono=bool(np.all(np.diff(r.H) >= 0) and np.all(np.diff(r.G) <= 0))))
c = [m["cbar"] for m in mean]
out["cbar_seasonal"] = dict(runs=mean, richardson=2 * c[-1] - c[-2])
# (e) autonomous case eps=0: simulation vs independent semi-wave solve
q = p.with_(eps=0.0)
sim0 = []
for dt in (0.125, 0.0625, 0.03125):
    r = simulate(q, 5.0, 2.0, 5 * T, dx=0.25, dt=dt, amp=0.1)
    sim0.append(dict(dt=dt, c=r.speed_over(3 * T, 5 * T)))
c0s = [s["c"] for s in sim0]
sw = []
for L, dz in ((60.0, 0.1), (60.0, 0.05), (80.0, 0.025)):
    c0, z, phi = semiwave_speed(q, 2.0, L, dz)
    sw.append(dict(L=L, dz=dz, c0=c0))
out["autonomous"] = dict(sim=sim0, sim_richardson=2 * c0s[-1] - c0s[-2], semiwave=sw,
                         semiwave_extrap=sw[-1]["c0"] + (sw[-1]["c0"] - sw[-2]["c0"]) / 3)
# (f) seasonal ratchet: monthly mean front speed in year 3 (dx .25, dt .0625)
r = simulate(p, 5.0, 2.0, 4 * T, dx=0.25, dt=0.0625, amp=0.1, rec_dt=1.0)
edges = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) + 2 * T
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
spd = [r.speed_over(edges[i], edges[i + 1]) for i in range(12)]
out["ratchet"] = dict(months=months, speed=spd, annual_km=float(np.interp(3 * T, r.t, r.H) - np.interp(2 * T, r.t, r.H)),
                      ratio=max(spd) / min(spd), t=r.t[::5], h=r.H[::5])
save("s02_convergence", out)
print(json.dumps({k: v for k, v in out.items() if k != "ratchet"}, indent=1, default=str)[:3000])
print("ratchet", [round(s, 4) for s in spd], out["ratchet"]["annual_km"], out["ratchet"]["ratio"])
