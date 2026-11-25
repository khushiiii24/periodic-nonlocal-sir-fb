"""Revision: independent check of the SEASONAL mean speed through the periodic semi-wave
of Wang, Li and Dai (front-fixed frame, adaptive DOP853), plus Heun (RK2) time stepping
of the free-boundary solver."""
from _c import *
from sirfb.core import *
from sirfb.ext import *
p = DENGUE
T = p.T
out = {}
# (a) grid refinement of the periodic semi-wave at mu = 2
ref = []
for eps in (0.0, 0.6):
    for L, dz in ((60.0, 0.1), (60.0, 0.05), (60.0, 0.025), (80.0, 0.025)):
        t0 = time.time()
        r = periodic_semiwave(p.with_(eps=eps), 2.0, L=L, dz=dz, years=5, rtol=1e-9, atol=1e-11)
        ref.append(dict(eps=eps, L=L, dz=dz, cbar=r["cbar"], change=r["change"], sec=time.time() - t0))
        print(ref[-1], flush=True)
out["refinement_mu2"] = ref
# (b) all mu values of Fig. 2b
rows = []
for mu in (0.5, 1, 2, 5, 10, 25, 60, 150, 400):
    a = periodic_semiwave(p, mu, L=60.0, dz=0.05, years=5)
    b = periodic_semiwave(p, mu, L=60.0, dz=0.025, years=5)
    a0 = periodic_semiwave(p.with_(eps=0.0), mu, L=60.0, dz=0.05, years=4)
    b0 = periodic_semiwave(p.with_(eps=0.0), mu, L=60.0, dz=0.025, years=4)
    cs = 2 * b["cbar"] - a["cbar"]          # Richardson (first order in dz at the boundary)
    c0 = 2 * b0["cbar"] - a0["cbar"]
    rows.append(dict(mu=mu, cbar_dz05=a["cbar"], cbar_dz025=b["cbar"], cbar_extrap=cs,
                     c0_dz05=a0["cbar"], c0_dz025=b0["cbar"], c0_extrap=c0, ratio=cs / c0,
                     reduction_pct=100 * (1 - cs / c0), periodic_change=b["change"]))
    print(rows[-1], flush=True)
out["mu_sweep"] = rows
# (c) profile and c(t) over one period at mu = 2 (for the figure)
r = periodic_semiwave(p, 2.0, L=60.0, dz=0.05, years=5, return_profile=True)
out["profile"] = dict(z=r["z"][::4], U=r["U"][::4], t=r["t"], c=r["c"])
# (d) Heun (RK2) time stepping of the moving-grid solver: years 3-5 mean speed
rk = []
for dt in (0.5, 0.25, 0.125, 0.0625):
    s = simulate_rk2(p, 5.0, 2.0, 5 * T, dx=0.25, dt=dt, amp=0.1)
    rk.append(dict(dt=dt, cbar=s.speed_over(3 * T, 5 * T), mono=bool(np.all(np.diff(s.H) >= 0))))
    print("rk2", rk[-1], flush=True)
eu = load("s02_convergence")["cbar_seasonal"]["runs"]
out["rk2"] = rk
out["euler"] = eu
# observed orders
c = [x["cbar"] for x in rk]
out["rk2_order"] = [math.log2(abs(c[i] - c[i + 1]) / abs(c[i + 1] - c[i + 2])) for i in range(len(c) - 2)]
print("orders", out["rk2_order"])
save("r01_semiwave", out)
