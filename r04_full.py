"""Revision: full S-I-R model.  (a) stall map over mu and d1; (b) diagnostics of the yearly
advance (why year 4 jumps back); (c) boundary driven by I + rho R; (d) agreement of the
full and reduced models near mu*, for more mu values and three focus sizes."""
from _c import *
from sirfb.core import *
from sirfb.ext import *
D70 = 1 / (70 * 365)
q0 = DENGUE.with_(delta=D70)
hs = h0_star(q0)
T = 365.0
out = {"h0": 2 * hs, "delta": D70}
VRED = {1: 0.147, 2: 0.219, 3: 0.27, 5: 0.327, 10: 0.41, 20: 0.49, 50: 0.58}
# ---------------- (a) stall map ----------------
mus = [1, 2, 3, 5, 10, 20, 50]
d1s = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
STALL = []
M5 = np.zeros((len(d1s), len(mus)))
M6 = np.zeros_like(M5)
Y1 = np.zeros_like(M5)
for i, d1 in enumerate(d1s):
    for j, mu in enumerate(mus):
        o = run_full(q0.with_(d1=d1), 2 * hs, mu, 6, vmax=1.15 * VRED[mu])
        a = o["annual_adv"]
        Y1[i, j] = a[0]
        M5[i, j] = 0.5 * (a[4] + a[5])
        M6[i, j] = a[5]
        print("d1", d1, "mu", mu, np.round(a, 2), "edge", o["edge"], flush=True)
        STALL.append(dict(d1=d1, mu=mu, annual_adv=a))
out["map"] = dict(mu=mus, d1=d1s, adv_year1=Y1, adv_years56=M5, adv_year6=M6,
                  reduced_km_per_year=[VRED[m] * 365 for m in mus])
save("r04_stall_table", dict(rows=STALL))
save("r04_full", out)
# ---------------- (b) diagnostics ----------------
diag = {}
for name, q, mu in (("base", q0, 2.0), ("d1=0", q0.with_(d1=0.0), 2.0),
                    ("delta=1/5y", q0.with_(delta=1 / (5 * 365)), 2.0),
                    ("eps=0", q0.with_(eps=0.0), 2.0)):
    o = run_full(q, 2 * hs, mu, 8, vmax=0.3)
    sl = slice(None, None, 2)
    diag[name] = dict(annual_adv=o["annual_adv"], t=o["t"][sl], h=o["h"][sl], Imax=o["Imax"][sl],
                      Smean=o["Smean"][sl], Sfront=o["Sfront"][sl], Rfront=o["Rfront"][sl])
    # S near the front at the start of each high season (day 182 of each year) and annual peak of I
    yrs = np.arange(int(o["t"][-1] // T))
    diag[name]["S_front_jul1"] = [float(np.interp(y * T + 182, o["t"], o["Sfront"])) for y in yrs]
    diag[name]["Imax_peak"] = [float(o["Imax"][(o["t"] >= y * T) & (o["t"] < (y + 1) * T)].max()) for y in yrs]
    print(name, np.round(o["annual_adv"], 3), np.round(diag[name]["S_front_jul1"], 3),
          ["%.1e" % v for v in diag[name]["Imax_peak"]], flush=True)
out["diagnostics"] = diag
save("r04_full", out)
# ---------------- (c) boundary law ----------------
bl = {}
for rho in (0.0, 0.01, 0.1, 1.0):
    o = run_full(q0, 2 * hs, 2.0, 6, rho=rho, vmax=0.6)
    bl[str(rho)] = dict(annual_adv=o["annual_adv"], t=o["t"][::2], h=o["h"][::2])
    print("rho", rho, np.round(o["annual_adv"], 2), flush=True)
out["boundary_law"] = bl
save("r04_full", out)
# ---------------- (d) agreement near mu* ----------------
agree = []
for h0 in (0.6, 1.0, 1.4):
    ms = mu_star(q0, h0, hs, amp=0.01, dx=0.05, dt=0.125, rtol=1e-3, hi=256.0)
    for f in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0):
        mu = f * ms
        rr = simulate(q0, h0, mu, 8 * T, dx=0.05, dt=0.125, amp=0.01, spread_len=2 * hs, vanish_tol=1e-60)
        rf = simulate(q0, h0, mu, 8 * T, dx=0.05, dt=0.125, amp=0.01, model="full", vanish_tol=1e-60,
                      L=60.0, spread_len=2 * hs + 10, keep_state=True)
        red = "spreading" if rr.status == "spreading" or (rr.h - rr.g) > 2 * hs else "vanishing"
        if rf.status == "spreading":
            full = "spreading"
        elif rf.status == "vanishing" or (rf.h - rf.g) <= 2 * hs:
            full = "vanishing"
        else:
            full = "undecided"
        agree.append(dict(h0=h0, mu_star_red=ms, factor=f, mu=mu, reduced=red, full=full,
                          full_width=rf.h - rf.g, red_width=rr.h - rr.g, full_status=rf.status))
        print(agree[-1], flush=True)
out["agreement"] = agree
out["agreement_summary"] = dict(n=len(agree), same=sum(a["reduced"] == a["full"] for a in agree))
save("r04_full", out)
