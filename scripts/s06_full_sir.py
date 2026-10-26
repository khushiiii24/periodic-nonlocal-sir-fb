"""Full three-component S-I-R runs (S on the whole line, births/deaths delta = 1/(70 y))
versus the reduced Fisher-KPP surrogate."""
from _common import *
from sirfb.core import *

DELTA = 1 / (70 * 365)
out = {"delta": DELTA}
rows = []
for q0 in (DENGUE, INFLUENZA, CHOLERA):
    q = q0.with_(delta=DELTA)
    hq = h0_star(q)
    dx = min(0.5, q.kernel.s / 4)
    h0 = 2 * hq
    rr = simulate(q, h0, 2.0, 5 * q.T, dx=dx, dt=0.125, amp=0.1, model="reduced")
    rf = simulate(q, h0, 2.0, 5 * q.T, dx=dx, dt=0.125, amp=0.1, model="full", keep_state=True)
    row = dict(name=q.name, h0=h0, reduced_speed=rr.speed_over(3 * q.T, 5 * q.T),
               full_speed=rf.speed_over(3 * q.T, 5 * q.T), full_status=rf.status,
               reduced_peakI=float(rr.Imax[-1]), full_peakI_late=float(np.max(rf.Imax[-365:])))
    row["ratio"] = row["full_speed"] / row["reduced_speed"]
    # final susceptible fraction behind the front and SIR final-size check
    inside = (rf.x > rf.g + 20) & (rf.x < rf.h - 20)
    row["S_behind_median"] = float(np.median(rf.S[inside])) if inside.any() else None
    rows.append(row)
    print(row, flush=True)
    if q.name == "dengue":
        out["profile"] = dict(x=rf.x[::4], S=rf.S[::4], I=rf.I[::4], R=rf.R[::4], g=rf.g, h=rf.h, t=rf.t_end)
        rprof = simulate(q, h0, 2.0, 5 * q.T, dx=dx, dt=0.125, amp=0.1, model="reduced", keep_state=True)
        out["profile_reduced"] = dict(x=rprof.x[::4], I=rprof.I[::4], g=rprof.g, h=rprof.h)
        out["traj"] = dict(t=rf.t[::7], h_full=rf.H[::7], h_red=np.interp(rf.t[::7], rr.t, rr.H))
out["speeds"] = rows

# thresholds: dichotomy outcome of full vs reduced around mu*_reduced (dengue, h0 = 1, amp 0.01)
q = DENGUE.with_(delta=DELTA)
hs = h0_star(q)
ms = mu_star(q, 1.0, hs, amp=0.01, dx=0.05, dt=0.125, rtol=1e-3)
cmp = []
for f in (0.8, 0.9, 0.95, 1.05, 1.1, 1.25):
    rr = simulate(q, 1.0, f * ms, 8 * q.T, dx=0.05, dt=0.125, amp=0.01, spread_len=2 * hs, vanish_tol=1e-60)
    rf = simulate(q, 1.0, f * ms, 8 * q.T, dx=0.05, dt=0.125, amp=0.01, model="full", vanish_tol=1e-60,
                  L=60.0, spread_len=2 * hs + 10)
    cmp.append(dict(factor=f, mu=f * ms, reduced=rr.status, full=rf.status, full_width=rf.h - rf.g,
                    red_width=rr.h - rr.g))
    print(cmp[-1], flush=True)
out["threshold_compare"] = dict(mu_star_reduced=ms, rows=cmp)
save("s06_full_sir", out)
