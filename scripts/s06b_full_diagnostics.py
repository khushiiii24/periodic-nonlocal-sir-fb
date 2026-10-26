"""Diagnostics for the full S-I-R front: why does the free boundary stall?
Varies seasonality, mu, demographic turnover delta, S-dispersal and the time step."""
from _common import *
from sirfb.core import *

D70 = 1 / (70 * 365)
cases = {
    "base (eps=.6, mu=2, delta=1/70y)": dict(),
    "no seasonality (eps=0)": dict(eps=0.0),
    "mu=50": dict(mu=50.0),
    "mu=400": dict(mu=400.0),
    "fast turnover delta=1/1y": dict(delta=1 / 365),
    "fast turnover delta=1/5y": dict(delta=1 / (5 * 365)),
    "no S dispersal (d1=0)": dict(d1=0.0),
    "dt=0.0625": dict(dt=0.0625),
}
out = {}
T = 365.0
for name, c in cases.items():
    c = dict(c)
    mu = c.pop("mu", 2.0)
    dt = c.pop("dt", 0.125)
    q = DENGUE.with_(delta=c.pop("delta", D70), **c)
    hs = h0_star(q)
    yrs = 6
    L = 3.7 + 1.1 * 1.0 * yrs * T + 100 if mu < 60 else 3.7 + 1.0 * yrs * T * 1.1 + 100
    r = simulate(q, 2 * hs, mu, yrs * T, dx=0.5, dt=dt, amp=0.1, model="full", rec_dt=5.0,
                 vanish_tol=1e-60, L=L, keep_state=True)
    yrs_ = np.arange(0, int(r.t_end // T) + 1)
    out[name] = dict(status=r.status, t_end=r.t_end, years=yrs_, h=np.interp(yrs_ * T, r.t, r.H),
                     annual_disp=np.diff(np.interp(yrs_ * T, r.t, r.H)), t=r.t[::6], H=r.H[::6],
                     Imax=r.Imax[::6], Iend_max=float(r.I.max()), S_min=float(r.S.min()))
    print(name, r.status, np.round(out[name]["annual_disp"], 2), "Imax(end)=%.2e" % r.I.max(), flush=True)
save("s06b_full_diagnostics", out)
