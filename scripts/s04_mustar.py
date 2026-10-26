"""Critical expansion coefficient mu*: amplitude scaling, mu*(h0), introduction-date and
seasonal-amplitude effects, (date x eps) map, three-disease table."""
from _common import *
from sirfb.core import *

AMP = 0.01   # reference initial prevalence at the focus centre (I0 = AMP cos(pi x / 2h0))
out = {"AMP": AMP}
p = DENGUE
hs = h0_star(p)
kw = dict(dx=0.05, dt=0.125, rtol=1e-3)

# (a) amplitude scaling at h0 = 1
out["amp_scaling"] = []
for amp in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
    m = mu_star(p, 1.0, hs, amp=amp, **kw)
    out["amp_scaling"].append(dict(amp=amp, mu_star=m, kappa=amp * m))
    print(out["amp_scaling"][-1], flush=True)

# (b) mu*(h0)
out["mu_vs_h0"] = []
for h0 in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.7, 1.75, 1.8, 1.83):
    m = mu_star(p, h0, hs, amp=AMP, hi=256.0, **kw)
    out["mu_vs_h0"].append(dict(h0=h0, mu_star=m))
    print(out["mu_vs_h0"][-1], flush=True)

# (c) introduction date: simulation starts at t=0 = introduction; peak at day 265 of the year
def shifted(q, t_in):
    return q.with_(phi=(q.phi - t_in) % q.T)

out["intro_date"] = []
for t_in in np.arange(0, 365, 365 / 24):
    m = mu_star(shifted(p, t_in), 1.0, hs, amp=AMP, hi=256.0, **kw)
    out["intro_date"].append(dict(t_in=float(t_in), mu_star=m))
    print(out["intro_date"][-1], flush=True)

# (d) seasonal amplitude at fixed introduction on 1 Jan
out["eps_sweep"] = []
for eps in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
    m = mu_star(p.with_(eps=eps), 1.0, hs, amp=AMP, hi=256.0, **kw)
    out["eps_sweep"].append(dict(eps=eps, mu_star=m))
    print(out["eps_sweep"][-1], flush=True)

# (e) map: introduction month x eps
months = np.arange(12) * 365 / 12 + 365 / 24
eps_vals = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
M = np.zeros((len(eps_vals), 12))
for i, eps in enumerate(eps_vals):
    for j, t_in in enumerate(months):
        M[i, j] = mu_star(shifted(p.with_(eps=eps), t_in), 1.0, hs, amp=AMP, hi=1024.0, **kw)
    print("eps", eps, np.round(M[i], 3), flush=True)
out["map"] = dict(t_in=months, eps=eps_vals, mu_star=M)

# (f) three diseases
out["diseases"] = []
for q in (DENGUE, INFLUENZA, CHOLERA):
    hq = h0_star(q)
    h0 = hq / 2
    dx = h0 / 20
    m = mu_star(q, h0, hq, amp=AMP, dx=dx, dt=0.125, rtol=1e-3, hi=256.0)
    sig = q.kernel.s
    r = simulate(q, 2 * hq, 2.0, 5 * q.T, dx=min(0.25, sig / 8), dt=0.0625, amp=0.1)
    cb = r.speed_over(3 * q.T, 5 * q.T)
    cs, _ = kpp_speed(q)
    out["diseases"].append(dict(name=q.name, R0T=q.R0T, R0min=q.beta_bar * (1 - q.eps) / q.gamma,
                                R0max=q.beta_bar * (1 + q.eps) / q.gamma, h0_star=hq, mu_star=m,
                                cbar=cb, cstar=cs, km_per_year=cb * q.T, ratio=cb / cs))
    print(out["diseases"][-1], flush=True)
save("s04_mustar", out)
