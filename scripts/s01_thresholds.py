"""Threshold quantities: R0T, lambda_p, R0D, h0*, c*; Proposition 1 check (Floquet)."""
from _common import *
from scipy.integrate import solve_ivp
from sirfb.core import *

out = {}
for p in (DENGUE, INFLUENZA, CHOLERA):
    cs, lam = kpp_speed(p)
    r0t = p.beta(np.linspace(0, p.T, 3651)) / (p.gamma + p.delta)
    out[p.name] = dict(R0T=p.R0T, R0t_min=r0t.min(), R0t_max=r0t.max(), h0_star=h0_star(p),
                       c_star=cs, lam_star=lam, rbar=p.rbar)

p = DENGUE
radii = [0.5, 1.0, 1.5, out["dengue"]["h0_star"], 2.5, 4.0, 8.0, 16.0, 32.0, 64.0, 80.0]
tab = []
for h in radii:
    lp, r = lambda_p(p, h), R0D(p, h)
    tab.append(dict(h0=h, lambda_p=lp, R0D=r, agree=bool(np.sign(round(lp, 12)) == np.sign(round(r - 1, 12)))))
out["identity_table"] = tab
grid = np.geomspace(0.25, 80, 22)
r0d = np.array([R0D(p, h) for h in grid])
out["monotonicity"] = dict(radii=grid, R0D=r0d, strictly_increasing=bool(np.all(np.diff(r0d) > 0)),
                           limit_R0T=p.R0T)

# Proposition 1: Floquet exponent of the discretised periodic linear problem equals
# d2(mu1-1)+rbar (the time-average), independent of eps and phi.
prop = []
for h0 in (1.0, 4.0):
    n = 121
    x = np.linspace(-h0, h0, n); dx = x[1] - x[0]
    w = np.full(n, dx); w[0] = w[-1] = dx / 2
    K = p.kernel.J(x[:, None] - x[None, :]) * w[None, :]
    for eps in (0.0, 0.6, 0.95):
        q = p.with_(eps=eps)
        f = lambda t, u: q.d2 * (K @ u - u) + (q.beta(t) * q.S_star - q.gamma - q.delta) * u
        u = np.ones(n)
        logs = []
        for k in range(6):
            sol = solve_ivp(f, (k * q.T, (k + 1) * q.T), u, rtol=1e-11, atol=1e-40, method="DOP853")
            un = sol.y[:, -1]
            logs.append(math.log(np.linalg.norm(un) / np.linalg.norm(u)) / q.T)
            u = un / np.linalg.norm(un)
        prop.append(dict(h0=h0, eps=eps, floquet=logs[-1], formula=lambda_p(q, h0, n=n)))
out["proposition1_check"] = prop
save("s01_thresholds", out)
for k, v in out.items():
    print(k, v if not isinstance(v, list) else v[:3])
