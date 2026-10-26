"""Sensitivity analysis.
(1) Local normalised sensitivity indices  Y_p^Q = (dQ/dp)(p/Q)  by +/-5% central differences.
(2) Global: Latin-hypercube sample (n=200) + partial rank correlation coefficients (PRCC)
    for the mean front speed c_bar^T and the critical expansion coefficient mu*."""
from _common import *
from scipy import stats
from sirfb.core import *

p0 = DENGUE
T = p0.T
AMP = 0.01

def build(theta):
    b, g, d2, sig, eps = theta[:5]
    return p0.with_(beta_bar=b, gamma=g, d1=d2, d2=d2, d3=d2, kernel=gaussian(sig), eps=eps)

def Q_static(q):
    cs, _ = kpp_speed(q)
    return dict(R0T=q.R0T, h0_star=h0_star(q), c_star=cs)

def Q_speed(q, mu):
    hq = h0_star(q)
    r = simulate(q, max(5.0, 2 * hq), mu, 4 * T, dx=min(0.5, q.kernel.s / 4), dt=0.125, amp=0.1)
    return r.speed_over(2 * T, 4 * T)

def Q_mustar(q, t_in=0.0):
    hq = h0_star(q)
    qq = q.with_(phi=(q.phi - t_in) % q.T)
    return mu_star(qq, 0.5 * hq, hq, amp=AMP, dx=0.5 * hq / 20, dt=0.125, rtol=2e-3, hi=256.0)

names = ["beta_bar", "gamma", "d2", "sigma", "eps", "mu"]
base = np.array([0.35, 1 / 7, 0.5, 2.0, 0.6, 2.0])
h = 0.05
local = {}
Q0 = Q_static(build(base)); Q0["cbar"] = Q_speed(build(base), base[5]); Q0["mu_star"] = Q_mustar(build(base))
local["base"] = Q0
print("base", Q0, flush=True)
for i, nm in enumerate(names):
    up, dn = base.copy(), base.copy()
    up[i] *= 1 + h; dn[i] *= 1 - h
    qu, qd = build(up), build(dn)
    su, sd = Q_static(qu), Q_static(qd)
    su["cbar"] = Q_speed(qu, up[5]); sd["cbar"] = Q_speed(qd, dn[5])
    if nm != "mu":
        su["mu_star"] = Q_mustar(qu); sd["mu_star"] = Q_mustar(qd)
    else:
        su["mu_star"] = sd["mu_star"] = Q0["mu_star"]
    local[nm] = {k: (su[k] - sd[k]) / (2 * h * Q0[k]) if Q0[k] != 0 else 0.0 for k in su}
    print(nm, {k: round(v, 3) for k, v in local[nm].items()}, flush=True)
# introduction date: absolute effect of +/-15 days on mu*
local["intro_date_pm15d"] = dict(plus=Q_mustar(build(base), 15.0), minus=Q_mustar(build(base), -15.0))

# ---------------- global LHS / PRCC ----------------
lo = np.array([0.25, 1 / 10, 0.25, 1.0, 0.2, 0.5, 0.0])
hi = np.array([0.45, 1 / 5, 1.00, 4.0, 0.9, 8.0, 365.0])
gnames = names + ["intro_day"]
n = 200
rng = np.random.default_rng(20261025)
sampler = stats.qmc.LatinHypercube(d=7, seed=rng)
X = lo + sampler.random(n) * (hi - lo)
Y = np.zeros((n, 2))
t0 = time.time()
for s in range(n):
    q = build(X[s])
    Y[s, 0] = Q_speed(q, X[s, 5])
    Y[s, 1] = Q_mustar(q, X[s, 6])
    if s % 20 == 0:
        print(s, np.round(X[s], 3), np.round(Y[s], 4), f"{time.time() - t0:.0f}s", flush=True)

def prcc(X, y):
    R = np.column_stack([stats.rankdata(c) for c in X.T])
    ry = stats.rankdata(y)
    out = []
    for j in range(R.shape[1]):
        others = np.delete(R, j, axis=1)
        A = np.column_stack([np.ones(len(ry)), others])
        rx = R[:, j] - A @ np.linalg.lstsq(A, R[:, j], rcond=None)[0]
        rr = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        c = np.corrcoef(rx, rr)[0, 1]
        tstat = c * np.sqrt((len(ry) - 2 - others.shape[1]) / (1 - c * c))
        pval = 2 * stats.t.sf(abs(tstat), len(ry) - 2 - others.shape[1])
        out.append(dict(param=gnames[j], prcc=c, p=pval))
    return out

# intro day is periodic: use distance (days) from introduction to the seasonal peak instead
days_to_peak = (base_phi := p0.phi) - X[:, 6]
days_to_peak = np.mod(days_to_peak, 365.0)
Xp = X.copy(); Xp[:, 6] = days_to_peak
gl = dict(n=n, lo=lo, hi=hi, names=gnames, X=X, Y=Y,
          prcc_cbar=prcc(X[:, :6], Y[:, 0]),          # intro date irrelevant for asymptotic speed
          prcc_mustar=prcc(Xp, np.log(Y[:, 1])))
for r in gl["prcc_cbar"]: print("cbar", r)
for r in gl["prcc_mustar"]: print("mu*", r)
save("s07_sensitivity", dict(local=local, global_=gl))
