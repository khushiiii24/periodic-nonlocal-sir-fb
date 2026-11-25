"""Revision: rough fit of the seasonal shape (eps) and timing (phi) of beta(t) to monthly
dengue counts reported by the Municipal Corporation of Delhi, January 2021 - January 2023.
In the low-prevalence regime d log I/dt = beta(t) S_eff - gamma, so that, with a constant
background B of sporadic cases, the count in month m is
  C_m = B + kappa * int_{month m} exp( r t + (A T / 2 pi) sin(2 pi (t - phi)/T) ) dt,
which corresponds to beta(t) S_eff = gamma + r + A cos(2 pi (t - phi)/T), i.e.
eps = A / (gamma + r) and peak day phi.  We take r = 0 (recurrent seasons) and let the
epidemic level change between seasons (1 April), when it is far below the background."""
from _c import *
from scipy.optimize import least_squares
T = 365.0
GAMMA = 1 / 7
DATA = {2021: [0, 2, 5, 10, 12, 7, 16, 72, 217, 1196, 6739, 1337],
        2022: [23, 16, 22, 20, 30, 32, 26, 75, 693, 1238, 1420, 874],
        2023: [14]}                       # January 2023 up to 28 January
ml = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
edges = [0.0]
for y in (2021, 2022):
    for d in ml:
        edges.append(edges[-1] + d)
edges.append(edges[-1] + 28.0)            # partial January 2023
edges = np.array(edges)
C = np.array(DATA[2021] + DATA[2022] + DATA[2023], float)
widths = np.diff(edges)
tq = [np.linspace(edges[m], edges[m + 1], 41) for m in range(C.size)]

SEASON_START = (90.0, 455.0)           # 1 April 2021 and 1 April 2022
def level(t, lk0, lk1, lk2):
    return np.where(t < SEASON_START[0], lk0, np.where(t < SEASON_START[1], lk1, lk2))

def model(th):
    A, phi, lk0, lk1, lk2, lB = th
    out = np.empty(C.size)
    for m in range(C.size):
        t = tq[m]
        e = np.exp(level(t, lk0, lk1, lk2) + A * T / (2 * np.pi) * np.sin(2 * np.pi * (t - phi) / T))
        out[m] = math.exp(lB) * widths[m] / 30.4 + np.trapezoid(e, t)
    return out

def resid(th, c):
    return np.log(c + 1.0) - np.log(model(th) + 1.0)

LO = [0.0, 100.0, -40.0, -40.0, -40.0, -5.0]
HI = [0.4, 330.0, 30.0, 30.0, 30.0, 6.0]

def fit(c):
    best = None
    for phi0 in (170, 200, 230, 260):
        for A0 in (0.04, 0.08, 0.12):
            x0 = [A0, phi0, 0.0, 0.0, 0.0, math.log(10.0)]
            r = least_squares(resid, x0, args=(c,), bounds=(LO, HI), max_nfev=20000)
            if best is None or r.cost < best.cost:
                best = r
    return best

r = fit(C)
A, phi, lk0, lk1, lk2, lB = r.x
rr = 0.0
res = dict(A=A, phi=phi, r=rr, eps=A / (GAMMA + rr), B=math.exp(lB), levels=[lk0, lk1, lk2], rmse_log=float(np.sqrt(np.mean(r.fun ** 2))),
           model=model(r.x), edges=edges, counts=C)
rng = np.random.default_rng(11)
boots = []
mod = model(r.x)
for b in range(400):
    e = rng.choice(r.fun, C.size, replace=True)
    cb = np.maximum(np.exp(np.log(mod + 1.0) + e) - 1.0, 0.0)
    rb = least_squares(resid, r.x, args=(cb,), bounds=(LO, HI), max_nfev=5000)
    boots.append((rb.x[0] / GAMMA, rb.x[1]))
boots = np.array(boots)
res["eps_ci"] = np.percentile(boots[:, 0], [2.5, 97.5])
res["phi_ci"] = np.percentile(boots[:, 1], [2.5, 97.5])
# sensitivity: drop the background (pure log-sinusoid) and Poisson deviance alternative
def resid_nob(th, c):
    return np.log(c + 1.0) - np.log(model(list(th) + [-30.0]) + 1.0)
bestn = None
for phi0 in (170, 200, 230, 260):
    for A0 in (0.02, 0.04, 0.08):
        rn = least_squares(resid_nob, [A0, phi0, 0.0, 0.0, 0.0], args=(C,), bounds=(LO[:5], HI[:5]), max_nfev=20000)
        if bestn is None or rn.cost < bestn.cost:
            bestn = rn
rn = bestn
res["no_background"] = dict(eps=rn.x[0] / GAMMA, phi=rn.x[1], rmse_log=float(np.sqrt(np.mean(rn.fun ** 2))))
def pois(th, c):
    m = model(th)
    return np.sqrt(2 * np.maximum(m - c + np.where(c > 0, c * np.log(np.maximum(c, 1e-12) / m), 0.0), 0.0))
rp = least_squares(pois, r.x, args=(C,), bounds=(LO, HI), max_nfev=20000)
res["poisson"] = dict(eps=rp.x[0] / GAMMA, phi=rp.x[1])
# implied instantaneous growth rate and beta(t) S_eff
tt = np.arange(0, 2 * T + 28, 1.0)
res["curve_t"] = tt
res["curve"] = np.exp(level(tt, lk0, lk1, lk2) + A * T / (2 * np.pi) * np.sin(2 * np.pi * (tt - phi) / T)) * 30.4 + math.exp(lB)
print({k: (np.round(v, 4) if not isinstance(v, (dict, np.ndarray)) else v) for k, v in res.items()
       if k not in ("model", "edges", "counts", "curve", "curve_t")})
print("model", np.round(res["model"], 1))
print("data ", C)
save("r03_fit", dict(fit=res, gamma=GAMMA, data=DATA))
