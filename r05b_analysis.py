"""Revision: analysis of the global sensitivity samples (PRCC with bootstrap intervals and
first-order variance-based indices by the given-data method), with convergence in n."""
from _c import *
from scipy import stats
g = load("s07_sensitivity")["global_"]
Xs, Ys = [np.array(g["X"], float)], [np.array(g["Y"], float)]
for a0 in (0, 200):                       # the 400 new samples, computed in two halves
    try:
        part = load(f"r05_part_{a0}")
        Xs.append(np.array(part["X"], float)); Ys.append(np.array(part["Y"], float))
    except FileNotFoundError:
        pass
X = np.vstack(Xs); Y = np.vstack(Ys); names = g["names"]
smp = dict(lo=g["lo"], hi=g["hi"])
n_all = len(Y)
ok = np.isfinite(Y[:, 1]) & (Y[:, 1] > 0)  # mu* = 0 (h0* = 0) samples have no log mu*
n_excl = int((~ok).sum())
X, Y = X[ok], Y[ok]
save("r05_samples", dict(X=X, Y=Y, names=names, lo=g["lo"], hi=g["hi"], n_all=n_all, n_excluded=n_excl))
PHI = 265.0
def prcc(X, y):
    R = np.column_stack([stats.rankdata(c) for c in X.T]); ry = stats.rankdata(y)
    out = []
    for j in range(R.shape[1]):
        A = np.column_stack([np.ones(len(ry)), np.delete(R, j, axis=1)])
        rx = R[:, j] - A @ np.linalg.lstsq(A, R[:, j], rcond=None)[0]
        rr = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        out.append(np.corrcoef(rx, rr)[0, 1])
    return np.array(out)
def S1_given_data(X, y, M):
    n = len(y); V = y.var(); out = []
    b = (M - 1) / (n - 1)
    for j in range(X.shape[1]):
        idx = np.argsort(X[:, j]); parts = np.array_split(idx, M)
        Vc = sum(len(p) * (y[p].mean() - y.mean()) ** 2 for p in parts) / n
        s = Vc / V
        out.append((s - b) / (1 - b))
    return np.array(out)
def analyse(X, Y, col, transform, M, nboot=500, seed=1):
    y = transform(Y[:, col])
    Xp = X.copy(); Xp[:, 6] = np.mod(PHI - X[:, 6], 365.0)     # days from arrival to the peak
    pr = prcc(Xp, y); s1 = S1_given_data(X, y, M)
    rng = np.random.default_rng(seed); PB, SB = [], []
    for b in range(nboot):
        i = rng.integers(0, len(y), len(y))
        PB.append(prcc(Xp[i], y[i])); SB.append(S1_given_data(X[i], y[i], M))
    PB, SB = np.array(PB), np.array(SB)
    pv = []
    for j in range(len(pr)):
        k = X.shape[1] - 1
        t = pr[j] * np.sqrt((len(y) - 2 - k) / (1 - pr[j] ** 2))
        pv.append(2 * stats.t.sf(abs(t), len(y) - 2 - k))
    return dict(prcc=[dict(param=names[j], prcc=pr[j], lo=np.percentile(PB[:, j], 2.5), hi=np.percentile(PB[:, j], 97.5), p=pv[j]) for j in range(len(pr))],
                # normal intervals from the bootstrap standard deviation: resampling with replacement
                # duplicates inputs, which biases given-data estimates upward, so percentile intervals
                # of the resampled values are shifted; their spread is still a fair error estimate
                S1=[dict(param=names[j], S1=s1[j], sd=SB[:, j].std(), lo=s1[j] - 1.96 * SB[:, j].std(),
                         hi=s1[j] + 1.96 * SB[:, j].std()) for j in range(len(s1))])
n = len(Y)
M = int(round(np.sqrt(n) * 0.8))
out = dict(n=n, M=M, names=names, lo=smp["lo"], hi=smp["hi"], n_all=n_all, n_excluded=n_excl)
a = analyse(X, Y, 1, np.log, M)
out["prcc_logmu"], out["S1_logmu"] = a["prcc"], a["S1"]
b = analyse(X[:, :6], Y, 0, lambda v: v, int(round(np.sqrt(n) * 0.8))) if False else None
# speed: intro day is irrelevant, keep it as a dummy (negative control)
a2 = analyse(X, Y, 0, lambda v: v, M)
out["prcc_cbar"], out["S1_cbar"] = a2["prcc"], a2["S1"]
# convergence with n
conv = []
for nn in [100, 200, 300, 400, 500, n]:
    if nn > n: break
    Mn = int(round(np.sqrt(nn) * 0.8))
    conv.append(dict(n=nn, S1=S1_given_data(X[:nn], np.log(Y[:nn, 1]), Mn),
                     prcc=prcc(np.column_stack([X[:nn, :6], np.mod(PHI - X[:nn, 6], 365)]), np.log(Y[:nn, 1]))))
out["convergence"] = conv
for r in out["prcc_logmu"]: print("prcc logmu", r["param"], round(r["prcc"], 3), round(r["lo"], 3), round(r["hi"], 3), "%.1e" % r["p"])
for r in out["S1_logmu"]: print("S1 logmu", r["param"], round(r["S1"], 3), round(r["lo"], 3), round(r["hi"], 3))
for r in out["S1_cbar"]: print("S1 cbar", r["param"], round(r["S1"], 3), round(r["lo"], 3), round(r["hi"], 3))
for c in conv: print(c["n"], np.round(c["S1"], 3), np.round(c["prcc"], 2))
print("logmu range", np.log10(Y[:, 1].min()), np.log10(Y[:, 1].max()))
save("r05_analysis", out)
