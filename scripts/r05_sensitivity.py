"""Revision: global sensitivity.  Extends the 200-point Latin hypercube of s07 by 400 new
points (independent LHS), then reports PRCC with bootstrap intervals, first-order variance
based (Sobol) indices by the given-data method, and their convergence with sample size."""
from _c import *
from scipy import stats
from sirfb.core import *
from sirfb.ext import *
p0 = DENGUE
T = p0.T
AMP = 0.01
old = load("s07_sensitivity")["global_"]
lo, hi = np.array(old["lo"]), np.array(old["hi"])
names = old["names"]
def build(th):
    b, g, d2, sig, eps = th[:5]
    return p0.with_(beta_bar=b, gamma=g, d1=d2, d2=d2, d3=d2, kernel=gaussian(sig), eps=eps)
def safe_h0s(q):
    from sirfb.core import lambda_p
    if lambda_p(q, 1e-3) >= 0:
        return 0.0
    return h0_star(q, hi=400.0)
def Q_speed(q, mu):
    hq = safe_h0s(q)
    r = simulate(q, max(5.0, 2 * hq), mu, 4 * T, dx=min(0.5, q.kernel.s / 4), dt=0.125, amp=0.1)
    return r.speed_over(2 * T, 4 * T)
def Q_mustar(q, t_in):
    hq = safe_h0s(q)
    if hq == 0.0:
        return float("nan")      # every focus invades: mu* = 0, log mu* undefined
    qq = q.with_(phi=(q.phi - t_in) % q.T)
    return mu_star(qq, 0.5 * hq, hq, amp=AMP, dx=0.5 * hq / 20, dt=0.125, rtol=2e-3, hi=256.0)
n_new = 400
sampler = stats.qmc.LatinHypercube(d=7, seed=np.random.default_rng(20261126))
Xall = lo + sampler.random(n_new) * (hi - lo)
a0, a1 = int(sys.argv[1]), int(sys.argv[2])
Xn = Xall[a0:a1]
Yn = np.zeros((len(Xn), 2))
t0 = time.time()
for s in range(len(Xn)):
    q = build(Xn[s])
    Yn[s, 0] = Q_speed(q, Xn[s, 5])
    Yn[s, 1] = Q_mustar(q, Xn[s, 6])
    if s % 25 == 0:
        print(s, f"{time.time() - t0:.0f}s", flush=True)
        save(f"r05_part_{a0}", dict(X=Xn[:s + 1], Y=Yn[:s + 1]))
save(f"r05_part_{a0}", dict(X=Xn, Y=Yn, done=True))
