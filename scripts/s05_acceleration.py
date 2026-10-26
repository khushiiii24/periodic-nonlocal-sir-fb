"""Long-horizon kernel study: finite speed iff the first moment is finite (Du & Ni).
Seasonal dengue coefficients, h0 = 30 km (> h0* for every kernel; max h0* = 12.6 km for a = 1.5),
mu = 2, dx = 1 km, dt = 0.25 d."""
from _common import *
from sirfb.core import *
from sirfb.semiwave import semiwave_speed
import sys

p = DENGUE
T = p.T
which = sys.argv[1] if len(sys.argv) > 1 else "all"
kernels = {
    "gaussian": (gaussian(2.0), 40),
    "laplace": (laplace(2.0), 40),
    "sexp_half": (stretched_exp(0.5, 2.0), 40),
    "alg_a4": (algebraic(4, 2.0), 40),
    "alg_a3": (algebraic(3, 2.0), 40),
    "alg_a2.5": (algebraic(2.5, 2.0), 40),
    "cauchy": (cauchy(2.0), 30),
    "alg_a1.75": (algebraic(1.75, 2.0), 12),
    "alg_a1.5": (algebraic(1.5, 2.0), 8),
}
res = {}
names = list(kernels) if which == "all" else which.split(",")
for name in names:
    k, yrs = kernels[name]
    res[f"{name}|h0_star"] = h0_star(p.with_(kernel=k), n=1201, hi=2000.0)
    for eps in (0.6, 0.0):
        q = p.with_(kernel=k, eps=eps)
        t0 = time.time()
        r = simulate_heavy(q, 30.0, 2.0, yrs * T, dx=1.0, dt=0.25, amp=0.1, rec_dt=T / 4, L_max=1.2e5,
                           vanish_tol=1e-60)
        n = np.arange(1, int(r.t_end // T) + 1)
        hn = np.interp(n * T, r.t, r.H)
        res[f"{name}|eps={eps}"] = dict(kernel=k.name, eps=eps, first_moment_finite=k.first_moment_finite,
                                        exp_moment_finite=k.exp_moment_finite, years=n, h=hn,
                                        h_over_t=hn / (n * T), status=r.status, t_end=r.t_end,
                                        secs=time.time() - t0)
        print(name, eps, r.status, round(r.t_end / T, 2),
              np.round(hn / (n * T), 4)[[0, len(n) // 2, -1]] if len(n) else "-",
              f"{time.time() - t0:.0f}s", flush=True)
    if k.first_moment_finite:
        try:
            c0, _, _ = semiwave_speed(p.with_(kernel=k, eps=0.0), 2.0, 80.0, 0.05)
        except Exception as e:  # pragma: no cover
            c0 = float("nan")
        res[f"{name}|semiwave"] = dict(c0=c0)
        print(name, "semi-wave c0 =", c0, flush=True)
save(f"s05_acceleration_{which.replace(',', '_')}", res)
