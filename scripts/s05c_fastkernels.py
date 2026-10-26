"""Accelerating kernels at coarse resolution and large box: local growth exponent d ln h / d ln t.
dx = 5 km (cell weights keep the mass exact), L = 2e6 km."""
from _common import *
from sirfb.core import *
p = DENGUE
import sys
out = {}
a, yrs, eps = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
for _ in [0]:
    if True:
        q = p.with_(kernel=algebraic(a, 2.0), eps=eps)
        t0 = time.time()
        r = simulate_heavy(q, 30.0, 2.0, yrs * 365, dx=5.0, dt=0.25, amp=0.1, rec_dt=365 / 12, L_max=2e6, vanish_tol=1e-60)
        t, h = r.t[1:], r.H[1:]
        n = np.arange(1, int(t[-1] // 365) + 1)
        hy = np.interp(n * 365.0, t, h)
        loc = np.diff(np.log(hy)) / np.diff(np.log(n * 365.0))
        out[f"a{a}|eps={eps}"] = dict(years=n, h=hy, local_exponent=loc, predicted=1 / (a - 1), status=r.status, t_end=r.t_end)
        print(a, eps, r.status, round(r.t_end / 365, 2), np.round(loc, 2), f"{time.time()-t0:.0f}s", flush=True)
save(f"s05c_fastkernels_a{a}_eps{eps}", out)
