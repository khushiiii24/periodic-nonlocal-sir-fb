"""Mean front speed vs expansion coefficient mu: seasonal c_bar^T, autonomous semi-wave c0, KPP bound c*."""
from _common import *
from sirfb.core import *
from sirfb.semiwave import semiwave_speed

p = DENGUE
T = p.T
cs, _ = kpp_speed(p)
rows = []
for mu in (0.5, 1, 2, 5, 10, 25, 60, 150, 400):
    yrs = 5 if mu < 60 else 4
    dx = 0.25 if mu <= 10 else 0.5
    r = simulate(p, 5.0, mu, yrs * T, dx=dx, dt=0.0625, amp=0.1, L=5 + 1.05 * cs * yrs * T + 100)
    cbar = r.speed_over((yrs - 2) * T, yrs * T)
    c0, _, _ = semiwave_speed(p.with_(eps=0.0), mu, 60.0, 0.05)
    rows.append(dict(mu=mu, cbar=cbar, c0=c0, cstar=cs, ratio_seasonal=cbar / cs, ratio_auton=c0 / cs,
                     seasonal_over_auton=cbar / c0, status=r.status))
    print(rows[-1], flush=True)
save("s03_speed_mu", dict(rows=rows, cstar=cs))
