"""
Semi-wave solver for the autonomous reduced problem (eps = 0).

Find (c, phi) with phi on (-inf, 0], phi(-inf) = u*, phi(0) = 0, and
  d2 [ int_{-inf}^0 J(z-y) phi(y) dy - phi(z) ] + c phi'(z) + f(phi(z)) = 0 ,  z < 0
  c = mu int_{-inf}^0 phi(z) G(-z) dz            (Du, Li & Zhou; Du & Ni)
with f(u) = (beta_bar S* - gamma - delta) u - beta_bar S* u^2.

For fixed c the profile is obtained by Newton's method on a truncated
interval (-L, 0] (far field phi = u* for z < -L, second-order upwind for
phi'); then c is fixed by solving c = mu * int phi_c G(-z) dz with brentq.
This gives an independent check of the time-dependent free-boundary solver.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import optimize
from scipy.linalg import solve

from .core import Params, kpp_speed


def profile(p: Params, c: float, L: float = 60.0, dz: float = 0.05, phi0=None, tol=1e-12):
    k = p.kernel
    r = p.rbar
    b = p.beta_bar * p.S_star
    ustar = r / b
    z = np.arange(-L, 0.0 + 1e-12, dz)          # includes z = 0
    N = z.size - 1                               # unknowns 0..N-1, phi_N = 0
    zi = z[:N]
    # mass-conserving cell weights: node j carries the cell [z_j - dz/2, z_j + dz/2]
    d = np.abs(zi[:, None] - zi[None, :])
    K = k.G(np.maximum(d - dz / 2, 0.0)) - k.G(d + dz / 2)
    np.fill_diagonal(K, 1.0 - 2.0 * k.G(dz / 2))
    far = ustar * k.G(zi + L + dz / 2)               # mass from y < -L - dz/2, phi = u*
    D = np.zeros((N, N))
    for i in range(N):
        if i <= N - 3:
            D[i, i] = -3; D[i, i + 1] = 4; D[i, i + 2] = -1
            D[i] /= (2 * dz)
        else:
            D[i, i] = -1 / dz
            if i + 1 < N:
                D[i, i + 1] = 1 / dz
            # phi_N = 0 contributes nothing
    A = p.d2 * (K - np.eye(N)) + c * D
    phi = ustar * (1 - np.exp(zi / 2.0)) if phi0 is None else phi0.copy()
    for it in range(60):
        F = A @ phi + p.d2 * far + r * phi - b * phi ** 2
        Jm = A + np.diag(r - 2 * b * phi)
        step = solve(Jm, -F)
        phi += step
        if np.max(np.abs(step)) < tol:
            break
    full = np.append(phi, 0.0)
    return z, full


def speed_functional(p: Params, mu: float, z, phi):
    """c = mu int_{-inf}^0 phi(z) G(-z) dz : midpoint sum over the node cells plus the
    far-field part u* int_{L+dz/2}^inf G(r) dr."""
    from scipy.integrate import quad
    dz = z[1] - z[0]
    L = -z[0]
    ustar = p.rbar / (p.beta_bar * p.S_star)
    far, _ = quad(lambda r: p.kernel.G(r), L + dz / 2, np.inf, limit=400)
    return mu * (np.sum(phi[:-1] * p.kernel.G(-z[:-1])) * dz + ustar * far)


def semiwave_speed(p: Params, mu: float, L: float = 60.0, dz: float = 0.05):
    cstar, _ = kpp_speed(p)
    hi = 0.999 * cstar if math.isfinite(cstar) else 50.0

    def F(c):
        z, phi = profile(p, c, L, dz)
        return speed_functional(p, mu, z, phi) - c

    c0 = optimize.brentq(F, 1e-4, hi, xtol=1e-10)
    z, phi = profile(p, c0, L, dz)
    return c0, z, phi
