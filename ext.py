"""
sirfb.ext -- additions made for the revision.

  * simulate_rk2 : reduced model with Heun's method (explicit, second order in time)
  * classify_x / mu_star_x : threshold search with a choice of time scheme, an
    extinction (minimum density) cut-off and a configurable spreading length
  * periodic_semiwave : independent computation of the periodic semi-wave of
    Wang, Li and Dai in a front-fixed frame, integrated with an adaptive
    8th-order Runge-Kutta method until it becomes T-periodic
  * floquet_lambda : periodic principal eigenvalue for position-dependent
    seasonality, beta(t,x), by direct Floquet computation
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp, quad
from scipy.linalg import eigh

from .core import (Params, Run, STATUS, _G_scalar, cell_weights, initial_profile,
                   h0_star, kpp_speed, simulate)


# ----------------------------------------------------------------------------
# Heun (RK2) integrator for the reduced model, thin kernels
# ----------------------------------------------------------------------------
@njit(cache=True)
def _active(x, g, h):
    N = x.size
    x0 = x[0]
    dx = x[1] - x[0]
    i0 = int(math.ceil((g - x0) / dx))
    if i0 < 0:
        i0 = 0
    while i0 < N and x[i0] <= g:
        i0 += 1
    i1 = int(math.floor((h - x0) / dx))
    if i1 > N - 1:
        i1 = N - 1
    while i1 >= 0 and x[i1] >= h:
        i1 -= 1
    return i0, i1


@njit(cache=True)
def _rhs_red(x, I, i0, i1, g, h, Jk, m, d2, bt, S, gam, delta, ktype, ks, ka, out):
    hdot = 0.0
    gdot = 0.0
    for i in range(i0, i1 + 1):
        acc = 0.0
        jlo = max(i - m, i0)
        jhi = min(i + m, i1)
        for j in range(jlo, jhi + 1):
            acc += Jk[j - i + m] * I[j]
        Ii = I[i]
        out[i] = d2 * (acc - Ii) + bt * S * Ii * (1.0 - Ii) - (gam + delta) * Ii
        hdot += Ii * _G_scalar(h - x[i], ktype, ks, ka)
        gdot += Ii * _G_scalar(x[i] - g, ktype, ks, ka)
    return hdot, gdot


@njit(cache=True)
def _reduced_thin_rk2(x, I, g, h, Jk, m, dx, dt, nsteps, d2, bbar, eps, phi, T, gam, delta, S,
                      mu, ktype, ks, ka, spread_len, vanish_tol, rec_every, out):
    N = x.size
    k1 = np.zeros(N)
    k2 = np.zeros(N)
    Ip = np.zeros(N)
    nrec = 0
    status = 0
    w = 2.0 * math.pi / T
    step = 0
    for step in range(nsteps):
        t = step * dt
        i0, i1 = _active(x, g, h)
        if i0 <= m + 1 or i1 >= N - m - 2:
            status = 2
            break
        b1 = bbar * (1.0 + eps * math.cos(w * (t - phi)))
        hd1, gd1 = _rhs_red(x, I, i0, i1, g, h, Jk, m, d2, b1, S, gam, delta, ktype, ks, ka, k1)
        hp = h + dt * mu * hd1 * dx
        gp = g - dt * mu * gd1 * dx
        for i in range(i0, i1 + 1):
            v = I[i] + dt * k1[i]
            Ip[i] = v if v > 0.0 else 0.0
        j0, j1 = _active(x, gp, hp)
        for i in range(j0, i0):
            Ip[i] = 0.0
        for i in range(i1 + 1, j1 + 1):
            Ip[i] = 0.0
        b2 = bbar * (1.0 + eps * math.cos(w * (t + dt - phi)))
        hd2, gd2 = _rhs_red(x, Ip, j0, j1, gp, hp, Jk, m, d2, b2, S, gam, delta, ktype, ks, ka, k2)
        hn = h + 0.5 * dt * mu * (hd1 + hd2) * dx
        gn = g - 0.5 * dt * mu * (gd1 + gd2) * dx
        imax = 0.0
        for i in range(j0, j1 + 1):
            a = k1[i] if (i >= i0 and i <= i1) else 0.0
            v = I[i] + 0.5 * dt * (a + k2[i])
            if v < 0.0:
                v = 0.0
            if x[i] <= gn or x[i] >= hn:
                v = 0.0
            I[i] = v
            if v > imax:
                imax = v
        for i in range(j0, j1 + 1):
            k1[i] = 0.0
            k2[i] = 0.0
            Ip[i] = 0.0
        h = hn
        g = gn
        if step % rec_every == 0 and nrec < out.shape[0]:
            out[nrec, 0] = (step + 1) * dt
            out[nrec, 1] = g
            out[nrec, 2] = h
            out[nrec, 3] = imax
            nrec += 1
        if h - g > spread_len:
            status = 1
            break
        if imax < vanish_tol:
            status = -1
            break
    return status, (step + 1) * dt, g, h, nrec


def simulate_rk2(p: Params, h0, mu, t_end, dx=0.5, dt=0.25, L=None, amp=0.1, shape="cos",
                 spread_len=math.inf, vanish_tol=1e-14, rec_dt=1.0, keep_state=False) -> Run:
    k = p.kernel
    W = k.trunc_width()
    assert math.isfinite(W), "thin kernels only"
    m = int(math.ceil(W / dx))
    if L is None:
        L = h0 + 1.2 * t_end + 6 * W
    N = int(round(2 * L / dx)) + 1
    if N % 2 == 0:
        N += 1
    x = (np.arange(N) - (N - 1) // 2) * float(dx)
    Jk = cell_weights(k, dx, m)
    I = initial_profile(x, h0, amp, shape)
    nsteps = int(round(t_end / dt))
    rec_every = max(1, int(round(rec_dt / dt)))
    out = np.zeros((nsteps // rec_every + 2, 4))
    st, te, g, h, nrec = _reduced_thin_rk2(x, I, -h0, h0, Jk, m, dx, dt, nsteps, p.d2, p.beta_bar,
                                           p.eps, p.phi, p.T, p.gamma, p.delta, p.S_star, mu,
                                           k.ktype, k.s, k.a, spread_len, vanish_tol, rec_every, out)
    o = out[:nrec]
    t = np.concatenate([[0.0], o[:, 0]])
    G = np.concatenate([[-h0], o[:, 1]])
    H = np.concatenate([[h0], o[:, 2]])
    Im = np.concatenate([[amp], o[:, 3]])
    r = Run(STATUS[st], te, g, h, t, G, H, Im)
    if keep_state:
        r.x, r.I = x, I
    return r


# ----------------------------------------------------------------------------
# Threshold search with options
# ----------------------------------------------------------------------------
def classify_x(p, h0, mu, h0s, t_max=8 * 365, vanish_tol=1e-60, spread_len=None,
               scheme="euler", **kw):
    """+1 spreading / -1 vanishing.  spread_len defaults to 2 h0* (Theorem 2)."""
    sl = 2 * h0s if spread_len is None else spread_len
    sim = simulate if scheme == "euler" else simulate_rk2
    r = sim(p, h0, mu, t_max, spread_len=sl, rec_dt=30.0, vanish_tol=vanish_tol, **kw)
    if r.status == "spreading":
        return 1
    if r.status == "vanishing":
        return -1
    return 1 if (r.h - r.g) > 2 * h0s and spread_len is None else -1


def mu_star_x(p, h0, h0s=None, lo=1e-3, hi=64.0, rtol=1e-3, t_max=8 * 365, mu_cap=1e4, **kw):
    if h0s is None:
        h0s = h0_star(p)
    if h0 >= h0s:
        return 0.0
    if classify_x(p, h0, lo, h0s, t_max, **kw) == 1:
        return lo
    while classify_x(p, h0, hi, h0s, t_max, **kw) == -1:
        hi *= 2
        if hi > mu_cap:
            return math.inf
    a, b = math.log(lo), math.log(hi)
    while (b - a) > rtol:
        mid = 0.5 * (a + b)
        if classify_x(p, h0, math.exp(mid), h0s, t_max, **kw) == 1:
            b = mid
        else:
            a = mid
    return math.exp(0.5 * (a + b))


# ----------------------------------------------------------------------------
# Periodic semi-wave (front-fixed frame), independent of the moving-grid solver
# ----------------------------------------------------------------------------
def ode_periodic_state(p: Params, n=20001):
    """Positive T-periodic solution u*(t) of u' = (beta(t)S*-gamma-delta)u - beta(t)S* u^2.
    With v = 1/u: v' = -a(t) v + b(t), solved exactly by quadrature."""
    T = p.T
    t = np.linspace(0, T, n)
    a = p.beta(t) * p.S_star - p.gamma - p.delta
    b = p.beta(t) * p.S_star
    A = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(t))])
    eA = np.exp(A)
    Ib = np.concatenate([[0.0], np.cumsum(0.5 * (b[1:] * eA[1:] + b[:-1] * eA[:-1]) * np.diff(t))])
    v0 = Ib[-1] / (eA[-1] - 1.0)          # periodicity v(T) = v(0)
    v = (v0 + Ib) / eA
    return t, 1.0 / v


def periodic_semiwave(p: Params, mu: float, L=60.0, dz=0.05, years=12, rtol=1e-9, atol=1e-11,
                      phi0=None, return_profile=False):
    """Integrate U_t = d2[int_{-inf}^0 J(z-y)U dy - U] + c(t) U_z + f(t,U) on (-L,0],
    U(t,0)=0, U = u*(t) for z<-L, c(t) = mu int_{-inf}^0 U G(-z) dz, until T-periodic.
    Returns the mean of c over the final period and the period-to-period change."""
    from scipy.signal import fftconvolve
    k = p.kernel
    z = np.arange(-L, 1e-12, dz)
    N = z.size - 1
    zi = z[:N]
    # cell weights J_k for offsets -(N-1)..(N-1)
    off = np.arange(-(N - 1), N) * dz
    dd = np.abs(off)
    Jw = k.G(np.maximum(dd - dz / 2, 0.0)) - k.G(dd + dz / 2)
    Jw[N - 1] = 1.0 - 2.0 * k.G(dz / 2)
    farw = k.G(zi + L + dz / 2)               # mass from y < -L - dz/2
    Gz = k.G(-zi)
    tt, us = ode_periodic_state(p)
    far_tail, _ = quad(lambda r: k.G(r), L + dz / 2, np.inf, limit=400)
    S, gam, dl = p.S_star, p.gamma, p.delta

    def ustar(t):
        return np.interp(t % p.T, tt, us)

    def cfun(U, t):
        return mu * (np.dot(U, Gz) * dz + ustar(t) * far_tail)

    def rhs(t, U):
        conv = fftconvolve(U, Jw)[N - 1:2 * N - 1]
        c = cfun(U, t)
        Uz = np.empty(N)
        Uz[:N - 2] = (-3 * U[:N - 2] + 4 * U[1:N - 1] - U[2:N]) / (2 * dz)
        Uz[N - 2] = (-3 * U[N - 2] + 4 * U[N - 1] - 0.0) / (2 * dz)
        Uz[N - 1] = (0.0 - U[N - 1]) / dz
        bt = p.beta_bar * (1 + p.eps * math.cos(2 * math.pi * (t - p.phi) / p.T))
        return (p.d2 * (conv + ustar(t) * farw - U) + c * Uz
                + (bt * S - gam - dl) * U - bt * S * U * U)

    if phi0 is None:
        U = ustar(0.0) * (1 - np.exp(zi / 2.0))
    else:
        U = np.interp(zi, phi0[0], phi0[1])
    cmeans = []
    for yr in range(years):
        ts = np.linspace(yr * p.T, (yr + 1) * p.T, 731)
        sol = solve_ivp(rhs, (yr * p.T, (yr + 1) * p.T), U, method="DOP853", t_eval=ts,
                        rtol=rtol, atol=atol)
        cs = np.array([cfun(sol.y[:, j], ts[j]) for j in range(ts.size)])
        cbar = np.trapezoid(cs, ts) / p.T
        cmeans.append(cbar)
        U = sol.y[:, -1]
    out = dict(cbar=cmeans[-1], history=cmeans, change=abs(cmeans[-1] - cmeans[-2]))
    if return_profile:
        out.update(z=zi, U=U, t=ts - (years - 1) * p.T, c=cs)
    return out


# ----------------------------------------------------------------------------
# Position-dependent seasonality: periodic principal eigenvalue by Floquet theory
# ----------------------------------------------------------------------------
def floquet_lambda(p: Params, ell: float, beta_tx, n=121, periods=6):
    """beta_tx(t, x) -> transmission rate; returns the principal Floquet exponent of
    u_t = d2(K u - u) + (beta(t,x) S* - gamma - delta) u on (-ell, ell)."""
    x = np.linspace(-ell, ell, n)
    dx = x[1] - x[0]
    w = np.full(n, dx)
    w[0] = w[-1] = dx / 2
    K = p.kernel.J(x[:, None] - x[None, :]) * w[None, :]

    def f(t, u):
        return p.d2 * (K @ u - u) + (beta_tx(t, x) * p.S_star - p.gamma - p.delta) * u

    u = np.ones(n)
    lam = None
    for kper in range(periods):
        sol = solve_ivp(f, (kper * p.T, (kper + 1) * p.T), u, rtol=1e-10, atol=1e-40,
                        method="DOP853")
        un = sol.y[:, -1]
        lam = math.log(np.linalg.norm(un) / np.linalg.norm(u)) / p.T
        u = un / np.linalg.norm(un)
    return lam


# ----------------------------------------------------------------------------
# Full S-I-R model with a generalised boundary law
#     h' = mu int (I + rho R) G(h - x) dx   (rho = 0 is problem (P))
# Integrated in chunks so that diagnostics can be recorded between chunks.
# ----------------------------------------------------------------------------
@njit(cache=True)
def _full_rho(x, S, I, R, g, h, Jk, m, dx, dt, nsteps, d1, d2, d3, bbar, eps, phi, T, gam,
              delta, Sstar, mu, rho, ktype, ks, ka):
    N = x.size
    dS = np.zeros(N)
    dI = np.zeros(N)
    dR = np.zeros(N)
    w = 2.0 * math.pi / T
    Lam = delta * Sstar
    edge = 0
    for step in range(nsteps):
        t = step * dt
        bt = bbar * (1.0 + eps * math.cos(w * (t - phi)))
        i0, i1 = _active(x, g, h)
        if i0 <= 3 * m or i1 >= N - 3 * m - 1:
            edge = 1
            break
        for i in range(N):
            acc = 0.0
            jlo = max(i - m, 0)
            jhi = min(i + m, N - 1)
            for j in range(jlo, jhi + 1):
                acc += Jk[j - i + m] * (S[j] - Sstar)
            dS[i] = d1 * (Sstar + acc - S[i]) + Lam - delta * S[i] - bt * S[i] * I[i]
        hdot = 0.0
        gdot = 0.0
        for i in range(i0, i1 + 1):
            accI = 0.0
            accR = 0.0
            jlo = max(i - m, i0)
            jhi = min(i + m, i1)
            for j in range(jlo, jhi + 1):
                accI += Jk[j - i + m] * I[j]
                accR += Jk[j - i + m] * R[j]
            dI[i] = d2 * (accI - I[i]) + bt * S[i] * I[i] - (gam + delta) * I[i]
            dR[i] = d3 * (accR - R[i]) + gam * I[i] - delta * R[i]
            push = I[i] + rho * R[i]
            hdot += push * _G_scalar(h - x[i], ktype, ks, ka)
            gdot += push * _G_scalar(x[i] - g, ktype, ks, ka)
        for i in range(N):
            v = S[i] + dt * dS[i]
            S[i] = v if v > 0.0 else 0.0
        for i in range(i0, i1 + 1):
            v = I[i] + dt * dI[i]
            I[i] = v if v > 0.0 else 0.0
            v = R[i] + dt * dR[i]
            R[i] = v if v > 0.0 else 0.0
        h += dt * mu * hdot * dx
        g -= dt * mu * gdot * dx
    return g, h, edge


def run_full(p: Params, h0, mu, years, dx=0.5, dt=0.125, amp=0.1, rho=0.0, L=None, chunk=5.0,
             margin=400.0, vmax=1.0):
    """Full model, chunked; returns daily-ish diagnostics."""
    k = p.kernel
    W = k.trunc_width()
    m = int(math.ceil(W / dx))
    T = p.T
    if L is None:
        L = h0 + vmax * years * T * 1.1 + margin
    N = int(round(2 * L / dx)) + 1
    if N % 2 == 0:
        N += 1
    x = (np.arange(N) - (N - 1) // 2) * float(dx)
    Jk = cell_weights(k, dx, m)
    I = initial_profile(x, h0, amp)
    S = np.full(N, p.S_star)
    R = np.zeros(N)
    g, h = -h0, h0
    nch = int(round(years * T / chunk))
    ns = int(round(chunk / dt))
    rec = dict(t=[0.0], h=[h], g=[g], Imax=[I.max()], Smean=[1.0], Sfront=[1.0], Imass=[np.sum(I) * dx],
               Rfront=[0.0])
    edge = 0
    for c in range(nch):
        t0 = c * chunk
        g, h, edge = _full_rho(x, S, I, R, g, h, Jk, m, dx, dt, ns, p.d1, p.d2, p.d3, p.beta_bar,
                               p.eps, (p.phi - t0) % T, T, p.gamma, p.delta, p.S_star, mu, rho,
                               k.ktype, k.s, k.a)
        ins = (x > g) & (x < h)
        fr = (x > h - 5.0) & (x < h)
        rec["t"].append(t0 + chunk); rec["h"].append(h); rec["g"].append(g)
        rec["Imax"].append(float(I[ins].max()) if ins.any() else 0.0)
        rec["Smean"].append(float(S[ins].mean()) if ins.any() else 1.0)
        rec["Sfront"].append(float(S[fr].mean()) if fr.any() else 1.0)
        rec["Rfront"].append(float(R[fr].mean()) if fr.any() else 0.0)
        rec["Imass"].append(float(I[ins].sum() * dx))
        if edge:
            break
    out = {kk: np.array(v) for kk, v in rec.items()}
    out["edge"] = edge
    yrs = np.arange(0, int(out["t"][-1] // T) + 1)
    out["annual_adv"] = np.diff(np.interp(yrs * T, out["t"], out["h"]))
    out["x"], out["S"], out["I"], out["R"] = x, S, I, R
    return out
