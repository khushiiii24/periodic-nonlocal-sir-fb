"""
sirfb.core -- time-periodic nonlocal SIR model with double free boundaries.

Model (one space dimension, Omega(t) = (g(t), h(t))):

  S_t = d1 [ (J*S)(x) - S ] + Lambda - delta S - beta(t) S I          x in R
  I_t = d2 [ int_{g}^{h} J(x-y) I(t,y) dy - I ] + beta(t) S I - (gamma+delta) I   x in Omega(t)
  R_t = d3 [ int_{g}^{h} J(x-y) R(t,y) dy - R ] + gamma I - delta R   x in Omega(t)
  h'(t) =  mu int_{g}^{h} I(t,x) G(h - x) dx,   G(r) = int_r^inf J(z) dz
  g'(t) = -mu int_{g}^{h} I(t,x) G(x - g) dx

  beta(t) = beta_bar [ 1 + eps cos( 2 pi (t - phi) / T ) ]

The "reduced" model freezes S at the disease-free state S* and replaces
beta S I by the monostable surrogate beta S* I (1 - I) (Fisher-KPP type).

Numerics: explicit Euler in time, Nystrom (rectangle) quadrature in space,
direct convolution for thin kernels (numba) and FFT convolution for
heavy-tailed kernels; the double Stefan integral is collapsed onto the
analytic tail function G (one quadrature instead of two).

Everything here is deliberately self-contained (numpy / scipy / numba).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace, field

import numpy as np
from numba import njit
from scipy import optimize, special
from scipy.linalg import eigh

# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------
KT_GAUSS, KT_LAPLACE, KT_ALG, KT_CAUCHY, KT_SEXP = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class Kernel:
    """Even, unit-mass dispersal kernel J with analytic tail G(r)=int_r^inf J."""
    ktype: int
    s: float          # length scale (km): sigma for Gaussian
    a: float = 0.0    # algebraic exponent, or stretched-exponential power p
    label: str = ""

    # --- density -----------------------------------------------------------
    def J(self, z):
        z = np.abs(np.asarray(z, dtype=float))
        s, a = self.s, self.a
        if self.ktype == KT_GAUSS:
            return np.exp(-z**2 / (2 * s * s)) / math.sqrt(2 * math.pi * s * s)
        if self.ktype == KT_LAPLACE:
            return np.exp(-z / s) / (2 * s)
        if self.ktype == KT_ALG:
            return (a - 1) / (2 * s) * (1 + z / s) ** (-a)
        if self.ktype == KT_CAUCHY:
            return s / (math.pi * (s * s + z * z))
        if self.ktype == KT_SEXP:
            p = a
            C = 1.0 / (2 * s * math.gamma(1 + 1 / p))
            return C * np.exp(-(z / s) ** p)
        raise ValueError

    def G(self, r):
        r0 = np.asarray(r, dtype=float)
        out = _G_vec(np.maximum(np.atleast_1d(r0), 0.0).ravel(), self.ktype, self.s, self.a)
        return out.reshape(r0.shape) if r0.ndim else float(out[0])

    # --- moment conditions -------------------------------------------------
    @property
    def first_moment_finite(self) -> bool:
        """(J2)/(Du-Ni J1): int_0^inf z J(z) dz < inf  -> finite spreading speed."""
        if self.ktype == KT_ALG:
            return self.a > 2
        if self.ktype == KT_CAUCHY:
            return False
        return True

    @property
    def exp_moment_finite(self) -> bool:
        """(J3): exists lambda>0 with int J(z) e^{lambda z} dz < inf (thin tail)."""
        return self.ktype in (KT_GAUSS, KT_LAPLACE) or (self.ktype == KT_SEXP and self.a >= 1)

    def mgf(self, lam):
        """int_R J(z) e^{lam z} dz (inf if it diverges)."""
        if self.ktype == KT_GAUSS:
            return math.exp(0.5 * lam * lam * self.s * self.s)
        if self.ktype == KT_LAPLACE:
            return 1.0 / (1 - (lam * self.s) ** 2) if lam * self.s < 1 else math.inf
        return math.inf

    def lam_max(self):
        if self.ktype == KT_LAPLACE:
            return 1.0 / self.s
        if self.ktype == KT_GAUSS:
            return math.inf
        return 0.0

    def trunc_width(self, tol=1e-14):
        """Half-width beyond which the kernel can be dropped (inf for heavy tails)."""
        if self.ktype == KT_GAUSS:
            return self.s * math.sqrt(2) * special.erfcinv(2 * tol)
        if self.ktype == KT_LAPLACE:
            return -self.s * math.log(2 * tol)
        return math.inf

    @property
    def name(self):
        if self.label:
            return self.label
        return {KT_GAUSS: f"Gaussian(s={self.s})", KT_LAPLACE: f"Laplace(s={self.s})",
                KT_ALG: f"algebraic(a={self.a},s={self.s})", KT_CAUCHY: f"Cauchy(s={self.s})",
                KT_SEXP: f"stretched-exp(p={self.a},s={self.s})"}[self.ktype]


def cell_weights(kernel: "Kernel", dx: float, m: int) -> np.ndarray:
    """Mass-conserving cell-averaged kernel weights w_k = int_{(k-1/2)dx}^{(k+1/2)dx} J,
    k = -m..m (sum -> 1 as m -> inf).  Point sampling J(k dx) dx over-counts mass badly
    for kernels with a cusp/peak at 0 (e.g. algebraic a=4, s=2, dx=1 gives sum 1.23)."""
    k = np.arange(0, m + 1, dtype=float)
    up = kernel.G(np.maximum(k - 0.5, 0.0) * dx)
    lo = kernel.G((k + 0.5) * dx)
    w = up - lo
    w[0] = 1.0 - 2.0 * kernel.G(0.5 * dx)
    return np.concatenate([w[:0:-1], w])


def gaussian(s):
    return Kernel(KT_GAUSS, s)


def laplace(s):
    return Kernel(KT_LAPLACE, s)


def algebraic(a, s):
    return Kernel(KT_ALG, s, a)


def cauchy(s):
    return Kernel(KT_CAUCHY, s)


def stretched_exp(p, s):
    assert abs(1 / p - round(1 / p)) < 1e-12, "only 1/p integer supported (closed-form tail)"
    return Kernel(KT_SEXP, s, p)


@njit(cache=True)
def _G_scalar(r, ktype, s, a):
    if r < 0.0:
        r = 0.0
    if ktype == 0:
        return 0.5 * math.erfc(r / (s * math.sqrt(2.0)))
    if ktype == 1:
        return 0.5 * math.exp(-r / s)
    if ktype == 2:
        return 0.5 * (1.0 + r / s) ** (1.0 - a)
    if ktype == 3:
        return 0.5 - math.atan(r / s) / math.pi
    if ktype == 4:
        n = int(round(1.0 / a))
        u = (r / s) ** a
        acc = 0.0
        term = 1.0
        for k in range(n):
            if k > 0:
                term *= u / k
            acc += term
        return 0.5 * math.exp(-u) * acc
    return 0.0


@njit(cache=True)
def _G_vec(r, ktype, s, a):
    out = np.empty(r.size)
    rf = r.ravel()
    for i in range(rf.size):
        out[i] = _G_scalar(rf[i], ktype, s, a)
    return out.reshape(r.shape)


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    beta_bar: float = 0.35      # mean transmission rate (1/day)
    eps: float = 0.60           # seasonal amplitude
    phi: float = 265.0          # peak day of year
    T: float = 365.0            # period (days)
    gamma: float = 1 / 7        # recovery rate (1/day)
    delta: float = 0.0          # natural removal rate (1/day)
    d1: float = 0.50            # dispersal rate of S (1/day)
    d2: float = 0.50            # dispersal rate of I (1/day)
    d3: float = 0.50            # dispersal rate of R (1/day)
    S_star: float = 1.0         # disease-free density (normalised)
    kernel: Kernel = field(default_factory=lambda: gaussian(2.0))
    name: str = "dengue"

    def beta(self, t):
        return self.beta_bar * (1 + self.eps * np.cos(2 * np.pi * (np.asarray(t) - self.phi) / self.T))

    @property
    def R0T(self):
        return self.beta_bar * self.S_star / (self.gamma + self.delta)

    @property
    def rbar(self):
        return self.beta_bar * self.S_star - self.gamma - self.delta

    def with_(self, **kw):
        return replace(self, **kw)


DENGUE = Params()
INFLUENZA = Params(beta_bar=0.42, eps=0.35, phi=15.0, gamma=1 / 4, d1=1.0, d2=1.0, d3=1.0,
                   kernel=gaussian(5.0), name="influenza")
CHOLERA = Params(beta_bar=0.30, eps=0.80, phi=205.0, gamma=1 / 5, d1=0.30, d2=0.30, d3=0.30,
                 kernel=gaussian(1.5), name="cholera")

# ----------------------------------------------------------------------------
# Spectral / threshold quantities
# ----------------------------------------------------------------------------


def mu1(kernel: Kernel, h0: float, n: int = 901, rule: str = "trap") -> float:
    """Principal eigenvalue of K phi = int_{-h0}^{h0} J(x-y) phi(y) dy (Nystrom).

    rule="trap": trapezoidal nodes including endpoints (symmetrised);
    rule="mid" : midpoint nodes.
    """
    if rule == "mid":
        dx = 2 * h0 / n
        x = -h0 + (np.arange(n) + 0.5) * dx
        w = np.full(n, dx)
    else:
        x = np.linspace(-h0, h0, n)
        dx = x[1] - x[0]
        w = np.full(n, dx)
        w[0] = w[-1] = dx / 2
    sw = np.sqrt(w)
    A = kernel.J(x[:, None] - x[None, :]) * sw[:, None] * sw[None, :]
    return float(eigh(A, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0])


def lambda_p(p: Params, h0: float, n: int = 901, rule="trap") -> float:
    """Periodic principal eigenvalue on (-h0,h0).

    Because the coefficients depend on t only, the T-periodic eigenfunction
    separates as phi(t,x)=q(t)psi(x) with q(t)=exp(int_0^t (beta(s)-beta_bar)S* ds),
    so lambda_p = d2(mu1-1) + beta_bar S* - gamma - delta  (exact; see Prop. 1).
    """
    return p.d2 * (mu1(p.kernel, h0, n, rule) - 1) + p.rbar


def R0D(p: Params, h0: float, n: int = 901, rule="trap") -> float:
    """Domain-restricted reproduction number = spectral radius of F V^{-1}."""
    m = mu1(p.kernel, h0, n, rule)
    return p.beta_bar * p.S_star / (p.gamma + p.delta + p.d2 * (1 - m))


def h0_star(p: Params, n: int = 901, rule="trap", hi: float = 200.0) -> float:
    """Critical half-length: lambda_p(h0*) = 0  <=>  R0D(h0*) = 1."""
    if p.R0T <= 1:
        return math.inf
    return optimize.brentq(lambda h: lambda_p(p, h, n, rule), 1e-3, hi, xtol=1e-10)


def kpp_speed(p: Params):
    """Linear (KPP) speed c* = inf_{lam>0} [d2(M(lam)-1) + rbar]/lam  (inf if (J3) fails)."""
    k = p.kernel
    if not k.exp_moment_finite:
        return math.inf, math.nan
    lmax = k.lam_max()
    hi = min(lmax * (1 - 1e-9), 5.0) if math.isfinite(lmax) else 5.0
    f = lambda lam: (p.d2 * (k.mgf(lam) - 1) + p.rbar) / lam
    res = optimize.minimize_scalar(f, bounds=(1e-6, hi), method="bounded",
                                   options={"xatol": 1e-12})
    return float(res.fun), float(res.x)


# ----------------------------------------------------------------------------
# Free-boundary integrators
# ----------------------------------------------------------------------------
STATUS = {0: "running/horizon", 1: "spreading", -1: "vanishing", 2: "box-edge"}


@njit(cache=True)
def _reduced_thin(x, I, g, h, Jk, m, dx, dt, nsteps, d2, bbar, eps, phi, T, gam, delta, S,
                  mu, ktype, ks, ka, spread_len, vanish_tol, rec_every, out):
    """Reduced (Fisher-KPP surrogate) model; thin kernel; direct convolution."""
    N = x.size
    tmp = np.zeros(N)
    x0 = x[0]
    nrec = 0
    status = 0
    two_pi_T = 2.0 * math.pi / T
    step = 0
    for step in range(nsteps):
        t = step * dt
        bt = bbar * (1.0 + eps * math.cos(two_pi_T * (t - phi)))
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
        if i0 <= m + 1 or i1 >= N - m - 2:
            status = 2
            break
        hdot = 0.0
        gdot = 0.0
        for i in range(i0, i1 + 1):
            acc = 0.0
            jlo = i - m
            if jlo < i0:
                jlo = i0
            jhi = i + m
            if jhi > i1:
                jhi = i1
            for j in range(jlo, jhi + 1):
                acc += Jk[j - i + m] * I[j]
            Ii = I[i]
            tmp[i] = d2 * (acc - Ii) + bt * S * Ii * (1.0 - Ii) - (gam + delta) * Ii
            hdot += Ii * _G_scalar(h - x[i], ktype, ks, ka)
            gdot += Ii * _G_scalar(x[i] - g, ktype, ks, ka)
        imax = 0.0
        for i in range(i0, i1 + 1):
            v = I[i] + dt * tmp[i]
            if v < 0.0:
                v = 0.0
            I[i] = v
            if v > imax:
                imax = v
        h += dt * mu * hdot * dx
        g -= dt * mu * gdot * dx
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


@njit(cache=True)
def _full_thin(x, S, I, R, g, h, Jk, m, dx, dt, nsteps, d1, d2, d3, bbar, eps, phi, T, gam,
               delta, Sstar, mu, ktype, ks, ka, spread_len, vanish_tol, rec_every, out):
    """Full S-I-R model; S on the whole box (J*S over R = S* + J*(S-S*)); thin kernel."""
    N = x.size
    dS = np.zeros(N)
    dI = np.zeros(N)
    dR = np.zeros(N)
    x0 = x[0]
    nrec = 0
    status = 0
    two_pi_T = 2.0 * math.pi / T
    Lam = delta * Sstar
    step = 0
    for step in range(nsteps):
        t = step * dt
        bt = bbar * (1.0 + eps * math.cos(two_pi_T * (t - phi)))
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
        if i0 <= 3 * m or i1 >= N - 3 * m - 1:
            status = 2
            break
        # S on the whole box: only update where S deviates (cheap mask by range)
        for i in range(N):
            acc = 0.0
            jlo = i - m
            if jlo < 0:
                jlo = 0
            jhi = i + m
            if jhi > N - 1:
                jhi = N - 1
            for j in range(jlo, jhi + 1):
                acc += Jk[j - i + m] * (S[j] - Sstar)
            # J*S over R  =  S* + J*(S-S*)   (S = S* outside the box)
            dS[i] = d1 * (Sstar + acc - S[i]) + Lam - delta * S[i] - bt * S[i] * I[i]
        hdot = 0.0
        gdot = 0.0
        for i in range(i0, i1 + 1):
            accI = 0.0
            accR = 0.0
            jlo = i - m
            if jlo < i0:
                jlo = i0
            jhi = i + m
            if jhi > i1:
                jhi = i1
            for j in range(jlo, jhi + 1):
                accI += Jk[j - i + m] * I[j]
                accR += Jk[j - i + m] * R[j]
            dI[i] = d2 * (accI - I[i]) + bt * S[i] * I[i] - (gam + delta) * I[i]
            dR[i] = d3 * (accR - R[i]) + gam * I[i] - delta * R[i]
            hdot += I[i] * _G_scalar(h - x[i], ktype, ks, ka)
            gdot += I[i] * _G_scalar(x[i] - g, ktype, ks, ka)
        imax = 0.0
        for i in range(N):
            v = S[i] + dt * dS[i]
            S[i] = v if v > 0.0 else 0.0
        for i in range(i0, i1 + 1):
            v = I[i] + dt * dI[i]
            if v < 0.0:
                v = 0.0
            I[i] = v
            if v > imax:
                imax = v
            v = R[i] + dt * dR[i]
            R[i] = v if v > 0.0 else 0.0
        h += dt * mu * hdot * dx
        g -= dt * mu * gdot * dx
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


@njit(cache=True)
def _rhs_reduced_local(Iloc, conv, xloc, g, h, d2, bt, S, gam, delta, ktype, ks, ka):
    n = Iloc.size
    dI = np.empty(n)
    hdot = 0.0
    gdot = 0.0
    for i in range(n):
        Ii = Iloc[i]
        dI[i] = d2 * (conv[i] - Ii) + bt * S * Ii * (1.0 - Ii) - (gam + delta) * Ii
        hdot += Ii * _G_scalar(h - xloc[i], ktype, ks, ka)
        gdot += Ii * _G_scalar(xloc[i] - g, ktype, ks, ka)
    return dI, hdot, gdot


@dataclass
class Run:
    status: str
    t_end: float
    g: float
    h: float
    t: np.ndarray
    G: np.ndarray
    H: np.ndarray
    Imax: np.ndarray
    x: np.ndarray | None = None
    I: np.ndarray | None = None
    S: np.ndarray | None = None
    R: np.ndarray | None = None

    def speed_over(self, t_a, t_b):
        ha = np.interp(t_a, self.t, self.H)
        hb = np.interp(t_b, self.t, self.H)
        return (hb - ha) / (t_b - t_a)


def initial_profile(x, h0, amp, shape="cos"):
    I0 = np.zeros_like(x, dtype=float)
    inside = (x > -h0) & (x < h0)
    if shape == "cos":
        I0[inside] = amp * np.cos(np.pi * x[inside] / (2 * h0))
    elif shape == "para":
        I0[inside] = amp * (1 - (x[inside] / h0) ** 2)
    elif shape == "flat":
        I0[inside] = amp
    else:
        raise ValueError(shape)
    return I0


def simulate(p: Params, h0: float, mu: float, t_end: float, dx: float = 0.5, dt: float = 0.25,
             L: float | None = None, amp: float = 0.1, shape: str = "cos", model: str = "reduced",
             spread_len: float = math.inf, vanish_tol: float = 1e-14, rec_dt: float = 1.0,
             keep_state: bool = False) -> Run:
    """Integrate the reduced or full model with a thin (truncated) kernel."""
    k = p.kernel
    W = k.trunc_width()
    if not math.isfinite(W):
        return simulate_heavy(p, h0, mu, t_end, dx, dt, L, amp, shape, spread_len, vanish_tol,
                              rec_dt, keep_state)
    m = int(math.ceil(W / dx))
    if L is None:
        L = h0 + 1.2 * 1.0 * t_end + 6 * W  # generous: speeds here are < 1 km/day
    N = int(round(2 * L / dx)) + 1
    if N % 2 == 0:
        N += 1
    x = (np.arange(N) - (N - 1) // 2) * float(dx)
    Jk = cell_weights(k, dx, m)
    I = initial_profile(x, h0, amp, shape)
    nsteps = int(round(t_end / dt))
    rec_every = max(1, int(round(rec_dt / dt)))
    out = np.zeros((nsteps // rec_every + 2, 4))
    if model == "reduced":
        st, te, g, h, nrec = _reduced_thin(x, I, -h0, h0, Jk, m, dx, dt, nsteps, p.d2,
                                           p.beta_bar, p.eps, p.phi, p.T, p.gamma, p.delta,
                                           p.S_star, mu, k.ktype, k.s, k.a, spread_len,
                                           vanish_tol, rec_every, out)
        S = R = None
    else:
        S = np.full(N, p.S_star)
        R = np.zeros(N)
        st, te, g, h, nrec = _full_thin(x, S, I, R, -h0, h0, Jk, m, dx, dt, nsteps, p.d1, p.d2,
                                        p.d3, p.beta_bar, p.eps, p.phi, p.T, p.gamma, p.delta,
                                        p.S_star, mu, k.ktype, k.s, k.a, spread_len, vanish_tol,
                                        rec_every, out)
    o = out[:nrec]
    t = np.concatenate([[0.0], o[:, 0]])
    G = np.concatenate([[-h0], o[:, 1]])
    H = np.concatenate([[h0], o[:, 2]])
    Im = np.concatenate([[I.max() if nrec == 0 else amp], o[:, 3]])
    r = Run(STATUS[st], te, g, h, t, G, H, Im)
    if keep_state:
        r.x, r.I, r.S, r.R = x, I, S, R
    return r


def simulate_heavy(p: Params, h0, mu, t_end, dx=1.0, dt=0.25, L=None, amp=0.1, shape="cos",
                   spread_len=math.inf, vanish_tol=1e-14, rec_dt=1.0, keep_state=False,
                   L_max=2.0e5) -> Run:
    """Reduced model with an untruncated (heavy-tailed) kernel: FFT convolution on the
    active window, kernel spectra cached per power-of-two window size."""
    from scipy import fft as sfft
    k = p.kernel
    if L is None:
        L = h0 + 50.0
    N = int(round(2 * L_max / dx)) + 1
    if N % 2 == 0:
        N += 1
    c = (N - 1) // 2
    xfun = lambda i: (i - c) * dx
    I = np.zeros(N)
    xs = xfun(np.arange(N))
    I[:] = initial_profile(xs, h0, amp, shape)
    g, h = -h0, h0
    cache = {}

    def kspec(B):
        if B not in cache:
            P = 2 * B
            w = cell_weights(k, dx, B - 1)          # offsets -(B-1)..(B-1)
            kk2 = np.zeros(P)
            kk2[:B] = w[B - 1:]                      # offsets 0..B-1
            kk2[P - B + 1:] = w[:B - 1]              # offsets -(B-1)..-1 wrap to the end
            cache[B] = sfft.rfft(kk2)
        return cache[B]

    nsteps = int(round(t_end / dt))
    rec_every = max(1, int(round(rec_dt / dt)))
    ts, Gs, Hs, Ms = [0.0], [g], [h], [amp]
    status = 0
    two_pi_T = 2 * math.pi / p.T
    for step in range(nsteps):
        t = step * dt
        bt = p.beta_bar * (1 + p.eps * math.cos(two_pi_T * (t - p.phi)))
        i0 = int(math.ceil(g / dx)) + c
        if xfun(i0) <= g:
            i0 += 1
        i1 = int(math.floor(h / dx)) + c
        if xfun(i1) >= h:
            i1 -= 1
        if i0 <= 1 or i1 >= N - 2:
            status = 2
            break
        n = i1 - i0 + 1
        B = 1 << max(4, int(math.ceil(math.log2(n))))
        P = 2 * B
        buf = np.zeros(P)
        buf[:n] = I[i0:i1 + 1]
        conv = sfft.irfft(sfft.rfft(buf) * kspec(B), n=P)[:n]
        xloc = xs[i0:i1 + 1]
        dI, hdot, gdot = _rhs_reduced_local(I[i0:i1 + 1].copy(), conv, xloc, g, h, p.d2, bt,
                                            p.S_star, p.gamma, p.delta, k.ktype, k.s, k.a)
        newI = np.maximum(I[i0:i1 + 1] + dt * dI, 0.0)
        I[i0:i1 + 1] = newI
        h += dt * mu * hdot * dx
        g -= dt * mu * gdot * dx
        imax = newI.max()
        if step % rec_every == 0:
            ts.append((step + 1) * dt); Gs.append(g); Hs.append(h); Ms.append(imax)
        if h - g > spread_len:
            status = 1
            break
        if imax < vanish_tol:
            status = -1
            break
    r = Run(STATUS[status], (step + 1) * dt, g, h, np.array(ts), np.array(Gs), np.array(Hs),
            np.array(Ms))
    if keep_state:
        r.x, r.I = xs, I
    return r


# ----------------------------------------------------------------------------
# Threshold solvers
# ----------------------------------------------------------------------------

def classify(p: Params, h0: float, mu: float, h0s: float, t_max: float = 8 * 365,
             vanish_tol: float = 1e-60, **kw) -> int:
    """+1 spreading (h-g exceeded 2 h0*, which forces spreading), -1 vanishing.

    vanish_tol must be tiny: under seasonal forcing I can fall by e^-25 in the
    low season and regrow by e^+20 in the high season, and the boundary keeps
    creeping during each regrowth (a 1e-14 cut-off biased mu* by ~10%)."""
    r = simulate(p, h0, mu, t_max, spread_len=2 * h0s, rec_dt=30.0, vanish_tol=vanish_tol, **kw)
    if r.status == "spreading":
        return 1
    if r.status == "vanishing":
        return -1
    # horizon reached without either event: domain frozen below 2h0* => vanishing
    return 1 if (r.h - r.g) > 2 * h0s else -1


def mu_star(p: Params, h0: float, h0s: float | None = None, lo: float = 1e-3, hi: float = 64.0,
            rtol: float = 2e-3, t_max: float = 8 * 365, **kw) -> float:
    """Critical expansion coefficient by bisection in log(mu)."""
    if h0s is None:
        h0s = h0_star(p)
    if h0 >= h0s:
        return 0.0
    if classify(p, h0, lo, h0s, t_max, **kw) == 1:
        return lo
    while classify(p, h0, hi, h0s, t_max, **kw) == -1:
        hi *= 2
        if hi > 1e4:
            return math.inf
    a, b = math.log(lo), math.log(hi)
    while (b - a) > rtol:
        mid = 0.5 * (a + b)
        if classify(p, h0, math.exp(mid), h0s, t_max, **kw) == 1:
            b = mid
        else:
            a = mid
    return math.exp(0.5 * (a + b))
