# periodic-nonlocal-sir-fb

Code accompanying the manuscript draft by **Group G3**: Khushi (LCB2024007), Diyanshu Kumar (LCI2024015), Ramesh Bijarniya (LCI2024045), Satyavrat Rai (LCI2024022).

> **Seasonal forcing in a nonlocal diffusive SIR epidemic model with double free boundaries: thresholds, front speed and the timing of introduction**

It simulates and analyses the time-periodic nonlocal SIR model with double free boundaries

```
S_t = d1 [ (J*S) - S ] + Λ - δS - β(t) S I                    x ∈ ℝ
I_t = d2 [ ∫_{g}^{h} J(x-y) I(t,y) dy - I ] + β(t) S I - (γ+δ) I   x ∈ (g(t), h(t))
R_t = d3 [ ∫_{g}^{h} J(x-y) R(t,y) dy - R ] + γ I - δ R            x ∈ (g(t), h(t))
h'(t) =  μ ∫_{g}^{h} I(t,x) G(h-x) dx ,   g'(t) = -μ ∫_{g}^{h} I(t,x) G(x-g) dx ,   G(r) = ∫_r^∞ J
β(t) = β̄ [1 + ε cos(2π(t-φ)/T)]
```

together with the reduced (Fisher–KPP) surrogate in which `S ≡ S*` and `β S I` is replaced by `β S* I (1 - I)`.

The model extends Zhang & Wang, *Nonlinear Anal. RWA* 81 (2025) 104208 (constant coefficients) to seasonal transmission. The numerical scheme is our own; the analysis it is checked against is due to Cao, Du, Li & Li (2019), Du & Ni (2022, 2025) and Wang, Li & Dai (2023). See the manuscript's reference list.

## Layout

| path | content |
|---|---|
| `sirfb/core.py` | kernel library (Gaussian, Laplace, algebraic, Cauchy, stretched exponential, each with an analytic tail `G`); parameter sets (dengue, influenza, cholera); principal eigenvalue `λp`, `R0^D`, `h0*`, KPP bound `c*`; free-boundary integrators (reduced and full S–I–R; direct convolution for thin kernels with numba, FFT for heavy tails); bisection for `μ*` |
| `sirfb/semiwave.py` | independent semi-wave solver (Newton on the profile, `brentq` on the speed) for the autonomous problem |
| `scripts/s01_thresholds.py` | `λp`, `R0^D`, `h0*`, `c*`; check of Proposition 1 against a direct Floquet computation |
| `scripts/s02_convergence.py` | grid / time-step / domain refinement; semi-wave cross-check; seasonal "ratchet" |
| `scripts/s03_speed_mu.py` | mean front speed against `μ`, seasonal vs autonomous |
| `scripts/s04_mustar.py` | `μ*`: amplitude scaling, `μ*(h0)`, introduction date, seasonal amplitude, date × ε map, three diseases |
| `scripts/s05_acceleration.py` | 8–40-year runs for eight kernels: finite speed iff `∫ z J(z) dz < ∞` |
| `scripts/s06_full_sir.py` | full three-component runs against the reduced surrogate |
| `scripts/s07_sensitivity.py` | local normalised sensitivity indices and LHS–PRCC global sensitivity |
| `scripts/s08_figures.py` | every figure in the manuscript and slides, from `results/*.json` |
| `results/` | JSON output of every script (no hand-entered numbers) |
| `figures/` | PNG figures |

## Reproduce

```bash
pip install -r requirements.txt
cd scripts
python s01_thresholds.py      # seconds
python s02_convergence.py     # ~5 min
python s03_speed_mu.py        # ~5 min
python s04_mustar.py          # ~3 min
python s05_acceleration.py    # ~20 min (heavy-tailed kernels, FFT)
python s06_full_sir.py        # ~5 min
python s07_sensitivity.py     # ~15 min
python s08_figures.py
```

Numerical settings used for the reported numbers: explicit Euler, `Δt = 0.0625–0.25 d`, `Δx = 0.05 km` (threshold runs) to `1 km` (40-year kernel runs); first-order convergence in `Δt` is documented in `results/s02_convergence.json` and the reported mean speed is Richardson-extrapolated.

## Data

Case counts quoted in the manuscript (India 2023: 289,235 dengue cases, 485 deaths; Uttar Pradesh 2023: 35,402 cases, 36 deaths) are from the National Center for Vector Borne Diseases Control, Government of India: <https://ncvbdc.mohfw.gov.in/Doc/Dengue-Cases-deaths-2021-2026-till-April.pdf>. The three parameter sets are illustrative calibrations inside published ranges, **not** fits to these data.

## Related open-source code

* `py-pde` (Zwicker, *J. Open Source Softw.* 5 (2020) 2158) — finite-difference solver for local reaction–diffusion PDEs, including an SIR example: <https://github.com/zwicker-group/py-pde>. It solves local (Laplacian) models on fixed domains; this repository handles the nonlocal kernel and the moving Stefan-type boundaries.

## Licence

MIT — see `LICENSE`.
