<div align="center">

# Seasonal Nonlocal SIR Model with Two Free Boundaries

**How does a seasonal disease like dengue spread into new areas, and does the time of year matter?**

**Python** · **NumPy** · **SciPy** · **Numba** · **MIT License** · **Group G3**

</div>

---

## 👥 Team (Group G3)

| # | Name | Roll number |
|:-:|:-----|:-----------:|
| 1 | Khushi | LCB2024007 |
| 2 | Divyanshu Kumar | LCI2024015 |
| 3 | Ramesh Bijarniya | LCI2024045 |
| 4 | Satyavrat Rai | LCI2024022 |

**Research topic:** *Dynamics for a nonlocal diffusive SIR epidemic model with double free boundaries* (Zhang and Wang, Nonlinear Analysis: RWA 81, 2025), extended to seasonal transmission.

---

## 📌 The idea in one picture

- People are split into three groups: **S** (can catch the disease), **I** (infected) and **R** (recovered).
- The infected area is an interval from **g(t)** to **h(t)**. Its two ends move outward as the infection pushes them.
- People can jump long distances (nonlocal dispersal), not only to nearby places.
- The infection rate **β(t)** goes up and down every year, like dengue after the monsoon.

We ask three questions:

1. 🦠 Will the infection **spread forever or die out**?
2. 🏃 If it spreads, **how fast** does the front move?
3. 📅 Does the **date the infection arrives** change the answer?

---

## 🧮 The model

**Infected people** (only inside the infected area):

$$
\frac{\partial I}{\partial t} = d_2\left(\int_{g(t)}^{h(t)} J(x-y)\,I(t,y)\,dy - I\right) + \beta(t)\,S\,I - (\gamma+\delta)\,I
$$

**Susceptible people** (everywhere):

$$
\frac{\partial S}{\partial t} = d_1\left(\int_{\mathbb{R}} J(x-y)\,S(t,y)\,dy - S\right) + \Lambda - \delta S - \beta(t)\,S\,I
$$

**Recovered people** (only inside the infected area):

$$
\frac{\partial R}{\partial t} = d_3\left(\int_{g(t)}^{h(t)} J(x-y)\,R(t,y)\,dy - R\right) + \gamma I - \delta R
$$

**How the right edge moves** (the left edge is the mirror image):

$$
h'(t) = \mu \int_{g(t)}^{h(t)} I(t,x)\,G\big(h(t)-x\big)\,dx, \qquad G(r) = \int_r^{\infty} J(z)\,dz
$$

**Seasonal infection rate:**

$$
\beta(t) = \bar\beta\left[1 + \varepsilon \cos\left(\frac{2\pi (t-\varphi)}{365}\right)\right]
$$

<details>
<summary><b>📖 What each symbol means (click to open)</b></summary>

| Symbol | Meaning |
|:------:|:--------|
| $J(x-y)$ | chance of jumping from place $y$ to place $x$ (the kernel) |
| $d_1, d_2, d_3$ | how often S, I and R people move |
| $\bar\beta$ | average infection rate over the year |
| $\varepsilon$ | strength of the season (0 means no season) |
| $\varphi$ | day of the year when infection peaks |
| $\gamma$ | recovery rate |
| $\delta$ | birth and death rate |
| $\mu$ | how easily the edge of the infected area moves (units: 1/day) |
| $h_0$ | half width of the first infected area |

</details>

We also study a **simple model** that keeps $S$ fixed and uses one equation for $I$.

---

## 🔑 Main results

| Result | What we found |
|:-------|:--------------|
| 📐 **Thresholds** | They depend only on the **yearly average** infection rate (Proposition 1), with a full proof of spread-or-vanish for the simple model (Appendix B). |
| 🐢 **Speed** | Seasons make the front **8 to 13% slower**, confirmed by an independent periodic semi-wave calculation (0.15% agreement). |
| 📅 **Arrival date** | The critical value μ* (units: 1/day) changes **about 700 times** with the arrival date (0.060 in July, 42 in February). Treating very small infected numbers as extinct makes this **larger** (up to 4000 times). |
| 🦠 **Three diseases** | The arrival-date effect is about 650 times for dengue, 1100 for cholera and 18 for influenza. |
| 🚀 **Speeding up** | The front speeds up **only** when the average jump length is infinite (tested on 9 kernels). |
| 🛑 **Full model** | With lifelong immunity the front **stalls** for small μ, while the simple model keeps going. The stall comes from the boundary rule: if recovered people also push the boundary, it no longer stalls. |
| 📊 **Real data** | A fit to monthly dengue counts from Delhi puts the peak of transmission around **1 August**. |

### Three example diseases

| Disease | Average R₀ | Critical half width (km) | Front speed (km per year) |
|:--------|:----------:|:------------------------:|:-------------------------:|
| 🦟 Dengue | 2.45 | 1.84 | 80 |
| 🤧 Influenza | 1.68 | 9.93 | 150 |
| 💧 Cholera | 1.50 | 1.73 | 27 |

---

## 📊 Figures

| | |
|:-:|:-:|
| ![Thresholds](figures/fig1_thresholds.png) | ![Speed](figures/fig2_speed_validation.png) |
| **Fig 1.** Seasonal R₀ and the critical size | **Fig 2.** Checking the solver and speed vs μ |
| ![Ratchet](figures/fig3_ratchet.png) | ![Kernels](figures/fig6_kernels.png) |
| **Fig 3.** Fast in autumn, almost still in spring | **Fig 6.** Which kernels make the front speed up |

<details>
<summary><b>More figures (click to open)</b></summary>

**Fig 4.** Arrival date and the critical value μ*

![Arrival date](figures/fig4_mustar.png)

**Fig 5.** Extinction threshold and the three diseases

![Extinction](figures/fig5_extinction_diseases.png)

**Fig 7.** Full model compared with the simple model

![Full model](figures/fig7_full_sir.png)

**Fig 8.** Where the full model stalls

![Stall map](figures/fig8_stall_map.png)

**Fig 9.** Sensitivity analysis

![Sensitivity](figures/fig9_sensitivity.png)

**Fig 10.** Fit to Delhi dengue data

![Data fit](figures/fig10_dengue_fit.png)

</details>

---

## 🚀 Quick start

```bash
# 1. Download the code
git clone https://github.com/YOUR-USERNAME/periodic-nonlocal-sir-fb.git
cd periodic-nonlocal-sir-fb

# 2. Install the packages
pip install -r requirements.txt

# 3. Run the studies (each one saves its results in results/)
cd scripts
python s01_thresholds.py
python r08_figures.py     # all figures of the revised paper
```

<details>
<summary><b>⏱️ All scripts and how long they take</b></summary>

| Script | What it does | Time |
|:-------|:-------------|:----:|
| `s01_thresholds.py` | Thresholds R₀, h₀* and the proof check | seconds |
| `s02_convergence.py` | Checks the solver with finer grids | ~5 min |
| `s03_speed_mu.py` | Front speed for different μ | ~5 min |
| `s04_mustar.py` | Critical value μ* and arrival date | ~3 min |
| `s05_acceleration.py` | Nine kernels over 8 to 40 years | ~20 min |
| `s05b_rates.py` | Growth rates for the kernel runs | seconds |
| `s05c_fastkernels.py` | Very heavy kernels on a large grid | ~5 min each |
| `s06_full_sir.py` | Full model vs simple model | ~5 min |
| `s06b_full_diagnostics.py` | Why the full model front stops | ~3 min |
| `s07_sensitivity.py` | Sensitivity analysis (200 samples) | ~15 min |
| `s08_figures.py` | Figures of the first version | seconds |
| `r01_semiwave.py` | Independent periodic semi-wave check of the seasonal speed; second-order (Heun) time stepping | ~10 min |
| `r02_mustar.py` | Convergence of μ*, extinction-threshold test, μ* vs arrival date for three diseases | ~10 min |
| `r03_fit.py` | Fit of the seasonal shape and timing of β(t) to Delhi monthly dengue counts | ~1 min |
| `r04_full.py` | Stall map over μ and d₁, yearly-advance diagnostics, boundary rule with I + ρR, agreement near μ* | ~25 min |
| `r05_sensitivity.py` | 400 more sensitivity samples (`python r05_sensitivity.py 0 200` and `python r05_sensitivity.py 200 400`) | ~16 min each |
| `r05b_analysis.py` | PRCC and first-order (Sobol) indices from all samples | ~2 min |
| `r06_spatial.py` | Critical size when the season changes with position | ~3 min |
| `r07_fitted.py` | μ* and speed for the season fitted to Delhi data | ~3 min |
| `r08_figures.py` | Makes all ten figures of the revised paper | ~1 min |

</details>

---

## 📁 Folder layout

```
periodic-nonlocal-sir-fb/
├── sirfb/
│   ├── core.py        model, kernels, thresholds and the simulator
│   ├── semiwave.py    separate check of the front speed
│   └── ext.py         Heun stepping, periodic semi-wave, extinction test, full model with I + ρR boundary
├── scripts/           one script for each study (s01 to s08, r01 to r08)
├── results/           numbers saved by each script (JSON)
├── figures/           all figures used in the paper
├── data/              monthly Delhi dengue counts used for the fit
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📚 Main references

1. Q. Zhang, M. Wang, *Dynamics for a nonlocal diffusive SIR epidemic model with double free boundaries*, Nonlinear Anal. RWA 81 (2025) 104208.
2. J.-F. Cao, Y. Du, F. Li, W.-T. Li, *The dynamics of a Fisher-KPP nonlocal diffusion model with free boundaries*, J. Funct. Anal. 277 (2019).
3. Y. Du, W. Ni, *Spreading speed for some cooperative systems with nonlocal diffusion and free boundaries, part 1*, J. Differential Equations 308 (2022).
4. Y. Du, W. Ni, *Rate of propagation for the Fisher-KPP equation with nonlocal diffusion and free boundaries*, J. Eur. Math. Soc. 27 (2025).
5. T. Wang, Z. Li, B. Dai, *Semi-wave and spreading speed for a nonlocal diffusive Fisher-KPP model with free boundaries in time periodic environment*, DCDS-B 29 (2024).

**Data:** yearly dengue case counts from the [National Center for Vector Borne Diseases Control, India](https://ncvbdc.mohfw.gov.in/Doc/Dengue-Cases-deaths-2021-2026-till-April.pdf). Monthly Delhi counts (Municipal Corporation of Delhi, as reported by PTI) are in `data/delhi_dengue_monthly.csv` and are used to fit the timing and strength of the season. The other parameter values are examples and are not fitted.

---

<div align="center">

Made by **Group G3** · Released under the [MIT License](LICENSE)

</div>
