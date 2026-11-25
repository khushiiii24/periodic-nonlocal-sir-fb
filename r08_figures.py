"""All manuscript figures for the revision, from results/*.json.
Every series is distinguished by line style and marker as well as by colour, so the
figures remain readable when printed in black and white."""
from _c import *
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from sirfb.core import DENGUE, INFLUENZA, CHOLERA, h0_star

INK, MUTED, GRID = "#111111", "#555555", "#E4E4E4"
# dark-to-light so that greyscale order is preserved
C = ["#08306b", "#c2410c", "#4d4d4d", "#2f8f5b", "#7b3294", "#b08800", "#1f78b4", "#999999"]
LS = ["-", "--", ":", "-.", (0, (5, 1.5, 1, 1.5)), (0, (1, 1)), (0, (8, 2)), "-"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
GREYBLUE = LinearSegmentedColormap.from_list("gb", ["#f2f2f2", "#bdbdbd", "#6baed6", "#2171b5", "#08306b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.7, "lines.markersize": 4, "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MSTART = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
MU_UNIT = "day$^{-1}$"


def months_axis(ax, every=2):
    ax.set_xticks(MSTART[::every])
    ax.set_xticklabels(MONTHS[::every])
    ax.set_xlim(0, 365)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("ok", name)


def fig1():
    d = load("s01_thresholds")
    p = DENGUE
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    t = np.linspace(0, 365, 366)
    ax[0].plot(t, p.beta(t) / p.gamma, color=C[0], ls=LS[0], label=r"$R_0(t)=\beta(t)/\gamma$")
    ax[0].axhline(p.R0T, color=C[1], ls=LS[1], lw=1.4, label=r"$R_0^T=\bar\beta/\gamma=2.45$")
    ax[0].axhline(1, color=MUTED, lw=0.9, ls=LS[2])
    imin = int(np.argmin(p.beta(t)))
    ax[0].plot([imin], [p.beta(imin) / p.gamma], "v", color=INK, ms=5)
    ax[0].text(imin + 6, 0.62, "min, day %d" % imin, fontsize=7.5)
    months_axis(ax[0])
    ax[0].set_ylim(0.4, 4.4)
    ax[0].set_xlabel("day of the year")
    ax[0].set_ylabel("reproduction number")
    ax[0].legend(fontsize=7.5, loc="upper left")
    ax[0].set_title("(a) seasonal forcing, dengue")
    m = d["monotonicity"]
    hs = d["dengue"]["h0_star"]
    ax[1].semilogx(m["radii"], m["R0D"], marker=MK[0], ls=LS[0], color=C[0], ms=3.5, label="$R_0^D(h_0)$")
    ax[1].axhline(p.R0T, color=C[1], ls=LS[1], lw=1.4, label="$R_0^T=2.45$")
    ax[1].axhline(1, color=MUTED, lw=0.9, ls=LS[2])
    ax[1].axvline(hs, color=MUTED, lw=0.9, ls=LS[2])
    ax[1].text(hs * 1.1, 0.68, f"$h_0^*={hs:.3f}$ km", fontsize=7.5)
    ax[1].set_xlabel("initial half width $h_0$ (km)")
    ax[1].set_ylabel("$R_0^D(h_0)$ (dimensionless)")
    ax[1].legend(fontsize=7.5, loc="lower right")
    ax[1].set_title("(b) domain-restricted reproduction number")
    save(fig, "fig1_thresholds.png")


def fig2():
    c = load("s02_convergence")
    r1 = load("r01_semiwave")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    runs = c["cbar_seasonal"]["runs"]
    ax[0].plot([r["dt"] for r in runs], [r["cbar"] for r in runs], marker=MK[0], ls=LS[0], color=C[0],
               label="seasonal, Euler")
    rk = r1["rk2"]
    ax[0].plot([r["dt"] for r in rk], [r["cbar"] for r in rk], marker=MK[3], ls=LS[3], color=C[3],
               label="seasonal, Heun")
    sim0 = c["autonomous"]["sim"]
    ax[0].plot([r["dt"] for r in sim0], [r["c"] for r in sim0], marker=MK[1], ls=LS[1], color=C[1],
               label="no seasons, Euler")
    sw = sw_values(r1)
    ax[0].axhline(sw["seas"], color=C[0], ls=LS[2], lw=1.1)
    ax[0].axhline(sw["auto"], color=C[1], ls=LS[2], lw=1.1)
    ax[0].text(0.034, sw["seas"] + 0.0012, "periodic semi-wave (independent)", fontsize=7)
    ax[0].text(0.034, sw["auto"] - 0.0042, "semi-wave (independent)", fontsize=7)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("time step $\\Delta t$ (day)")
    ax[0].set_ylabel("mean front speed (km/day)")
    ax[0].set_ylim(0.205, 0.256)
    ax[0].legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax[0].set_title("(a) time-step convergence, $\\mu=2$ " + MU_UNIT)
    s = load("s03_speed_mu")
    rows = r1["mu_sweep"]
    mus = np.array([r["mu"] for r in rows])
    ext = lambda a, b: b + (b - a) / 3
    c0 = np.array([ext(r["c0_dz05"], r["c0_dz025"]) for r in rows])
    cs = np.array([ext(r["cbar_dz05"], r["cbar_dz025"]) for r in rows])
    ax[1].semilogx(mus, c0, ls=LS[1], color=C[1], marker=MK[1], label="no seasons, semi-wave $c_0$")
    ax[1].semilogx(mus, cs, ls=LS[0], color=C[0], label="seasonal, periodic semi-wave $\\bar c$")
    ax[1].semilogx([r["mu"] for r in s["rows"]], [r["cbar"] for r in s["rows"]], ls="none", marker=MK[0],
                   mfc="white", color=C[0], label="seasonal, simulation $\\bar c^T$")
    ax[1].axhline(s["cstar"], color=MUTED, ls=LS[2], lw=1.1)
    ax[1].text(0.45, s["cstar"] + 0.02, f"linear speed $c^*={s['cstar']:.3f}$ km/day", fontsize=7)
    ax[1].set_xlabel("expansion coefficient $\\mu$ (" + MU_UNIT + ")")
    ax[1].set_ylabel("front speed (km/day)")
    ax[1].set_ylim(0, 1.1)
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_title("(b) speed against $\\mu$")
    save(fig, "fig2_speed_validation.png")


def sw_values(r1):
    ref = r1["refinement_mu2"]
    def get(eps):
        a = [r for r in ref if r["eps"] == eps and r["L"] == 60.0 and r["dz"] == 0.05][0]["cbar"]
        b = [r for r in ref if r["eps"] == eps and r["L"] == 60.0 and r["dz"] == 0.025][0]["cbar"]
        return b + (b - a) / 3
    return dict(seas=get(0.6), auto=get(0.0))


def fig3():
    c = load("s02_convergence")["ratchet"]
    r1 = load("r01_semiwave")["profile"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    sp = np.array(c["speed"])
    ax[0].bar(range(12), sp, color="#9ecae1", edgecolor=C[0], width=0.72, label="simulation, year 3")
    # monthly means of c(t) from the periodic semi-wave
    t = np.array(r1["t"]); cc = np.array(r1["c"])
    edges = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    mm = [np.trapezoid(cc[(t >= edges[i]) & (t <= edges[i + 1])], t[(t >= edges[i]) & (t <= edges[i + 1])]) /
          (edges[i + 1] - edges[i]) for i in range(12)]
    ax[0].plot(range(12), mm, ls="none", marker="D", color=INK, ms=3.5, label="periodic semi-wave")
    for i in (int(np.argmin(sp)), int(np.argmax(sp))):
        ax[0].text(i, sp[i] + 0.012, f"{sp[i]:.3f}", ha="center", fontsize=7.5)
    ax[0].set_xticks(range(12)); ax[0].set_xticklabels(MONTHS, fontsize=7.5)
    ax[0].set_ylabel("mean front speed (km/day)")
    ax[0].set_ylim(0, 0.43)
    ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title(f"(a) monthly speed (max/min = {c['ratio']:.1f})")
    tt = np.array(c["t"]) / 365
    ax[1].plot(tt, c["h"], color=C[0])
    ax[1].set_xlabel("time (years)"); ax[1].set_ylabel("$h(t)$ (km)")
    ax[1].set_title(f"(b) right boundary: {c['annual_km']:.1f} km per year")
    save(fig, "fig3_ratchet.png")


def fig4():
    d = load("s04_mustar")
    fit = load("r07_fitted")
    fig = plt.figure(figsize=(7.4, 5.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, :])
    ii = d["intro_date"]
    tin = np.array([r["t_in"] for r in ii]); ms = np.array([r["mu_star"] for r in ii])
    eps0 = [r["mu_star"] for r in d["eps_sweep"] if r["eps"] == 0.0][0]
    ax0.semilogy(tin, ms, marker=MK[0], ls=LS[0], color=C[0], ms=3.2, label="illustrative ($\\varepsilon=0.6$, $\\varphi=265$)")
    fr = fit["deterministic"]["rows"]
    ax0.semilogy([r["t_in"] for r in fr], [r["mu_star"] for r in fr], marker=MK[1], ls=LS[1], color=C[1], ms=3.0,
                 label=f"fitted ($\\varepsilon={fit['eps']:.2f}$, $\\varphi={fit['phi']}$)")
    ax0.axhline(eps0, color=MUTED, ls=LS[2], lw=1.2, label="no seasons")
    months_axis(ax0)
    ax0.set_xlabel("date of introduction"); ax0.set_ylabel("$\\mu^*$ (" + MU_UNIT + ")")
    ax0.set_ylim(0.01, 600)
    ax0.legend(fontsize=6.6, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
    ax0.set_title("(a) $h_0=1$ km, $A=0.01$")
    a = d["amp_scaling"]
    amps = np.array([r["amp"] for r in a]); mm = np.array([r["mu_star"] for r in a])
    ax1.loglog(amps, mm, marker=MK[0], ls=LS[0], color=C[0], label="computed $\\mu^*$")
    k0 = a[0]["kappa"]
    ax1.loglog(amps, k0 / amps, ls=LS[1], color=C[1], lw=1.2, label=f"$\\kappa^*/A$, $\\kappa^*={k0:.3f}$ " + MU_UNIT)
    ax1.set_xlabel("initial peak prevalence $A$"); ax1.set_ylabel("$\\mu^*$ (" + MU_UNIT + ")")
    ax1.legend(fontsize=7)
    ax1.set_title("(b) $\\mu^*A\\to\\kappa^*$ (1 Jan, $h_0=1$ km)")
    M = np.array(d["map"]["mu_star"], dtype=float)
    eps = d["map"]["eps"]
    im = ax2.imshow(M, aspect="auto", origin="lower", cmap=GREYBLUE, norm=LogNorm(vmin=M.min(), vmax=M.max()))
    ax2.set_xticks(range(12)); ax2.set_xticklabels(MONTHS)
    ax2.set_yticks(range(len(eps))); ax2.set_yticklabels([f"{e:g}" for e in eps])
    ax2.set_xlabel("month of introduction (mid-month)"); ax2.set_ylabel("seasonal amplitude $\\varepsilon$")
    ax2.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax2.text(j, i, f"{v:.2g}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                     color="white" if v > np.sqrt(M.min() * M.max()) * 1.5 else INK)
    cb = fig.colorbar(im, ax=ax2, pad=0.01); cb.set_label("$\\mu^*$ (" + MU_UNIT + ", log scale)")
    ax2.set_title("(c) $\\mu^*$ by month of introduction and seasonal amplitude ($h_0=1$ km, $A=0.01$, $\\varphi=265$)")
    save(fig, "fig4_mustar.png")


def fig5():
    d = load("r02_mustar")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    md = d["min_density"]
    labs = [("1e-60", "deterministic ($10^{-60}$)"), ("1e-08", "$I_{\\min}=10^{-8}$"),
            ("1e-06", "$I_{\\min}=10^{-6}$"), ("0.0001", "$I_{\\min}=10^{-4}$")]
    for i, (k, lab) in enumerate(labs):
        rows = md[k]
        ax[0].semilogy([r["t_in"] for r in rows], [float(r["mu_star"]) for r in rows], marker=MK[i], ls=LS[i],
                       color=C[i], ms=3, label=lab)
    months_axis(ax[0])
    ax[0].set_xlabel("date of introduction"); ax[0].set_ylabel("$\\mu^*$ (" + MU_UNIT + ")")
    ax[0].legend(fontsize=6.8, loc="upper right")
    ax[0].set_ylim(0.03, 2000)
    ax[0].set_title("(a) extinction threshold, dengue, $h_0=1$ km")
    dd = d["diseases_date"]
    for i, (k, lab) in enumerate((("dengue", "dengue"), ("influenza", "influenza"), ("cholera", "cholera"))):
        rows = dd[k]["rows"]
        ax[1].semilogy([r["t_in"] for r in rows], np.array([r["mu_star"] for r in rows]) / dd[k]["no_season"],
                       marker=MK[i], ls=LS[i], color=C[i], ms=3,
                       label=f"{lab} (range {dd[k]['ratio']:.0f}-fold)")
    ax[1].axhline(1, color=MUTED, lw=0.9, ls=LS[2])
    months_axis(ax[1])
    ax[1].set_xlabel("date of introduction")
    ax[1].set_ylabel("$\\mu^*/\\mu^*_{\\varepsilon=0}$")
    ax[1].set_ylim(1e-3, 30)
    ax[1].legend(fontsize=6.6, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0), framealpha=0.9, frameon=True, edgecolor="none")
    ax[1].set_title("(b) three diseases, $h_0=h_0^*/2$")
    save(fig, "fig5_extinction_diseases.png")


def fig6():
    d = load("s05_acceleration_all")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    fin = [("gaussian", "Gaussian"), ("laplace", "Laplace"), ("sexp_half", "stretched exp $p=1/2$"),
           ("alg_a4", "algebraic $a=4$"), ("alg_a3", "algebraic $a=3$"), ("alg_a2.5", "algebraic $a=2.5$")]
    for i, (k, lab) in enumerate(fin):
        r = d[f"{k}|eps=0.6"]
        yrs = np.array(r["years"]); ho = np.array(r["h_over_t"])
        ax[0].plot(yrs, ho, color=C[i], ls=LS[i], lw=1.4, marker=MK[i], markevery=max(1, len(yrs) // 6), label=lab)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("time (years)"); ax[0].set_ylabel("$h(t)/t$ (km/day)")
    ax[0].legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax[0].set_title("(a) finite first moment: $h(t)/t$ levels off")
    rc = d["cauchy|eps=0.6"]
    ax[1].loglog(np.array(rc["years"]) * 365, rc["h"], marker=MK[0], ls=LS[0], ms=2.5, color=C[0], lw=1.3,
                 label="Cauchy ($\\gamma=2$)")
    for i, (a, lab) in enumerate(((1.75, "algebraic $a=1.75$"), (1.5, "algebraic $a=1.5$"))):
        rr = load(f"s05c_fastkernels_a{a}_eps0.6")[f"a{a}|eps=0.6"]
        ax[1].loglog(np.array(rr["years"]) * 365, rr["h"], marker=MK[i + 1], ls=LS[i + 1], ms=2.5, color=C[i + 1],
                     lw=1.3, label=lab)
    ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    tt = np.array([365, 365 * 30.0])
    ax[1].loglog(tt, 0.5 * tt, ":", color=MUTED, lw=1)
    ax[1].text(420, 0.5 * 420 * 1.6, "slope 1", fontsize=7, color=MUTED)
    ax[1].set_xlabel("time (days)"); ax[1].set_ylabel("$h(t)$ (km)")
    ax[1].set_title("(b) infinite first moment: acceleration")
    save(fig, "fig6_kernels.png")


def fig7():
    g = load("s06b_full_diagnostics")
    d = load("s06_full_sir")
    r4 = load("r04_full")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    keys = [("base (eps=.6, mu=2, delta=1/70y)", "full, $\\mu=2$", 0),
            ("mu=50", "full, $\\mu=50$", 3),
            ("fast turnover delta=1/1y", "full, $\\mu=2$, $\\delta=1/(1$ y$)$", 2)]
    for k, lab, i in keys:
        r = g[k]
        ax[0].plot(np.array(r["t"]) / 365, r["H"], color=C[i], ls=LS[i], label=lab)
    bl = r4["boundary_law"]["1.0"]
    ax[0].plot(np.array(bl["t"]) / 365, bl["h"], color=C[4], ls=LS[4], label="full, $\\mu=2$, $I+R$ move boundary")
    tr = d["traj"]
    ax[0].plot(np.array(tr["t"]) / 365, tr["h_red"], color=C[1], ls=LS[1], label="reduced, $\\mu=2$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("time (years)"); ax[0].set_ylabel("$h(t)$ (km)")
    ax[0].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax[0].set_ylim(1, 3000)
    ax[0].set_title("(a) right free boundary")
    for k, lab, i in keys + [("no seasonality (eps=0)", "full, $\\mu=2$, $\\varepsilon=0$", 5)]:
        r = g[k]
        ax[1].plot(np.array(r["t"]) / 365, np.maximum(r["Imax"], 1e-12), color=C[i], ls=LS[i], label=lab)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-8, 1)
    ax[1].set_xlabel("time (years)"); ax[1].set_ylabel("$\\max_x I(t,x)$")
    ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax[1].set_title("(b) peak prevalence inside $\\Omega(t)$")
    save(fig, "fig7_full_sir.png")


def stall_table():
    """annual advances for every (d1, mu) from the r04 log"""
    return {(r["d1"], r["mu"]): np.array(r["annual_adv"]) for r in load("r04_stall_table")["rows"]}


def fig8():
    r4 = load("r04_full")
    tab = stall_table()
    mus = r4["map"]["mu"]; d1s = r4["map"]["d1"]
    M = np.array([[tab[(d1, mu)][3:6].mean() for mu in mus] for d1 in d1s])
    red = np.array(r4["map"]["reduced_km_per_year"])
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    Mp = np.maximum(M, 0.01)
    im = ax[0].imshow(Mp, origin="lower", aspect="auto", cmap=GREYBLUE, norm=LogNorm(vmin=0.01, vmax=200))
    ax[0].set_xticks(range(len(mus))); ax[0].set_xticklabels([str(m) for m in mus])
    ax[0].set_yticks(range(len(d1s))); ax[0].set_yticklabels([f"{v:g}" for v in d1s])
    ax[0].set_xlabel("expansion coefficient $\\mu$ (" + MU_UNIT + ")")
    ax[0].set_ylabel("susceptible dispersal $d_1$ (" + MU_UNIT + ")")
    ax[0].grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; ratio = v / red[j]
            txt = (f"{v:.0f}" if v >= 10 else f"{v:.2g}") if v >= 0.01 else "0"
            ax[0].text(j, i, txt, ha="center", va="center", fontsize=6.3,
                       color="white" if v > 8 else INK, fontweight="bold" if ratio < 0.05 else "normal")
    cb = fig.colorbar(im, ax=ax[0], pad=0.01); cb.set_label("advance, years 4-6 (km/year)")
    ax[0].set_title("(a) mean yearly advance of the full model")
    dg = r4["diagnostics"]["base"]
    yrs = np.arange(1, len(dg["annual_adv"]) + 1)
    ax[1].bar(yrs - 0.2, dg["annual_adv"], width=0.38, color="#9ecae1", edgecolor=C[0], label="advance (km)")
    ax[1].set_xlabel("year"); ax[1].set_ylabel("yearly advance of $h$ (km)")
    ax2 = ax[1].twinx()
    ax2.plot(yrs, dg["S_front_jul1"][:len(yrs)], marker="o", ls="-", color=C[1], label="$S$ near front, 1 July")
    ax2.set_ylabel("$S$ within 5 km of $h$ on 1 July", color=INK)
    ax2.set_ylim(0.3, 0.9)
    ax2.spines["right"].set_visible(True); ax2.grid(False)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper right")
    ax[1].set_title("(b) $\\mu=2$, $d_1=0.5$: pulses of advance")
    save(fig, "fig8_stall_map.png")


def fig9():
    s = load("r05_analysis")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    labels = {"beta_bar": "$\\bar\\beta$", "gamma": "$\\gamma$", "d2": "$d$", "sigma": "$\\sigma$",
              "eps": "$\\varepsilon$", "mu": "$\\mu$ (control)", "intro_day": "arrival day"}
    names = s["names"]
    y = np.arange(len(names))[::-1]
    p = s["prcc_logmu"]; S1 = s["S1_logmu"]
    ax[0].barh(y, [r["prcc"] for r in p], xerr=[[r["prcc"] - r["lo"] for r in p], [r["hi"] - r["prcc"] for r in p]],
               color=["#9ecae1" if r["prcc"] >= 0 else "white" for r in p], edgecolor=C[0], hatch=None,
               height=0.62, error_kw=dict(lw=0.8, capsize=2))
    for yy, r in zip(y, p):
        ax[0].text(0.98 if r["prcc"] < 0.5 else -0.98, yy, (f"{r['prcc']:+.2f}" if abs(r['prcc']) >= 0.005 else "0.00"), va="center",
                   ha="right" if r["prcc"] < 0.5 else "left", fontsize=7)
    ax[0].axvline(0, color=MUTED, lw=0.8)
    ax[0].set_xlim(-1, 1)
    ax[0].set_xticks([-1, -0.5, 0, 0.5, 1])
    ax[0].set_yticks(y); ax[0].set_yticklabels([labels[n] for n in names])
    ax[0].set_xlabel("PRCC (monotone association)")
    ax[0].set_title("(a) PRCC for $\\log\\mu^*$")
    ax[1].barh(y, [r["S1"] for r in S1], xerr=[[r["S1"] - r["lo"] for r in S1], [r["hi"] - r["S1"] for r in S1]],
               color="#9ecae1", edgecolor=C[0], height=0.62, error_kw=dict(lw=0.8, capsize=2))
    for yy, r in zip(y, S1):
        ax[1].text(max(r["hi"], r["S1"]) + 0.02, yy, f"{r['S1']:.2f}", va="center", fontsize=7)
    ax[1].axvline(0, color=MUTED, lw=0.8)
    ax[1].set_xlim(-0.1, 0.75)
    ax[1].set_xlabel("first-order index $S_i$ (share of variance)")
    ax[1].set_title("(b) variance-based index for $\\log\\mu^*$")
    save(fig, "fig9_sensitivity.png")


def fig10():
    f = load("r03_fit")["fit"]
    fit = load("r07_fitted")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    edges = np.array(f["edges"]); cnt = np.array(f["counts"]); mod = np.array(f["model"])
    mid = 0.5 * (edges[1:] + edges[:-1])
    ax[0].semilogy(mid / 365 + 2021, np.maximum(cnt, 0.5), ls="none", marker="o", color=C[0], label="reported (MCD)")
    ax[0].semilogy(mid / 365 + 2021, mod, ls=LS[1], marker="s", mfc="white", color=C[1], label="fitted model")
    ax[0].axhline(f["B"], color=MUTED, ls=LS[2], lw=1, label="fitted background")
    ax[0].set_xlabel("year"); ax[0].set_ylabel("dengue cases per month")
    ax[0].set_xticks([2021, 2021.5, 2022, 2022.5, 2023]); ax[0].set_xticklabels(["Jan 21", "Jul 21", "Jan 22", "Jul 22", "Jan 23"])
    ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title("(a) Delhi, monthly counts, 2021-2022")
    t = np.linspace(0, 365, 366)
    p0 = DENGUE
    pf = DENGUE.with_(eps=fit["eps"], phi=float(fit["phi"]))
    ax[1].plot(t, p0.beta(t) / p0.beta_bar, ls=LS[0], color=C[0], label="illustrative ($\\varepsilon=0.6$, $\\varphi=265$)")
    ax[1].plot(t, pf.beta(t) / pf.beta_bar, ls=LS[1], color=C[1], label=f"fitted ($\\varepsilon={fit['eps']:.2f}$, $\\varphi={fit['phi']}$)")
    lo, hi = f["phi_ci"]
    ax[1].axvspan(lo, hi, color="#d9d9d9", alpha=0.7, lw=0, label="95% interval for $\\varphi$")
    months_axis(ax[1])
    ax[1].set_ylim(0, 2.3)
    ax[1].set_xlabel("day of the year"); ax[1].set_ylabel("$\\beta(t)/\\bar\\beta$")
    ax[1].legend(fontsize=6.5, loc="upper left")
    ax[1].set_title("(b) seasonal profile of transmission")
    save(fig, "fig10_dengue_fit.png")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9", "fig10"]
    for f in todo:
        globals()[f]()
