"""All manuscript figures from results/*.json (no hand-entered numbers)."""
from _common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from sirfb.core import DENGUE, R0D, h0_star

INK, MUTED, GRID = "#0F2A3F", "#5C6F7C", "#E2EAEF"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def panel_tag(ax, s):
    ax.set_title(s)


def fig1():
    d = load("s01_thresholds")
    p = DENGUE
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    t = np.linspace(0, 365, 366)
    ax[0].plot(t, p.beta(t) / p.gamma, color=CAT[0])
    ax[0].axhline(p.R0T, color=CAT[1], ls="--", lw=1.4)
    ax[0].axhline(1, color=MUTED, lw=1)
    ax[0].text(5, p.R0T + 0.08, r"$R_0^T=\bar\beta/\gamma=2.45$", color=INK, fontsize=8)
    ax[0].text(5, 3.75, r"$R_0(t)=\beta(t)/\gamma$", color=INK, fontsize=8)
    ax[0].set_xticks(np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])[::2])
    ax[0].set_xticklabels(MONTHS[::2])
    ax[0].set_ylabel("reproduction number")
    panel_tag(ax[0], "(a) seasonal forcing, dengue")
    m = d["monotonicity"]
    hs = d["dengue"]["h0_star"]
    ax[1].semilogx(m["radii"], m["R0D"], "o-", color=CAT[0], ms=3.5)
    ax[1].axhline(p.R0T, color=CAT[1], ls="--", lw=1.4)
    ax[1].axhline(1, color=MUTED, lw=1)
    ax[1].axvline(hs, color=MUTED, lw=1, ls=":")
    ax[1].text(hs * 1.08, 0.7, f"$h_0^*={hs:.3f}$ km", fontsize=8)
    ax[1].text(0.28, p.R0T - 0.17, "$R_0^T=2.45$", fontsize=8, color=INK)
    ax[1].set_xlabel("initial half-width $h_0$ (km)")
    ax[1].set_ylabel("$R_0^D(h_0)$")
    panel_tag(ax[1], "(b) domain-restricted reproduction number")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_thresholds.png"))


def fig2():
    c = load("s02_convergence")
    s = load("s03_speed_mu")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    runs = c["cbar_seasonal"]["runs"]
    dts = [r["dt"] for r in runs]
    ax[0].plot(dts, [r["cbar"] for r in runs], "o-", color=CAT[0], ms=4, label="seasonal, $\\varepsilon=0.6$")
    ax[0].axhline(c["cbar_seasonal"]["richardson"], color=CAT[0], ls=":", lw=1.2)
    sim0 = c["autonomous"]["sim"]
    ax[0].plot([r["dt"] for r in sim0], [r["c"] for r in sim0], "s-", color=CAT[1], ms=4, label="autonomous, $\\varepsilon=0$")
    ax[0].axhline(c["autonomous"]["semiwave_extrap"], color=CAT[1], ls="--", lw=1.2)
    ax[0].text(0.045, c["autonomous"]["semiwave_extrap"] - 0.0045, "semi-wave $c_0$ (independent solve)", fontsize=7.5, color=INK)
    ax[0].text(0.045, c["cbar_seasonal"]["richardson"] + 0.0015, "Richardson limit", fontsize=7.5, color=INK)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("time step $\\Delta t$ (day)")
    ax[0].set_ylabel("mean front speed (km/day)")
    ax[0].legend(fontsize=7.5, loc="center right")
    panel_tag(ax[0], "(a) time-step convergence, $\\mu=2$")
    rows = s["rows"]
    mus = [r["mu"] for r in rows]
    ax[1].semilogx(mus, [r["c0"] for r in rows], "s-", color=CAT[1], ms=4, label="autonomous $c_0$ (semi-wave)")
    ax[1].semilogx(mus, [r["cbar"] for r in rows], "o-", color=CAT[0], ms=4, label="seasonal $\\bar c^T$ (simulation)")
    ax[1].axhline(s["cstar"], color=MUTED, ls="--", lw=1.2)
    ax[1].text(25, s["cstar"] - 0.07, f"KPP bound $c^*={s['cstar']:.3f}$", fontsize=7.5)
    ax[1].set_xlabel("expansion coefficient $\\mu$ (km$^2$/day)")
    ax[1].set_ylabel("front speed (km/day)")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend(fontsize=7.5, loc="lower right")
    panel_tag(ax[1], "(b) speed against $\\mu$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_speed_validation.png"))


def fig3():
    c = load("s02_convergence")["ratchet"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    sp = np.array(c["speed"])
    ax[0].bar(range(12), sp, color=CAT[0], width=0.72)
    for i in (int(np.argmin(sp)), int(np.argmax(sp))):
        ax[0].text(i, sp[i] + 0.008, f"{sp[i]:.3f}", ha="center", fontsize=7.5)
    ax[0].set_xticks(range(12)); ax[0].set_xticklabels(MONTHS, fontsize=7.5)
    ax[0].set_ylabel("mean front speed (km/day)")
    panel_tag(ax[0], f"(a) monthly speed, year 3 (max/min = {c['ratio']:.1f})")
    t = np.array(c["t"]) / 365
    ax[1].plot(t, c["h"], color=CAT[0])
    ax[1].set_xlabel("time (years)"); ax[1].set_ylabel("$h(t)$ (km)")
    panel_tag(ax[1], f"(b) the ratchet: {c['annual_km']:.1f} km per year")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_ratchet.png"))


def fig4():
    d = load("s04_mustar")
    fig = plt.figure(figsize=(7.4, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, :])
    ii = d["intro_date"]
    tin = np.array([r["t_in"] for r in ii]); ms = np.array([r["mu_star"] for r in ii])
    eps0 = [r["mu_star"] for r in d["eps_sweep"] if r["eps"] == 0.0][0]
    ax0.semilogy(tin, ms, "o-", color=CAT[0], ms=3.5, label="seasonal, $\\varepsilon=0.6$")
    ax0.axhline(eps0, color=CAT[1], ls="--", lw=1.4, label="no seasonality, same $R_0^T$")
    ax0.axvline(265, color=MUTED, ls=":", lw=1)
    ax0.text(268, ms.max() * 0.9, "peak", fontsize=7.5, color=MUTED)
    ax0.set_xticks(np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])[::2]); ax0.set_xticklabels(MONTHS[::2])
    ax0.set_xlabel("date of introduction"); ax0.set_ylabel("$\\mu^*$ (km$^2$/day)")
    ax0.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.55, 1.0))
    panel_tag(ax0, f"(a) $h_0=1$ km: range {ms.min():.3f} to {ms.max():.1f}")
    a = d["amp_scaling"]
    amps = np.array([r["amp"] for r in a]); mm = np.array([r["mu_star"] for r in a])
    ax1.loglog(amps, mm, "o-", color=CAT[0], ms=4, label="computed $\\mu^*$")
    k0 = a[0]["kappa"]
    ax1.loglog(amps, k0 / amps, "--", color=CAT[1], lw=1.2, label=f"$\\kappa^*/I_0$, $\\kappa^*={k0:.3f}$")
    ax1.set_xlabel("initial peak prevalence $I_0$"); ax1.set_ylabel("$\\mu^*$")
    ax1.legend(fontsize=7)
    panel_tag(ax1, "(b) $\\mu^*\\,I_0\\to\\kappa^*$ (linear regime)")
    M = np.array(d["map"]["mu_star"], dtype=float)
    eps = d["map"]["eps"]; tm = d["map"]["t_in"]
    im = ax2.imshow(M, aspect="auto", origin="lower", cmap=BLUES, norm=LogNorm(vmin=M.min(), vmax=M.max()))
    ax2.set_xticks(range(12)); ax2.set_xticklabels(MONTHS)
    ax2.set_yticks(range(len(eps))); ax2.set_yticklabels([f"{e:g}" for e in eps])
    ax2.set_xlabel("month of introduction"); ax2.set_ylabel("seasonal amplitude $\\varepsilon$")
    ax2.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax2.text(j, i, f"{v:.2g}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                     color="white" if v > np.sqrt(M.min() * M.max()) * 1.5 else INK)
    cb = fig.colorbar(im, ax=ax2, pad=0.01); cb.set_label("$\\mu^*$ (log scale)")
    panel_tag(ax2, "(c) $\\mu^*$ by month of introduction and seasonal amplitude ($h_0=1$ km, $I_0=0.01$)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_mustar.png"))


def fig5():
    d = load("s05_acceleration_all")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    fin = [("gaussian", "Gaussian"), ("laplace", "Laplace"), ("sexp_half", "stretched exp $p=1/2$"),
           ("alg_a4", "algebraic $a=4$"), ("alg_a3", "algebraic $a=3$"), ("alg_a2.5", "algebraic $a=2.5$")]
    for i, (k, lab) in enumerate(fin):
        r = d[f"{k}|eps=0.6"]
        ax[0].plot(r["years"], r["h_over_t"], color=CAT[i], lw=1.6, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("time (years)"); ax[0].set_ylabel("$h(t)/t$ (km/day)")
    ax[0].legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    panel_tag(ax[0], "(a) finite first moment: $h(t)/t$ levels off")
    rc = d["cauchy|eps=0.6"]
    ax[1].loglog(np.array(rc["years"]) * 365, rc["h"], "o-", ms=2.5, color=CAT[0], lw=1.4, label="Cauchy ($\\gamma=2$)")
    for i, (a, lab) in enumerate(((1.75, "algebraic $a=1.75$"), (1.5, "algebraic $a=1.5$"))):
        rr = load(f"s05c_fastkernels_a{a}_eps0.6")[f"a{a}|eps=0.6"]
        ax[1].loglog(np.array(rr["years"]) * 365, rr["h"], "o-", ms=2.5, color=CAT[i + 1], lw=1.4, label=lab)
    ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    tt = np.array([365, 365 * 30.0])
    ax[1].loglog(tt, 0.5 * tt, ":", color=MUTED, lw=1)
    ax[1].text(420, 0.5 * 420 * 1.6, "slope 1", fontsize=7, color=MUTED)
    ax[1].set_xlabel("time (days)"); ax[1].set_ylabel("$h(t)$ (km)")
    panel_tag(ax[1], "(b) infinite first moment: acceleration")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_kernels.png"))


def fig6():
    d = load("s06_full_sir")
    g = load("s06b_full_diagnostics")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    keys = [("base (eps=.6, mu=2, delta=1/70y)", "full SIR, $\\mu=2$", CAT[0]),
            ("mu=50", "full SIR, $\\mu=50$", CAT[2]),
            ("fast turnover delta=1/1y", "full, $\\mu=2$, $\\delta=1/(1\\,\\mathrm{y})$", CAT[3])]
    for k, lab, col in keys:
        r = g[k]
        ax[0].plot(np.array(r["t"]) / 365, r["H"], color=col, label=lab)
    tr = d["traj"]
    ax[0].plot(np.array(tr["t"]) / 365, tr["h_red"], color=CAT[1], ls="--", label="reduced surrogate, $\\mu=2$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("time (years)"); ax[0].set_ylabel("$h(t)$ (km)")
    ax[0].legend(fontsize=6.5, loc="lower right")
    panel_tag(ax[0], "(a) right free boundary")
    ax[0].set_ylim(1, 3000)
    for k, lab, col in keys + [("no seasonality (eps=0)", "full SIR, $\\mu=2$, $\\varepsilon=0$", CAT[4])]:
        r = g[k]
        ax[1].plot(np.array(r["t"]) / 365, np.maximum(r["Imax"], 1e-12), color=col, label=lab)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-8, 1)
    ax[1].set_xlabel("time (years)"); ax[1].set_ylabel("max$_x\\,I(t,x)$")
    ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    panel_tag(ax[1], "(b) peak prevalence inside $\\Omega(t)$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_full_sir.png"))


def fig7():
    d = load("s07_sensitivity")["global_"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True)
    labels = {"beta_bar": "$\\bar\\beta$", "gamma": "$\\gamma$", "d2": "$d$", "sigma": "$\\sigma$",
              "eps": "$\\varepsilon$", "mu": "$\\mu$", "intro_day": "days to peak"}
    labels_mu = dict(labels, mu="$\\mu$ (control)")
    for a, key, title in ((ax[0], "prcc_cbar", "(a) mean front speed $\\bar c^T$"),
                          (ax[1], "prcc_mustar", "(b) $\\log\\mu^*$ at $h_0=h_0^*/2$")):
        rows = d[key]
        lab_ = labels_mu if key == "prcc_mustar" else labels
        names = [lab_[r["param"]] for r in rows]; vals = [r["prcc"] for r in rows]
        y = np.arange(len(rows))[::-1]
        a.barh(y, vals, color=[CAT[0] if v >= 0 else CAT[1] for v in vals], height=0.62)
        a.set_yticks(y); a.set_yticklabels(names)
        a.axvline(0, color=MUTED, lw=0.8)
        for yy, v, r in zip(y, vals, rows):
            star = "" if r["p"] < 0.01 else " (n.s.)"
            a.text(v + (0.03 if v >= 0 else -0.03), yy, f"{v:+.2f}{star}", va="center",
                   ha="left" if v >= 0 else "right", fontsize=7)
        a.set_xlim(-1.25, 1.25)
        a.set_xlabel("PRCC")
        panel_tag(a, title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_prcc.png"))


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7"]
    for f in todo:
        globals()[f]()
        print("ok", f)
