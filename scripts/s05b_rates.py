"""Growth-law diagnostics for the kernel study (reads s05_acceleration_all.json)."""
from _common import *

d = load("s05_acceleration_all")
T = 365.0
theory = {
    "gaussian": "finite c₀", "laplace": "finite c₀", "sexp_half": "finite c₀",
    "alg_a4": "finite; |h − c₀t| bdd", "alg_a3": "finite; c₀t − h ~ ln t", "alg_a2.5": "finite; c₀t − h ~ t^½",
    "cauchy": "h ~ t ln t", "alg_a1.75": "h ~ t^(4/3)", "alg_a1.5": "h ~ t²",
}
out = {}
for k in theory:
    r = d[f"{k}|eps=0.6"]
    yrs = np.array(r["years"], float)
    h = np.array(r["h"], float)
    t = yrs * T
    ht = h / t
    sw = d.get(f"{k}|semiwave", {}).get("c0")
    r0 = d.get(f"{k}|eps=0.0")
    entry = dict(theory=theory[k], final_year=int(yrs[-1]), final_h_over_t=float(ht[-1]))
    if r["first_moment_finite"]:
        n = len(ht)
        j = max(0, n - 11)
        chg = ht[-1] / ht[j] - 1
        entry["change_last_10y"] = float(chg)
        entry["auton_final_h_over_t"] = float(r0["h_over_t"][-1]) if r0 else None
        entry["semiwave_c0"] = sw
        entry["measured"] = f"levels off ({chg * 100:+.1f}% in last 10 yr)"
    elif k == "cauchy":
        m = yrs >= 5
        A = np.vstack([np.log(t[m]), np.ones(m.sum())]).T
        coef, res, *_ = np.linalg.lstsq(A, ht[m], rcond=None)
        pred = A @ coef
        r2 = 1 - np.sum((ht[m] - pred) ** 2) / np.sum((ht[m] - ht[m].mean()) ** 2)
        # local power exponent for comparison
        sl = np.polyfit(np.log(t[m]), np.log(h[m]), 1)[0]
        entry.update(slope_vs_lnt=float(coef[0]), r2=float(r2), loglog_slope=float(sl),
                     measured=f"h/t linear in ln t (R² = {r2:.3f})")
    else:
        a = {"alg_a1.75": 1.75, "alg_a1.5": 1.5}[k]
        rr = load(f"s05c_fastkernels_a{a}_eps0.6")[f"a{a}|eps=0.6"]
        loc = np.array(rr["local_exponent"], float)
        tail = loc[len(loc) // 2:]
        pred = 1 / (a - 1)
        entry.update(local_exponents=loc.tolist(), late_exponent=float(np.mean(tail)), predicted=pred,
                     years_coarse=int(rr["years"][-1]),
                     measured=f"exponent {loc[0]:.2f} → {loc[-1]:.2f}, falling")
    out[k] = entry
    print(k, entry)
save("s05_rates", out)
