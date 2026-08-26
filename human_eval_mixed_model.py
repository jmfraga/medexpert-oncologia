"""Human-evaluation mixed-effects model + leave-one-rater-out sensitivity.

Model: composite ~ tier, random intercept per rater. The case variance component
was dropped after it estimated to ~0 (boundary; non-convergence with statsmodels);
removing it yields clean convergence and identical fixed-effect CIs. All variance
is between-rater (severity) and residual.
"""
import pandas as pd, warnings
warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf

df = pd.read_csv("onco_human_eval_data.csv", dtype={"caso": str})
print(f"N={len(df)} | raters={df.evaluador.nunique()} | cases={df.caso.nunique()}\n")

def fit(data, outcome="composite"):
    # random intercept per rater only (captures severity); converges cleanly
    m = smf.mixedlm(f"{outcome} ~ C(tier, Treatment('sonnet'))", data,
                    groups="evaluador").fit(reml=True, method="lbfgs")
    out = {}
    for t in ("gemma4_31b", "gemma4_26b"):
        k = f"C(tier, Treatment('sonnet'))[T.{t}]"
        b, se = m.fe_params[k], m.bse[k]
        out[t] = (b, se, b - 1.96*se, b + 1.96*se, m.pvalues[k])
    return m, out

# ---- main model ----
m, out = fit(df)
print(f"Converged: {m.converged} | Var(rater)={float(m.cov_re.iloc[0,0]):.3f} | Var(resid)={m.scale:.3f}")
for t, tn in (("gemma4_31b", "Gemma4 31B (local)"), ("gemma4_26b", "Gemma4 26B MoE (local)")):
    b, se, lo, hi, p = out[t]
    print(f"  {tn} - Sonnet: delta={b:+.3f}  CI95%[{lo:+.3f},{hi:+.3f}]  p={p:.3f}")
b, se, lo, hi, p = out["gemma4_31b"]
print(f"  Non-inferiority 31B vs Sonnet (delta=0.5): CI_low={lo:+.3f} -> {'NON-INFERIOR' if lo > -0.5 else 'not met'}")

# ---- leave-one-rater-out sensitivity (addresses reviewer C1) ----
print("\nLeave-one-rater-out (composite, Gemma4 31B - Sonnet):")
print(f"  {'excluded':<14}{'delta':>8}{'CI_low':>9}   NI@0.5")
for r in ["(none)"] + sorted(df.evaluador.unique()):
    sub = df if r == "(none)" else df[df.evaluador != r]
    _, o = fit(sub)
    b, se, lo, hi, p = o["gemma4_31b"]
    print(f"  {r:<14}{b:>+8.3f}{lo:>+9.3f}   {'yes' if lo > -0.5 else 'NO'}")
