import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf
df=pd.read_csv("onco_human_eval_data.csv", dtype={"caso":str})
# Sanity: OLS con efectos fijos de evaluador (dummies) — no asume nada de random effects
m=smf.ols("composite ~ C(tier, Treatment('sonnet')) + C(evaluador)", df).fit(cov_type="cluster", cov_kwds={"groups":df["caso"]})
print("OLS efectos-fijos-evaluador (SE clusterizados por caso) — COMPUESTO:")
for t,tn in [("gemma4_31b","G31B"),("gemma4_26b","G26B")]:
    k=f"C(tier, Treatment('sonnet'))[T.{t}]"; b=m.params[k]; lo,hi=m.conf_int().loc[k]
    print(f"  {tn}−Sonnet: Δ={b:+.3f} IC95%[{lo:+.3f},{hi:+.3f}] p={m.pvalues[k]:.3f}")
