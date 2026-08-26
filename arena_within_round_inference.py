"""Within-round inference on the raw 480 Arena judgments (addresses reproducibility
of the automated Arena). Runs a mixed-effects model (random intercept per case) on
the clean single-round 'Matrix' factorial for the two local Gemma models, giving
base-vs-RAG / base-vs-FT contrasts with 95% CIs and p-values."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, statsmodels.formula.api as smf
df=pd.read_csv("supplementary_data/Supplementary_Data_2_arena_raw_480.csv")
m=df[(df["round"]=="Matrix") & (df["composite"].notna())].copy()
name={"Gemma4 31B base":"31B_base","Gemma4 31B + RAG":"31B_RAG","Gemma4 31B FT":"31B_FT","Gemma4 31B FT+RAG":"31B_FTRAG",
      "Gemma4 26B base":"26B_base","Gemma4 26B MoE + RAG":"26B_RAG","Gemma4 26B MoE FT":"26B_FT","Gemma4 26B FT+RAG":"26B_FTRAG"}
m["t"]=m["tier_name"].map(name); m["case_id"]=m["case_id"].astype(str)
for base,variants,lab in [("31B_base",["31B_RAG","31B_FT","31B_FTRAG"],"Gemma4 31B"),
                          ("26B_base",["26B_RAG","26B_FT","26B_FTRAG"],"Gemma4 26B MoE")]:
    sub=m[m["t"].isin([base]+variants)].copy(); sub["t"]=pd.Categorical(sub["t"],categories=[base]+variants)
    mod=smf.mixedlm("composite ~ C(t)",sub,groups="case_id").fit(reml=True,method="lbfgs")
    print(f"\n{lab} — contrasts vs base (random intercept per case, 95% CI):")
    for v in variants:
        k=f"C(t)[T.{v}]"; b=mod.fe_params[k]; se=mod.bse[k]; p=mod.pvalues[k]
        print(f"  {v:10s} Delta={b:+.2f}  CI[{b-1.96*se:+.2f},{b+1.96*se:+.2f}]  p={p:.4g}")
