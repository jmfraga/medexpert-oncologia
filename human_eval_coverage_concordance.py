"""Coverage and inter-rater concordance for the human evaluation.

Reads the de-identified dataset (`onco_human_eval_data.csv`, raters R1–R6) and reports:
  - ratings per rater and rater severity
  - case coverage (how many cases have >=2 raters scoring all three candidates)
  - Krippendorff's alpha (interval) on the composite and on each criterion
  - alpha after removing between-rater severity, to separate "who scores" from "what is scored"
"""
from collections import defaultdict
from itertools import combinations
import pandas as pd

df = pd.read_csv("onco_human_eval_data.csv", dtype={"caso": str})
CRIT = [("pdx", "Diagnostic accuracy"), ("guias", "Guideline adherence"),
        ("comp_crit", "Completeness"), ("util", "Clinical utility")]

print(f"N ratings: {len(df)} | raters: {df.evaluador.nunique()} | cases: {df.caso.nunique()}\n")

# ---- per-rater ----
print("Ratings and severity per rater:")
sev = df.groupby("evaluador").composite.mean()
for r in sorted(df.evaluador.unique()):
    sub = df[df.evaluador == r]
    cases = sorted(set(sub.caso), key=int)
    print(f"  {r}: {len(sub):3d} ratings | {len(cases)} cases {cases} | mean {sev[r]:.2f}")
print(f"  severity range: {sev.min():.2f} – {sev.max():.2f}")

# ---- coverage ----
rated = defaultdict(lambda: defaultdict(set))          # case -> rater -> {candidates}
cell = defaultdict(set)                                 # (case, candidate) -> {raters}
for _, r in df.iterrows():
    rated[r.caso][r.evaluador].add(r.cand)
    cell[(r.caso, r.cand)].add(r.evaluador)

full = [c for c in sorted(df.caso.unique(), key=int)
        if sum(1 for _, cs in rated[c].items() if cs >= {"A", "B", "C"}) >= 2]
pairs = sum(len(list(combinations(
    [e for e, cs in rated[c].items() if cs >= {"A", "B", "C"}], 2))) for c in full)
orphan = [c for c in sorted(df.caso.unique(), key=int) if len(rated[c]) < 2]

print(f"\nCoverage: {len(full)}/{df.caso.nunique()} cases with >=2 raters scoring all three "
      f"candidates ({pairs} fully-overlapping rater pairs)")
print(f"  cases with a single rater: {orphan or 'none'}")

# ---- Krippendorff's alpha (interval) ----
def alpha(units):
    units = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in units)
    Do = sum(sum((u[i] - u[j]) ** 2 for i in range(len(u)) for j in range(len(u)) if i != j)
             / (len(u) - 1) for u in units) / n
    vals = [v for u in units for v in u]
    N = len(vals)
    De = sum((vals[i] - vals[j]) ** 2 for i in range(N) for j in range(N) if i != j) / (N * (N - 1))
    return 1 - Do / De

def units_for(col, centred=False):
    u = defaultdict(list)
    for _, r in df.iterrows():
        v = r[col] - sev[r.evaluador] if centred else r[col]
        u[(r.caso, r.cand)].append(v)
    return list(u.values())

print("\nKrippendorff's alpha (interval):")
print(f"  composite            {alpha(units_for('composite')):+.3f}")
for c, label in CRIT:
    print(f"  {label:<21}{alpha(units_for(c)):+.3f}")
print(f"\n  composite, after removing rater severity: {alpha(units_for('composite', True)):+.3f}")
print("  (>0.80 excellent | 0.67–0.80 tentative | <0.67 weak)")
print("\nInterpretation: agreement is near chance and variance is dominated by the rater, not the\n"
      "case. The tier contrasts in human_eval_mixed_model.py control for this via the rater\n"
      "random intercept, which is why those — and not the raw means — are the valid estimates.")
