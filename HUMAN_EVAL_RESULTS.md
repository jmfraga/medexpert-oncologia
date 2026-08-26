# Human Evaluation by Oncologists — Results

**Version:** matches the manuscript submitted to npj Digital Medicine (26 Aug 2026).
**Analysis:** linear mixed-effects model (fixed effect for tier, random intercept per rater, REML), pre-specified non-inferiority test, leave-one-rater-out sensitivity, and Krippendorff's α.
**Source of truth:** `supplementary_data/Supplementary_Data_3_human_evaluation.xlsx` → `onco_human_eval_data.csv` (104 unique ratings).
**Reproduce:** `python human_eval_mixed_model.py` · `python human_eval_coverage_concordance.py` · `python human_eval_robustness_ols.py`

---

## 1. Design and sample

**Tiers evaluated** (blinded; candidates A/B/C randomised per case, fixed seed 42) — the three top performers under the automated judge, **all un-augmented base models**:

- Claude **Sonnet 4.6** (paid API) — reference
- **Gemma-4 31B** base (local, free)
- **Gemma-4 26B MoE** base (local, free)

Fine-tuned and RAG tiers did not reach the pre-specified quality gate (Opus composite ≥ 2.5/5) and were not submitted to human review.

**Raters:** 6 practising oncologists (medical oncology and radiation oncology); a seventh invited oncologist did not respond and contributed no data. **104 unique ratings** across all 15 cases, scored on the 4-criterion rubric (diagnostic accuracy 25%, guideline adherence 30%, completeness 25%, clinical utility 20%).

| Rater | Ratings | Mean severity /5 |
|---|---|---|
| R1 | 21 | 4.25 |
| R2 | 17 | 4.38 |
| R3 | 7 | 3.96 |
| R4 | 21 | 3.16 |
| R5 | 21 | 3.97 |
| R6 | 17 | 3.27 |

**Coverage:** 12 of 15 cases had ≥2 raters who each scored all three candidates (20 fully-overlapping rater pairs). Case 10 (Hodgkin lymphoma) obtained only one rater, as lymphoma is largely managed outside general medical oncology in our setting.

---

## 2. Primary result — composite score

Marginal means and contrasts against Sonnet 4.6 (reference):

| Tier | Marginal mean /5 | Δ vs Sonnet | 95% CI | p |
|---|---|---|---|---|
| **Sonnet 4.6 (API, paid)** | **4.03** | — | — | — |
| **Gemma-4 31B (local, free)** | **3.87** | −0.154 | −0.452 to +0.145 | 0.314 |
| **Gemma-4 26B MoE (local, free)** | **3.57** | −0.456 | −0.760 to −0.152 | 0.003 |

Model converged (`Converged: True`); Var(rater) = 0.234, Var(residual) = 0.406. A case variance component estimated to ~0 and was dropped for clean convergence.

### Non-inferiority (margin δ = 0.5 points)

The margin is one half of the smallest interpretable unit of the instrument: the rubric is anchored in whole-point steps, each a distinct qualitative level of clinical quality, so a difference below half a rubric point cannot move a response across an anchor category.

- **Gemma-4 31B vs Sonnet: no inferiority detected** — lower CI bound −0.452 > −0.5.
- **Gemma-4 26B MoE**: significantly inferior; does not meet non-inferiority.

---

## 3. By criterion

| Criterion | Sonnet 4.6 | Gemma-4 31B (Δ, p) | Gemma-4 26B MoE (Δ, p) |
|---|---|---|---|
| Diagnostic accuracy | 4.20 | +0.08 (0.624) → 4.28 | −0.21 (0.212) → 3.99 |
| Guideline adherence | 3.97 | −0.17 (0.399) → 3.80 | −0.52 (**0.010**) → 3.44 |
| Completeness | 3.93 | −0.22 (0.231) → 3.71 | −0.50 (**0.008**) → 3.43 |
| Clinical utility | 4.03 | −0.34 (0.064) → 3.69 | −0.61 (**0.001**) → 3.41 |

Gemma-4 31B did not differ significantly from Sonnet on any criterion. Gemma-4 26B MoE was significantly below Sonnet on guideline adherence, completeness and clinical utility.

**Agreement with the automated judge:** the ranking is reproduced **at the aggregate (group-mean) level** — Opus 4.6 gave Sonnet 4.47 > Gemma-4 31B 4.17 > Gemma-4 26B 3.60; the oncologists gave Sonnet 4.03 > Gemma-4 31B 3.87 > Gemma-4 26B 3.57. This is a weaker statement than case-by-case agreement, which was near zero (see §4).

---

## 4. Inter-rater agreement (reported as a finding)

**Krippendorff's α = 0.008** on the composite — essentially chance-level agreement. Variance is dominated by the rater (0.234) rather than the case (~0): the main source of variability is *who* scores, not *what* is scored. Rater severity spanned **3.16 to 4.38** out of 5.

The tier contrasts in §2–§3 already control for this through the rater random intercept, which is why they — and not the raw means — are the valid estimates. The low absolute agreement is itself a result: subjective clinical scoring of AI-generated responses varies substantially between oncologists, consistent with the inter-observer-variability literature. No calibration or anchoring session was held before the evaluation.

---

## 5. Sensitivity — leave-one-rater-out

| Excluded | Δ (31B − Sonnet) | Lower CI | Non-inferiority at δ=0.5 |
|---|---|---|---|
| (none) | −0.154 | −0.452 | yes |
| R1 | −0.222 | **−0.572** | **NO** |
| R2 | −0.165 | −0.486 | yes |
| R3 | −0.176 | −0.498 | yes |
| R4 | −0.094 | −0.428 | yes |
| R5 | −0.172 | −0.474 | yes |
| R6 | −0.091 | −0.423 | yes |

The point estimate stays near zero throughout (−0.09 to −0.22), and **non-inferiority holds in 5 of 6 analyses**. It fails only when R1 (a lenient rater, mean 4.25/5) is excluded — largely a loss-of-power effect, since each leave-one-out analysis retains only 5 of the 6 raters. We therefore report the local-vs-API comparison as *"no inferiority detected"*, not as established equivalence, and rely on the automated Arena for the study's primary claims.

---

## 6. Conclusions

1. **A free, locally deployable base model (Gemma-4 31B) showed no detected inferiority to a paid frontier API** in blinded expert judgement — supporting privacy-preserving, essentially zero-marginal-cost clinical decision support.
2. **Model scale matters within the local option:** Gemma-4 26B MoE was significantly inferior on three of four criteria.
3. **Expert agreement on AI-generated clinical answers is poor**, which is a caution for the field and for the design of future evaluation panels.

## 7. Limitations

- Synthetic clinical cases, not real encounters.
- The panel is **confirmatory** and under-powered relative to the 480-judgment automated Arena; it corroborates the ranking but does not establish definitive effect sizes on its own.
- Non-inferiority is sensitive to individual raters (see §5).
- No pre-hoc calibration; severity ranged 3.16–4.38.
- Case 10 has a single rater and contributes nothing to concordance.
- Several co-authors served as raters; they were blinded throughout, reviewed the manuscript only after the analysis was complete, and had no role in model development, the automated evaluation or the statistical analysis.
