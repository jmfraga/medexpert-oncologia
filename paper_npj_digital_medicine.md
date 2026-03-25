# Low-Cost Clinical Fine-Tuning of a Large Language Model for Oncology: A Multi-Tier Evaluation with Automated and Expert Human Assessment

**Target journal:** npj Digital Medicine (Nature)

**Authors:** Juan Manuel Fraga Sastrías¹, [co-authors TBD]

**Affiliations:**
¹ [Institution TBD]

---

## Abstract

**Background:** Large language models (LLMs) show promise in clinical oncology, but frontier models remain costly and dependent on external APIs, limiting adoption in resource-constrained healthcare settings. Whether a small, locally-deployed model fine-tuned on domain-specific knowledge can approach the clinical utility of larger models remains an open question.

**Methods:** We constructed a bilingual oncology knowledge base comprising 134,478 text chunks from 663 sources including NCCN, ESMO, Mexican IMSS clinical practice guidelines, and 397 pharmaceutical drug monographs. From this corpus, 11,592 representative chunks were selected via stratified sampling across five clinical strata (treatment, pharmacology, diagnosis, supportive care, follow-up). A teacher model (Claude Sonnet 4.6) generated ~35,000 question-answer training pairs in Spanish. We fine-tuned Meta-Llama-3.1-8B-Instruct using QLoRA (rank 64, all layers) on consumer hardware (Mac Mini M4, 64 GB RAM) via the MLX framework. We evaluated five response tiers — (1) fine-tuned model alone, (2) fine-tuned model with retrieval-augmented generation (RAG), (3) MedGemma 27B with RAG, (4) MiniMax 2.7 with RAG, and (5) Claude Sonnet 4.6 with RAG — across 15 clinical oncology cases of varying complexity. Evaluation employed a dual assessment design: automated scoring by Claude Opus 4.6 and blinded human evaluation by clinical oncologists using a shared four-criteria rubric (diagnostic accuracy, guideline adherence, completeness, clinical utility), with inter-rater reliability assessed via Krippendorff's alpha.

**Results:** [TO BE COMPLETED]

**Conclusions:** [TO BE COMPLETED]

**Keywords:** large language models, oncology, fine-tuning, QLoRA, clinical decision support, retrieval-augmented generation, low-resource, Latin America

---

## Introduction

The application of large language models (LLMs) to clinical medicine has accelerated rapidly, with models demonstrating competence in medical licensing examinations, clinical reasoning⁹, and guideline-concordant treatment recommendations²⁶. In oncology specifically, LLMs have shown potential as clinical decision support tools for treatment planning, drug interaction assessment, and patient education²⁵˒²⁷, with a recent meta-analysis reporting an average accuracy of 76.2% across 56 studies and 15 cancer types²⁶.

However, the deployment of LLMs in clinical oncology faces three critical barriers. First, frontier models capable of nuanced clinical reasoning require costly API access, with per-query costs of $0.05–0.15 USD that become prohibitive at institutional scale⁸. Second, these models depend on external cloud infrastructure, raising concerns about patient data privacy, latency, and availability in regions with limited connectivity⁶. Third, the majority of clinical AI research and model training has been conducted in English, with limited attention to the clinical terminology, guidelines, and healthcare contexts of Latin American countries⁷.

The cautionary tale of IBM Watson for Oncology (WFO) illustrates the localization problem acutely. Despite significant investment, WFO showed concordance rates ranging from 48.9% for colon cancer in South Korea² to 93% for breast cancer in India¹, with a meta-analysis across 2,463 patients reporting 81.5% overall concordance but significantly lower rates for advanced-stage disease and cancer types underrepresented in its US-centric training data⁴. Critically, WFO failed to account for locally approved drugs, regional insurance policies, and country-specific treatment patterns³˒⁵, underscoring that clinical AI systems must be adapted to local healthcare contexts to be clinically useful.

An alternative paradigm is the fine-tuning of smaller, open-weight models on domain-specific clinical knowledge, enabling local deployment on consumer-grade hardware without API dependencies. Recent work has demonstrated that models in the 7–13B parameter range can achieve competitive performance on medical benchmarks when appropriately fine-tuned¹¹˒¹²˒¹³. However, systematic evaluations comparing fine-tuned small models against frontier models in realistic clinical oncology scenarios — particularly with expert human evaluation — remain scarce.

The retrieval-augmented generation (RAG) approach offers a complementary strategy, grounding model responses in authoritative clinical guidelines at inference time¹⁸˒¹⁹. In oncology, RAG has shown promise for personalized treatment recommendations and clinical trial matching²⁰. Whether RAG provides additive value to a model that has already internalized domain knowledge through fine-tuning is an important practical question for system design.

Recent advances in parameter-efficient fine-tuning, particularly QLoRA²³, have made it feasible to fine-tune large models on consumer hardware by combining 4-bit quantization with low-rank adaptation²², dramatically reducing the compute requirements for domain specialization²⁴. Meanwhile, the emerging practice of using LLMs as automated evaluators ("LLM-as-judge") has shown promise for scalable assessment¹⁵, with recent work validating this approach in clinical AI contexts¹⁶, though important caveats about evaluation validity remain¹⁷.

In this study, we present a complete, reproducible pipeline for creating a domain-specialized oncology LLM at a total cost under $200 USD, running entirely on consumer hardware. We evaluate five response strategies spanning the spectrum from local fine-tuned model to frontier API, using a dual assessment framework combining automated LLM-based evaluation with blinded expert oncologist review. Our evaluation employs 15 clinical cases covering 14 cancer types at three complexity levels, assessed on four clinically relevant criteria.

Our primary research questions are:
1. Can a fine-tuned 8B-parameter model achieve clinically acceptable performance in oncology, and how does it compare to frontier models?
2. Does RAG provide additional value when the model has already internalized clinical knowledge through fine-tuning?
3. How well does automated LLM evaluation correlate with expert human judgment in clinical oncology?

---

## Methods

### Ethics Statement

This study was approved by [Institution Ethics Committee — TO BE COMPLETED]. All clinical cases were synthetically constructed and do not contain real patient data. No patient consent was required.

### Knowledge Base Construction

We assembled a bilingual clinical oncology knowledge base from six authoritative sources (Table 1).

**Table 1.** Sources comprising the oncology knowledge base.

| Source | Chunks | Language | Content |
|--------|--------|----------|---------|
| NCCN Guidelines | 62,004 | English | Cancer-specific treatment guidelines (breast, lung, colorectal, prostate, melanoma, etc.) |
| ESMO Clinical Practice Guidelines | 40,414 | English | European evidence-based oncology guidelines |
| IMSS Clinical Practice Guidelines (GPC) | 15,066 | Spanish | Mexican national healthcare system guidelines |
| Pharmaceutical Drug Monographs | 11,292 | Spanish | 397 oncology drug monographs from a cancer center pharmacy service |
| Mexican Breast Cancer Consensus 2025 | 2,717 | Spanish | National consensus on breast cancer management |
| ASCO/NCI | 250 | English | Selected American and NCI guidelines |
| **Total** | **134,478** | **Bilingual** | **663 unique sources** |

Documents in PDF, DOCX, and TXT formats were processed through a standardized pipeline: text extraction, clinical-aware chunking (500 characters, 100-character overlap, section boundary preservation), automated metadata extraction (source, society, category, section hierarchy, document type), sentence-transformer embedding, and storage in ChromaDB with cosine similarity indexing.

The retrieval system supports bilingual search (Spanish queries are translated to English for NCCN/ESMO retrieval), synonym expansion across 137 brand-generic drug name pairs (e.g., Keytruda ↔ pembrolizumab), society-level filtering, and result diversification (maximum 3 hits per source document).

### Training Data Generation

#### Stratified Sampling

From the full corpus of 134,478 chunks, we selected 11,592 representative chunks via stratified sampling across five clinical strata designed to ensure balanced coverage of oncological knowledge domains (Table 2).

**Table 2.** Stratified sampling design.

| Stratum | Proportion | Target | Content |
|---------|-----------|--------|---------|
| Treatment | 30% | 3,600 | Chemotherapy protocols, radiotherapy, surgery, immunotherapy |
| Pharmacology | 25% | 3,000 | Dosing, mechanisms of action, toxicities, interactions |
| Diagnosis | 20% | 2,400 | TNM staging, biomarkers, molecular pathology |
| Supportive care | 15% | 1,800 | Adverse effect management, palliative care |
| Follow-up | 10% | 1,200 | Post-treatment surveillance, survivorship |

Quality filters were applied prior to sampling: exclusion of low-information chunks (bibliography sections, conflict-of-interest statements, copyright notices, chunks <80 characters), content-hash deduplication (first 150 characters), and round-robin source diversification to prevent over-representation of any single document. The final sample comprised chunks from 568 distinct source documents.

#### Synthetic Question-Answer Generation

For each sampled chunk, a teacher model (Claude Sonnet 4.6, Anthropic) generated three question-answer pairs of increasing clinical complexity:

1. **Direct knowledge**: A factual question about the chunk content
2. **Clinical vignette**: A brief case (age, sex, staging) requiring clinical reasoning
3. **Therapeutic decision**: A comparison between treatment options or management decision

All responses were generated in Spanish regardless of source language, strictly grounded in the source chunk content, and included source attribution. Quality filters excluded responses shorter than 30 tokens. The generation process produced approximately 35,000 training pairs, split 80/10/10 into training, validation, and test sets.

The total cost of synthetic data generation was approximately $130 USD in API fees.

### Model Fine-Tuning

#### Base Model and Configuration

We fine-tuned Meta-Llama-3.1-8B-Instruct (4-bit quantized, sourced from mlx-community) using QLoRA²³ with the configuration detailed in Table 3.

**Table 3.** QLoRA fine-tuning hyperparameters.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 64 | Dense medical domain requires higher capacity than standard rank 8–16 |
| Target layers | All (-1) | Medical knowledge is distributed across all transformer layers |
| Batch size | 4 | Balance between throughput and training stability |
| Learning rate | 2 × 10⁻⁵ | Standard for LoRA on 8B models |
| Training iterations | 3,000 | Calibrated for ~35K training examples |
| Max sequence length | 2,048 | Sufficient for clinical Q&A |
| Dropout | 0.05 | Light regularization for specialized dataset |
| LoRA alpha (scale) | 32.0 | 0.5× rank, recommended for narrow-domain fine-tuning |
| Gradient checkpointing | Enabled | Reduces peak memory (~15 GB of 64 GB available) |
| Prompt masking | Enabled | Loss computed only on response tokens |

#### Hardware and Framework

Training was performed on a Mac Mini M4 (Apple Silicon, 64 GB unified memory) using the MLX framework (mlx-lm 0.31.1), which provides native Apple Silicon optimization. Estimated training time was 18–28 hours. After training, LoRA adapters were fused into the base model using `mlx_lm.fuse` to produce a standalone model (Llama8B-MedExpert-Oncologia) deployable without adapter overhead.

**Total fine-tuning compute cost: $0 USD** (local consumer hardware).

### Evaluation Design

#### Clinical Test Cases

We designed 15 clinical oncology cases with balanced distribution across complexity levels and cancer types (Table 4).

**Table 4.** Distribution of clinical test cases.

| Complexity | Count | Description |
|------------|-------|-------------|
| Simple | 5 | Clear diagnosis, well-defined standard treatment |
| Moderate | 4 | Requires integration of multiple clinical factors |
| Complex | 6 | Specific biomarkers, multiple treatment options, controversial scenarios |

Cases covered 14 distinct cancer types: breast (2), lung (2), colorectal (2), prostate, cervical, gastric, Hodgkin lymphoma, melanoma, ovarian, thyroid, chronic myeloid leukemia, and pancreatic cancer. Each case included a realistic clinical presentation (age, sex, symptoms, workup), complete staging, relevant biomarkers, and a gold standard consisting of the correct diagnosis plus 6–8 key guideline-based recommendations with source references.

#### Response Tiers

Five response strategies were evaluated, representing a spectrum from fully local to frontier API-based approaches (Table 5).

**Table 5.** Response tiers evaluated.

| Tier | Model | Provider | Approach | RAG |
|------|-------|----------|----------|-----|
| Light | Llama8B-MedExpert-Oncologia | Local (fine-tuned) | Internalized knowledge | No |
| Light+RAG | Llama8B-MedExpert-Oncologia | Local (fine-tuned) | Internalized + retrieval | Yes |
| Basic A | MedGemma 27B | Google | Medical pre-trained model | Yes |
| Basic B | MiniMax 2.7 | MiniMax | Competitive generalist | Yes |
| Premium | Claude Sonnet 4.6 | Anthropic | Frontier generalist | Yes |

For tiers with RAG, the retrieval system queried the ChromaDB knowledge base with the clinical case, retrieving relevant guideline excerpts that were prepended to the model prompt.

#### Automated Evaluation (LLM Judge)

Each of the 75 case-tier combinations (15 cases × 5 tiers) was evaluated by Claude Opus 4.6 (Anthropic) with access to RAG context and the gold standard. The judge model did not participate as a response candidate, maintaining impartiality.

Evaluation used a structured rubric with four criteria scored on a 0–5 scale (Table 6).

**Table 6.** Evaluation rubric (shared between automated and human evaluation).

| Criterion | Description | Weight |
|-----------|-------------|--------|
| Diagnostic accuracy | Agreement of diagnosis and staging with gold standard | 25% |
| Guideline adherence | Consistency with NCCN/ESMO/IMSS guidelines; correct citations | 30% |
| Completeness | Coverage of relevant aspects (treatment, alternatives, follow-up, adverse effects) | 25% |
| Clinical utility | Practical, actionable value for a practicing oncologist | 20% |

The composite score per evaluation was the weighted average across all four criteria (range 0–5).

#### Human Expert Evaluation

##### Panel Design

Clinical oncologists performed blinded evaluation of all response tiers. Responses were presented anonymized (Candidate A/B/C/D/E) with randomized assignment per case using a fixed seed for reproducibility. Evaluators were unaware of which model generated each response.

##### Assignment Design

Cases were assigned to evaluators using a balanced incomplete block design (BIBD), ensuring that each case was evaluated by exactly two oncologists and each evaluator received a proportional distribution of simple, moderate, and complex cases. Three evaluation scenarios were pre-specified based on panel size (Table 7).

**Table 7.** Evaluator assignment scenarios.

| Scenario | Oncologists | Cases/evaluator | Evaluators/case | Scores/evaluator | Feasibility |
|----------|-------------|-----------------|-----------------|------------------|-------------|
| A (minimum) | 3 | 10 | 2 | 200 | Feasible, high load |
| B (recommended) | 4 | 8 | ~2 | 160 | Good balance |
| C (ideal) | 5 | 6 | 2 | 120 | Light load |

The evaluation instrument was administered via Google Forms using the four-criteria rubric (Table 6) plus free-text comments per response.

##### Inter-Rater Reliability

Agreement between evaluators was quantified using Krippendorff's alpha (α), reported globally and per criterion:
- α > 0.80: excellent agreement
- 0.67 < α ≤ 0.80: acceptable agreement
- α ≤ 0.67: low agreement — discrepancies investigated and documented

#### Cross-Evaluation Analysis

Automated (Opus) and human scores were reported separately and compared using:

1. **Pearson and Spearman correlation** between Opus composite score and mean human composite score per case-tier combination
2. **Bland-Altman analysis**: difference (Opus − human) vs. average, to detect systematic bias
3. **Discrepancy analysis**: cases where |Opus − human| > 1.0 points on the composite score were flagged for qualitative review

A correlation coefficient r > 0.80 between Opus and human consensus would support the use of LLM-based evaluation as a reliable proxy for expert clinical judgment in this domain.

#### Additional Performance Metrics

For each response, the following operational metrics were recorded: time to first token (TTFT, ms), generation throughput (tokens/s), total response time (s), and per-query cost (USD).

### Hypotheses

Seven pre-specified hypotheses guided the analysis:

- **H1**: The Light tier (fine-tuned, no RAG) achieves ≥70% of the Premium tier score on simple cases
- **H2**: Basic tiers with RAG outperform Light on complex cases requiring specific guideline knowledge
- **H3**: Premium is not significantly superior to Basic tiers on simple cases (supporting tiered deployment)
- **H4**: Light achieves TTFT <2s vs. >5s for Premium (superior user experience)
- **H5**: Per-query cost of Light is <$0.001 vs. $0.05–0.15 for Basic/Premium
- **H6**: Opus evaluation correlates r > 0.80 with human consensus (LLM judge validation)
- **H7**: Light+RAG outperforms Light alone on complex cases (RAG adds value beyond internalized knowledge)

### Statistical Analysis

Score comparisons between tiers were performed using bootstrap confidence intervals (n = 15 cases, 10,000 resamples). Subgroup analyses by complexity level and evaluation criterion were pre-specified. Cost-effectiveness was assessed as score-per-dollar across tiers.

### Reproducibility and Cost Summary

The complete pipeline — from knowledge base construction through model deployment — was designed for reproducibility on consumer hardware. Table 8 summarizes costs.

**Table 8.** Total project cost.

| Phase | Duration | Cost (USD) |
|-------|----------|------------|
| Knowledge base construction | ~8 hours | $0 (open-access sources) |
| Stratified sampling | ~1 hour | $0 (local compute) |
| Training data generation (Sonnet 4.6) | ~43 hours | ~$130 |
| Fine-tuning (QLoRA, MLX) | ~18–28 hours | $0 (local) |
| Model fusion | ~10 minutes | $0 |
| Arena: candidate execution | ~1 hour | ~$3 |
| Arena: Opus evaluation | ~1 hour | ~$42 |
| Human evaluation | ~1 week (async) | $0 |
| **Total** | **~4 days + 1 week** | **~$175** |

All code, configurations, and evaluation materials are available at [repository URL — TO BE COMPLETED].

---

## Results

### Response Quality by Tier

**Table 9.** Mean composite score (weighted) by tier, overall and by complexity level. Scores on 0–5 scale; 95% CI from bootstrap (10,000 resamples).

| Tier | Overall | 95% CI | Simple (n=5) | Moderate (n=4) | Complex (n=6) |
|------|---------|--------|--------------|----------------|----------------|
| Light | — | — | — | — | — |
| Light+RAG | — | — | — | — | — |
| Básico A (MedGemma 27B) | — | — | — | — | — |
| Básico B (MiniMax 2.7) | — | — | — | — | — |
| Premium (Sonnet 4.6) | — | — | — | — | — |

### Performance by Evaluation Criterion

**Table 10.** Mean Opus score per tier × evaluation criterion (0–5 scale).

| Tier | Diagnostic Accuracy | Guideline Adherence | Completeness | Clinical Utility |
|------|--------------------|--------------------|--------------|-----------------|
| Light | — | — | — | — |
| Light+RAG | — | — | — | — |
| Básico A (MedGemma 27B) | — | — | — | — |
| Básico B (MiniMax 2.7) | — | — | — | — |
| Premium (Sonnet 4.6) | — | — | — | — |

### Complexity Subgroup Analysis

**Table 11.** Mean composite score by complexity level and tier.

| Complexity | Light | Light+RAG | Básico A | Básico B | Premium |
|------------|-------|-----------|----------|----------|---------|
| Simple (n=5) | — | — | — | — | — |
| Moderate (n=4) | — | — | — | — | — |
| Complex (n=6) | — | — | — | — | — |

**Figure 1.** [Bar chart: composite score by tier, grouped by complexity level — to be generated]

### Human Expert Evaluation

**Table 12.** Mean human expert scores by tier (0–5 scale, n evaluators per case = 2).

| Tier | Diagnostic Accuracy | Guideline Adherence | Completeness | Clinical Utility | Composite |
|------|--------------------|--------------------|--------------|-----------------|-----------|
| Light | — | — | — | — | — |
| Light+RAG | — | — | — | — | — |
| Básico A (MedGemma 27B) | — | — | — | — | — |
| Básico B (MiniMax 2.7) | — | — | — | — | — |
| Premium (Sonnet 4.6) | — | — | — | — | — |

**Table 13.** Inter-rater reliability (Krippendorff's alpha).

| Criterion | α | Interpretation |
|-----------|---|---------------|
| Diagnostic Accuracy | — | — |
| Guideline Adherence | — | — |
| Completeness | — | — |
| Clinical Utility | — | — |
| **Global (all criteria)** | **—** | **—** |

### Automated vs. Human Evaluation Agreement

**Table 14.** Correlation between Opus and human composite scores.

| Metric | Value | 95% CI |
|--------|-------|--------|
| Pearson r | — | — |
| Spearman ρ | — | — |
| Mean difference (Opus − Human) | — | — |
| Limits of agreement (Bland-Altman) | — | — |
| Discrepant cases (\|Δ\| > 1.0) | —/75 | — |

**Figure 2.** [Scatter plot: Opus composite score vs. mean human composite score, with regression line and 95% CI — to be generated]

**Figure 3.** [Bland-Altman plot: difference (Opus − Human) vs. average — to be generated]

### Operational Performance

**Table 15.** Operational metrics by tier (mean across 15 cases).

| Tier | TTFT (ms) | Throughput (tok/s) | Response Time (s) | Cost/Query (USD) |
|------|-----------|-------------------|-------------------|-----------------|
| Light | — | — | — | ~$0 |
| Light+RAG | — | — | — | ~$0 |
| Básico A (MedGemma 27B) | — | — | — | ~$0 |
| Básico B (MiniMax 2.7) | — | — | — | — |
| Premium (Sonnet 4.6) | — | — | — | — |

### Hypothesis Testing

**Table 16.** Pre-specified hypothesis results.

| Hypothesis | Description | Result | Supported? |
|------------|-------------|--------|------------|
| H1 | Light achieves ≥70% of Premium score on simple cases | Light/Premium ratio: —% | — |
| H2 | Basic tiers + RAG outperform Light on complex cases | Δ score: — | — |
| H3 | Premium not significantly superior to Basic on simple cases | Δ score: — (95% CI: —) | — |
| H4 | Light TTFT <2s vs. Premium >5s | Light: —ms, Premium: —ms | — |
| H5 | Light cost <$0.001 vs. $0.05–0.15 for Basic/Premium | Light: $—, Premium: $— | — |
| H6 | Opus–Human correlation r > 0.80 | r = — | — |
| H7 | Light+RAG outperforms Light alone on complex cases | Δ score: — | — |

---

## Discussion

[TO BE COMPLETED]

Key discussion points to address:

### Fine-Tuned Small Models vs. Frontier Models
- How does Light compare to Premium? Position relative to PMC-LLaMA¹¹, MEDITRON¹², BioMistral¹³, and Woollie²⁵
- [TO BE COMPLETED with actual results]

### The Value of RAG on Top of Fine-Tuning
- H7 result: does Light+RAG outperform Light alone?
- Comparison with Almanac (RAG for clinical medicine)¹⁹ and oncology RAG applications²⁰
- [TO BE COMPLETED with actual results]

### LLM-as-Judge Validity in Clinical Contexts
- H6 result: Opus–human correlation
- Compare with Croxford et al.¹⁶ who found ICC 0.818 for LLM-as-judge in clinical summaries
- Address evaluation illusion concerns raised by Agrawal et al.¹⁷
- [TO BE COMPLETED with actual results]

### Localization and the Watson Lesson
- WFO's failure due to US-centric training¹˒²˒³˒⁴˒⁵ motivates our approach: building with local guidelines (IMSS GPC, Mexican Breast Cancer Consensus) from the ground up rather than adapting an imported system
- Our bilingual knowledge base (134K chunks, 663 sources) explicitly includes Mexican and Latin American clinical standards alongside NCCN/ESMO
- The $175 total cost makes this approach accessible to institutions in resource-constrained settings⁶˒⁷

### Cost-Accessibility for Latin America
- Total pipeline cost <$200 USD on consumer hardware (Mac Mini M4, $600)
- Compare with Med-PaLM⁹˒¹⁰ (required Google TPU clusters), WFO (millions in licensing), frontier API costs at institutional scale⁸
- Per-query cost of Light tier: effectively $0 (local inference) vs. $0.05–0.15 for Premium
- Implications for healthcare equity in LMICs⁷

### Limitations
- Synthetic clinical cases, not real patient encounters — limits generalizability
- Small sample (15 cases) — powered for trend detection, not definitive conclusions
- Single specialty (oncology) — may not generalize to other medical domains
- Human evaluation panel size (3–5 oncologists) — limited statistical power for inter-rater analysis
- Fine-tuning on synthetic Q&A pairs generated by a frontier model — potential teacher bias propagation
- Model quantized to 4-bit — possible quality loss vs. full-precision fine-tuning

### Future Directions
- Prospective clinical validation with real patient cases (IRB-approved)
- Expansion to additional oncology subspecialties and cancer types
- Multi-institutional evaluation across different Latin American healthcare settings
- Integration into clinical workflow with physician-in-the-loop assessment
- Extension of the fine-tuning approach to other medical specialties (MedExpert Universal)

---

## Conclusions

[TO BE COMPLETED]

---

## Data Availability

All clinical test cases, evaluation rubrics, anonymized scores, and analysis code will be made available upon publication at [repository URL]. The fine-tuned model weights will be released under [license TBD]. Training data (synthetic Q&A pairs) will be released for reproducibility. The underlying clinical guidelines are publicly available from their respective organizations (NCCN, ESMO, IMSS).

## Code Availability

The complete pipeline code — sampling, data generation, fine-tuning, evaluation, and arena — is available at [repository URL].

## Acknowledgments

[TO BE COMPLETED]

## Author Contributions

[TO BE COMPLETED]

## Competing Interests

The authors declare no competing interests.

## References

### IBM Watson for Oncology — Failures and Localization

1. Somashekhar SP, Sepulveda MJ, Puglielli S, Norden AD, Shortliffe EH, Rohit Kumar C, et al. Watson for Oncology and breast cancer treatment recommendations: agreement with an expert multidisciplinary tumor board. Ann Oncol. 2018;29(2):418-423. DOI: 10.1093/annonc/mdx781

2. Lee WS, Ahn SM, Chung JW, Kim KO, Kwon KA, Kim Y, et al. Assessing concordance with Watson for Oncology, a cognitive computing decision support system for colon cancer treatment in Korea. JCO Clin Cancer Inform. 2018;2:1-8. DOI: 10.1200/CCI.17.00109

3. Yao S, et al. Real world study for the concordance between IBM Watson for Oncology and clinical practice in advanced non-small cell lung cancer patients at a lung cancer center in China. Thorac Cancer. 2020;11(5):1265-1270. DOI: 10.1111/1759-7714.13391

4. Jie Z, Zhiying Z, Li L. A meta-analysis of Watson for Oncology in clinical application. Sci Rep. 2021;11:5792. DOI: 10.1038/s41598-021-84973-5

5. Tupasela A, Di Nucci E. Concordance as evidence in the Watson for Oncology decision-support system. AI Soc. 2020;35(4):811-818. DOI: 10.1007/s00146-020-00945-9

### Clinical AI Localization / Global South

6. Hussain SA, Bresnahan M, Zhuang J. Can artificial intelligence revolutionize healthcare in the Global South? A scoping review. Digit Health. 2025;11:20552076251348024. DOI: 10.1177/20552076251348024

7. Acosta JN, et al. Large language models and global health equity: a roadmap for equitable adoption in LMICs. Lancet Reg Health West Pac. 2025. DOI: 10.1016/S2666-6065(25)00246-9

8. Thirunavukarasu AJ, Ting DSJ, Elangovan K, Gutierrez L, Tan TF, Ting DSW. Large language models in medicine. Nat Med. 2023;29(8):1930-1940. DOI: 10.1038/s41591-023-02448-8

### Fine-Tuning Small LLMs for Medical Domains

9. Singhal K, Azizi S, Tu T, Mahdavi SS, et al. Large language models encode clinical knowledge. Nature. 2023;620(7972):172-180. DOI: 10.1038/s41586-023-06291-2

10. Singhal K, Tu T, et al. Towards expert-level medical question answering with large language models. Nat Med. 2024. DOI: 10.1038/s41591-024-03423-7

11. Wu C, Lin W, Zhang X, Zhang Y, Xie W, Wang Y. PMC-LLaMA: toward building open-source language models for medicine. J Am Med Inform Assoc. 2024;31(9):1833-1843. DOI: 10.1093/jamia/ocae045

12. Chen Z, Hernandez-Cano A, Romanou A, Bonnet A, Matoba K, Salvi F, et al. MEDITRON-70B: Scaling medical pretraining for large language models. arXiv preprint. 2023;arXiv:2311.16079.

13. Labrak Y, Bazoge A, Morin E, Gourraud PA, Rouvier M, Dufour R. BioMistral: A collection of open-source pretrained large language models for medical domains. In: Findings of ACL 2024. 2024:5848-5864. DOI: 10.18653/v1/2024.findings-acl.348

14. Google Research and Google DeepMind. MedGemma Technical Report. arXiv preprint. 2025;arXiv:2507.05201.

### LLM-as-Judge in Clinical Evaluation

15. Zheng L, Chiang WL, Sheng Y, Zhuang S, Wu Z, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In: NeurIPS 2023;36.

16. Croxford E, Gao Y, First E, Pellegrino N, Schnier M, et al. Evaluating clinical AI summaries with large language models as judges. npj Digit Med. 2025;8(1):640. DOI: 10.1038/s41746-025-02005-2

17. Agrawal M, Chen IY, Gulamali F, Joshi S. The evaluation illusion of large language models in medicine. npj Digit Med. 2025;8:600. DOI: 10.1038/s41746-025-01963-x

### RAG in Clinical/Medical Settings

18. Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. In: NeurIPS 2020;33:9459-9474.

19. Zakka C, Shad R, Chaurasia A, et al. Almanac — Retrieval-augmented language models for clinical medicine. NEJM AI. 2024;1(2):AIoa2300068. DOI: 10.1056/AIoa2300068

20. Zarfati M, Soffer S, Nadkarni GN, Klang E. Retrieval-Augmented Generation: Advancing personalized care and research in oncology. Eur J Cancer. 2025;220:115341. DOI: 10.1016/j.ejca.2025.115341

21. Gao Y, Xiong Y, Gao X, Jia K, Pan J, Bi Y, et al. Retrieval-augmented generation for large language models: A survey. arXiv preprint. 2024;arXiv:2312.10997.

### QLoRA and Efficient Fine-Tuning

22. Hu EJ, Shen Y, Wallis P, Allen-Zhu Z, Li Y, Wang S, et al. LoRA: Low-rank adaptation of large language models. In: ICLR 2022.

23. Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. QLoRA: Efficient finetuning of quantized LLMs. In: NeurIPS 2023;36. DOI: 10.5555/3666122.3666563

24. Hou Y, Bert C, Gomaa A, Lahmer G, Hofler D, Weissmann T, et al. Fine-tuning a local LLaMA-3 large language model for automated privacy-preserving physician letter generation in radiation oncology. Front Artif Intell. 2024;7:1493716. DOI: 10.3389/frai.2024.1493716

### Clinical AI in Oncology — High-Impact Journals

25. Zhu M, Lin H, Jiang J, Jinia AJ, Jee J, Pichotta K, et al. Large language model trained on clinical oncology data predicts cancer progression. npj Digit Med. 2025;8:397. DOI: 10.1038/s41746-025-01780-2

26. Hao Y, Qiu Z, Holmes J, et al. Large language model integrations in cancer decision-making: a systematic review and meta-analysis. npj Digit Med. 2025. DOI: 10.1038/s41746-025-01824-7

27. Wu C, Qiu P, Liu J, et al. Towards evaluating and building versatile large language models for medicine. npj Digit Med. 2025;8:58. DOI: 10.1038/s41746-024-01390-4
