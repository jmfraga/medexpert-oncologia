# Low-Cost Clinical Fine-Tuning of Large Language Models for Oncology: An Iterative Multi-Tier Evaluation with Automated Assessment

**Target journal:** npj Digital Medicine (Nature)

**Authors:** Juan Manuel Fraga Sastrías¹, [co-authors TBD]

**Affiliations:**
¹ [Institution TBD]

---

## Abstract

**Background:** Large language models (LLMs) show promise in clinical oncology, but frontier models remain costly and dependent on external APIs, limiting adoption in resource-constrained healthcare settings. Whether small, locally-deployed models fine-tuned on domain-specific knowledge can approach the clinical utility of larger models remains an open question.

**Methods:** We constructed a bilingual oncology knowledge base comprising 134,478 text chunks from 663 sources including NCCN, ESMO, Mexican IMSS clinical practice guidelines, and 397 pharmaceutical drug monographs. From this corpus, 11,592 representative chunks (~8.6%) were selected via stratified sampling, and a teacher model (Claude Sonnet 4.6) generated 34,728 question-answer training pairs in Spanish. We fine-tuned two architecturally distinct models — Meta-Llama-3.1-8B-Instruct (dense, 8B parameters) and gpt-oss-20b (Mixture-of-Experts, 20B parameters) — using QLoRA (rank 64, all layers) on consumer hardware (Mac Mini M4, 64 GB RAM) via the MLX framework. We evaluated six response strategies in an iterative Arena design: (1) llama8b base + RAG, (2) llama8b-onco fine-tuned, (3) gpt-oss-20b base + RAG, (4) gpt-oss-20b-onco fine-tuned, (5) MiniMax-M1-80K + RAG, and (6) Claude Sonnet 4 + RAG — across 15 clinical oncology cases. Three evaluation rounds were conducted with progressive prompt engineering refinements. Automated scoring used Claude Opus 4.6 as judge. Blinded human evaluation by clinical oncologists is planned for models reaching a composite score threshold of 2.5/5.

**Results:** Across three iterative rounds (270 total evaluations), Sonnet 4 + RAG consistently led (3.10–3.17/5), followed closely by MiniMax + RAG (2.75–2.79/5). Fine-tuned local models scored significantly lower: gpt-oss-20b-onco achieved 1.28–1.32/5 and llama8b-onco 0.65–0.66/5. Prompt engineering successfully controlled response length (184→960 tokens, 5× increase for llama8b-onco) but did not improve quality scores, demonstrating that the bottleneck is knowledge content, not response format. The fine-tuned models showed reasonable diagnostic accuracy (gpt-oss-onco: 2.27–2.53/5) but poor guideline adherence and completeness, suggesting the training dataset (~8.6% corpus coverage) is insufficient for comprehensive knowledge internalization. Notably, the MoE architecture (20B) consistently outperformed the dense architecture (8B) on the same training data.

**Conclusions:** Fine-tuning small models on consumer hardware is feasible at <$200 USD total cost, but current corpus coverage (~8.6%) is insufficient for clinically useful knowledge internalization. Prompt engineering controls response format but not knowledge quality — these are separable problems. MiniMax + RAG emerges as the most cost-effective production strategy at 2.79/5, only 0.31 points below Sonnet. Future work will explore MedGemma 27B (medical pre-trained), expanded dataset coverage (25–30%), and prospective human evaluation once the 2.5/5 automated threshold is met.

**Keywords:** large language models, oncology, fine-tuning, QLoRA, clinical decision support, retrieval-augmented generation, iterative evaluation, low-resource, Latin America

---

## Introduction

The application of large language models (LLMs) to clinical medicine has accelerated rapidly, with models demonstrating competence in medical licensing examinations, clinical reasoning⁹, and guideline-concordant treatment recommendations²⁶. In oncology specifically, LLMs have shown potential as clinical decision support tools for treatment planning, drug interaction assessment, and patient education²⁵˒²⁷, with a recent meta-analysis reporting an average accuracy of 76.2% across 56 studies and 15 cancer types²⁶.

However, the deployment of LLMs in clinical oncology faces three critical barriers. First, frontier models capable of nuanced clinical reasoning require costly API access, with per-query costs of $0.05–0.15 USD that become prohibitive at institutional scale⁸. Second, these models depend on external cloud infrastructure, raising concerns about patient data privacy, latency, and availability in regions with limited connectivity⁶. Third, the majority of clinical AI research and model training has been conducted in English, with limited attention to the clinical terminology, guidelines, and healthcare contexts of Latin American countries⁷.

The cautionary tale of IBM Watson for Oncology (WFO) illustrates the localization problem acutely. Despite significant investment, WFO showed concordance rates ranging from 48.9% for colon cancer in South Korea² to 93% for breast cancer in India¹, with a meta-analysis across 2,463 patients reporting 81.5% overall concordance but significantly lower rates for advanced-stage disease and cancer types underrepresented in its US-centric training data⁴. Critically, WFO failed to account for locally approved drugs, regional insurance policies, and country-specific treatment patterns³˒⁵, underscoring that clinical AI systems must be adapted to local healthcare contexts to be clinically useful.

An alternative paradigm is the fine-tuning of smaller, open-weight models on domain-specific clinical knowledge, enabling local deployment on consumer-grade hardware without API dependencies. Recent work has demonstrated that models in the 7–13B parameter range can achieve competitive performance on medical benchmarks when appropriately fine-tuned¹¹˒¹²˒¹³. However, systematic evaluations comparing fine-tuned small models against frontier models in realistic clinical oncology scenarios — particularly with iterative refinement — remain scarce.

The retrieval-augmented generation (RAG) approach offers a complementary strategy, grounding model responses in authoritative clinical guidelines at inference time¹⁸˒¹⁹. In oncology, RAG has shown promise for personalized treatment recommendations and clinical trial matching²⁰. Whether RAG or fine-tuning provides superior results — and whether the answer depends on model scale and architecture — is an important practical question for system design.

In this study, we present a complete, reproducible pipeline for creating domain-specialized oncology LLMs at a total cost under $200 USD, running entirely on consumer hardware. We compare two fine-tuned architectures (dense 8B vs. MoE 20B) against two API-based models with RAG, using an iterative evaluation framework with three rounds of prompt engineering refinement. Our evaluation employs 15 clinical cases covering 14 cancer types at three complexity levels, assessed by an automated LLM judge (Opus 4.6) on four clinically relevant criteria.

Our primary research questions are:
1. **RAG vs. fine-tuning**: Which strategy yields better clinical responses — retrieval augmentation on base models, or domain fine-tuning without retrieval?
2. **Dense vs. MoE**: Does a 20B Mixture-of-Experts model outperform an 8B dense model when both are fine-tuned on the same dataset?
3. **Format vs. knowledge**: Can prompt engineering improve fine-tuned model quality, or is the training dataset the binding constraint?
4. **Local vs. API**: How large is the quality gap between local and API-based models, and what is the cost-effectiveness trade-off?

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

From the full corpus of 134,478 chunks, we selected 11,592 representative chunks (~8.6%) via stratified sampling across five clinical strata designed to ensure balanced coverage of oncological knowledge domains (Table 2).

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

All responses were generated in Spanish regardless of source language, strictly grounded in the source chunk content, and included source attribution. Quality filters excluded responses shorter than 30 tokens. This initial generation produced 34,728 training pairs used for Arena v1 and v2 evaluations.

The total cost of initial synthetic data generation was approximately $175 USD in API fees.

#### Expanded Dataset Generation

Based on Arena v2b findings that the initial dataset (~8.6% corpus coverage) was the binding constraint on fine-tuned model quality, we performed a second, comprehensive generation pass. From the full corpus of 134,478 chunks, 98,133 high-quality chunks were selected after excluding low-information content (<80 characters, bibliography, copyright notices). A cost-effective teacher model (MiniMax M2.7) was used for this expanded generation, having demonstrated response quality comparable to Claude Sonnet in prior Arena evaluation at approximately 15× lower cost per token. Generation was parallelized across two concurrent batch processes (15 workers each), producing approximately 294,000 additional question-answer pairs at an estimated cost of ~$65 USD.

The combined raw dataset (~329,000 examples from both generation passes) was then subjected to a multi-stage quality funnel designed to maximize training signal while minimizing noise, following the principle that smaller, high-quality datasets consistently outperform larger, unfiltered ones²⁹˒³⁰˒³².

#### Dataset Quality Funnel

We applied a sequential filtering pipeline ordered by computational cost, such that inexpensive deterministic filters reduce the corpus volume before applying costlier model-based assessment³¹ (Table 2b).

**Table 2b.** Dataset quality funnel — sequential filtering stages.

| Stage | Filter | Type | Rationale |
|-------|--------|------|-----------|
| 1 | Encoding and format validation | Deterministic | Remove corrupted entries, malformed UTF-8, structural artifacts |
| 2 | Minimum response length (≥300 characters) | Deterministic | Short responses lack clinical depth; Arena v2b confirmed that brevity correlates with low utility scores |
| 3 | Evasive response detection | Regex | Remove refusals ("como modelo de lenguaje", "consulte a su médico") that introduce refusal contamination²⁸ |
| 4 | Question quality (≥30 characters) | Deterministic | Exclude trivially short or degenerate questions |
| 5 | Semantic deduplication (cosine similarity >0.95) | Embedding | Redundant examples degrade fine-tuning performance; applied using sentence-transformer embeddings |
| 6 | Cross-dataset deduplication | Embedding | Merge initial (34K) and expanded (294K) datasets, retaining the longer response when thematic overlap is detected |
| 7 | LLM-as-filter quality scoring | Model-based | A frontier model (Claude Opus or Sonnet) scores a stratified random sample (~5%) on clinical accuracy, completeness, and guideline adherence (0–5 scale); examples scoring <3/5 are discarded³⁰˒³³ |

Stages 1–6 are computationally free (deterministic or local embedding-based), allowing aggressive volume reduction before the cost-intensive Stage 7. This funnel design follows the dataset curation principles established by LIMA²⁹, AlpaGasus³⁰, and Deita³¹, adapted to the clinical domain with medical-specific filters (Stages 2–3).

The quality funnel also provides a documented audit trail for each filtering decision, enabling reproducibility and transparency in the dataset construction process.

#### Thematic Balance Verification

After filtering, the dataset distribution was verified across clinical strata (treatment, pharmacology, diagnosis, supportive care, follow-up) and source provenance (NCCN, ESMO, IMSS, pharmaceutical monographs) to ensure no single category dominated >40% of the final training set. Over-represented categories were subsampled to maintain balanced clinical coverage.

The final curated dataset was split 90/5/5 into training, validation, and test sets, with stratification by source and clinical stratum to ensure proportional representation in each partition.

### Model Fine-Tuning

#### Base Models

Two architecturally distinct models were fine-tuned on the same dataset to enable direct comparison (Table 3a).

**Table 3a.** Base models for fine-tuning.

| Model | Architecture | Parameters | Source | Peak Memory |
|-------|-------------|-----------|--------|-------------|
| Meta-Llama-3.1-8B-Instruct (4-bit) | Dense | 8B | mlx-community | 11.2 GB |
| gpt-oss-20b (4-bit) | Mixture-of-Experts | 20B | InferenceIllusionist | 30.8 GB |

#### Configuration

Both models were fine-tuned using QLoRA²³ with the configuration detailed in Table 3b.

**Table 3b.** QLoRA fine-tuning hyperparameters.

| Parameter | llama8b | gpt-oss-20b | Rationale |
|-----------|---------|-------------|-----------|
| LoRA rank | 64 | 64 | Dense medical domain requires higher capacity than standard rank 8–16 |
| Target layers | All (-1) | All (-1) | Medical knowledge is distributed across all transformer layers |
| Batch size | 4 | 4 | Balance between throughput and training stability |
| Learning rate | 2 × 10⁻⁵ | 1 × 10⁻⁵ | Reduced for larger model to prevent instability |
| Training iterations | 3,000 | 3,000 | Calibrated for ~35K training examples |
| Max sequence length | 2,048 | 2,048 | Sufficient for clinical Q&A |
| Dropout | 0.05 | 0.05 | Light regularization for specialized dataset |
| LoRA alpha (scale) | 32.0 | 32.0 | 0.5× rank, recommended for narrow-domain fine-tuning |
| Gradient checkpointing | Enabled | Enabled | Reduces peak memory |
| Prompt masking | Enabled | Enabled | Loss computed only on response tokens |

#### Hardware and Framework

Training was performed on a Mac Mini M4 (Apple Silicon, 64 GB unified memory) using the MLX framework (mlx-lm 0.31.1), which provides native Apple Silicon optimization. llama8b training completed in ~11 hours; gpt-oss-20b in ~8.5 hours. Both models showed optimal validation loss at iteration 2,500–2,600, with the best checkpoint used for subsequent evaluation.

**Total fine-tuning compute cost: $0 USD** (local consumer hardware).

#### Training Results

**Table 3c.** Training outcomes.

| Metric | llama8b-onco | gpt-oss-20b-onco |
|--------|-------------|------------------|
| Final train loss | 0.860 | 0.844 |
| Final val loss (iter 3000) | 0.881 | 0.885 |
| Best val loss | 0.874 (iter 2600) | 0.863 (iter 2600) |
| Tokens trained | 2,112,717 | 1,880,841 |
| Training speed | 53 tok/s | 65 tok/s |
| Training duration | ~11 hours | ~8.5 hours |

Neither model showed signs of overfitting. The MoE model achieved a lower optimal validation loss (0.863 vs. 0.874) despite having more parameters, suggesting that the MoE architecture is more parameter-efficient for domain-specific knowledge absorption.

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

Six response strategies were evaluated, representing a factorial design crossing model scale, training strategy (base vs. fine-tuned), and knowledge source (RAG vs. internalized) (Table 5).

**Table 5.** Response tiers evaluated in Arena v2.

| # | Tier | Model | Fine-tuned | RAG | Type |
|---|------|-------|-----------|-----|------|
| 1 | llama8b base + RAG | Meta-Llama-3.1-8B-Instruct-4bit | No | Yes | Local (dense) |
| 2 | llama8b-onco | llama8b-onco LoRA | Yes | No | Local (dense) |
| 3 | gpt-oss-20b base + RAG | gpt-oss-20b-MLX-4bit | No | Yes | Local (MoE) |
| 4 | gpt-oss-20b-onco | gpt-oss-20b-onco LoRA | Yes | No | Local (MoE) |
| 5 | MiniMax-M1-80K + RAG | MiniMax-Text-01 | No | Yes | API |
| 6 | Sonnet 4 + RAG | Claude Sonnet 4 | No | Yes | API |

This design enables three key comparisons: RAG vs. fine-tuning (tiers 1 vs. 2, tiers 3 vs. 4), dense vs. MoE (tiers 2 vs. 4), and local vs. API (tiers 4 vs. 5 vs. 6).

#### Iterative Evaluation Rounds

A key methodological contribution of this study is the iterative Arena design, where evaluation rounds inform progressive refinements (Table 5b).

**Table 5b.** Iterative evaluation rounds.

| Round | Tiers | Changes | Evaluations | Cost |
|-------|-------|---------|-------------|------|
| Arena v1 | 5 (original design) | Baseline with MedGemma 27B (discarded: timeout >300s) | 75 | ~$5 |
| Arena v2 | 6 (redesigned) | Replaced MedGemma with gpt-oss-20b; added RAG vs ft comparison | 90 | $10.25 |
| Arena v2b | 6 | Improved prompt (SAER format, 500+ words); best checkpoint (iter 2500); max_tokens 2000→4000 | 90 | $10.86 |
| **Total** | | | **255** | **~$26** |

The iterative design allowed us to distinguish format problems (solvable with prompt engineering) from knowledge problems (requiring dataset expansion), a distinction that would not have been apparent from a single evaluation round.

#### Automated Evaluation (LLM Judge)

Each of the 90 case-tier combinations per round (15 cases × 6 tiers) was evaluated by Claude Opus 4.6 (Anthropic) with access to RAG context and the gold standard. The judge model did not participate as a response candidate, maintaining impartiality.

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

##### Quality Gate

Based on iterative automated evaluation, we established a quality threshold of **composite score ≥ 2.5/5** from the Opus judge before submitting responses for human expert evaluation. This prevents wasting expert clinician time on models that are clearly inadequate and focuses human assessment on the most promising candidates.

##### Panel Design

Clinical oncologists will perform blinded evaluation of tiers meeting the quality threshold. Responses will be presented anonymized (Candidate A–F) with randomized assignment per case using a fixed seed for reproducibility. Evaluators will be unaware of which model generated each response.

The evaluation instrument will be administered via Google Forms using the four-criteria rubric (Table 6) plus free-text comments per response. Inter-rater agreement will be assessed via Krippendorff's alpha.

---

## Results

### Training Convergence

Both models converged smoothly without overfitting (Figure 1 — to be generated). Validation loss reached its minimum at iteration ~2600 for both architectures, after which slight increases suggested early overfitting onset. The best checkpoints (iteration 2500, nearest to optimal) were selected for Arena v2b evaluation.

### Arena v2: Baseline Six-Tier Evaluation

**Table 9.** Mean composite score by tier (Arena v2, Opus judge, 0–5 scale).

| Tier | Overall | Simple (n=5) | Moderate (n=4) | Complex (n=6) |
|------|---------|--------------|----------------|---------------|
| Sonnet 4 + RAG | **3.17** | 3.06 | 3.09 | 3.32 |
| MiniMax + RAG | **2.75** | 2.64 | 2.92 | 2.73 |
| llama8b base + RAG | 1.37 | 1.49 | 1.57 | 1.14 |
| gpt-oss-20b-onco (ft) | 1.28 | 1.54 | 1.64 | 0.82 |
| llama8b-onco (ft) | 0.65 | 1.05 | 0.34 | 0.53 |
| gpt-oss-20b base + RAG | 0.39 | 0.96 | 0.06 | 0.14 |

### Arena v2b: Prompt Engineering Effect

After observing that fine-tuned models generated very short responses (~180 tokens vs. ~800 for RAG tiers), we improved the prompt to explicitly request detailed SAER-format responses of 500+ words and increased max_tokens from 2,000 to 4,000.

**Table 9b.** Response length comparison (mean tokens output).

| Tier | v2 | v2b | Change |
|------|-----|------|--------|
| llama8b-onco (ft) | 184 | **960** | **5.2×** |
| gpt-oss-20b-onco (ft) | 179 | 193 | 1.1× |
| llama8b base + RAG | 703 | 717 | — |
| gpt-oss base + RAG | 2,000 | 3,387 | 1.7× |
| MiniMax + RAG | 882 | 805 | — |
| Sonnet + RAG | 912 | 964 | — |

**Table 9c.** Quality scores: v2 vs. v2b comparison.

| Tier | v2 Score | v2b Score | Δ |
|------|----------|-----------|---|
| Sonnet 4 + RAG | 3.17 | 3.10 | −0.07 |
| MiniMax + RAG | 2.75 | 2.79 | +0.04 |
| llama8b base + RAG | 1.37 | 1.29 | −0.08 |
| gpt-oss-20b-onco (ft) | 1.28 | 1.32 | +0.04 |
| **llama8b-onco (ft)** | **0.65** | **0.66** | **+0.01** |
| gpt-oss-20b base + RAG | 0.39 | 0.50 | +0.11 |

The critical finding: despite a 5.2× increase in response length for llama8b-onco, quality scores remained unchanged (0.65→0.66). This demonstrates that **prompt engineering controls response format but not knowledge quality** — the bottleneck is the training data, not the prompt.

### Performance by Evaluation Criterion

**Table 10.** Mean score per tier × criterion (Arena v2b, 0–5 scale).

| Criterion | llama8b + RAG | llama8b-onco | gptoss + RAG | gptoss-onco | MiniMax | Sonnet |
|-----------|--------------|-------------|-------------|-------------|---------|--------|
| Diagnostic accuracy | 2.60 | 1.53 | 0.87 | 2.27 | 4.20 | **4.33** |
| Guideline adherence | 0.80 | 0.47 | 0.33 | 1.27 | **2.60** | 2.40 |
| Completeness | 1.07 | 0.47 | 0.53 | 0.93 | 2.13 | **2.80** |
| Clinical utility | 1.00 | 0.27 | 0.33 | 1.00 | 2.67 | **3.33** |

Notable findings:
- Fine-tuned models show reasonable diagnostic accuracy (gpt-oss-onco: 2.27) but fail on guideline adherence and completeness
- MiniMax matches or exceeds Sonnet on guideline adherence (2.60 vs. 2.40), likely due to its large context window effectively leveraging RAG content
- The largest quality gaps between local and API models are in completeness and clinical utility

### RAG vs. Fine-Tuning

**Table 10b.** Direct comparison of strategies by architecture.

| Architecture | Base + RAG | Fine-tuned (no RAG) | Better strategy |
|-------------|-----------|-------------------|----------------|
| llama8b (8B, dense) | 1.29 | 0.66 | **RAG** (2.0×) |
| gpt-oss-20b (20B, MoE) | 0.50 | 1.32 | **Fine-tune** (2.6×) |

The optimal strategy depends on model scale: smaller models benefit more from RAG (external knowledge retrieval), while larger models better internalize fine-tuned knowledge. However, neither strategy approaches API-tier quality.

### Dense vs. MoE Architecture

When fine-tuned on the same dataset:
- gpt-oss-20b-onco (MoE, 20B): 1.32/5
- llama8b-onco (dense, 8B): 0.66/5

The MoE model achieves **2× the score**, consistent with its lower validation loss (0.863 vs. 0.874) and suggesting that larger, more capable architectures extract more value from the same training data.

### Operational Performance

**Table 15.** Operational metrics by tier (Arena v2b, mean across 15 cases).

| Tier | Response Time (s) | Throughput (tok/s) | Tokens Output | Cost/Query (USD) |
|------|-------------------|-------------------|---------------|-----------------|
| llama8b base + RAG | 18.6 | ~39 | 717 | $0 |
| llama8b-onco (ft) | 35.3 | ~27 | 960 | $0 |
| gpt-oss-20b base + RAG | 44.0 | ~77 | 3,387 | $0 |
| gpt-oss-20b-onco (ft) | 4.9 | ~39 | 193 | $0 |
| MiniMax + RAG | 22.9 | ~35 | 805 | ~$0.01 |
| Sonnet 4 + RAG | 19.0 | ~51 | 964 | ~$0.05 |

### Quality Gate Status

**Table 16.** Tier qualification for human evaluation (threshold: ≥2.5/5 Opus composite).

| Tier | Best Score | Threshold Met | Status |
|------|-----------|--------------|--------|
| Sonnet 4 + RAG | 3.17 | Yes | Ready for human evaluation |
| MiniMax + RAG | 2.79 | Yes | Ready for human evaluation |
| All local models | ≤1.37 | No | Requires improvement |

---

## Discussion

### Key Findings

#### 1. Format vs. Knowledge: A Separable Problem

Our iterative evaluation design revealed a critical distinction between response format and response quality. When prompt engineering increased llama8b-onco's output length by 5.2× (184→960 tokens), the composite quality score remained unchanged (0.65→0.66). The model generated more text but more "filler" — additional words without additional clinical substance. This demonstrates that prompt engineering and knowledge content are orthogonal dimensions, and that the bottleneck for fine-tuned model quality is the training dataset, not the instruction format.

This finding has practical implications for the broader fine-tuning community: researchers should not conflate response length improvements with quality improvements when evaluating domain-specialized models.

#### 2. Dataset Coverage as the Binding Constraint

Our training dataset sampled only ~8.6% (11,592 of 134,478) of the available clinical chunks. The fine-tuned models showed reasonable diagnostic accuracy (gpt-oss-onco: 2.27/5) — suggesting they learned some core oncological concepts — but failed on guideline adherence (1.27/5) and completeness (0.93/5), indicating that the breadth of clinical protocols, dosing regimens, and management algorithms was insufficiently represented.

For context, oncology clinical guidelines span dozens of cancer types, each with multiple staging-dependent treatment protocols, biomarker-driven therapeutic decisions, and continuously evolving evidence. An 8.6% sample necessarily leaves large gaps in coverage. We estimate that expanding coverage to 25–30% of the corpus would require generating ~65,000–85,000 additional Q&A pairs at a cost of approximately $250–325 USD (see Cost Projections below).

#### 3. MiniMax as a Cost-Effective Alternative

MiniMax-M1-80K + RAG achieved 2.79/5 — only 0.31 points below Sonnet 4 + RAG (3.10/5) — while being significantly less expensive per query. MiniMax showed particular strength in guideline adherence (2.60 vs. Sonnet's 2.40), possibly because its large context window (80K tokens) enables more effective use of RAG-retrieved guideline excerpts. For production deployment, MiniMax + RAG represents the optimal cost-quality trade-off for most clinical queries, with Sonnet reserved for complex cases.

#### 4. Architecture Matters for Fine-Tuning

The MoE gpt-oss-20b model consistently outperformed the dense llama8b on the same training data (1.32 vs. 0.66 composite, 2× improvement). This aligns with emerging evidence that MoE architectures are more parameter-efficient for domain-specific knowledge absorption, as different expert subnetworks can specialize in different clinical domains without interference.

### Cost Analysis and Projections

**Table 17.** Cumulative project costs to date.

| Phase | Cost (USD) |
|-------|------------|
| Knowledge base construction | $0 |
| Training data generation (34,728 Q&A pairs) | $175 |
| Fine-tuning: llama8b-onco (11 hours) | $0 (local) |
| Fine-tuning: gpt-oss-20b-onco (8.5 hours) | $0 (local) |
| Arena v1 evaluation (75 judgments) | ~$5 |
| Arena v2 evaluation (90 judgments) | $10.25 |
| Arena v2b evaluation (90 judgments) | $10.86 |
| **Total to date** | **~$201** |

**Table 17b.** Projected costs for dataset expansion.

| Coverage | Chunks | Q&A Pairs | Additional Pairs | Generation Cost | Total New Cost |
|----------|--------|-----------|-----------------|----------------|----------------|
| Current (8.6%) | 11,592 | 34,728 | — | $175 (done) | — |
| 25% | 33,620 | 100,860 | ~66,000 | ~$250 | ~$250 |
| 30% | 40,344 | 121,032 | ~86,000 | ~$325 | ~$325 |
| 50% | 67,239 | 201,717 | ~167,000 | ~$630 | ~$630 |

Each additional Arena evaluation round costs approximately $10 USD (Opus judge).

### Limitations

- Synthetic clinical cases, not real patient encounters — limits generalizability
- Small sample (15 cases) — powered for trend detection, not definitive conclusions
- Single specialty (oncology) — may not generalize to other medical domains
- Human evaluation not yet conducted — pending quality threshold achievement
- Fine-tuning on synthetic Q&A pairs generated by a frontier model — potential teacher bias propagation
- Models quantized to 4-bit — possible quality loss vs. full-precision fine-tuning
- Automated evaluation only (Opus judge) — human expert validation pending for all tiers

### Future Directions

1. **MedGemma 27B fine-tuning**: Google's medical pre-trained model may provide a stronger base for oncology specialization, potentially reaching the 2.5 threshold with the current dataset
2. **Dataset expansion**: Increase corpus coverage from 8.6% to 25–30%, targeting gaps identified in guideline adherence and treatment protocol completeness
3. **Human expert evaluation**: Once a local model achieves ≥2.5/5 automated composite score, conduct blinded evaluation with 4–7 clinical oncologists using the BIBD design
4. **Prospective clinical validation**: Real patient cases with IRB approval
5. **Multi-institutional evaluation**: Diverse Latin American healthcare settings
6. **Clinical workflow integration**: Physician-in-the-loop assessment with tiered routing (local for simple queries, API for complex cases)

---

## Conclusions

We demonstrate a complete, reproducible pipeline for fine-tuning oncology LLMs on consumer hardware at <$200 USD total cost. Through iterative evaluation (3 rounds, 255 total assessments), we establish three key findings: (1) prompt engineering controls response format but not knowledge quality — these are separable problems; (2) training dataset coverage (~8.6% of available corpus) is the binding constraint for fine-tuned model quality, not model architecture or prompt design; and (3) MiniMax + RAG achieves 89% of Sonnet's quality at a fraction of the cost, making it the optimal production strategy for clinical oncology decision support.

No fine-tuned local model currently meets our quality threshold of 2.5/5 for human expert evaluation. Future work will explore medical-domain pre-trained models (MedGemma 27B) and expanded dataset coverage (25–30%) to close the gap between local and API-based models, with the goal of enabling high-quality, privacy-preserving clinical AI in resource-constrained healthcare settings.

---

## Data Availability

All clinical test cases, evaluation rubrics, anonymized scores, and analysis code are available at https://github.com/jmfraga/medexpert-oncologia. Training data (synthetic Q&A pairs) will be released for reproducibility. The underlying clinical guidelines are publicly available from their respective organizations (NCCN, ESMO, IMSS).

## Code Availability

The complete pipeline code — sampling, data generation, fine-tuning, evaluation, and arena — is available at https://github.com/jmfraga/medexpert-oncologia.

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

### Synthetic Data Generation and Physician Alignment

25. Toma TP, Lawler PR, Ba J, Krishnan RG, Rubin BB, Wang B. Clinical Camel: An open-source expert-level medical language model with dialogue-based knowledge encoding. arXiv preprint. 2023;arXiv:2305.12031.

26. Fleming SL, Lozano A, Habber WJ, Jindal A, Reis EP, Paranjape A, et al. MedAlign: A clinician-generated dataset and benchmark for instruction following with electronic medical records. In: AAAI 2024;38(16):17960-17969. DOI: 10.1609/aaai.v38i16.29754

27. Wang Y, Kordi Y, Mishra S, Liu A, Smith NA, Khashabi D, et al. Self-Instruct: Aligning language models with self-generated instructions. In: ACL 2023. DOI: 10.18653/v1/2023.acl-long.754

28. Xu C, Sun Q, Zheng K, Geng X, Zhao P, Feng J, et al. WizardLM: Empowering large language models to follow complex instructions. In: ICLR 2024.

### Dataset Curation and Quality Filtering

29. Zhou C, Liu P, Xu P, Iyer S, Sun J, Mao Y, et al. LIMA: Less is more for alignment. In: NeurIPS 2023;36.

30. Chen L, Li S, Yan J, Wang H, Gunaratna K, Ez-Zizi V, et al. AlpaGasus: Training a better Alpaca with fewer data. In: ICLR 2024.

31. Liu M, Zeng H, Dong L, Hao Y, Wang W, Shao J, et al. What makes good data for alignment? A comprehensive study of automatic data selection in instruction tuning. In: ICLR 2024.

32. Gunasekar S, Zhang Y, Anber J, Bubeck S, et al. Textbooks are all you need. arXiv preprint. 2023;arXiv:2306.11644.

33. Zhang T, Lin Z, Pan A, Jiang S, et al. UltraMedical: Building specialized generalists in biomedicine. arXiv preprint. 2024;arXiv:2406.03949.

### Clinical AI in Oncology — High-Impact Journals

34. Zhu M, Lin H, Jiang J, Jinia AJ, Jee J, Pichotta K, et al. Large language model trained on clinical oncology data predicts cancer progression. npj Digit Med. 2025;8:397. DOI: 10.1038/s41746-025-01780-2

35. Hao Y, Qiu Z, Holmes J, et al. Large language model integrations in cancer decision-making: a systematic review and meta-analysis. npj Digit Med. 2025. DOI: 10.1038/s41746-025-01824-7

36. Wu C, Qiu P, Liu J, et al. Towards evaluating and building versatile large language models for medicine. npj Digit Med. 2025;8:58. DOI: 10.1038/s41746-024-01390-4
