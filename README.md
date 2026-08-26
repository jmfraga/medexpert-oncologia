# RAG and Fine-Tuning Do Not Improve Base LLMs for Clinical Oncology

Reproducible pipeline and analysis code for the study:
**"Retrieval-Augmentation and Fine-Tuning Do Not Improve — and Can Degrade — Base Language Models for Clinical Oncology: A Multi-Tier Automated Arena with Blinded Oncologist Confirmation."**

> **Key finding.** Across a five-round automated "Arena" (Claude Opus 4 as blinded judge, 480 evaluations) and a blinded panel of oncologists, **an un-augmented base model was the best strategy for every model tested** — both retrieval-augmented generation (RAG) and QLoRA fine-tuning *reduced* clinical answer quality. The three top tiers advanced to human evaluation were therefore all base models, and a free local model (Gemma-4 31B) was statistically **non-inferior** to the paid Claude Sonnet 4 API.

## Why this repository exists

Everything here runs on a single consumer workstation (Mac Mini M4, 64 GB) for under US$300. The repo lets others reproduce the pipeline and the analysis. Note the two deliberate exclusions below (data and weights) — both for good reasons.

## Repository structure

```
01_sample_chromadb.py            Stratified sampling from the RAG knowledge base
02_generate_dataset.py           Synthetic Q&A generation (teacher model)
02b_clean_dataset.py             Seven-stage dataset quality funnel
03..08_finetune_*.sh             QLoRA fine-tuning (Llama-3.1-8B, gpt-oss-20B, Gemma-4 26B/31B)
arena_serve.sh / run_arena_final.sh   Serve tiers and run the automated Arena
05_evaluate_model.py             Opus-judge scoring
human_eval_*.py                  Human-evaluation analysis (mixed model, concordance, robustness)
clinical_cases_gold_standard.md  15 synthetic clinical cases + gold standards
rubrica_evaluadores.md           4-criterion evaluation rubric
ARENA_ALL_RESULTS.md             Consolidated Arena scores (all 5 rounds)
HUMAN_EVAL_RESULTS.md            Human-evaluation results and statistics
onco_human_eval_data.csv         De-identified human ratings (104 ratings, 6 raters)
supplementary_data/              Supplementary Data 1–4 (cases, arena, human eval, key)
figures/                         Paper figures
```

## Reproducing the analysis

The human-evaluation statistics reproduce from the included data with no external dependencies beyond `pandas`, `statsmodels` and `matplotlib`:

```bash
pip install pandas statsmodels matplotlib openpyxl
python human_eval_mixed_model.py          # mixed-effects model + non-inferiority
python human_eval_coverage_concordance.py # coverage + Krippendorff's alpha
python human_eval_robustness_ols.py       # robustness check
```

The `arena/` directory contains the Arena runner and Opus-judge code (with the judge rubric/prompt). The raw 480 per-judgment scores are in `supplementary_data/Supplementary_Data_2_arena_raw_480.csv`; reproduce the within-round inference with `arena_within_round_inference.py`. The fine-tuning and Arena scripts require the base models (Hugging Face) and a locally built RAG index (see below).

## ⚠️ Data availability (copyright)

The RAG knowledge base is built from **third-party copyrighted clinical guidelines** (NCCN, ESMO, Mexican IMSS, national consensus, drug monographs). **That corpus, its chunked text, and its embeddings are NOT redistributable and are not included in this repository.** The construction pipeline (sources, chunking, indexing, retrieval) is fully described in the paper and reproducible with the public guidelines from their respective organizations. The teacher-generated synthetic training dataset is derived from that copyrighted text and is likewise withheld; the generation code is provided.

Synthetic clinical cases, the rubric, consolidated scores, and the de-identified human-evaluation data **are** included (they contain no copyrighted or patient data).

## Fine-tuned model adapters

The four QLoRA adapters are hosted on Hugging Face and released **for reproducibility and as a documented negative result** (fine-tuning degraded quality) — not as production models:

- `jmfraga/gemma4-31b-onco-lora`
- `jmfraga/gemma4-26b-moe-onco-lora`
- `jmfraga/gpt-oss-20b-onco-lora`
- `jmfraga/llama-3.1-8b-onco-lora`

## ⚠️ Not for clinical use

This is research code. Nothing here is a validated medical device and it must not be used for clinical, diagnostic, or treatment decisions.

## Citation

Fraga Sastrías, J. M. et al. *Retrieval-Augmentation and Fine-Tuning Do Not Improve — and Can Degrade — Base Language Models for Clinical Oncology* (2026). See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). Clinical guidelines and model base weights retain their own licenses.
