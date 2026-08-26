SUPPLEMENTARY DATA — npj Digital Medicine submission
Paper: Base Language Models Outperform RAG and Fine-Tuning for Clinical Oncology…

These are the four Supplementary Data files that go DIRECTLY to the journal
(small, synthetic, no third-party copyright, evaluators de-identified).

- Supplementary_Data_1_clinical_cases.xlsx
    15 synthetic clinical oncology cases + gold-standard diagnosis, key
    recommendations, guidelines, complexity and cancer type.
- Supplementary_Data_2_arena_raw_480.csv
    RAW per-judgment Arena data: all 480 case×tier evaluations (5 rounds), with
    the four criterion scores, composite, and clinical-safety flags (errores_graves,
    alucinaciones). Enables full inference (see arena_within_round_inference.py).
- Supplementary_Data_3_human_evaluation.xlsx
    104 blinded oncologist ratings (4 criteria + composite). Raters are
    DE-IDENTIFIED as "Rater 1–6". Includes blinded candidate letter and the
    unblinded true model for reproducibility.
- Supplementary_Data_4_anonymization_key.xlsx
    Case × candidate → true model mapping (blinding key, seed 42).

Total size < 40 KB (npj limits: 30 MB/file, 150 MB total).

NOT included here (go to external repositories, see SUPPLEMENTARY_PLAN.md):
  - Fine-tuned LoRA adapters  -> Hugging Face + Zenodo DOI
  - Code / RAG pipeline       -> GitHub + Zenodo DOI
  - RAG ChromaDB index & synthetic dataset -> NOT redistributable
    (derived from copyrighted NCCN/ESMO content); code + description only.
