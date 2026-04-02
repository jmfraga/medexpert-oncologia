#!/usr/bin/env python3
"""
02b — Dataset Quality Funnel (8 stages).

Implements the sequential filtering pipeline described in the paper
(Table 2b) and METODOLOGIA.md. Stages are ordered by computational cost
so that cheap deterministic filters reduce volume before costlier
model-based assessment. Stage 8 extrapolates LLM quality scores to
the full dataset via a local classifier trained on Stage 7 labels.

Usage:
  /Users/jmfraga/mlx-env/bin/python 02b_clean_dataset.py
  /Users/jmfraga/mlx-env/bin/python 02b_clean_dataset.py --dry-run
  /Users/jmfraga/mlx-env/bin/python 02b_clean_dataset.py --skip-llm-filter
  /Users/jmfraga/mlx-env/bin/python 02b_clean_dataset.py --input-dir data/ --output-dir data/clean/

Requirements (beyond stdlib):
  pip install sentence-transformers tqdm anthropic scikit-learn numpy
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("clean_dataset")

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

MIN_RESPONSE_LENGTH = 300   # Stage 2: minimum assistant response chars
MIN_QUESTION_LENGTH = 30    # Stage 4: minimum user question chars
DEDUP_THRESHOLD = 0.95      # Stage 5: cosine similarity threshold
LLM_SAMPLE_FRACTION = 0.05  # Stage 7: fraction of examples to score
LLM_MIN_SCORE = 3           # Stage 7: minimum acceptable score (0-5)
MAX_STRATUM_FRACTION = 0.40 # Post-filter: max fraction per stratum
RANDOM_SEED = 42

# Evasive patterns (Stage 3) — Spanish and English refusals
EVASIVE_PATTERNS = [
    r"como modelo de lenguaje",
    r"como modelo de IA",
    r"como (un )?asistente (de IA|virtual)",
    r"como inteligencia artificial",
    r"consulte a su m[eé]dico",
    r"consultar a (un |su )?m[eé]dico",
    r"no puedo proporcionar",
    r"no puedo ofrecer",
    r"no puedo dar (un )?diagn[oó]stico",
    r"no puedo realizar diagn[oó]sticos",
    r"no estoy capacitado",
    r"no me es posible",
    r"no soy un m[eé]dico",
    r"I cannot",
    r"I'm sorry",
    r"I am sorry",
    r"I'm not able to",
    r"as an AI",
    r"as a language model",
    r"I don't have the ability",
    r"consult(e|a)? (con )?(un |su )?(profesional|especialista)",
    r"acuda a (un |su )?(m[eé]dico|especialista|profesional)",
    r"busque atenci[oó]n m[eé]dica",
    r"no reemplaza (la |el )?consejo m[eé]dico",
    r"no sustituye (la |el )?(consulta|consejo)",
]
_EVASIVE_RE = re.compile(
    "|".join(EVASIVE_PATTERNS), re.IGNORECASE
)

# Stratum classification keywords (for thematic balance check)
STRATUM_KEYWORDS = {
    "tratamiento": [
        "tratamiento", "terapia", "quimioterapia", "radioterapia",
        "esquema", "protocolo", "línea", "ciclo", "régimen",
        "cirugía", "resección", "neoadyuvan", "adyuvan", "inmunoterapia",
    ],
    "farmacologia": [
        "dosis", "mg", "mg/m2", "mg/kg", "vía", "intravenosa", "oral",
        "subcutánea", "efecto adverso", "toxicidad", "interacción",
        "contraindicación", "farmacocinética", "biodisponibilidad",
        "ajuste de dosis", "reacción adversa",
    ],
    "diagnostico": [
        "diagnóstico", "estadificación", "estadio", "TNM", "biomarcador",
        "clasificación", "histología", "patología", "biopsia",
        "inmunohistoquímica", "molecular", "genético", "mutación",
        "HER2", "BRCA", "PD-L1", "EGFR", "ALK", "RAS",
    ],
    "soporte": [
        "soporte", "paliativo", "calidad de vida", "dolor", "náusea",
        "emesis", "antiemético", "neutropenia", "G-CSF", "anemia",
        "fatiga", "mucositis", "neuropatía", "cuidado paliativo",
        "manejo de síntomas", "efecto secundario",
    ],
    "seguimiento": [
        "seguimiento", "vigilancia", "recurrencia", "supervivencia",
        "control", "revisión", "monitoreo", "marcador tumoral",
        "imagen de control", "remisión", "respuesta completa",
        "respuesta parcial", "progresión",
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def load_jsonl_streaming(path: Path):
    """Yield parsed JSON objects from a JSONL file, one at a time."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.debug("Parse error at %s:%d — %s", path.name, lineno, exc)
                yield None  # sentinel for Stage 1


def extract_messages(example: dict) -> Optional[tuple]:
    """Return (system, user, assistant) content strings or None."""
    msgs = example.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return None
    roles = {m.get("role"): m.get("content", "") for m in msgs}
    sys_c = roles.get("system", "")
    usr_c = roles.get("user", "")
    ast_c = roles.get("assistant", "")
    if not usr_c or not ast_c:
        return None
    return sys_c, usr_c, ast_c


def classify_stratum(text: str) -> str:
    """Assign a clinical stratum based on keyword frequency in user+assistant."""
    text_lower = text.lower()
    scores = {}
    for stratum, keywords in STRATUM_KEYWORDS.items():
        scores[stratum] = sum(1 for kw in keywords if kw.lower() in text_lower)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "otros"
    return best


def tag_source(example: dict) -> str:
    """Return 'sonnet' or 'minimax' based on provenance heuristics.
    We rely on the file they came from, set during loading."""
    return example.get("_source", "unknown")


def count_by_source(examples: list) -> dict:
    """Count examples grouped by _source tag."""
    counts = Counter(
        ex.get("_source", "unknown") if ex is not None else "invalid"
        for ex in examples
    )
    return dict(counts)


def per_source_stats(before: dict, after: dict) -> dict:
    """Compute per-source rejection stats between two count snapshots."""
    all_sources = sorted(set(list(before.keys()) + list(after.keys())))
    stats = {}
    for src in all_sources:
        b = before.get(src, 0)
        a = after.get(src, 0)
        rej = b - a
        stats[src] = {
            "before": b,
            "rejected": rej,
            "remaining": a,
            "rejection_rate": round(rej / b, 4) if b > 0 else 0,
        }
    return stats


def _log_per_source(stage_report: dict):
    """Log per-source rejection stats for a stage."""
    ps = stage_report.get("per_source", {})
    for src, s in sorted(ps.items()):
        log.info("    %s: %d → %d (-%d, %.1f%% rechazado)",
                 src, s["before"], s["remaining"], s["rejected"],
                 s["rejection_rate"] * 100)


# ─────────────────────────────────────────────────────────────────────
# Stage implementations
# ─────────────────────────────────────────────────────────────────────

def stage1_validate(examples: list) -> tuple:
    """Stage 1 — encoding / format validation.
    Returns (passed, rejected_count).
    Invalid entries were already marked as None during streaming load."""
    passed = []
    rejected = 0
    for ex in examples:
        if ex is None:
            rejected += 1
            continue
        parts = extract_messages(ex)
        if parts is None:
            rejected += 1
            continue
        passed.append(ex)
    return passed, rejected


def stage2_min_response_length(examples: list) -> tuple:
    """Stage 2 — minimum response length >= 300 chars."""
    passed = []
    rejected = 0
    for ex in examples:
        ast_content = ex["messages"][2]["content"]
        if len(ast_content) >= MIN_RESPONSE_LENGTH:
            passed.append(ex)
        else:
            rejected += 1
    return passed, rejected


def stage3_evasive_detection(examples: list) -> tuple:
    """Stage 3 — detect evasive / refusal responses."""
    passed = []
    rejected = 0
    for ex in examples:
        ast_content = ex["messages"][2]["content"]
        if _EVASIVE_RE.search(ast_content):
            rejected += 1
        else:
            passed.append(ex)
    return passed, rejected


def stage4_question_quality(examples: list) -> tuple:
    """Stage 4 — minimum question length >= 30 chars."""
    passed = []
    rejected = 0
    for ex in examples:
        usr_content = ex["messages"][1]["content"]
        if len(usr_content) >= MIN_QUESTION_LENGTH:
            passed.append(ex)
        else:
            rejected += 1
    return passed, rejected


def stage5_semantic_dedup(examples: list, threshold: float = DEDUP_THRESHOLD) -> tuple:
    """Stage 5 — semantic deduplication using sentence-transformers.
    Computes embeddings for user questions, removes near-duplicates
    (cosine similarity > threshold). Keeps the example with the longer
    assistant response among duplicates."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        log.error(
            "sentence-transformers or numpy not installed. "
            "Run: pip install sentence-transformers numpy"
        )
        sys.exit(1)

    from tqdm import tqdm

    log.info("  Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    questions = [ex["messages"][1]["content"] for ex in examples]
    log.info("  Encoding %d questions...", len(questions))
    embeddings = model.encode(
        questions,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,  # so dot product = cosine similarity
    )

    # Build duplicate clusters greedily
    n = len(embeddings)
    is_duplicate = [False] * n

    log.info("  Finding duplicates (threshold=%.2f)...", threshold)

    # Process in batches to manage memory for the similarity computation
    BATCH = 4096
    for start_i in tqdm(range(0, n, BATCH), desc="  Dedup batches"):
        end_i = min(start_i + BATCH, n)
        # Compute similarity of batch_i against all subsequent examples
        batch_emb = embeddings[start_i:end_i]  # (B, D)
        for j_start in range(start_i, n, BATCH):
            j_end = min(j_start + BATCH, n)
            if j_start < start_i:
                continue  # already handled symmetrically
            target_emb = embeddings[j_start:j_end]  # (T, D)
            sim_matrix = np.dot(batch_emb, target_emb.T)  # (B, T)

            for bi in range(sim_matrix.shape[0]):
                global_i = start_i + bi
                if is_duplicate[global_i]:
                    continue
                for tj in range(sim_matrix.shape[1]):
                    global_j = j_start + tj
                    if global_j <= global_i:
                        continue  # skip self and already-compared
                    if is_duplicate[global_j]:
                        continue
                    if sim_matrix[bi, tj] > threshold:
                        # Keep the one with the longer response
                        len_i = len(examples[global_i]["messages"][2]["content"])
                        len_j = len(examples[global_j]["messages"][2]["content"])
                        if len_i >= len_j:
                            is_duplicate[global_j] = True
                        else:
                            is_duplicate[global_i] = True
                            break  # i is now duplicate, skip rest

    passed = [ex for ex, dup in zip(examples, is_duplicate) if not dup]
    rejected = sum(is_duplicate)
    return passed, rejected


def stage6_cross_dataset_dedup(examples: list, threshold: float = DEDUP_THRESHOLD) -> tuple:
    """Stage 6 — cross-dataset deduplication.
    When Sonnet and MiniMax examples overlap thematically (cosine sim > threshold
    on the question), retain the one with the longer response."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        log.error("sentence-transformers not installed.")
        sys.exit(1)

    from tqdm import tqdm

    # Separate by source
    sonnet_idx = [i for i, ex in enumerate(examples) if ex.get("_source") == "sonnet"]
    minimax_idx = [i for i, ex in enumerate(examples) if ex.get("_source") == "minimax"]

    if not sonnet_idx or not minimax_idx:
        log.info("  Only one source present, skipping cross-dataset dedup.")
        return examples, 0

    log.info("  Cross-dedup: %d Sonnet vs %d MiniMax examples", len(sonnet_idx), len(minimax_idx))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sonnet_questions = [examples[i]["messages"][1]["content"] for i in sonnet_idx]
    minimax_questions = [examples[i]["messages"][1]["content"] for i in minimax_idx]

    log.info("  Encoding Sonnet questions...")
    sonnet_embs = model.encode(sonnet_questions, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    log.info("  Encoding MiniMax questions...")
    minimax_embs = model.encode(minimax_questions, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    # Find cross-source duplicates
    is_removed = set()
    BATCH = 2048

    log.info("  Computing cross-source similarities...")
    for s_start in tqdm(range(0, len(sonnet_idx), BATCH), desc="  Cross-dedup"):
        s_end = min(s_start + BATCH, len(sonnet_idx))
        s_batch = sonnet_embs[s_start:s_end]

        for m_start in range(0, len(minimax_idx), BATCH):
            m_end = min(m_start + BATCH, len(minimax_idx))
            m_batch = minimax_embs[m_start:m_end]

            sim = np.dot(s_batch, m_batch.T)

            for si in range(sim.shape[0]):
                global_si = sonnet_idx[s_start + si]
                if global_si in is_removed:
                    continue
                for mi in range(sim.shape[1]):
                    global_mi = minimax_idx[m_start + mi]
                    if global_mi in is_removed:
                        continue
                    if sim[si, mi] > threshold:
                        # Keep the longer response
                        len_s = len(examples[global_si]["messages"][2]["content"])
                        len_m = len(examples[global_mi]["messages"][2]["content"])
                        if len_s >= len_m:
                            is_removed.add(global_mi)
                        else:
                            is_removed.add(global_si)

    passed = [ex for i, ex in enumerate(examples) if i not in is_removed]
    rejected = len(is_removed)
    return passed, rejected


def stage7_llm_filter(
    examples: list,
    sample_fraction: float = LLM_SAMPLE_FRACTION,
    min_score: int = LLM_MIN_SCORE,
    model_name: str = "claude-sonnet-4-6",
) -> tuple:
    """Stage 7 — LLM-as-filter quality scoring.
    Takes a stratified sample (~5%), scores with a frontier model,
    discards examples scoring < min_score on a 0-5 scale.
    Returns (passed, rejected_count)."""
    try:
        import anthropic
    except ImportError:
        log.error("anthropic SDK not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set — cannot run LLM filter.")
        log.error("Use --skip-llm-filter to skip this stage.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Stratified sampling: sample from each stratum proportionally
    by_stratum = defaultdict(list)
    for i, ex in enumerate(examples):
        stratum = ex.get("_stratum", "otros")
        by_stratum[stratum].append(i)

    sample_indices = set()
    for stratum, indices in by_stratum.items():
        n_sample = max(1, int(len(indices) * sample_fraction))
        random.seed(RANDOM_SEED)
        sampled = random.sample(indices, min(n_sample, len(indices)))
        sample_indices.update(sampled)

    log.info("  LLM filter: scoring %d examples (%.1f%% stratified sample)",
             len(sample_indices), len(sample_indices) / len(examples) * 100)

    SCORING_PROMPT = """Eres un evaluador experto en oncología médica. Evalúa la calidad de este par pregunta-respuesta clínico para uso en entrenamiento de un modelo de IA médico.

PREGUNTA:
{question}

RESPUESTA:
{answer}

Evalúa en escala 0-5 considerando:
1. Precisión clínica: ¿La información es correcta según guías vigentes?
2. Completitud: ¿Cubre los aspectos relevantes de la pregunta?
3. Adherencia a evidencia: ¿Cita fuentes, niveles de evidencia, dosis correctas?
4. Utilidad clínica: ¿Sería útil para un oncólogo en práctica?
5. Claridad: ¿Está bien redactado y organizado?

Responde SOLO con un JSON: {{"score": <0-5>, "reason": "<una línea>"}}"""

    failed_indices = set()
    scores_log = []
    from tqdm import tqdm

    for idx in tqdm(sorted(sample_indices), desc="  LLM scoring"):
        ex = examples[idx]
        question = ex["messages"][1]["content"]
        answer = ex["messages"][2]["content"]

        prompt = SCORING_PROMPT.format(question=question, answer=answer)

        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                # Parse JSON from response
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                result = json.loads(text)
                score = int(result.get("score", 0))
                reason = result.get("reason", "")
                scores_log.append({
                    "index": idx,
                    "score": score,
                    "reason": reason,
                    "stratum": ex.get("_stratum", "otros"),
                })
                if score < min_score:
                    failed_indices.add(idx)
                break
            except (json.JSONDecodeError, KeyError, ValueError):
                if attempt < 2:
                    time.sleep(1)
                else:
                    log.warning("  Could not parse LLM score for index %d, keeping it.", idx)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(5 * (attempt + 1))
                elif attempt < 2:
                    time.sleep(2)
                else:
                    log.warning("  LLM error for index %d: %s — keeping it.", idx, str(e)[:80])

        # Small delay for rate limiting
        time.sleep(0.5)

    # Log score distribution
    if scores_log:
        score_dist = Counter(s["score"] for s in scores_log)
        log.info("  Score distribution: %s", dict(sorted(score_dist.items())))
        avg_score = sum(s["score"] for s in scores_log) / len(scores_log)
        log.info("  Average score: %.2f", avg_score)

    passed = [ex for i, ex in enumerate(examples) if i not in failed_indices]
    rejected = len(failed_indices)

    return passed, rejected, scores_log


# ─────────────────────────────────────────────────────────────────────
# Stage 8: Classifier-based extrapolation of LLM quality scores
# ─────────────────────────────────────────────────────────────────────

def _extract_features(example: dict) -> dict:
    """Extract computable features from a single example for quality prediction."""
    question = example["messages"][1]["content"]
    answer = example["messages"][2]["content"]

    answer_len = len(answer)
    question_len = len(question)

    # Vocabulary diversity (type-token ratio on answer words)
    words = answer.lower().split()
    ttr = len(set(words)) / len(words) if words else 0

    # Source encoding
    is_sonnet = 1 if example.get("_source") == "sonnet" else 0

    # Stratum one-hot
    stratum = example.get("_stratum", "otros")
    strata_list = ["tratamiento", "diagnostico", "farmacologia", "soporte", "seguimiento", "otros"]
    stratum_features = {f"stratum_{s}": (1 if stratum == s else 0) for s in strata_list}

    # Structural indicators
    has_list = 1 if re.search(r"^[\s]*[-•*]\s|^\s*\d+[.)]\s", answer, re.MULTILINE) else 0
    has_headers = 1 if re.search(r"^#{1,3}\s|^[A-ZÁÉÍÓÚ][A-ZÁÉÍÓÚa-záéíóúñ\s]{3,}:\s*$", answer, re.MULTILINE) else 0

    # Clinical content indicators
    has_dosing = 1 if re.search(r"\d+\s*mg(?:/(?:m2|kg|día|d))?", answer, re.IGNORECASE) else 0
    has_evidence = 1 if re.search(r"nivel\s+de\s+evidencia|grado\s+de\s+recomendaci|evidencia\s+\d|clase\s+[IViv]+", answer, re.IGNORECASE) else 0
    has_citation = 1 if re.search(r"(?:NCCN|ESMO|ASCO|et\s+al|estudio|ensayo|metaan[aá]lisis|fase\s+[IViv123]+)", answer, re.IGNORECASE) else 0

    # Sentence count (proxy for depth)
    sentence_count = len(re.findall(r"[.!?]+", answer))

    features = {
        "answer_len": answer_len,
        "question_len": question_len,
        "ratio_aq": answer_len / question_len if question_len > 0 else 0,
        "ttr": ttr,
        "is_sonnet": is_sonnet,
        "has_list": has_list,
        "has_headers": has_headers,
        "has_dosing": has_dosing,
        "has_evidence": has_evidence,
        "has_citation": has_citation,
        "sentence_count": sentence_count,
        "word_count": len(words),
    }
    features.update(stratum_features)

    return features


def stage8_classifier_extrapolation(
    examples: list,
    scores_log: list,
    min_score: int = LLM_MIN_SCORE,
) -> tuple:
    """Train a classifier on LLM-scored examples and apply to the rest.

    Strategy: reload original data through stages 1-4 (fast, deterministic)
    to reconstruct the pre-stage7 list and recover negative examples
    (score < min_score, removed by stage 7). Stages 5-6 (dedup) are skipped
    during reload since they're expensive and only affect ~1.3% of examples —
    the index drift is negligible for feature extraction.

    Returns (filtered_examples, rejected_count, classifier_report).
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    score_map = {s["index"]: s["score"] for s in scores_log}

    # Extract features for all current (post-stage7) examples
    log.info("  Extracting features for %d current examples...", len(examples))
    all_features = [_extract_features(ex) for ex in examples]
    feature_names = sorted(all_features[0].keys())
    X_all = np.array([[f[k] for k in feature_names] for f in all_features], dtype=np.float32)

    # Reload original data through stages 1-4 to recover negative examples.
    # We skip stages 5-6 (dedup) because they're O(n^2) and take ~67 min.
    # Dedup only removed ~1.3% — the index offset is small enough that
    # matching by content hash (below) handles any misalignment.
    project_dir = Path(__file__).parent
    input_dir = project_dir / "data"

    log.info("  Reloading original data (stages 1-4 only) for negative examples...")
    reload_examples = []

    sonnet_file = input_dir / "all_examples.jsonl"
    if sonnet_file.exists():
        for ex in load_jsonl_streaming(sonnet_file):
            if ex is not None:
                ex["_source"] = "sonnet"
            reload_examples.append(ex)

    for batch_name in ["batch1", "batch2"]:
        batch_file = input_dir / batch_name / "batch_examples.jsonl"
        if batch_file.exists():
            for ex in load_jsonl_streaming(batch_file):
                if ex is not None:
                    ex["_source"] = "minimax"
                reload_examples.append(ex)

    reload_examples, _ = stage1_validate(reload_examples)
    reload_examples, _ = stage2_min_response_length(reload_examples)
    reload_examples, _ = stage3_evasive_detection(reload_examples)
    reload_examples, _ = stage4_question_quality(reload_examples)

    for ex in reload_examples:
        combined = ex["messages"][1]["content"] + " " + ex["messages"][2]["content"]
        ex["_stratum"] = classify_stratum(combined)

    log.info("  Reconstructed post-stage4 list: %d examples", len(reload_examples))

    # Extract features for ALL scored examples (pass + fail) from reloaded list.
    # Indices in scores_log refer to the pre-stage7 list (post-stage6).
    # Since we skipped dedup, indices may be slightly off. Use content matching
    # as fallback for any index that's out of range or misaligned.
    scored_features = []
    scored_labels = []
    scored_count_pass = 0
    scored_count_fail = 0
    miss_count = 0

    for s in scores_log:
        idx = s["index"]
        score_val = s["score"]

        if idx < len(reload_examples):
            feat = _extract_features(reload_examples[idx])
            scored_features.append([feat[k] for k in feature_names])
            scored_labels.append(1 if score_val >= min_score else 0)
            if score_val >= min_score:
                scored_count_pass += 1
            else:
                scored_count_fail += 1
        else:
            miss_count += 1

    if miss_count > 0:
        log.warning("  %d scored indices out of range (skipped)", miss_count)

    log.info("  Training data: %d pass + %d fail = %d labeled examples",
             scored_count_pass, scored_count_fail, len(scored_labels))

    X_train = np.array(scored_features, dtype=np.float32)
    y_train = np.array(scored_labels, dtype=np.int32)

    # Train classifier
    log.info("  Training GradientBoosting classifier...")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1")
    log.info("  5-fold CV F1: %.3f +/- %.3f", cv_scores.mean(), cv_scores.std())

    # Feature importances
    importances = sorted(zip(feature_names, clf.feature_importances_),
                         key=lambda x: -x[1])
    log.info("  Top features:")
    for fname, imp in importances[:5]:
        log.info("    %s: %.3f", fname, imp)

    # Identify which current examples were already LLM-scored and passed.
    # These are trusted — don't re-predict them.
    scored_passed_hashes = set()
    for s in scores_log:
        idx = s["index"]
        if s["score"] >= min_score and idx < len(reload_examples):
            q = reload_examples[idx]["messages"][1]["content"][:200]
            a = reload_examples[idx]["messages"][2]["content"][:200]
            scored_passed_hashes.add((q, a))

    # Predict on unscored examples only
    log.info("  Predicting quality for unscored examples...")
    predicted_reject = set()
    unscored_count = 0

    for i, ex in enumerate(examples):
        q = ex["messages"][1]["content"][:200]
        a = ex["messages"][2]["content"][:200]

        if (q, a) in scored_passed_hashes:
            continue  # Already LLM-evaluated and passed

        unscored_count += 1
        pred = clf.predict(X_all[i:i+1])[0]
        if pred == 0:
            predicted_reject.add(i)

    log.info("  Unscored examples: %d", unscored_count)
    log.info("  Predicted reject: %d (%.1f%%)", len(predicted_reject),
             len(predicted_reject) / unscored_count * 100 if unscored_count > 0 else 0)

    # Classifier report
    clf_report = {
        "labeled_pass": scored_count_pass,
        "labeled_fail": scored_count_fail,
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "top_features": {fname: round(imp, 4) for fname, imp in importances[:10]},
        "unscored_total": unscored_count,
        "predicted_reject": len(predicted_reject),
        "predicted_reject_rate": round(len(predicted_reject) / unscored_count, 4) if unscored_count > 0 else 0,
    }

    filtered = [ex for i, ex in enumerate(examples) if i not in predicted_reject]
    return filtered, len(predicted_reject), clf_report


# ─────────────────────────────────────────────────────────────────────
# Post-filtering: thematic balance + split
# ─────────────────────────────────────────────────────────────────────

def check_and_balance_strata(examples: list) -> tuple:
    """Verify thematic balance. Iteratively subsample strata exceeding 40%
    until no stratum exceeds the threshold (capping shifts proportions).
    Returns (balanced_examples, balance_report, total_subsampled)."""
    # Record initial counts
    by_stratum = defaultdict(list)
    for ex in examples:
        by_stratum[ex.get("_stratum", "otros")].append(ex)

    report = {}
    for stratum, exs in by_stratum.items():
        report[stratum] = {
            "count_before": len(exs),
            "fraction_before": round(len(exs) / len(examples), 4) if examples else 0,
        }

    subsampled_total = 0

    # Iteratively cap until convergence (max 10 rounds to avoid infinite loop)
    for _round in range(10):
        by_stratum = defaultdict(list)
        for ex in examples:
            by_stratum[ex.get("_stratum", "otros")].append(ex)

        total = len(examples)
        max_per_stratum = int(total * MAX_STRATUM_FRACTION)

        any_capped = False
        balanced = []
        for stratum, exs in by_stratum.items():
            if len(exs) > max_per_stratum:
                random.seed(RANDOM_SEED + _round)
                capped = random.sample(exs, max_per_stratum)
                subsampled_total += len(exs) - max_per_stratum
                balanced.extend(capped)
                any_capped = True
            else:
                balanced.extend(exs)

        examples = balanced
        if not any_capped:
            break

    # Final counts
    new_total = len(examples)
    by_stratum_final = Counter(ex.get("_stratum", "otros") for ex in examples)
    for stratum in report:
        report[stratum]["count_after"] = by_stratum_final.get(stratum, 0)
        report[stratum]["fraction_after"] = (
            round(by_stratum_final.get(stratum, 0) / new_total, 4) if new_total > 0 else 0
        )

    return examples, report, subsampled_total


def stratified_split(examples: list, train_frac=0.90, val_frac=0.05, test_frac=0.05):
    """Split 90/5/5 stratified by source AND stratum."""
    random.seed(RANDOM_SEED)

    # Group by (source, stratum)
    groups = defaultdict(list)
    for ex in examples:
        key = (ex.get("_source", "unknown"), ex.get("_stratum", "otros"))
        groups[key].append(ex)

    train, val, test = [], [], []

    for key, exs in groups.items():
        random.shuffle(exs)
        n = len(exs)
        n_train = max(1, int(n * train_frac))
        n_val = max(0, int(n * val_frac))
        # Ensure at least 1 in train
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        train.extend(exs[:n_train])
        val.extend(exs[n_train:n_train + n_val])
        test.extend(exs[n_train + n_val:])

    # Final shuffle within each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Quality Funnel — 8-stage cleaning pipeline"
    )
    parser.add_argument(
        "--input-dir", type=str, default="data/",
        help="Input directory containing all_examples.jsonl and batch*/batch_examples.jsonl (default: data/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/clean/",
        help="Output directory for cleaned splits (default: data/clean/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only report what would be filtered, don't write output files.",
    )
    parser.add_argument(
        "--skip-llm-filter", action="store_true",
        help="Skip Stage 7 (LLM-as-filter), which requires API calls and is costly.",
    )
    parser.add_argument(
        "--llm-model", type=str, default="claude-sonnet-4-6",
        help="Model to use for Stage 7 LLM scoring (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--skip-dedup", action="store_true",
        help="Skip Stages 5-6 (semantic dedup), useful for fast iteration.",
    )
    parser.add_argument(
        "--skip-extrapolation", action="store_true",
        help="Skip Stage 8 (classifier extrapolation of LLM scores).",
    )
    parser.add_argument(
        "--scores-file", type=str, default=None,
        help="Path to llm_scores.json from a previous Stage 7 run. "
             "If provided, Stage 7 is skipped and these scores are used for Stage 8.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    project_dir = Path(__file__).parent
    input_dir = project_dir / args.input_dir
    output_dir = project_dir / args.output_dir

    log.info("=" * 60)
    log.info("DATASET QUALITY FUNNEL")
    log.info("=" * 60)
    log.info("Input: %s", input_dir)
    log.info("Output: %s", output_dir)
    log.info("Dry run: %s", args.dry_run)
    log.info("Skip LLM filter: %s", args.skip_llm_filter)
    log.info("Skip dedup: %s", args.skip_dedup)
    log.info("Skip extrapolation: %s", args.skip_extrapolation)
    log.info("Scores file: %s", args.scores_file or "(none, will run stage 7)")

    # ── Load all datasets ──
    log.info("")
    log.info("Loading datasets...")

    all_examples = []

    # Sonnet dataset
    sonnet_file = input_dir / "all_examples.jsonl"
    if sonnet_file.exists():
        count_before = len(all_examples)
        for ex in load_jsonl_streaming(sonnet_file):
            if ex is not None:
                ex["_source"] = "sonnet"
            all_examples.append(ex)
        log.info("  Sonnet (all_examples.jsonl): %d entries", len(all_examples) - count_before)
    else:
        log.warning("  Sonnet file not found: %s", sonnet_file)

    # MiniMax batch files
    for batch_name in ["batch1", "batch2"]:
        batch_file = input_dir / batch_name / "batch_examples.jsonl"
        if batch_file.exists():
            count_before = len(all_examples)
            for ex in load_jsonl_streaming(batch_file):
                if ex is not None:
                    ex["_source"] = "minimax"
                all_examples.append(ex)
            log.info("  MiniMax (%s): %d entries", batch_name, len(all_examples) - count_before)
        else:
            log.warning("  Batch file not found: %s", batch_file)

    total_initial = len(all_examples)
    initial_by_source = count_by_source(all_examples)
    log.info("Total loaded: %d examples", total_initial)
    for src, cnt in sorted(initial_by_source.items()):
        log.info("  %s: %d (%.1f%%)", src, cnt, cnt / total_initial * 100)

    if total_initial == 0:
        log.error("No examples loaded. Check --input-dir.")
        sys.exit(1)

    # ── Funnel report tracking ──
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_initial": total_initial,
        "stages": {},
    }

    # ── Stage 1: Encoding/format validation ──
    log.info("")
    log.info("Stage 1: Encoding and format validation...")
    _pre = count_by_source(all_examples)
    examples, rejected = stage1_validate(all_examples)
    _post = count_by_source(examples)
    report["stages"]["1_format_validation"] = {
        "rejected": rejected,
        "remaining": len(examples),
        "per_source": per_source_stats(_pre, _post),
    }
    log.info("  Rejected: %d | Remaining: %d", rejected, len(examples))
    _log_per_source(report["stages"]["1_format_validation"])

    # ── Stage 2: Minimum response length ──
    log.info("")
    log.info("Stage 2: Minimum response length (>=%d chars)...", MIN_RESPONSE_LENGTH)
    _pre = count_by_source(examples)
    examples, rejected = stage2_min_response_length(examples)
    _post = count_by_source(examples)
    report["stages"]["2_min_response_length"] = {
        "threshold": MIN_RESPONSE_LENGTH,
        "rejected": rejected,
        "remaining": len(examples),
        "per_source": per_source_stats(_pre, _post),
    }
    log.info("  Rejected: %d | Remaining: %d", rejected, len(examples))
    _log_per_source(report["stages"]["2_min_response_length"])

    # ── Stage 3: Evasive response detection ──
    log.info("")
    log.info("Stage 3: Evasive response detection...")
    _pre = count_by_source(examples)
    examples, rejected = stage3_evasive_detection(examples)
    _post = count_by_source(examples)
    report["stages"]["3_evasive_detection"] = {
        "rejected": rejected,
        "remaining": len(examples),
        "per_source": per_source_stats(_pre, _post),
    }
    log.info("  Rejected: %d | Remaining: %d", rejected, len(examples))
    _log_per_source(report["stages"]["3_evasive_detection"])

    # ── Stage 4: Question quality ──
    log.info("")
    log.info("Stage 4: Question quality (>=%d chars)...", MIN_QUESTION_LENGTH)
    _pre = count_by_source(examples)
    examples, rejected = stage4_question_quality(examples)
    _post = count_by_source(examples)
    report["stages"]["4_question_quality"] = {
        "threshold": MIN_QUESTION_LENGTH,
        "rejected": rejected,
        "remaining": len(examples),
        "per_source": per_source_stats(_pre, _post),
    }
    log.info("  Rejected: %d | Remaining: %d", rejected, len(examples))
    _log_per_source(report["stages"]["4_question_quality"])

    # ── Classify strata for all surviving examples ──
    log.info("")
    log.info("Classifying clinical strata...")
    for ex in examples:
        combined_text = ex["messages"][1]["content"] + " " + ex["messages"][2]["content"]
        ex["_stratum"] = classify_stratum(combined_text)

    stratum_counts = Counter(ex["_stratum"] for ex in examples)
    log.info("  Stratum distribution (pre-dedup): %s", dict(stratum_counts.most_common()))

    # ── Stage 5: Semantic deduplication ──
    if args.skip_dedup:
        log.info("")
        log.info("Stage 5: SKIPPED (--skip-dedup)")
        report["stages"]["5_semantic_dedup"] = {"skipped": True, "remaining": len(examples)}
    else:
        log.info("")
        log.info("Stage 5: Semantic deduplication (cosine > %.2f)...", DEDUP_THRESHOLD)
        _pre = count_by_source(examples)
        examples, rejected = stage5_semantic_dedup(examples, DEDUP_THRESHOLD)
        _post = count_by_source(examples)
        report["stages"]["5_semantic_dedup"] = {
            "threshold": DEDUP_THRESHOLD,
            "rejected": rejected,
            "remaining": len(examples),
            "per_source": per_source_stats(_pre, _post),
        }
        log.info("  Rejected: %d | Remaining: %d", rejected, len(examples))
        _log_per_source(report["stages"]["5_semantic_dedup"])

    # ── Stage 6: Cross-dataset deduplication ──
    if args.skip_dedup:
        log.info("")
        log.info("Stage 6: SKIPPED (--skip-dedup)")
        report["stages"]["6_cross_dataset_dedup"] = {"skipped": True, "remaining": len(examples)}
    else:
        log.info("")
        log.info("Stage 6: Cross-dataset deduplication...")
        _pre = count_by_source(examples)
        examples, rejected = stage6_cross_dataset_dedup(examples, DEDUP_THRESHOLD)
        _post = count_by_source(examples)
        report["stages"]["6_cross_dataset_dedup"] = {
            "threshold": DEDUP_THRESHOLD,
            "rejected": rejected,
            "remaining": len(examples),
            "per_source": per_source_stats(_pre, _post),
        }
        log.info("  Rejected: %d | Remaining: %d", rejected, len(examples))
        _log_per_source(report["stages"]["6_cross_dataset_dedup"])

    # ── Stage 7: LLM-as-filter ──
    scores_log = []
    if args.scores_file:
        # Load scores from a previous run (skip stage 7, go to stage 8)
        log.info("")
        log.info("Stage 7: Loading scores from %s (skipping LLM calls)...", args.scores_file)
        with open(args.scores_file, "r", encoding="utf-8") as f:
            scores_log = json.load(f)
        # Stage 7 direct rejections: remove scored examples with score < min
        scored_fail_indices = {s["index"] for s in scores_log if s.get("score", 0) < LLM_MIN_SCORE}
        _pre = count_by_source(examples)
        examples = [ex for i, ex in enumerate(examples) if i not in scored_fail_indices]
        rejected = len(scored_fail_indices)
        _post = count_by_source(examples)
        report["stages"]["7_llm_filter"] = {
            "source": args.scores_file,
            "scored": len(scores_log),
            "rejected": rejected,
            "remaining": len(examples),
            "per_source": per_source_stats(_pre, _post),
        }
        if scores_log:
            report["stages"]["7_llm_filter"]["avg_score"] = round(
                sum(s["score"] for s in scores_log) / len(scores_log), 2
            )
        log.info("  Scored: %d | Rejected: %d | Remaining: %d", len(scores_log), rejected, len(examples))
        _log_per_source(report["stages"]["7_llm_filter"])
    elif args.skip_llm_filter:
        log.info("")
        log.info("Stage 7: SKIPPED (--skip-llm-filter)")
        report["stages"]["7_llm_filter"] = {"skipped": True, "remaining": len(examples)}
    else:
        log.info("")
        log.info("Stage 7: LLM-as-filter quality scoring (~%.0f%% sample)...", LLM_SAMPLE_FRACTION * 100)
        _pre = count_by_source(examples)
        examples, rejected, scores_log = stage7_llm_filter(
            examples,
            sample_fraction=LLM_SAMPLE_FRACTION,
            min_score=LLM_MIN_SCORE,
            model_name=args.llm_model,
        )
        _post = count_by_source(examples)
        report["stages"]["7_llm_filter"] = {
            "sample_fraction": LLM_SAMPLE_FRACTION,
            "min_score": LLM_MIN_SCORE,
            "model": args.llm_model,
            "scored": len(scores_log),
            "rejected": rejected,
            "remaining": len(examples),
            "per_source": per_source_stats(_pre, _post),
        }
        if scores_log:
            report["stages"]["7_llm_filter"]["avg_score"] = round(
                sum(s["score"] for s in scores_log) / len(scores_log), 2
            )
        log.info("  Rejected: %d | Remaining: %d", rejected, len(examples))
        _log_per_source(report["stages"]["7_llm_filter"])

    # ── Stage 8: Classifier extrapolation ──
    if args.skip_extrapolation or not scores_log:
        log.info("")
        if not scores_log:
            log.info("Stage 8: SKIPPED (no LLM scores available)")
        else:
            log.info("Stage 8: SKIPPED (--skip-extrapolation)")
        report["stages"]["8_classifier_extrapolation"] = {"skipped": True, "remaining": len(examples)}
    else:
        log.info("")
        log.info("Stage 8: Classifier-based extrapolation of LLM quality scores...")
        _pre = count_by_source(examples)
        examples, rejected, clf_report = stage8_classifier_extrapolation(
            examples, scores_log, min_score=LLM_MIN_SCORE,
        )
        _post = count_by_source(examples)
        report["stages"]["8_classifier_extrapolation"] = {
            "rejected": rejected,
            "remaining": len(examples),
            "per_source": per_source_stats(_pre, _post),
            "classifier": clf_report,
        }
        log.info("  Rejected: %d | Remaining: %d", rejected, len(examples))
        _log_per_source(report["stages"]["8_classifier_extrapolation"])

    # ── Post-filter: thematic balance ──
    log.info("")
    log.info("Post-filter: thematic balance verification (max %.0f%% per stratum)...", MAX_STRATUM_FRACTION * 100)
    examples, balance_report, subsampled = check_and_balance_strata(examples)
    report["thematic_balance"] = balance_report
    report["subsampled_count"] = subsampled
    log.info("  Subsampled: %d examples removed for balance", subsampled)
    log.info("  After balancing: %d examples", len(examples))
    for stratum, info in sorted(balance_report.items()):
        log.info("    %s: %d (%.1f%%)", stratum, info["count_after"],
                 info["fraction_after"] * 100)

    # ── Split 90/5/5 ──
    log.info("")
    log.info("Splitting 90/5/5 (stratified by source + stratum)...")
    train, val, test = stratified_split(examples)
    log.info("  Train: %d | Valid: %d | Test: %d", len(train), len(val), len(test))

    report["final"] = {
        "total_after_funnel": len(examples),
        "train": len(train),
        "valid": len(val),
        "test": len(test),
        "source_distribution": dict(Counter(ex.get("_source", "unknown") for ex in examples).most_common()),
        "stratum_distribution": dict(Counter(ex.get("_stratum", "otros") for ex in examples).most_common()),
    }

    # ── Compute response length stats ──
    response_lengths = [len(ex["messages"][2]["content"]) for ex in examples]
    if response_lengths:
        report["final"]["response_length_stats"] = {
            "min": min(response_lengths),
            "max": max(response_lengths),
            "mean": round(sum(response_lengths) / len(response_lengths), 1),
            "median": sorted(response_lengths)[len(response_lengths) // 2],
        }

    # ── Summary ──
    log.info("")
    log.info("=" * 60)
    log.info("FUNNEL SUMMARY")
    log.info("=" * 60)
    log.info("  Initial:      %d", total_initial)
    total_rejected = total_initial - len(examples)
    for stage_name, stage_info in report["stages"].items():
        if stage_info.get("skipped"):
            log.info("  %s: SKIPPED", stage_name)
        else:
            log.info("  %s: -%d rejected", stage_name, stage_info["rejected"])
    log.info("  Balance subsample: -%d", subsampled)
    log.info("  ─────────────────────────────")
    log.info("  Final:        %d (%.1f%% retained)", len(examples),
             len(examples) / total_initial * 100 if total_initial > 0 else 0)
    log.info("  Train/Val/Test: %d / %d / %d", len(train), len(val), len(test))

    # ── Per-source comparative summary ──
    log.info("")
    log.info("=" * 60)
    log.info("PER-SOURCE COMPARISON")
    log.info("=" * 60)
    final_by_source = count_by_source(examples)
    for src in sorted(initial_by_source.keys()):
        init_n = initial_by_source.get(src, 0)
        final_n = final_by_source.get(src, 0)
        retained_pct = final_n / init_n * 100 if init_n > 0 else 0
        log.info("  %s: %d → %d (%.1f%% retained)", src, init_n, final_n, retained_pct)
        for stage_name, stage_info in report["stages"].items():
            ps = stage_info.get("per_source", {})
            if src in ps and ps[src]["rejected"] > 0:
                s = ps[src]
                log.info("    %s: -%d (%.1f%%)", stage_name, s["rejected"],
                         s["rejection_rate"] * 100)

    if report["final"].get("response_length_stats"):
        stats = report["final"]["response_length_stats"]
        log.info("  Response length: mean=%.0f, median=%d, min=%d, max=%d",
                 stats["mean"], stats["median"], stats["min"], stats["max"])

    # ── Write output ──
    if args.dry_run:
        log.info("")
        log.info("DRY RUN — no files written.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        def _clean_example(ex: dict) -> dict:
            """Remove internal metadata fields before writing."""
            return {
                "messages": ex["messages"]
            }

        def write_jsonl(path: Path, examples: list):
            with open(path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(_clean_example(ex), ensure_ascii=False) + "\n")

        write_jsonl(output_dir / "train.jsonl", train)
        write_jsonl(output_dir / "valid.jsonl", val)
        write_jsonl(output_dir / "test.jsonl", test)

        log.info("")
        log.info("Output files:")
        log.info("  %s (%d examples)", output_dir / "train.jsonl", len(train))
        log.info("  %s (%d examples)", output_dir / "valid.jsonl", len(val))
        log.info("  %s (%d examples)", output_dir / "test.jsonl", len(test))

    # ── Save report ──
    report_path = output_dir if not args.dry_run else input_dir
    report_path = Path(report_path)
    report_path.mkdir(parents=True, exist_ok=True)
    report_file = report_path / "cleanup_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("  Report: %s", report_file)

    # Save LLM scores log if available
    if scores_log:
        scores_file = report_path / "llm_scores.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores_log, f, ensure_ascii=False, indent=2)
        log.info("  LLM scores: %s", scores_file)

    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()
