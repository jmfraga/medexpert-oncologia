#!/usr/bin/env python3
"""
MedExpert Arena — JudgeRunner (Opus 4.6)

Toma las respuestas generadas por arena_runner.py y las evalúa con
Opus 4.6 + RAG usando la rúbrica de 4 criterios clínicos.

Ejecutar en M1:
    cd ~/Projects/medexpert-admin
    python arena/arena_judge.py arena/results/responses_YYYYMMDD_HHMMSS.json
    python arena/arena_judge.py arena/results/responses_*.json --cases 1 2 3
    python arena/arena_judge.py arena/results/responses_*.json --tiers light premium

Output:
    arena/results/opus_scores_YYYYMMDD_HHMMSS.json
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

import database as db
from rag_engine import get_rag_for_expert
from bot_brain import (
    _translate_query_to_english, _diversify_results, _detect_society,
    _expand_synonyms,
)

logging.basicConfig(
    format="%(asctime)s [arena-judge] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("arena-judge")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# ─────────────────────────────────────────────
# Judge configuration
# ─────────────────────────────────────────────

JUDGE_MODEL = "claude-opus-4-6-20250514"  # Claude Opus 4.6
EXPERT_SLUG = "oncologia"

RUBRIC = {
    "precision_diagnostica": {
        "name": "Precisión diagnóstica",
        "weight": 0.25,
        "description": (
            "¿El diagnóstico y estadificación coinciden con el gold standard? "
            "¿Se identifican correctamente los biomarcadores y subtipos relevantes?"
        ),
    },
    "apego_guias": {
        "name": "Apego a guías",
        "weight": 0.30,
        "description": (
            "¿Las recomendaciones son consistentes con NCCN/ESMO/IMSS? "
            "¿Se citan correctamente las guías? ¿Se mencionan niveles de evidencia?"
        ),
    },
    "completitud": {
        "name": "Completitud",
        "weight": 0.25,
        "description": (
            "¿Se cubren todos los aspectos relevantes? Tratamiento primario, "
            "alternativas, seguimiento, efectos adversos, consideraciones especiales."
        ),
    },
    "utilidad_clinica": {
        "name": "Utilidad clínica",
        "weight": 0.20,
        "description": (
            "¿La respuesta es práctica y accionable para un oncólogo? "
            "¿Incluye dosis, esquemas, tiempos? ¿Es clara y bien organizada?"
        ),
    },
}


# ─────────────────────────────────────────────
# RAG context for judge (same as runner)
# ─────────────────────────────────────────────

def build_rag_context(query: str) -> str:
    """Build RAG context for the judge."""
    rag = get_rag_for_expert(EXPERT_SLUG)
    expanded = _expand_synonyms(query, EXPERT_SLUG)

    rag_es = rag.search_detailed(expanded, n_results=15)
    english_query = _translate_query_to_english(expanded)
    rag_en = rag.search_detailed(english_query, n_results=15) if english_query else []

    seen = set()
    merged = []
    for hit in rag_es + rag_en:
        key = (hit.get("source", ""), hit.get("text", "")[:100])
        if key not in seen:
            seen.add(key)
            merged.append(hit)

    diverse_hits = _diversify_results(merged, max_per_source=3, total=10)

    context_parts = []
    for hit in diverse_hits:
        source = hit.get("source", "Guia clinica")
        section = hit.get("section_path", "")
        society = hit.get("society", "") or _detect_society(source)
        label = source
        if section:
            label += f" > {section}"
        if society:
            label = f"[{society}] {label}"
        context_parts.append(f"[{label}]: {hit['text'][:500]}")

    return "\n\n".join(context_parts) if context_parts else "(Sin guias disponibles)"


# ─────────────────────────────────────────────
# Judge prompt
# ─────────────────────────────────────────────

def build_judge_prompt(case: dict, candidate_response: str,
                       rag_context: str) -> tuple[str, str]:
    """Build the system + user prompt for Opus judge.

    Returns (system_prompt, user_prompt).
    """
    rubric_text = "\n".join(
        f"  {i+1}. {r['name']} (peso {r['weight']*100:.0f}%): {r['description']}"
        for i, r in enumerate(RUBRIC.values())
    )

    gold = case["gold_standard"]
    gold_text = (
        f"Diagnóstico correcto: {gold['diagnosis']}\n"
        f"Recomendaciones clave:\n"
        + "\n".join(f"  - {rec}" for rec in gold["key_recommendations"])
        + f"\nGuías de referencia: {', '.join(gold['key_guidelines'])}"
    )

    system = (
        "Eres un evaluador experto en oncología clínica. Tu tarea es calificar "
        "la respuesta de un asistente clínico a un caso oncológico.\n\n"
        "Evalúas con objetividad, rigor clínico y apego a guías internacionales. "
        "No eres un candidato — eres el juez imparcial.\n\n"
        "GUÍAS CLÍNICAS DE REFERENCIA (para tu evaluación):\n"
        f"{rag_context}"
    )

    user = (
        "CASO CLÍNICO:\n"
        f"{case['case_text']}\n\n"
        "GOLD STANDARD (respuesta esperada):\n"
        f"{gold_text}\n\n"
        "RESPUESTA DEL CANDIDATO A EVALUAR:\n"
        f"{candidate_response}\n\n"
        "RÚBRICA DE EVALUACIÓN (escala 0-5 por criterio):\n"
        f"{rubric_text}\n\n"
        "Escala:\n"
        "  0 = Nulo/incorrecto\n"
        "  1 = Muy deficiente\n"
        "  2 = Deficiente\n"
        "  3 = Aceptable\n"
        "  4 = Bueno\n"
        "  5 = Excelente\n\n"
        "INSTRUCCIONES:\n"
        "1. Evalúa la respuesta del candidato comparándola con el gold standard y las guías clínicas\n"
        "2. Califica cada criterio de 0 a 5\n"
        "3. Justifica brevemente cada calificación\n"
        "4. Identifica errores clínicos graves si los hay\n"
        "5. Comenta sobre alucinaciones (información no respaldada por guías)\n\n"
        "Responde EXCLUSIVAMENTE con un JSON válido con esta estructura:\n"
        "```json\n"
        "{\n"
        '  "precision_diagnostica": {"score": <0-5>, "justification": "<texto>"},\n'
        '  "apego_guias": {"score": <0-5>, "justification": "<texto>"},\n'
        '  "completitud": {"score": <0-5>, "justification": "<texto>"},\n'
        '  "utilidad_clinica": {"score": <0-5>, "justification": "<texto>"},\n'
        '  "score_compuesto": <float>,\n'
        '  "errores_graves": ["<error1>", ...],\n'
        '  "alucinaciones": ["<alucinacion1>", ...],\n'
        '  "feedback_general": "<resumen de 2-3 oraciones>"\n'
        "}\n"
        "```\n"
        "El score_compuesto es el promedio ponderado: "
        "precisión×0.25 + apego×0.30 + completitud×0.25 + utilidad×0.20"
    )

    return system, user


# ─────────────────────────────────────────────
# Call Opus judge
# ─────────────────────────────────────────────

def call_judge(system: str, user: str) -> dict:
    """Call Opus 4.6 to judge a response. Returns parsed JSON scores."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.time()

    try:
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=120.0,
        )
        text = resp.content[0].text
        elapsed = time.time() - start

        # Parse JSON from response (handle markdown code blocks)
        json_text = text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        scores = json.loads(json_text.strip())

        # Validate and compute weighted score if not present
        if "score_compuesto" not in scores:
            scores["score_compuesto"] = round(
                scores["precision_diagnostica"]["score"] * 0.25
                + scores["apego_guias"]["score"] * 0.30
                + scores["completitud"]["score"] * 0.25
                + scores["utilidad_clinica"]["score"] * 0.20,
                2
            )

        return {
            "status": "success",
            "scores": scores,
            "raw_response": text,
            "judge_time_s": round(elapsed, 2),
            "tokens_input": resp.usage.input_tokens if resp.usage else 0,
            "tokens_output": resp.usage.output_tokens if resp.usage else 0,
        }

    except json.JSONDecodeError as e:
        elapsed = time.time() - start
        logger.error(f"JSON parse error: {e}")
        return {
            "status": "error",
            "scores": None,
            "raw_response": text if 'text' in dir() else "",
            "judge_time_s": round(elapsed, 2),
            "error": f"JSON parse error: {e}",
        }

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Judge error: {e}")
        return {
            "status": "error",
            "scores": None,
            "raw_response": "",
            "judge_time_s": round(elapsed, 2),
            "error": str(e),
        }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def judge_responses(responses_path: str, case_ids: list[int] = None,
                    tier_keys: list[str] = None) -> dict:
    """Judge all responses in a responses file.

    Returns a judgment results dict.
    """
    db.init_db()

    with open(responses_path, encoding="utf-8") as f:
        responses = json.load(f)

    cases = responses["cases"]
    if case_ids:
        cases = [c for c in cases if c["case_id"] in case_ids]

    all_tiers = list(responses["tiers"].keys())
    tiers_to_judge = tier_keys or all_tiers

    total = sum(
        1 for c in cases
        for tk in tiers_to_judge
        if tk in c["responses"] and c["responses"][tk].get("status") == "success"
    )
    logger.info(f"Judging {total} responses ({len(cases)} cases × {len(tiers_to_judge)} tiers)")

    results = {
        "arena_version": "1.0",
        "judge": JUDGE_MODEL,
        "rubric": {k: {"name": v["name"], "weight": v["weight"]} for k, v in RUBRIC.items()},
        "source_responses": responses_path,
        "timestamp": datetime.now().isoformat(),
        "judgments": [],
    }

    total_cost = 0
    judge_count = 0

    for case in cases:
        case_id = case["case_id"]
        print(f"\n{'='*60}")
        print(f"CASO {case_id}: {case['title']}")
        print(f"{'='*60}")

        # Build RAG context for judge (once per case)
        rag_context = build_rag_context(case["case_text"])

        for tk in tiers_to_judge:
            if tk not in case["responses"]:
                continue
            resp_data = case["responses"][tk]
            if resp_data.get("status") != "success":
                logger.warning(f"  Skipping {tk} (status: {resp_data.get('status')})")
                continue

            candidate_response = resp_data["response"]

            # Build judge prompt
            system, user = build_judge_prompt(case, candidate_response, rag_context)

            logger.info(f"  Judging {resp_data.get('tier_name', tk)}...")
            judgment = call_judge(system, user)

            judge_count += 1
            if judgment["status"] == "success":
                scores = judgment["scores"]
                sc = scores.get("score_compuesto", 0)
                # Estimate cost (~$15/MTok input, ~$75/MTok output for Opus)
                cost = (judgment.get("tokens_input", 0) * 15 / 1e6
                        + judgment.get("tokens_output", 0) * 75 / 1e6)
                total_cost += cost

                print(f"  {resp_data.get('tier_name', tk):30s} "
                      f"Score: {sc:.2f}/5.00 "
                      f"[P:{scores['precision_diagnostica']['score']} "
                      f"A:{scores['apego_guias']['score']} "
                      f"C:{scores['completitud']['score']} "
                      f"U:{scores['utilidad_clinica']['score']}] "
                      f"({judgment['judge_time_s']:.1f}s, ~${cost:.3f})")
            else:
                print(f"  {resp_data.get('tier_name', tk):30s} ERROR: {judgment.get('error')}")

            results["judgments"].append({
                "case_id": case_id,
                "case_title": case["title"],
                "complexity": case["complexity"],
                "tier": tk,
                "tier_name": resp_data.get("tier_name", tk),
                **judgment,
            })

    # Summary
    _print_judge_summary(results, tiers_to_judge)
    print(f"\nCosto estimado total del jueceo: ${total_cost:.2f} USD ({judge_count} evaluaciones)")

    return results


def _print_judge_summary(results: dict, tier_keys: list[str]):
    """Print summary of judge scores."""
    print(f"\n\n{'='*60}")
    print("RESUMEN DE CALIFICACIONES (Opus Judge)")
    print(f"{'='*60}")

    # Collect scores by tier
    tier_scores = {tk: [] for tk in tier_keys}
    for j in results["judgments"]:
        if j["status"] == "success" and j["tier"] in tier_scores:
            tier_scores[j["tier"]].append(j["scores"]["score_compuesto"])

    # By tier
    print(f"\n{'Tier':<35s} {'Score':>8s} {'Min':>6s} {'Max':>6s} {'N':>4s}")
    print("-" * 60)
    for tk in tier_keys:
        scores = tier_scores[tk]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"{tk:<35s} {avg:>7.2f} {min(scores):>6.2f} {max(scores):>6.2f} {len(scores):>4d}")

    # By complexity
    print(f"\n{'Complejidad':<15s}", end="")
    for tk in tier_keys:
        print(f" {tk:>15s}", end="")
    print()
    print("-" * (15 + 16 * len(tier_keys)))

    for complexity in ["simple", "moderado", "complejo"]:
        row = f"{complexity:<15s}"
        for tk in tier_keys:
            scores = [
                j["scores"]["score_compuesto"]
                for j in results["judgments"]
                if j["status"] == "success" and j["tier"] == tk
                and j["complexity"] == complexity
            ]
            avg = sum(scores) / len(scores) if scores else 0
            row += f" {avg:>14.2f}" if scores else f" {'N/A':>14s}"
        print(row)

    # By criterion
    print(f"\n{'Criterio':<25s}", end="")
    for tk in tier_keys:
        print(f" {tk:>15s}", end="")
    print()
    print("-" * (25 + 16 * len(tier_keys)))

    for criterion, info in RUBRIC.items():
        row = f"{info['name']:<25s}"
        for tk in tier_keys:
            scores = [
                j["scores"][criterion]["score"]
                for j in results["judgments"]
                if j["status"] == "success" and j["tier"] == tk
                and criterion in j["scores"]
            ]
            avg = sum(scores) / len(scores) if scores else 0
            row += f" {avg:>14.2f}" if scores else f" {'N/A':>14s}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="MedExpert Arena — Opus Judge")
    parser.add_argument("responses_file", help="Path to responses JSON from arena_runner.py")
    parser.add_argument("--cases", nargs="+", type=int, default=None,
                        help="Case IDs to judge (default: all)")
    parser.add_argument("--tiers", nargs="+", default=None,
                        help="Tiers to judge (default: all in responses file)")
    args = parser.parse_args()

    if not Path(args.responses_file).exists():
        print(f"ERROR: File not found: {args.responses_file}")
        sys.exit(1)

    results = judge_responses(
        args.responses_file,
        case_ids=args.cases,
        tier_keys=args.tiers,
    )

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"opus_scores_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nCalificaciones guardadas en: {out_path}")


if __name__ == "__main__":
    main()
