#!/usr/bin/env python3
"""
MedExpert Arena — CandidateRunner

Ejecuta los 4 tiers (Light, Básico A, Básico B, Premium) contra los 15 casos
clínicos y guarda respuestas + métricas de rendimiento.

Ejecutar en M1 (donde está ChromaDB + bot_brain.py):
    cd ~/Projects/medexpert-admin
    python arena/arena_runner.py                    # Todos los casos, todos los tiers
    python arena/arena_runner.py --cases 1 2 3      # Casos específicos
    python arena/arena_runner.py --tiers light premium  # Tiers específicos
    python arena/arena_runner.py --dry-run           # Solo muestra qué haría

Requiere:
    - ChromaDB oncología con 134K+ chunks
    - API keys en .env: ANTHROPIC_API_KEY, OPENAI_API_KEY (MiniMax via OpenRouter)
    - llama8b-onco corriendo en MLX en M4 (<MLX_HOST>) — solo para tier light
    - MedGemma 27B accesible via Ollama/API — solo para tier basico_a

Output:
    arena/results/responses_YYYYMMDD_HHMMSS.json
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add parent dir so we can import bot_brain, rag_engine, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

import database as db
from rag_engine import get_rag_for_expert
from bot_brain import (
    BotBrain, DEFAULT_SYSTEM_PROMPT,
    _translate_query_to_english, _diversify_results, _detect_society,
    _expand_synonyms,
)

logging.basicConfig(
    format="%(asctime)s [arena-runner] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("arena-runner")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# ─────────────────────────────────────────────
# Tier definitions
# ─────────────────────────────────────────────

TIERS = {
    "light": {
        "name": "Light (llama8b-onco)",
        "description": "Fine-tuned llama8b-onco sin RAG — conocimiento internalizado",
        "provider": "ollama",  # MLX served via ollama-compatible endpoint on M4
        "model": "llama8b-onco",
        "uses_rag": False,
        "base_url": "http://<MLX_HOST>:8080/v1",  # MLX server on M4
        "max_tokens": 2000,
    },
    "light_rag": {
        "name": "Light+RAG (llama8b-onco + RAG)",
        "description": "Fine-tuned llama8b-onco CON RAG — conocimiento internalizado + retrieval",
        "provider": "ollama",
        "model": "llama8b-onco",
        "uses_rag": True,
        "base_url": "http://<MLX_HOST>:8080/v1",
        "max_tokens": 2000,
    },
    "basico_a": {
        "name": "Básico A (MedGemma 27B)",
        "description": "Modelo médico pre-entrenado + RAG",
        "provider": "ollama",
        "model": "mlx-community/medgemma-27b-text-it-4bit",
        "uses_rag": True,
        "base_url": "http://<MLX_HOST>:8092/v1",  # MLX server on M4
        "max_tokens": 2000,
    },
    "basico_b": {
        "name": "Básico B (MiniMax 2.7)",
        "description": "Modelo generalista competitivo + RAG",
        "provider": "openai",  # Via OpenRouter or direct API
        "model": "minimax/minimax-m1-80k",
        "uses_rag": True,
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/v1",
        "max_tokens": 2000,
    },
    "premium": {
        "name": "Premium (Sonnet 4.6)",
        "description": "Modelo frontier generalista + RAG",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6-20250514",
        "uses_rag": True,
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 2000,
    },
}

EXPERT_SLUG = "oncologia"


# ─────────────────────────────────────────────
# RAG context builder (reused from llm_benchmark)
# ─────────────────────────────────────────────

def build_rag_context(query: str) -> tuple[str, list[str], list[dict]]:
    """Build RAG context using bilingual search.

    Returns (context_string, citations_list, raw_hits).
    """
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
    citations = []
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
        if label not in citations:
            citations.append(label)

    context = "\n\n".join(context_parts) if context_parts else "(Sin guias disponibles)"
    return context, citations, diverse_hits


# ─────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────

def _get_client(tier_cfg: dict):
    """Create an LLM client for the given tier."""
    provider = tier_cfg["provider"]

    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.getenv(tier_cfg.get("api_key_env", "ANTHROPIC_API_KEY")))

    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(
            api_key=os.getenv(tier_cfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url=tier_cfg.get("base_url"),
        )

    elif provider == "ollama":
        from openai import OpenAI
        return OpenAI(
            base_url=tier_cfg.get("base_url", "http://localhost:11434/v1"),
            api_key="ollama",
        )

    elif provider == "groq":
        from openai import OpenAI
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv(tier_cfg.get("api_key_env", "GROQ_API_KEY")),
        )


def _call_tier(client, tier_key: str, tier_cfg: dict,
               system: str, user_msg: str) -> dict:
    """Call a tier's LLM and return response + metrics."""
    provider = tier_cfg["provider"]
    model_id = tier_cfg["model"]
    max_tokens = tier_cfg.get("max_tokens", 2000)
    start = time.time()

    try:
        if provider == "anthropic":
            resp = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                timeout=120.0,
            )
            text = resp.content[0].text
            tokens_in = resp.usage.input_tokens if resp.usage else 0
            tokens_out = resp.usage.output_tokens if resp.usage else 0

        else:
            # OpenAI-compatible (OpenAI, Ollama, Groq, MiniMax)
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                timeout=120.0,
            )
            text = resp.choices[0].message.content or ""
            tokens_in = getattr(resp.usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(resp.usage, "completion_tokens", 0) or 0

        elapsed = time.time() - start
        return {
            "status": "success",
            "response": text,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "response_time_s": round(elapsed, 2),
            "tokens_per_second": round(tokens_out / elapsed, 1) if elapsed > 0 else 0,
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[{tier_key}] Error: {e}")
        return {
            "status": "error",
            "response": f"ERROR: {e}",
            "tokens_input": 0,
            "tokens_output": 0,
            "response_time_s": round(elapsed, 2),
            "tokens_per_second": 0,
            "error": str(e),
        }


# ─────────────────────────────────────────────
# Build prompts
# ─────────────────────────────────────────────

def build_prompts(case_text: str, rag_context: str = None) -> tuple[str, str]:
    """Build system + user prompts for a clinical case.

    If rag_context is None (light tier), the system prompt omits RAG.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")

    if rag_context:
        system = DEFAULT_SYSTEM_PROMPT.format(
            current_date=current_date,
            rag_context=rag_context,
        )
    else:
        # Light tier: no RAG, rely on internalized knowledge
        system = (
            "Eres MedExpert Onco, un asistente clínico especializado en oncología médica.\n"
            f"Fecha: {current_date}\n\n"
            "INSTRUCCIONES:\n"
            "- Responde en español médico profesional\n"
            "- Basa tus respuestas en tu conocimiento de guías clínicas\n"
            "- Cita las fuentes cuando sea posible\n"
            "- Si no tienes información suficiente, indícalo claramente\n"
            "- NO inventes datos clínicos ni estadísticas"
        )

    user_msg = (
        "Consulta clínica:\n\n"
        f"{case_text}\n\n"
        "ESTRUCTURA SAER (OBLIGATORIO):\n"
        "Organiza tu respuesta así:\n\n"
        "SITUACIÓN:\n"
        "Resumen del caso o pregunta con datos clínicos relevantes\n\n"
        "ANTECEDENTES:\n"
        "Contexto clínico: epidemiología, fisiopatología, clasificación, factores pronósticos\n\n"
        "EVALUACIÓN:\n"
        "Análisis basado en guías clínicas con niveles de evidencia. "
        "Cita fuentes: [NCCN 2024], [ESMO], [CMCM 2025]\n\n"
        "RECOMENDACIONES:\n"
        "Conducta sugerida: opciones terapéuticas, dosis, seguimiento\n\n"
        "GUÍAS CONSULTADAS:\n"
        "Lista de guías citadas con año"
    )

    return system, user_msg


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def load_cases(cases_path: str = None) -> list[dict]:
    """Load clinical cases from JSON."""
    if cases_path is None:
        cases_path = str(Path(__file__).parent / "clinical_cases.json")
    with open(cases_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def run_arena(case_ids: list[int] = None, tier_keys: list[str] = None,
              dry_run: bool = False) -> dict:
    """Run the arena: all tiers × selected cases.

    Returns a results dict ready for JSON export.
    """
    db.init_db()
    cases = load_cases()

    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]

    if tier_keys is None:
        tier_keys = list(TIERS.keys())

    # Validate tiers
    for tk in tier_keys:
        if tk not in TIERS:
            logger.error(f"Unknown tier: {tk}. Valid: {list(TIERS.keys())}")
            sys.exit(1)

    logger.info(f"Arena: {len(cases)} cases × {len(tier_keys)} tiers = {len(cases) * len(tier_keys)} evaluations")

    if dry_run:
        for c in cases:
            for tk in tier_keys:
                print(f"  [DRY-RUN] Case {c['id']} ({c['complexity']}) × {TIERS[tk]['name']}")
        return {}

    # Initialize clients
    clients = {}
    for tk in tier_keys:
        try:
            clients[tk] = _get_client(TIERS[tk])
            logger.info(f"Client ready: {TIERS[tk]['name']}")
        except Exception as e:
            logger.error(f"Cannot initialize {tk}: {e}")
            sys.exit(1)

    results = {
        "arena_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "tiers": {tk: TIERS[tk]["name"] for tk in tier_keys},
        "total_cases": len(cases),
        "total_evaluations": len(cases) * len(tier_keys),
        "cases": [],
    }

    for case in cases:
        case_id = case["id"]
        print(f"\n{'='*70}")
        print(f"CASO {case_id}: {case['title']} ({case['complexity']})")
        print(f"{'='*70}")

        # Build RAG context once (shared by all tiers that use RAG)
        rag_context = None
        rag_citations = []
        rag_time = 0
        needs_rag = any(TIERS[tk]["uses_rag"] for tk in tier_keys)

        if needs_rag:
            rag_start = time.time()
            rag_context, rag_citations, _ = build_rag_context(case["case_text"])
            rag_time = round(time.time() - rag_start, 2)
            logger.info(f"RAG: {len(rag_citations)} citations ({rag_time}s)")

        case_result = {
            "case_id": case_id,
            "title": case["title"],
            "complexity": case["complexity"],
            "case_text": case["case_text"],
            "gold_standard": case["gold_standard"],
            "rag_time_s": rag_time,
            "rag_citations": rag_citations,
            "responses": {},
        }

        for tk in tier_keys:
            tier_cfg = TIERS[tk]
            uses_rag = tier_cfg["uses_rag"]

            # Build prompts
            system, user_msg = build_prompts(
                case["case_text"],
                rag_context=rag_context if uses_rag else None,
            )

            logger.info(f"  Calling {tier_cfg['name']}...")
            response = _call_tier(clients[tk], tk, tier_cfg, system, user_msg)

            case_result["responses"][tk] = {
                "tier": tk,
                "tier_name": tier_cfg["name"],
                "uses_rag": uses_rag,
                **response,
            }

            status = "OK" if response["status"] == "success" else "ERR"
            print(f"  {tier_cfg['name']:30s} {status} {response['response_time_s']:.1f}s "
                  f"({response['tokens_input']}in/{response['tokens_output']}out) "
                  f"{response['tokens_per_second']} t/s")

        results["cases"].append(case_result)

    # Summary
    _print_summary(results, tier_keys)

    return results


def _print_summary(results: dict, tier_keys: list[str]):
    """Print a summary table."""
    print(f"\n\n{'='*70}")
    print("RESUMEN DE ARENA")
    print(f"{'='*70}")

    header = f"{'Caso':<40s}"
    for tk in tier_keys:
        header += f" {TIERS[tk]['name']:>20s}"
    print(header)
    print("-" * len(header))

    for case in results["cases"]:
        row = f"{case['title'][:38]:<40s}"
        for tk in tier_keys:
            r = case["responses"].get(tk, {})
            t = r.get("response_time_s", 0)
            s = "" if r.get("status") == "success" else " ERR"
            row += f" {t:>17.1f}s{s}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'PROMEDIO':<40s}"
    for tk in tier_keys:
        times = [c["responses"].get(tk, {}).get("response_time_s", 0)
                 for c in results["cases"]
                 if c["responses"].get(tk, {}).get("status") == "success"]
        avg = sum(times) / len(times) if times else 0
        avg_row += f" {avg:>17.1f}s"
    print(avg_row)

    # Success rate
    ok_row = f"{'ÉXITO':<40s}"
    for tk in tier_keys:
        oks = sum(1 for c in results["cases"]
                  if c["responses"].get(tk, {}).get("status") == "success")
        ok_row += f" {oks:>14d}/{len(results['cases']):<3d}"
    print(ok_row)


def main():
    parser = argparse.ArgumentParser(description="MedExpert Arena — CandidateRunner")
    parser.add_argument("--cases", nargs="+", type=int, default=None,
                        help="Case IDs to run (default: all 15)")
    parser.add_argument("--tiers", nargs="+", default=None,
                        choices=list(TIERS.keys()),
                        help="Tiers to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    results = run_arena(
        case_ids=args.cases,
        tier_keys=args.tiers,
        dry_run=args.dry_run,
    )

    if not results:
        return

    # Save results
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"responses_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResultados guardados en: {out_path}")
    print(f"Total: {results['total_evaluations']} evaluaciones")


if __name__ == "__main__":
    main()
