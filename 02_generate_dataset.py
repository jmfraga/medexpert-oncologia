#!/usr/bin/env python3
"""
Fase 2 — Generación de dataset de entrenamiento con Sonnet como teacher.

Ejecutar en M4 (mlx-env):
  export ANTHROPIC_API_KEY=sk-ant-...
  /Users/jmfraga/mlx-env/bin/python 02_generate_dataset.py --input data/sampled_chunks.json --output-dir data

Genera pares instrucción/respuesta en formato chat JSONL para mlx-lm.
"""

import json
import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ── Logging ──
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("dataset_gen")

# ── Config ──
TEACHER_MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
RETRY_DELAY = 5
BATCH_SIZE = 50  # Save progress every N chunks
RATE_LIMIT_DELAY = 1.2  # seconds between API calls (max 50 req/min Anthropic)

# System prompt for the teacher
SYSTEM_PROMPT_TEACHER = """Eres un generador de datos de entrenamiento para un modelo de IA especializado en oncología médica.

Tu tarea: dado un fragmento de una guía clínica o ficha técnica de oncología, genera pares de pregunta-respuesta clínicos de alta calidad.

REGLAS:
1. Las preguntas deben ser las que haría un oncólogo en práctica clínica real
2. Las respuestas deben basarse ESTRICTAMENTE en la información del fragmento proporcionado
3. NO inventes datos, cifras o recomendaciones que no estén en el fragmento
4. Responde en ESPAÑOL (las guías pueden estar en inglés pero las Q&A deben ser en español)
5. Usa terminología médica apropiada
6. Incluye niveles de evidencia y grados de recomendación cuando estén disponibles
7. Si el fragmento menciona dosis, inclúyelas exactamente como aparecen

FORMATO DE SALIDA (JSON array estricto):
[
  {"type": "conocimiento", "question": "...", "answer": "..."},
  {"type": "caso_clinico", "question": "...", "answer": "..."},
  {"type": "decision", "question": "...", "answer": "..."}
]

Tipos:
- conocimiento: Pregunta directa de conocimiento factual
- caso_clinico: Caso clínico breve que requiere razonamiento (incluye edad, género, estadio)
- decision: Pregunta de comparación o decisión terapéutica entre opciones"""

# System prompt for the fine-tuned model (included in training data)
MODEL_SYSTEM_PROMPT = (
    "Eres MedExpert Onco, un asistente clínico especializado en oncología médica, "
    "entrenado en guías NCCN, ESMO, IMSS y consensos mexicanos. "
    "Responde siempre en español, basándote en evidencia clínica actualizada. "
    "Cita la fuente y nivel de evidencia cuando esté disponible."
)


def create_teacher_prompt(chunk: dict) -> str:
    """Build the prompt for the teacher model."""
    source = chunk.get("source", "Guía clínica")
    society = chunk.get("society", "")
    section = chunk.get("section_path", "")
    stratum = chunk.get("stratum", "")
    text = chunk["text"]

    context_label = source
    if society:
        context_label = "[" + society + "] " + context_label
    if section:
        context_label += " > " + section

    # Adjust instruction based on stratum
    stratum_hints = {
        "tratamiento": "Enfócate en protocolos de tratamiento, esquemas, secuencia terapéutica.",
        "farmacologia": "Enfócate en dosis, vía de administración, efectos adversos, interacciones, contraindicaciones.",
        "diagnostico": "Enfócate en criterios diagnósticos, estadificación, biomarcadores, clasificación.",
        "soporte": "Enfócate en manejo de efectos adversos, cuidado paliativo, calidad de vida.",
        "seguimiento": "Enfócate en protocolos de seguimiento, vigilancia, detección de recurrencia.",
    }
    hint = stratum_hints.get(stratum, "")

    return (
        "FUENTE: " + context_label + "\n"
        "TIPO DE CONTENIDO: " + stratum + "\n\n"
        "FRAGMENTO:\n" + text + "\n\n"
        "Genera 3 pares pregunta-respuesta clínicos basados en este fragmento. "
        + hint + "\n"
        "Responde SOLO con el JSON array, sin texto adicional."
    )


def call_teacher(client, prompt: str) -> list:
    """Call Haiku 4.5 to generate Q&A pairs."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=TEACHER_MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT_TEACHER,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Parse JSON - handle common issues
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            pairs = json.loads(text)
            if isinstance(pairs, list) and len(pairs) > 0:
                return pairs
            log.warning("Empty or invalid response, retrying...")

        except json.JSONDecodeError as e:
            log.warning("JSON parse error (attempt %d): %s", attempt + 1, str(e)[:100])
        except Exception as e:
            log.warning("API error (attempt %d): %s", attempt + 1, str(e)[:100])
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(RETRY_DELAY * (attempt + 2))
            else:
                time.sleep(RETRY_DELAY)

    return []


def format_training_example(question: str, answer: str, source: str, society: str = "") -> dict:
    """Format a Q&A pair as a chat training example for mlx-lm."""
    # Add source attribution to the answer
    if source and not source.lower() in answer.lower():
        society_label = ""
        if society:
            society_label = ", " + society
        answer = answer.rstrip() + "\n\n[Fuente: " + source + society_label + "]"

    return {
        "messages": [
            {"role": "system", "content": MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def estimate_cost(num_chunks: int) -> dict:
    """Estimate API cost for dataset generation."""
    avg_input_tokens = 800   # chunk text + prompt
    avg_output_tokens = 600  # 3 Q&A pairs
    total_input = num_chunks * avg_input_tokens
    total_output = num_chunks * avg_output_tokens

    # Sonnet 4.6 pricing
    input_cost = total_input / 1_000_000 * 3.00   # $3.00/MTok input
    output_cost = total_output / 1_000_000 * 15.00  # $15.00/MTok output
    total_cost = input_cost + output_cost

    return {
        "chunks": num_chunks,
        "est_input_tokens": total_input,
        "est_output_tokens": total_output,
        "est_cost_usd": round(total_cost, 2),
        "est_training_examples": num_chunks * 3,
        "est_time_minutes": round(num_chunks * RATE_LIMIT_DELAY / 60, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate fine-tuning dataset from sampled chunks")
    parser.add_argument("--input", required=True, help="Path to sampled_chunks.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of chunks to process (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from progress file")
    parser.add_argument("--dry-run", action="store_true", help="Only estimate cost, don't call API")
    parser.add_argument("--split", default="80/10/10", help="Train/valid/test split (default: 80/10/10)")
    args = parser.parse_args()

    # Load sampled chunks
    log.info("Loading sampled chunks from %s", args.input)
    with open(args.input) as f:
        data = json.load(f)
    chunks = data["chunks"]
    log.info("Loaded %d chunks", len(chunks))

    if args.limit > 0:
        chunks = chunks[:args.limit]
        log.info("Limited to %d chunks", len(chunks))

    # Cost estimate
    est = estimate_cost(len(chunks))
    log.info("=== COST ESTIMATE ===")
    log.info("  Chunks: %d", est["chunks"])
    log.info("  Est. input tokens: %s", f"{est['est_input_tokens']:,}")
    log.info("  Est. output tokens: %s", f"{est['est_output_tokens']:,}")
    log.info("  Est. cost: $%.2f USD", est["est_cost_usd"])
    log.info("  Est. training examples: %s", f"{est['est_training_examples']:,}")
    log.info("  Est. time: %.0f minutes", est["est_time_minutes"])

    if args.dry_run:
        log.info("Dry run — exiting")
        return

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "progress.json"
    all_examples_file = output_dir / "all_examples.jsonl"

    # Resume support
    processed_ids = set()
    all_examples = []
    if args.resume and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        processed_ids = set(progress.get("processed_ids", []))
        # Reload existing examples
        if all_examples_file.exists():
            with open(all_examples_file) as f:
                for line in f:
                    all_examples.append(json.loads(line))
        log.info("Resuming: %d chunks already processed, %d examples", len(processed_ids), len(all_examples))

    # Init Anthropic client
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # Process chunks
    total_tokens_in = 0
    total_tokens_out = 0
    failed = 0
    start_time = time.time()

    remaining = [c for c in chunks if c["id"] not in processed_ids]
    log.info("Processing %d remaining chunks...", len(remaining))

    for i, chunk in enumerate(remaining):
        prompt = create_teacher_prompt(chunk)
        pairs = call_teacher(client, prompt)

        if not pairs:
            failed += 1
            log.warning("Failed chunk %s (total failed: %d)", chunk["id"][:20], failed)
        else:
            source_label = chunk.get("source", "")
            society_label = chunk.get("society", "")
            for pair in pairs:
                q = pair.get("question", "")
                a = pair.get("answer", "")
                if q and a and len(a) > 30:
                    example = format_training_example(q, a, source_label, society_label)
                    all_examples.append(example)

        processed_ids.add(chunk["id"])

        # Progress logging
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            log.info(
                "Progress: %d/%d (%.1f/min) | Examples: %d | Failed: %d",
                i + 1, len(remaining), rate, len(all_examples), failed,
            )

        # Save checkpoint
        if (i + 1) % BATCH_SIZE == 0:
            with open(progress_file, "w") as f:
                json.dump({"processed_ids": list(processed_ids), "failed": failed}, f)
            with open(all_examples_file, "w") as f:
                for ex in all_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            log.info("Checkpoint saved (%d examples)", len(all_examples))

        time.sleep(RATE_LIMIT_DELAY)

    # ── Final save ──
    log.info("Generation complete. Total examples: %d, Failed chunks: %d", len(all_examples), failed)

    # Shuffle
    random.seed(42)
    random.shuffle(all_examples)

    # Split
    split_parts = [int(x) for x in args.split.split("/")]
    total_ex = len(all_examples)
    train_end = int(total_ex * split_parts[0] / 100)
    valid_end = train_end + int(total_ex * split_parts[1] / 100)

    splits = {
        "train.jsonl": all_examples[:train_end],
        "valid.jsonl": all_examples[train_end:valid_end],
        "test.jsonl": all_examples[valid_end:],
    }

    for filename, examples in splits.items():
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("Wrote %s: %d examples", filename, len(examples))

    # Save all examples too
    with open(all_examples_file, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    elapsed_total = time.time() - start_time
    log.info("=== FINAL STATS ===")
    log.info("  Total chunks processed: %d", len(processed_ids))
    log.info("  Total examples generated: %d", len(all_examples))
    log.info("  Failed chunks: %d (%.1f%%)", failed, failed / max(len(chunks), 1) * 100)
    log.info("  Train: %d | Valid: %d | Test: %d", len(splits["train.jsonl"]), len(splits["valid.jsonl"]), len(splits["test.jsonl"]))
    log.info("  Time: %.1f minutes", elapsed_total / 60)


if __name__ == "__main__":
    main()
