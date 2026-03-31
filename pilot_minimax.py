#!/usr/bin/env python3
"""
Piloto: generar Q&A pairs con MiniMax M2.7 para comparar vs Sonnet 4.6.

Toma 50 chunks del sampled_chunks.json existente y genera Q&A con MiniMax.
Luego muestra comparación lado a lado con los pares generados por Sonnet.

Uso:
  MINIMAX_API_KEY=sk-api-... /Users/jmfraga/mlx-env/bin/python pilot_minimax.py
"""

import json
import os
import sys
import time
import random
import logging
from pathlib import Path
from openai import OpenAI

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("pilot")

# ── Same prompts as 02_generate_dataset.py ──
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

MODEL_SYSTEM_PROMPT = (
    "Eres MedExpert Onco, un asistente clínico especializado en oncología médica, "
    "entrenado en guías NCCN, ESMO, IMSS y consensos mexicanos. "
    "Responde siempre en español, basándote en evidencia clínica actualizada. "
    "Cita la fuente y nivel de evidencia cuando esté disponible."
)

STRATUM_HINTS = {
    "tratamiento": "Enfócate en protocolos de tratamiento, esquemas, secuencia terapéutica.",
    "farmacologia": "Enfócate en dosis, vía de administración, efectos adversos, interacciones, contraindicaciones.",
    "diagnostico": "Enfócate en criterios diagnósticos, estadificación, biomarcadores, clasificación.",
    "soporte": "Enfócate en manejo de efectos adversos, cuidado paliativo, calidad de vida.",
    "seguimiento": "Enfócate en protocolos de seguimiento, vigilancia, detección de recurrencia.",
}


def create_teacher_prompt(chunk: dict) -> str:
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

    hint = STRATUM_HINTS.get(stratum, "")

    return (
        "FUENTE: " + context_label + "\n"
        "TIPO DE CONTENIDO: " + stratum + "\n\n"
        "FRAGMENTO:\n" + text + "\n\n"
        "Genera 3 pares pregunta-respuesta clínicos basados en este fragmento. "
        + hint + "\n"
        "Responde SOLO con el JSON array, sin texto adicional."
    )


def extract_json_from_response(text: str) -> str:
    """Strip <think> tags and markdown fences from MiniMax reasoning output."""
    # Strip reasoning block
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return text


def normalize_pair(pair: dict) -> dict:
    """Normalize key names (MiniMax sometimes uses Spanish keys)."""
    q = pair.get("question") or pair.get("pregunta", "")
    a = pair.get("answer") or pair.get("respuesta", "")
    t = pair.get("type") or pair.get("tipo", "conocimiento")
    return {"type": t, "question": q, "answer": a}


def call_minimax(client: OpenAI, prompt: str, retries: int = 3) -> list:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="MiniMax-M2.7",
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEACHER},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            text = extract_json_from_response(text)

            pairs = json.loads(text)
            if isinstance(pairs, list) and len(pairs) > 0:
                pairs = [normalize_pair(p) for p in pairs]
                tokens_in = resp.usage.prompt_tokens if resp.usage else 0
                tokens_out = resp.usage.completion_tokens if resp.usage else 0
                return pairs, tokens_in, tokens_out

        except json.JSONDecodeError as e:
            log.warning("JSON parse error (attempt %d): %s | text: %s", attempt + 1, str(e)[:80], text[:100])
        except Exception as e:
            log.warning("API error (attempt %d): %s", attempt + 1, str(e)[:200])
            time.sleep(2 * (attempt + 1))

    return [], 0, 0


def format_example(question, answer, source, society=""):
    if source and source.lower() not in answer.lower():
        soc = ", " + society if society else ""
        answer = answer.rstrip() + "\n\n[Fuente: " + source + soc + "]"
    return {
        "messages": [
            {"role": "system", "content": MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        log.error("MINIMAX_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.minimax.io/v1")

    # Load sampled chunks
    chunks_file = Path(__file__).parent / "data" / "sampled_chunks.json"
    log.info("Loading chunks from %s", chunks_file)
    with open(chunks_file) as f:
        data = json.load(f)
    all_chunks = data["chunks"]
    log.info("Total chunks available: %d", len(all_chunks))

    # Sample 50 random chunks (fixed seed for reproducibility)
    random.seed(2026)
    pilot_chunks = random.sample(all_chunks, 50)
    log.info("Selected %d chunks for pilot", len(pilot_chunks))

    # Process
    results = []
    total_in = 0
    total_out = 0
    failed = 0

    for i, chunk in enumerate(pilot_chunks):
        prompt = create_teacher_prompt(chunk)
        pairs, tok_in, tok_out = call_minimax(client, prompt)

        if not pairs:
            failed += 1
            log.warning("Failed chunk %d/%d: %s", i + 1, 50, chunk.get("id", "?")[:30])
        else:
            total_in += tok_in
            total_out += tok_out
            for pair in pairs:
                q = pair.get("question", "")
                a = pair.get("answer", "")
                if q and a and len(a) > 30:
                    ex = format_example(q, a, chunk.get("source", ""), chunk.get("society", ""))
                    results.append(ex)

        if (i + 1) % 10 == 0:
            log.info("Progress: %d/50 | Examples: %d | Failed: %d", i + 1, len(results), failed)

        time.sleep(0.5)  # Light rate limiting

    # Save results
    out_file = Path(__file__).parent / "data" / "pilot_minimax_50.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Also save chunk IDs for cross-reference
    meta_file = Path(__file__).parent / "data" / "pilot_minimax_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "pilot_chunk_ids": [c["id"] for c in pilot_chunks],
            "total_examples": len(results),
            "failed_chunks": failed,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "cost_input": round(total_in / 1_000_000 * 0.30, 4),
            "cost_output": round(total_out / 1_000_000 * 1.20, 4),
            "cost_total": round(total_in / 1_000_000 * 0.30 + total_out / 1_000_000 * 1.20, 4),
        }, f, indent=2)

    # Print summary
    cost_in = total_in / 1_000_000 * 0.30
    cost_out = total_out / 1_000_000 * 1.20
    log.info("=== PILOT COMPLETE ===")
    log.info("  Chunks processed: %d/50", 50 - failed)
    log.info("  Examples generated: %d", len(results))
    log.info("  Failed: %d", failed)
    log.info("  Input tokens: %s", f"{total_in:,}")
    log.info("  Output tokens: %s", f"{total_out:,}")
    log.info("  Cost: $%.4f (in: $%.4f + out: $%.4f)", cost_in + cost_out, cost_in, cost_out)
    log.info("  Output file: %s", out_file)

    # Print 3 sample examples for quick review
    log.info("\n=== SAMPLE MINIMAX EXAMPLES ===")
    for ex in results[:3]:
        msgs = ex["messages"]
        log.info("---")
        log.info("Q: %s", msgs[1]["content"][:200])
        log.info("A: %s", msgs[2]["content"][:300])


if __name__ == "__main__":
    main()
