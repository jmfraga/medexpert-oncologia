#!/usr/bin/env python3
"""
Batch generation of Q&A pairs with MiniMax M2.7 (async, parallel).

Usage:
  MINIMAX_API_KEY=sk-api-... /Users/jmfraga/mlx-env/bin/python batch_minimax.py \
    --batch 1 --workers 15

Batch 1 = first half of remaining chunks
Batch 2 = second half
"""

import asyncio
import json
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "data" / "batch_minimax.log"),
    ],
)
log = logging.getLogger("batch")

# ── Same prompts as pilot / 02_generate_dataset.py ──
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


def extract_json(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def normalize_pair(pair: dict) -> dict:
    q = pair.get("question") or pair.get("pregunta", "")
    a = pair.get("answer") or pair.get("respuesta", "")
    t = pair.get("type") or pair.get("tipo", "conocimiento")
    return {"type": t, "question": q, "answer": a}


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


class BatchProcessor:
    def __init__(self, api_key: str, workers: int = 15, max_rpm: int = 200):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.minimax.io/v1")
        self.workers = workers
        self.semaphore = asyncio.Semaphore(workers)
        self.rpm_delay = 60.0 / max_rpm  # min delay between requests
        self.last_request = 0.0

        # Stats
        self.processed = 0
        self.failed = 0
        self.examples = 0
        self.total_in = 0
        self.total_out = 0
        self.start_time = None

        # Output
        self.results = []
        self.processed_ids = set()
        self.lock = asyncio.Lock()

    async def call_minimax(self, prompt: str, retries: int = 3):
        for attempt in range(retries):
            try:
                # Rate limiting
                now = time.time()
                wait = self.rpm_delay - (now - self.last_request)
                if wait > 0:
                    await asyncio.sleep(wait)
                self.last_request = time.time()

                resp = await self.client.chat.completions.create(
                    model="MiniMax-M2.7",
                    max_tokens=4000,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_TEACHER},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = resp.choices[0].message.content.strip()
                text = extract_json(text)
                pairs = json.loads(text)

                if isinstance(pairs, list) and len(pairs) > 0:
                    pairs = [normalize_pair(p) for p in pairs]
                    tok_in = resp.usage.prompt_tokens if resp.usage else 0
                    tok_out = resp.usage.completion_tokens if resp.usage else 0
                    return pairs, tok_in, tok_out

            except json.JSONDecodeError:
                if attempt < retries - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    await asyncio.sleep(5 * (attempt + 1))
                elif attempt < retries - 1:
                    await asyncio.sleep(2)

        return [], 0, 0

    async def process_chunk(self, chunk: dict):
        async with self.semaphore:
            prompt = create_teacher_prompt(chunk)
            pairs, tok_in, tok_out = await self.call_minimax(prompt)

            examples = []
            if pairs:
                source = chunk.get("source", "")
                society = chunk.get("society", "")
                for pair in pairs:
                    q = pair.get("question", "")
                    a = pair.get("answer", "")
                    if q and a and len(a) > 30:
                        examples.append(format_example(q, a, source, society))

            async with self.lock:
                self.processed += 1
                self.total_in += tok_in
                self.total_out += tok_out
                if examples:
                    self.results.extend(examples)
                    self.examples += len(examples)
                else:
                    self.failed += 1
                self.processed_ids.add(chunk["id"])

                if self.processed % 100 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.processed / elapsed * 60
                    cost_in = self.total_in / 1_000_000 * 0.30
                    cost_out = self.total_out / 1_000_000 * 1.20
                    log.info(
                        "Progress: %d/%d (%.0f/min) | Examples: %d | Failed: %d | Cost: $%.2f",
                        self.processed, self.total_chunks, rate, self.examples,
                        self.failed, cost_in + cost_out,
                    )

    async def run(self, chunks: list, output_dir: Path, checkpoint_every: int = 500):
        self.total_chunks = len(chunks)
        self.start_time = time.time()
        output_file = output_dir / "batch_examples.jsonl"
        progress_file = output_dir / "batch_progress.json"

        # Resume support
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
            self.processed_ids = set(progress.get("processed_ids", []))
            if output_file.exists():
                with open(output_file) as f:
                    for line in f:
                        self.results.append(json.loads(line))
                self.examples = len(self.results)
            self.processed = len(self.processed_ids)
            log.info("Resuming: %d already processed, %d examples", self.processed, self.examples)

        remaining = [c for c in chunks if c["id"] not in self.processed_ids]
        log.info("Processing %d remaining chunks with %d workers...", len(remaining), self.workers)

        # Process in batches for checkpointing
        for batch_start in range(0, len(remaining), checkpoint_every):
            batch = remaining[batch_start:batch_start + checkpoint_every]
            tasks = [self.process_chunk(chunk) for chunk in batch]
            await asyncio.gather(*tasks)

            # Checkpoint
            with open(output_file, "w", encoding="utf-8") as f:
                for ex in self.results:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            with open(progress_file, "w") as f:
                json.dump({
                    "processed_ids": list(self.processed_ids),
                    "failed": self.failed,
                    "total_in": self.total_in,
                    "total_out": self.total_out,
                    "timestamp": datetime.now().isoformat(),
                }, f)
            log.info("Checkpoint saved (%d examples, %d processed)", self.examples, self.processed)

        # Final stats
        elapsed = time.time() - self.start_time
        cost_in = self.total_in / 1_000_000 * 0.30
        cost_out = self.total_out / 1_000_000 * 1.20
        log.info("=== BATCH COMPLETE ===")
        log.info("  Processed: %d | Failed: %d", self.processed, self.failed)
        log.info("  Examples: %d", self.examples)
        log.info("  Input tokens: %s", f"{self.total_in:,}")
        log.info("  Output tokens: %s", f"{self.total_out:,}")
        log.info("  Cost: $%.2f (in: $%.2f + out: $%.2f)", cost_in + cost_out, cost_in, cost_out)
        log.info("  Time: %.1f hours", elapsed / 3600)
        log.info("  Rate: %.1f chunks/min", self.processed / elapsed * 60)
        log.info("  Output: %s", output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True, choices=[1, 2], help="Batch number (1=first half, 2=second half)")
    parser.add_argument("--workers", type=int, default=15, help="Concurrent workers (default: 15)")
    parser.add_argument("--max-rpm", type=int, default=200, help="Max requests per minute (default: 200)")
    args = parser.parse_args()

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        log.error("MINIMAX_API_KEY not set")
        sys.exit(1)

    data_dir = Path(__file__).parent / "data"

    # Load all quality chunks
    log.info("Loading all quality chunks...")
    all_chunks = []
    with open(data_dir / "all_quality_chunks.jsonl") as f:
        for line in f:
            all_chunks.append(json.loads(line))
    log.info("Total quality chunks: %d", len(all_chunks))

    # Load already-processed IDs (from original Sonnet dataset)
    with open(data_dir / "sampled_chunks.json") as f:
        existing = json.load(f)
    existing_ids = {c["id"] for c in existing["chunks"]}
    log.info("Already in Sonnet dataset: %d", len(existing_ids))

    # Filter remaining
    remaining = [c for c in all_chunks if c["id"] not in existing_ids]
    log.info("Remaining to process: %d", len(remaining))

    # Split into batches
    mid = len(remaining) // 2
    if args.batch == 1:
        chunks = remaining[:mid]
        output_dir = data_dir / "batch1"
    else:
        chunks = remaining[mid:]
        output_dir = data_dir / "batch2"

    output_dir.mkdir(exist_ok=True)
    log.info("Batch %d: %d chunks", args.batch, len(chunks))

    # Estimate cost
    cost_est = len(chunks) * 0.00104  # from pilot
    log.info("Estimated cost: $%.2f USD", cost_est)

    # Run
    processor = BatchProcessor(api_key, workers=args.workers, max_rpm=args.max_rpm)
    asyncio.run(processor.run(chunks, output_dir))


if __name__ == "__main__":
    main()
