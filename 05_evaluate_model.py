#!/usr/bin/env python3
"""
Fase 5 — Evaluación del modelo fine-tuned vs base.

Ejecutar en M4:
  cd /Users/jmfraga/Projects/medexpert-oncologia
  /Users/jmfraga/mlx-env/bin/python 05_evaluate_model.py

Compara respuestas del modelo base vs fine-tuned en preguntas de oncología.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path("/Users/jmfraga/Projects/medexpert-oncologia")
MLX_BIN = "/Users/jmfraga/mlx-env/bin/python"
BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
FUSED_MODEL = str(PROJECT_DIR / "models" / "Llama8B-MedExpert-Oncologia")

# Quick evaluation questions
EVAL_QUESTIONS = [
    "¿Cuál es el tratamiento de primera línea para cáncer de mama HER2+ estadio II?",
    "¿Qué dosis de trastuzumab se recomienda y cada cuánto se administra?",
    "¿Cuándo está indicado osimertinib en cáncer de pulmón?",
    "¿Cuál es el esquema adyuvante estándar para cáncer de colon estadio III?",
    "¿Qué rol tiene pembrolizumab en cáncer colorrectal MSI-H metastásico?",
    "Paciente de 55 años con melanoma BRAF V600E y metástasis cerebrales. ¿Tratamiento?",
    "¿Cuáles son los efectos adversos principales de cisplatino?",
    "¿Cuándo se recomienda olaparib en cáncer de ovario?",
    "¿Qué es el esquema FOLFIRINOX y cuándo se usa?",
    "¿Cuáles son los criterios de respuesta molecular en leucemia mieloide crónica?",
]

SYSTEM_PROMPT = (
    "Eres MedExpert Onco, un asistente clínico especializado en oncología médica. "
    "Respondes consultas basándote en guías clínicas internacionales. "
    "Respondes en español de forma clara y concisa."
)


def generate_response(model_path: str, question: str, max_tokens: int = 500) -> str:
    """Generate a response using mlx_lm.generate."""
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + SYSTEM_PROMPT
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + question
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    cmd = [
        MLX_BIN, "-m", "mlx_lm.generate",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        # Extract just the generated text (after the prompt echo)
        if "<|start_header_id|>assistant<|end_header_id|>" in output:
            output = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return output
    except subprocess.TimeoutExpired:
        return "(TIMEOUT)"
    except Exception as e:
        return "(ERROR: " + str(e) + ")"


def main():
    pilot = "--pilot" in sys.argv
    if pilot:
        fused = FUSED_MODEL + "-pilot"
    else:
        fused = FUSED_MODEL

    if not Path(fused).exists():
        print("ERROR: Fused model not found at", fused)
        print("Run 04_fuse_model.sh first.")
        sys.exit(1)

    results = []
    print("=" * 80)
    print("EVALUACIÓN: Base vs Fine-tuned")
    print("=" * 80)

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print("\n--- Pregunta", i, "---")
        print("Q:", question)

        print("\n[BASE MODEL]")
        base_response = generate_response(BASE_MODEL, question)
        print(base_response[:500])

        print("\n[FINE-TUNED]")
        ft_response = generate_response(fused, question)
        print(ft_response[:500])

        results.append({
            "question": question,
            "base_response": base_response,
            "finetuned_response": ft_response,
        })

    # Save results
    suffix = "-pilot" if pilot else ""
    output_file = PROJECT_DIR / "data" / ("eval_results" + suffix + ".json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n\nResults saved to", output_file)


if __name__ == "__main__":
    main()
