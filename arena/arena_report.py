#!/usr/bin/env python3
"""
MedExpert Arena — Report Generator

Genera:
1. Materiales anonimizados para evaluación humana (Google Forms)
2. CSV resumen de scores Opus (para análisis en R/Python)
3. HTML interactivo con respuestas lado a lado
4. Integración de scores humanos (cuando estén disponibles)
5. Análisis de concordancia Opus vs. humanos

Ejecutar en M1:
    cd ~/Projects/medexpert-admin

    # Generar materiales para Google Forms
    python arena/arena_report.py forms \\
        arena/results/responses_*.json

    # Generar reporte de scores Opus
    python arena/arena_report.py opus-report \\
        arena/results/opus_scores_*.json

    # Integrar scores humanos y generar análisis cruzado
    python arena/arena_report.py full-report \\
        arena/results/opus_scores_*.json \\
        --human-scores arena/results/human_scores.json

Output:
    arena/results/forms/               # Materiales para Google Forms
    arena/results/opus_report.csv      # CSV resumen Opus
    arena/results/opus_report.html     # HTML interactivo
    arena/results/full_report.html     # Reporte completo Opus+humanos
    arena/results/anonymization_map.json
"""

import json
import csv
import random
import argparse
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# Human evaluation scenarios (from METODOLOGIA)
# ─────────────────────────────────────────────

# Case complexity mapping
CASE_COMPLEXITY = {
    1: "simple", 2: "complejo", 3: "complejo", 4: "moderado", 5: "simple",
    6: "complejo", 7: "simple", 8: "moderado", 9: "moderado", 10: "simple",
    11: "complejo", 12: "complejo", 13: "simple", 14: "moderado", 15: "complejo",
}

EVAL_SCENARIOS = {
    3: {
        "description": "3 doctores, 10 casos cada uno, 2 evaluadores por caso",
        "assignments": {
            "D1": [1, 3, 5, 7, 8, 9, 10, 11, 13, 15],
            "D2": [1, 2, 4, 6, 7, 9, 10, 12, 14, 15],
            "D3": [2, 3, 4, 5, 6, 8, 11, 12, 13, 14],
        },
    },
    4: {
        "description": "4 doctores, 8 casos cada uno, ~2 evaluadores por caso",
        "assignments": {
            "D1": [1, 4, 5, 8, 9, 11, 13, 15],
            "D2": [2, 3, 6, 7, 10, 12, 14, 15],
            "D3": [1, 2, 5, 7, 8, 10, 11, 14],
            "D4": [3, 4, 6, 9, 12, 13, 14, 15],
        },
    },
    5: {
        "description": "5 doctores, 6 casos cada uno, 2 evaluadores por caso",
        "assignments": {
            "D1": [1, 4, 7, 9, 11, 14],
            "D2": [2, 5, 8, 10, 12, 15],
            "D3": [3, 6, 7, 9, 13, 14],
            "D4": [1, 4, 8, 10, 11, 15],
            "D5": [2, 3, 5, 6, 12, 13],
        },
    },
    7: {
        "description": "7 doctores, 6-7 casos cada uno, 3 evaluadores por caso",
        "assignments": {
            "D1": [1, 3, 4, 7, 11, 14],
            "D2": [1, 2, 6, 8, 10, 15],
            "D3": [1, 3, 5, 9, 11, 13, 15],
            "D4": [2, 7, 8, 11, 13, 14, 15],
            "D5": [3, 4, 5, 9, 10, 12],
            "D6": [6, 7, 8, 12, 13, 14],
            "D7": [2, 4, 5, 6, 9, 10, 12],
        },
    },
}

TIER_LABELS = ["Candidato A", "Candidato B", "Candidato C", "Candidato D", "Candidato E"]
TIER_KEYS = ["light", "light_rag", "basico_a", "basico_b", "premium"]

CRITERIA = [
    ("precision_diagnostica", "Precisión diagnóstica", 0.25),
    ("apego_guias", "Apego a guías clínicas", 0.30),
    ("completitud", "Completitud", 0.25),
    ("utilidad_clinica", "Utilidad clínica", 0.20),
]


# ─────────────────────────────────────────────
# Anonymization
# ─────────────────────────────────────────────

def generate_anonymization_map(case_ids: list[int], seed: int = 42) -> dict:
    """Generate a per-case randomized mapping of tier → anonymous label.

    Returns {case_id: {tier_key: "Candidato X", ...}, ...}
    """
    anon_map = {}
    for case_id in case_ids:
        rng = random.Random(seed + case_id)  # Reproducible per case
        shuffled = list(TIER_KEYS)
        rng.shuffle(shuffled)
        anon_map[case_id] = {
            tier: TIER_LABELS[i] for i, tier in enumerate(shuffled)
        }
    return anon_map


# ─────────────────────────────────────────────
# Google Forms materials
# ─────────────────────────────────────────────

def generate_forms_materials(responses_path: str, num_doctors: int = 3):
    """Generate anonymized materials for Google Forms human evaluation."""
    with open(responses_path, encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"]
    case_ids = [c["case_id"] for c in cases]
    anon_map = generate_anonymization_map(case_ids)

    scenario = EVAL_SCENARIOS.get(num_doctors)
    if not scenario:
        print(f"ERROR: No scenario for {num_doctors} doctors. Valid: 3, 4, 5")
        sys.exit(1)

    out_dir = Path(__file__).parent / "results" / "forms"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save anonymization map
    map_path = Path(__file__).parent / "results" / "anonymization_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        # Convert int keys to strings for JSON
        serializable = {str(k): v for k, v in anon_map.items()}
        json.dump({
            "seed": 42,
            "generated": datetime.now().isoformat(),
            "mapping": serializable,
            "reverse": {
                str(case_id): {label: tier for tier, label in mapping.items()}
                for case_id, mapping in anon_map.items()
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"Mapa de anonimización guardado en: {map_path}")

    # Generate per-doctor files
    cases_by_id = {c["case_id"]: c for c in cases}

    for doctor_id, assigned_cases in scenario["assignments"].items():
        doc_dir = out_dir / doctor_id
        doc_dir.mkdir(exist_ok=True)

        # Instructions file
        instructions = (
            f"# Evaluación MedExpert Arena — {doctor_id}\n\n"
            f"Estimado(a) evaluador(a),\n\n"
            f"Se le asignan {len(assigned_cases)} casos clínicos oncológicos. "
            f"Para cada caso, recibirá {len(TIER_LABELS)} respuestas generadas por distintos sistemas "
            f"de asistencia clínica (anonimizadas como {', '.join(TIER_LABELS[:-1])} y {TIER_LABELS[-1]}).\n\n"
            f"**No conoce qué sistema generó cada respuesta.**\n\n"
            f"## Rúbrica de evaluación (escala 0-5)\n\n"
            f"Para cada respuesta, califique los siguientes criterios:\n\n"
            f"1. **Precisión diagnóstica** (25%): ¿El diagnóstico y estadificación son correctos?\n"
            f"2. **Apego a guías** (30%): ¿Las recomendaciones son consistentes con NCCN/ESMO/IMSS?\n"
            f"3. **Completitud** (25%): ¿Se cubren todos los aspectos relevantes?\n"
            f"4. **Utilidad clínica** (20%): ¿La respuesta es práctica y accionable?\n\n"
            f"## Escala\n\n"
            f"- 0 = Nulo/incorrecto\n"
            f"- 1 = Muy deficiente\n"
            f"- 2 = Deficiente\n"
            f"- 3 = Aceptable\n"
            f"- 4 = Bueno\n"
            f"- 5 = Excelente\n\n"
            f"## Casos asignados\n\n"
        )

        for i, cid in enumerate(assigned_cases, 1):
            case = cases_by_id.get(cid)
            if not case:
                continue
            instructions += f"{i}. Caso {cid}: {case['title']} ({case['complexity']})\n"

        instructions += (
            f"\n## Tiempo estimado\n\n"
            f"~{len(assigned_cases) * 15}-{len(assigned_cases) * 20} minutos "
            f"({len(assigned_cases)} casos × {len(TIER_LABELS)} respuestas × {len(CRITERIA)} criterios)\n\n"
            f"Gracias por su participación.\n"
        )

        with open(doc_dir / "instrucciones.md", "w", encoding="utf-8") as f:
            f.write(instructions)

        # Per-case files with anonymized responses
        for cid in assigned_cases:
            case = cases_by_id.get(cid)
            if not case:
                continue

            mapping = anon_map[cid]
            case_file = f"caso_{cid:02d}.md"

            content = (
                f"# Caso {cid}: {case['title']}\n"
                f"**Complejidad**: {case['complexity']}\n\n"
                f"## Presentación clínica\n\n"
                f"{case['case_text']}\n\n"
                f"---\n\n"
            )

            # Add responses in anonymous order (A, B, C, D)
            label_to_tier = {label: tier for tier, label in mapping.items()}
            for label in TIER_LABELS:
                tier_key = label_to_tier[label]
                resp = case["responses"].get(tier_key, {})
                resp_text = resp.get("response", "(Sin respuesta)")

                content += (
                    f"## {label}\n\n"
                    f"{resp_text}\n\n"
                    f"---\n\n"
                )

            with open(doc_dir / case_file, "w", encoding="utf-8") as f:
                f.write(content)

        print(f"  {doctor_id}: {len(assigned_cases)} casos → {doc_dir}/")

    # Generate a single consolidated text file for easy copy-paste to Forms
    consolidated = out_dir / "consolidated_all_cases.md"
    with open(consolidated, "w", encoding="utf-8") as f:
        f.write("# MedExpert Arena — Todos los casos (para construir Google Forms)\n\n")
        for cid in sorted(case_ids):
            case = cases_by_id.get(cid)
            if not case:
                continue
            mapping = anon_map[cid]
            label_to_tier = {label: tier for tier, label in mapping.items()}

            f.write(f"{'='*60}\n")
            f.write(f"# CASO {cid}: {case['title']} ({case['complexity']})\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"{case['case_text']}\n\n")

            for label in TIER_LABELS:
                tier_key = label_to_tier[label]
                resp = case["responses"].get(tier_key, {})
                f.write(f"## {label}\n\n{resp.get('response', '(Sin respuesta)')}\n\n---\n\n")

    print(f"\nConsolidado: {consolidated}")
    print(f"Escenario: {scenario['description']}")


# ─────────────────────────────────────────────
# Opus report (CSV + HTML)
# ─────────────────────────────────────────────

def generate_opus_report(scores_path: str):
    """Generate CSV and HTML report from Opus judge scores."""
    with open(scores_path, encoding="utf-8") as f:
        data = json.load(f)

    judgments = [j for j in data["judgments"] if j["status"] == "success"]
    out_dir = Path(__file__).parent / "results"

    # ── CSV ──
    csv_path = out_dir / "opus_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "case_title", "complexity", "tier", "tier_name",
            "precision_diagnostica", "apego_guias", "completitud",
            "utilidad_clinica", "score_compuesto", "judge_time_s",
            "errores_graves", "alucinaciones", "feedback_general",
        ])
        for j in judgments:
            s = j["scores"]
            writer.writerow([
                j["case_id"], j["case_title"], j["complexity"],
                j["tier"], j["tier_name"],
                s["precision_diagnostica"]["score"],
                s["apego_guias"]["score"],
                s["completitud"]["score"],
                s["utilidad_clinica"]["score"],
                s["score_compuesto"],
                j["judge_time_s"],
                "; ".join(s.get("errores_graves", [])),
                "; ".join(s.get("alucinaciones", [])),
                s.get("feedback_general", ""),
            ])
    print(f"CSV: {csv_path}")

    # ── HTML ──
    html_path = out_dir / "opus_report.html"
    html = _build_opus_html(data, judgments)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML: {html_path}")


def _build_opus_html(data: dict, judgments: list) -> str:
    """Build an interactive HTML report for Opus scores."""
    # Group by case
    by_case = defaultdict(list)
    for j in judgments:
        by_case[j["case_id"]].append(j)

    # Tier averages
    tier_avgs = defaultdict(list)
    for j in judgments:
        tier_avgs[j["tier"]].append(j["scores"]["score_compuesto"])

    # Complexity averages
    complexity_tier = defaultdict(lambda: defaultdict(list))
    for j in judgments:
        complexity_tier[j["complexity"]][j["tier"]].append(j["scores"]["score_compuesto"])

    tiers_present = sorted(set(j["tier"] for j in judgments))

    rows_html = ""
    for case_id in sorted(by_case.keys()):
        case_judgments = by_case[case_id]
        first = case_judgments[0]
        rows_html += f'<tr class="case-header"><td colspan="{6 + len(CRITERIA)}">'
        rows_html += f'<strong>Caso {case_id}: {first["case_title"]}</strong> '
        rows_html += f'<span class="badge {first["complexity"]}">{first["complexity"]}</span>'
        rows_html += "</td></tr>\n"

        for j in sorted(case_judgments, key=lambda x: x["tier"]):
            s = j["scores"]
            sc = s["score_compuesto"]
            color_class = "high" if sc >= 4.0 else ("mid" if sc >= 3.0 else "low")
            rows_html += "<tr>"
            rows_html += f'<td>{j["tier_name"]}</td>'
            for crit, _, _ in CRITERIA:
                score = s[crit]["score"]
                rows_html += f'<td class="score">{score}</td>'
            rows_html += f'<td class="score {color_class}"><strong>{sc:.2f}</strong></td>'
            rows_html += f'<td class="feedback">{s.get("feedback_general", "")}</td>'
            rows_html += "</tr>\n"

    # Summary rows
    summary_html = ""
    for tk in tiers_present:
        scores = tier_avgs[tk]
        avg = sum(scores) / len(scores) if scores else 0
        summary_html += f"<tr><td>{tk}</td><td>{avg:.2f}</td><td>{min(scores):.2f}</td>"
        summary_html += f"<td>{max(scores):.2f}</td><td>{len(scores)}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>MedExpert Arena — Opus Judge Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #1a3a5c; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin: 20px 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }}
  th {{ background: #1a3a5c; color: white; }}
  .case-header {{ background: #e8eef4; }}
  .case-header td {{ font-size: 15px; padding: 10px 12px; }}
  .score {{ text-align: center; }}
  .high {{ color: #2d7a3a; font-weight: bold; }}
  .mid {{ color: #b8860b; }}
  .low {{ color: #c0392b; }}
  .feedback {{ font-size: 12px; color: #555; max-width: 300px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; color: white; margin-left: 8px; }}
  .simple {{ background: #27ae60; }}
  .moderado {{ background: #f39c12; }}
  .complejo {{ background: #c0392b; }}
  .meta {{ color: #777; font-size: 13px; }}
</style>
</head>
<body>
<h1>MedExpert Arena — Evaluación Opus</h1>
<p class="meta">Juez: {data.get('judge', 'Opus 4.6')} | {data.get('timestamp', '')}</p>

<h2>Resumen por Tier</h2>
<table>
<tr><th>Tier</th><th>Promedio</th><th>Mín</th><th>Máx</th><th>N</th></tr>
{summary_html}
</table>

<h2>Detalle por Caso</h2>
<table>
<tr>
  <th>Tier</th>
  {''.join(f'<th>{name}</th>' for _, name, _ in CRITERIA)}
  <th>Compuesto</th>
  <th>Feedback</th>
</tr>
{rows_html}
</table>

<p class="meta">Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body>
</html>"""


# ─────────────────────────────────────────────
# Full report: Opus + Human scores
# ─────────────────────────────────────────────

def generate_full_report(opus_path: str, human_path: str):
    """Integrate Opus + human scores and generate analysis.

    human_scores.json format (exported from Google Forms):
    {
        "evaluators": ["D1", "D2", ...],
        "scores": [
            {
                "evaluator": "D1",
                "case_id": 1,
                "candidate_label": "Candidato A",
                "precision_diagnostica": 4,
                "apego_guias": 3,
                "completitud": 4,
                "utilidad_clinica": 5,
                "comments": "..."
            },
            ...
        ]
    }
    """
    with open(opus_path, encoding="utf-8") as f:
        opus_data = json.load(f)
    with open(human_path, encoding="utf-8") as f:
        human_data = json.load(f)

    # Load anonymization map to de-anonymize human scores
    map_path = Path(__file__).parent / "results" / "anonymization_map.json"
    if not map_path.exists():
        print("ERROR: anonymization_map.json not found. Run 'forms' first.")
        sys.exit(1)
    with open(map_path, encoding="utf-8") as f:
        anon_data = json.load(f)
    reverse_map = anon_data["reverse"]  # {case_id_str: {label: tier}}

    # De-anonymize human scores
    human_by_case_tier = defaultdict(lambda: defaultdict(list))
    for entry in human_data["scores"]:
        cid = str(entry["case_id"])
        label = entry["candidate_label"]
        tier = reverse_map.get(cid, {}).get(label)
        if not tier:
            print(f"WARNING: Cannot de-anonymize case {cid} / {label}")
            continue

        weighted = (
            entry["precision_diagnostica"] * 0.25
            + entry["apego_guias"] * 0.30
            + entry["completitud"] * 0.25
            + entry["utilidad_clinica"] * 0.20
        )
        human_by_case_tier[int(cid)][tier].append({
            "evaluator": entry["evaluator"],
            "precision_diagnostica": entry["precision_diagnostica"],
            "apego_guias": entry["apego_guias"],
            "completitud": entry["completitud"],
            "utilidad_clinica": entry["utilidad_clinica"],
            "score_compuesto": round(weighted, 2),
            "comments": entry.get("comments", ""),
        })

    # Build Opus lookup
    opus_by_case_tier = {}
    for j in opus_data["judgments"]:
        if j["status"] == "success":
            opus_by_case_tier[(j["case_id"], j["tier"])] = j["scores"]["score_compuesto"]

    # Correlation analysis
    opus_scores = []
    human_avg_scores = []
    for (cid, tier), opus_sc in opus_by_case_tier.items():
        human_entries = human_by_case_tier.get(cid, {}).get(tier, [])
        if human_entries:
            human_avg = sum(e["score_compuesto"] for e in human_entries) / len(human_entries)
            opus_scores.append(opus_sc)
            human_avg_scores.append(human_avg)

    correlation = _pearson(opus_scores, human_avg_scores) if len(opus_scores) >= 3 else None
    spearman = _spearman(opus_scores, human_avg_scores) if len(opus_scores) >= 3 else None

    # Krippendorff's alpha (simplified — ordinal)
    alpha = _krippendorff_alpha(human_data["scores"])

    # Output
    out_dir = Path(__file__).parent / "results"

    # CSV with both scores
    csv_path = out_dir / "full_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "complexity", "tier",
            "opus_score", "human_avg_score", "human_n_evaluators",
            "human_scores_detail", "difference",
        ])
        for (cid, tier), opus_sc in sorted(opus_by_case_tier.items()):
            human_entries = human_by_case_tier.get(cid, {}).get(tier, [])
            human_avg = sum(e["score_compuesto"] for e in human_entries) / len(human_entries) if human_entries else None
            writer.writerow([
                cid, CASE_COMPLEXITY.get(cid, "?"), tier,
                opus_sc,
                f"{human_avg:.2f}" if human_avg else "N/A",
                len(human_entries),
                "; ".join(f"{e['evaluator']}:{e['score_compuesto']}" for e in human_entries),
                f"{opus_sc - human_avg:.2f}" if human_avg else "N/A",
            ])
    print(f"CSV completo: {csv_path}")

    # Summary to terminal
    print(f"\n{'='*60}")
    print("ANÁLISIS CRUZADO OPUS vs. HUMANOS")
    print(f"{'='*60}")
    print(f"Pares comparados: {len(opus_scores)}")
    if correlation is not None:
        print(f"Pearson r:  {correlation:.3f} {'✓ > 0.80' if correlation > 0.8 else '⚠ < 0.80'}")
    if spearman is not None:
        print(f"Spearman ρ: {spearman:.3f}")
    if alpha is not None:
        print(f"Krippendorff α (inter-evaluador humano): {alpha:.3f} "
              f"{'✓ excelente' if alpha > 0.8 else ('~ aceptable' if alpha > 0.67 else '⚠ baja')}")

    # Discrepancies
    print(f"\nDiscrepancias (|Opus - Humano| > 1.0):")
    for opus_sc, human_sc in zip(opus_scores, human_avg_scores):
        diff = opus_sc - human_sc
        if abs(diff) > 1.0:
            print(f"  Opus: {opus_sc:.2f}, Humano: {human_sc:.2f}, Δ: {diff:+.2f}")

    return {
        "pearson": correlation,
        "spearman": spearman,
        "krippendorff_alpha": alpha,
        "n_pairs": len(opus_scores),
    }


# ─────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────

def _pearson(x: list, y: list) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sx = (sum((xi - mx) ** 2 for xi in x) / (n - 1)) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y) / (n - 1)) ** 0.5
    if sx == 0 or sy == 0:
        return None
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    return cov / (sx * sy)


def _spearman(x: list, y: list) -> float:
    """Spearman rank correlation."""
    def _rank(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(sorted_vals):
            j = i
            while j < len(sorted_vals) and sorted_vals[j][1] == sorted_vals[i][1]:
                j += 1
            avg_rank = (i + j + 1) / 2  # 1-based average
            for k in range(i, j):
                ranks[sorted_vals[k][0]] = avg_rank
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    return _pearson(rx, ry)


def _krippendorff_alpha(scores: list) -> float:
    """Simplified Krippendorff's alpha for ordinal data.

    Input: list of dicts with evaluator, case_id, candidate_label, and score fields.
    """
    # Build matrix: (case_id, candidate_label) → [scores from different evaluators]
    units = defaultdict(list)
    for entry in scores:
        key = (entry["case_id"], entry["candidate_label"])
        weighted = (
            entry["precision_diagnostica"] * 0.25
            + entry["apego_guias"] * 0.30
            + entry["completitud"] * 0.25
            + entry["utilidad_clinica"] * 0.20
        )
        units[key].append(round(weighted, 2))

    # Filter to units with ≥2 evaluators
    multi = {k: v for k, v in units.items() if len(v) >= 2}
    if not multi:
        return None

    # Observed disagreement
    do = 0
    n_pairs = 0
    for scores_list in multi.values():
        for i in range(len(scores_list)):
            for j in range(i + 1, len(scores_list)):
                do += (scores_list[i] - scores_list[j]) ** 2
                n_pairs += 1

    if n_pairs == 0:
        return None

    do /= n_pairs

    # Expected disagreement (all scores pooled)
    all_scores = []
    for scores_list in multi.values():
        all_scores.extend(scores_list)

    de = 0
    total_pairs = 0
    for i in range(len(all_scores)):
        for j in range(i + 1, len(all_scores)):
            de += (all_scores[i] - all_scores[j]) ** 2
            total_pairs += 1

    if total_pairs == 0:
        return None

    de /= total_pairs

    if de == 0:
        return 1.0  # Perfect agreement

    return 1.0 - (do / de)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedExpert Arena — Report Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # forms
    p_forms = subparsers.add_parser("forms", help="Generate materials for Google Forms")
    p_forms.add_argument("responses_file", help="Path to responses JSON")
    p_forms.add_argument("--doctors", type=int, default=3, choices=[3, 4, 5, 7],
                         help="Number of evaluator doctors (default: 3)")

    # opus-report
    p_opus = subparsers.add_parser("opus-report", help="Generate Opus score report")
    p_opus.add_argument("scores_file", help="Path to Opus scores JSON")

    # full-report
    p_full = subparsers.add_parser("full-report", help="Generate Opus + human report")
    p_full.add_argument("opus_file", help="Path to Opus scores JSON")
    p_full.add_argument("--human-scores", required=True, help="Path to human scores JSON")

    args = parser.parse_args()

    if args.command == "forms":
        generate_forms_materials(args.responses_file, args.doctors)
    elif args.command == "opus-report":
        generate_opus_report(args.scores_file)
    elif args.command == "full-report":
        generate_full_report(args.opus_file, args.human_scores)


if __name__ == "__main__":
    main()
