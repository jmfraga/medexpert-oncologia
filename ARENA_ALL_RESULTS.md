# MedExpert Arena — Resultados Consolidados

Todas las evaluaciones realizadas entre 26-mar y 5-abr-2026.
15 casos clínicos oncológicos, judge: Claude Opus 4 (evaluación ciega).
Hardware: M4 64GB (MLX local), M1 (ChromaDB + admin).

---

## Tabla Maestra de Scores

| Arena | Modelo | Params | FT? | RAG? | Score /5 | Prec.Dx | Guías | Complet. | Utilidad | tok/s | Costo/eval |
|-------|--------|--------|-----|------|----------|---------|-------|----------|----------|-------|------------|
| **Matrix** | Gemma4 31B base | 31B | No | No | **4.13** | 4.9 | 3.7 | 3.9 | 4.3 | ~12 | $0 |
| **Final** | Sonnet 4.6 | — | No | No | **4.48** | 4.67 | 4.20 | 4.27 | 4.87 | API | ~$0.05 |
| **Matrix** | Gemma4 26B MoE base | 26B (4B act.) | No | No | **3.51** | 4.7 | 3.0 | 3.2 | 3.7 | ~70 | $0 |
| **v3** | Sonnet + RAG | — | No | Yes | 2.94 | 4.40 | 2.27 | 2.67 | 3.13 | API | ~$0.05 |
| **v3** | ChatGPT sin RAG | — | No | No | 3.00 | 4.20 | 2.80 | 2.27 | 3.13 | API | ~$0.03 |
| **v3** | MiniMax sin RAG | — | No | No | 2.95 | 4.07 | 2.47 | 2.47 | 3.20 | API | ~$0.01 |
| **v3** | ChatGPT + RAG | — | No | Yes | 2.75 | 3.93 | 2.67 | 2.07 | 2.73 | API | ~$0.03 |
| **v3** | MiniMax + RAG | — | No | Yes | 2.73 | 3.80 | 2.67 | 2.07 | 2.67 | API | ~$0.01 |
| **Final** | Gemma4 26B MoE + RAG | 26B | No | Yes | 2.63 | 4.13 | 2.33 | 2.27 | 2.33 | ~40 | $0 |
| **Matrix** | Gemma4 26B MoE + RAG | 26B | No | Yes | 2.55 | 4.0 | 2.3 | 2.1 | 2.3 | ~40 | $0 |
| **Final** | Gemma4 31B + RAG | 31B | No | Yes | 2.51 | 4.00 | 2.13 | 2.07 | 2.27 | ~12 | $0 |
| **Matrix** | Gemma4 31B + RAG | 31B | No | Yes | 2.52 | 4.0 | 2.2 | 2.1 | 2.3 | ~12 | $0 |
| **Matrix** | Gemma4 26B MoE FT | 26B | Yes | No | 2.30 | 3.9 | 1.9 | 1.8 | 2.1 | ~40 | $0 |
| **Final** | Gemma4 26B MoE FT | 26B | Yes | No | 2.27 | 3.93 | 1.93 | 1.73 | 2.00 | ~40 | $0 |
| **Matrix** | Gemma4 31B FT | 31B | Yes | No | 1.71 | 3.1 | 1.4 | 1.3 | 1.3 | ~12 | $0 |
| **Final** | Gemma4 31B FT | 31B | Yes | No | 1.70 | 3.07 | 1.33 | 1.33 | 1.33 | ~12 | $0 |
| **Matrix** | Gemma4 26B MoE FT+RAG | 26B | Yes | Yes | 1.59 | 2.9 | 1.2 | 1.3 | 1.3 | ~30 | $0 |
| **Matrix** | Gemma4 31B FT+RAG | 31B | Yes | Yes | 1.49 | 2.8 | 1.1 | 1.3 | 1.1 | ~6 | $0 |
| **Final** | gpt-oss 20B + RAG | 20B | No | Yes | 0.68 | 1.07 | 0.40 | 0.80 | 0.53 | ~64 | $0 |
| **v1** | llama8b-onco FT | 8B | Yes | No | 0.64 | — | — | — | — | ~53 | $0 |
| **v1** | gpt-oss 20B base | 20B | No | No | 0.51 | — | — | — | — | ~64 | $0 |
| **v1** | llama8b-onco FT+RAG | 8B | Yes | Yes | 0.44 | — | — | — | — | ~53 | $0 |
| **Final** | gpt-oss 20B FT | 20B | Yes | No | 0.14 | 0.27 | 0.20 | 0.07 | 0.07 | ~64 | $0 |

---

## Resumen por Arena

### Arena v1 (26-mar-2026) — 5 tiers, dataset 34K
Primera evaluación. Todos los modelos locales insuficientes.
- Sonnet 3.12, MiniMax 2.62, llama8b-onco 0.64, gpt-oss-20b 0.51

### Arena v2/v2b (27-mar-2026) — 6 tiers, dataset 34K
Fine-tuned vs base+RAG. FT generaba respuestas demasiado cortas (~180 tok).
- Sonnet 3.17, MiniMax 2.75, locales <1.37

### Arena v3 (31-mar-2026) — 6 tiers, RAG delta
Descubrimiento: RAG degrada TODOS los modelos (Sonnet 4.48→2.94, ChatGPT 3.00→2.75, MiniMax 2.95→2.73).

### Arena Final (5-abr-2026) — 7 tiers, dataset 142K
Fine-tuned (Gemma4 31B, 26B MoE, gpt-oss 20B) vs base+RAG vs Sonnet.
- Sonnet 4.48 domina. FT locales 1.70-2.27. gpt-oss desastre (0.14).

### Arena Matrix (5-abr-2026) — 8 tiers, Gemma4 only
Matriz 2×2 completa (base/FT × RAG/noRAG) para Gemma4 26B MoE y 31B.
- **31B base sin RAG: 4.13** — cerca de Sonnet (4.48)
- **26B MoE base sin RAG: 3.51** — mejor modelo local rápido
- RAG degrada todo. FT degrada todo. FT+RAG es lo peor.

---

## Conclusiones Principales

### 1. Los modelos base ya saben oncología
Gemma4 31B base (4.13) y 26B MoE base (3.51) sin ninguna intervención superan a ChatGPT (3.00) y MiniMax (2.95). El conocimiento médico ya está en los parámetros.

### 2. RAG hace más daño que bien (con nuestra pipeline actual)
En TODAS las comparaciones, RAG baja el score:
- Sonnet: 4.48 → 2.94 (-1.54)
- Gemma4 31B: 4.13 → 2.52 (-1.61)
- Gemma4 26B: 3.51 → 2.55 (-0.96)

### 3. Fine-tuning no agrega valor (con nuestro dataset/config actual)
- Gemma4 31B: 4.13 → 1.71 (-2.42)
- Gemma4 26B: 3.51 → 2.30 (-1.21)
El FT reduce completitud y utilidad dramáticamente.

### 4. La combinación FT+RAG es lo peor
- Gemma4 31B: 4.13 → 1.49 (-2.64)
- Gemma4 26B: 3.51 → 1.59 (-1.92)

### 5. Para producción: modelo base + buen prompt
El valor de MedExpert no está en RAG ni en fine-tuning, sino en:
- Selección del mejor modelo base
- Prompt engineering (formato SAER, instrucciones claras)
- Escalamiento a Sonnet/Opus para casos complejos

### 6. Velocidad importa para UX
| Modelo | tok/s | Viable para bot? |
|--------|-------|------------------|
| Gemma4 26B MoE | ~70 | ✅ Sí (18s/caso) |
| Gemma4 31B | ~12 | ❌ No (100s/caso) |
| Sonnet API | ~30 | ✅ Sí (~35s/caso) |

---

## Costos Totales de Evaluación

| Arena | Tiers | Evaluaciones | Costo Judge | Costo Runner |
|-------|-------|-------------|-------------|-------------|
| v1 | 5 | 75 | ~$10 | ~$3 |
| v2/v2b | 6 | 90 | ~$10 | ~$5 |
| v3 | 6 | 90 | $10.51 | ~$5 |
| Final | 7 | 105 | $12.55 | ~$2 |
| Matrix | 8 | 120 | $2.87 | $0 |
| **Total** | — | **480** | **~$46** | **~$15** |

---

*Generado 5-abr-2026. Judge: Claude Opus 4. Hardware: M4 64GB (MLX), M1 (ChromaDB).*
