# Dataset Quality Funnel Report

**Fecha**: 2026-04-02 (stages 1-6: 2026-03-31, stage 7: 2026-04-01/02, stage 8: 2026-04-02)
**Script**: `02b_clean_dataset.py`
**Duracion total**: ~21 horas (stages 1-6: ~67 min, stage 7: ~20.5h, stage 8: ~70 min)

---

## 1. Datasets de entrada

| Dataset | Archivo | Ejemplos | Tamano | Modelo teacher |
|---------|---------|----------|--------|----------------|
| Sonnet | `data/all_examples.jsonl` | 34,728 | 42 MB | Claude Sonnet 4.6 |
| MiniMax Batch 1 | `data/batch1/batch_examples.jsonl` | 129,135 | 130 MB | MiniMax M2.7 |
| MiniMax Batch 2 | `data/batch2/batch_examples.jsonl` | 128,999 | 136 MB | MiniMax M2.7 |
| **Total** | | **292,862** | **308 MB** | |

Composicion: Sonnet 11.9% / MiniMax 88.1%

---

## 2. Pipeline de filtrado (8 stages)

| Stage | Filtro | Tipo | Costo |
|-------|--------|------|-------|
| 1 | Validacion encoding/formato (UTF-8, JSON) | Deterministico | $0 |
| 2 | Longitud minima respuesta >= 300 chars | Deterministico | $0 |
| 3 | Deteccion respuestas evasivas (regex) | Regex | $0 |
| 4 | Calidad de pregunta >= 30 chars | Deterministico | $0 |
| 5 | Dedup semantica (cosine > 0.95, all-MiniLM-L6-v2) | Embedding | $0 (local) |
| 6 | Dedup cross-dataset (Sonnet vs MiniMax) | Embedding | $0 (local) |
| 7 | LLM-as-filter (5% muestra, Sonnet 4.6, escala 0-5, score < 3 rechazado) | Model-based | ~$51 USD |
| 8 | Clasificador extrapolado (GradientBoosting, entrenado con labels de stage 7) | Clasificador local | $0 |

**Costo total del pipeline**: ~$51 USD

---

## 3. Resultados por stage

### Totales

| Stage | Rechazados | Restantes | % restante |
|-------|-----------|-----------|------------|
| Inicial | — | 292,862 | 100.0% |
| 1. Encoding | 0 | 292,862 | 100.0% |
| 2. Longitud >= 300 | 28,225 | 264,637 | 90.4% |
| 3. Evasivas | 36 | 264,601 | 90.3% |
| 4. Pregunta >= 30 | 1 | 264,600 | 90.3% |
| 5. Dedup semantica | 3,543 | 261,057 | 89.1% |
| 6. Cross-dataset dedup | 0 | 261,057 | 89.1% |
| 7. LLM filter (muestra 5%) | 4,823 | 256,234 | 87.5% |
| 8. Clasificador extrapolado | 30,469 | 225,765 | 77.1% |
| Balance tematico | 67,949 | 157,816 | 53.9% |

### Por fuente

| Stage | Sonnet rechazados | Sonnet % | MiniMax rechazados | MiniMax % |
|-------|------------------|----------|-------------------|-----------|
| 1. Encoding | 0 | 0.0% | 0 | 0.0% |
| 2. Longitud >= 300 | 816 | 2.4% | 27,409 | 10.6% |
| 3. Evasivas | 9 | 0.03% | 27 | 0.01% |
| 4. Pregunta >= 30 | 0 | 0.0% | 1 | 0.0% |
| 5. Dedup semantica | 497 | 1.5% | 3,046 | 1.3% |
| 6. Cross-dataset dedup | 0 | 0.0% | 0 | 0.0% |
| 7. LLM filter | 254 | 0.8% | 4,569 | 2.0% |
| 8. Clasificador | 42 | 0.1% | 30,427 | 13.6% |

### Retencion final por fuente

| Fuente | Inicial | Final | % retenido |
|--------|---------|-------|------------|
| Sonnet | 34,728 | 26,900 | 77.5% |
| MiniMax | 258,134 | 130,916 | 50.7% |
| **Total** | **292,862** | **157,816** | **53.9%** |

---

## 4. Stage 7: LLM-as-filter — Detalle

**Modelo evaluador**: Claude Sonnet 4.6
**Muestra**: 12,967 ejemplos (5% estratificado pre-balance)
**Duracion**: ~20.5 horas (3 intentos, el 2o crasheo al 3%)
**Costo**: ~$51 USD

### Distribucion de scores

| Score | Cantidad | % |
|-------|----------|---|
| 0 | 2 | 0.0% |
| 1 | 713 | 5.5% |
| 2 | 4,108 | 31.7% |
| 3 | 5,718 | 44.1% |
| 4 | 2,330 | 18.0% |
| 5 | 96 | 0.7% |

**Score promedio**: 2.77/5
**Tasa de rechazo (score < 3)**: 37.2%

### Tasa de rechazo por estrato

| Estrato | Rechazados | Total | % rechazo |
|---------|-----------|-------|-----------|
| Tratamiento | 3,035 | 7,640 | 39.7% |
| Otros | 324 | 833 | 38.9% |
| Seguimiento | 275 | 796 | 34.5% |
| Diagnostico | 862 | 2,575 | 33.5% |
| Farmacologia | 263 | 900 | 29.2% |
| Soporte | 64 | 223 | 28.7% |

---

## 5. Stage 8: Clasificador extrapolado — Detalle

**Clasificador**: GradientBoosting (200 estimators, max_depth=5)
**Training data**: 8,144 pass + 4,823 fail = 12,967 labeled (de stage 7)
**5-fold CV F1**: 0.705 +/- 0.040
**Duracion**: ~2 minutos (local, $0)

### Feature importances (top 10)

| Feature | Importancia |
|---------|-------------|
| ratio_aq (respuesta/pregunta) | 0.214 |
| ttr (diversidad vocabulario) | 0.174 |
| answer_len | 0.168 |
| is_sonnet | 0.148 |
| word_count | 0.105 |
| sentence_count | 0.046 |
| has_citation | 0.035 |
| question_len | 0.030 |
| has_dosing | 0.022 |
| has_list | 0.017 |

### Resultados de extrapolacion

| Metrica | Valor |
|---------|-------|
| Ejemplos no evaluados por LLM | 248,267 |
| Predichos como baja calidad | 30,469 (12.3%) |
| MiniMax rechazados | 30,427 (13.6% de sus no evaluados) |
| Sonnet rechazados | 42 (0.1% de sus no evaluados) |

---

## 6. Distribucion tematica

### Despues del balance (cap 40%)

| Estrato | Ejemplos | % |
|---------|----------|---|
| Tratamiento | 64,219 | 40.0% |
| Diagnostico | 46,613 | 29.0% |
| Farmacologia | 16,016 | 10.0% |
| Otros | 15,360 | 9.6% |
| Seguimiento | 14,187 | 8.8% |
| Soporte | 4,143 | 2.6% |

67,949 ejemplos removidos por balance tematico.

---

## 7. Split final (90/5/5)

| Split | Ejemplos |
|-------|----------|
| Train | 142,030 |
| Valid | 7,885 |
| Test | 7,901 |
| **Total** | **157,816** |

Split estratificado por fuente (sonnet/minimax) y estrato clinico.

### Distribucion de fuente en dataset final

| Fuente | Ejemplos | % |
|--------|----------|---|
| MiniMax | 130,916 | 82.9% |
| Sonnet | 26,900 | 17.1% |

---

## 8. Estadisticas de longitud de respuesta

| Metrica | Valor |
|---------|-------|
| Minimo | 300 chars |
| Maximo | 6,333 chars |
| Media | 522 chars |
| Mediana | 503 chars |

---

## 9. Hallazgos clave

### MiniMax genera respuestas mas cortas
MiniMax M2.7 produjo 10.6% de respuestas con <300 chars, vs solo 2.4% de Sonnet.

### Cross-dataset dedup es nulo
Sonnet y MiniMax generaron desde chunks diferentes del corpus ChromaDB. Los datasets son complementarios, no redundantes.

### El LLM revelo calidad inferior a lo esperado
Score promedio de 2.77/5 y 37.2% de rechazo en la muestra. El factor principal: respuestas genericas sin evidencia clinica especifica, dosis correctas o citacion de fuentes.

### El clasificador confirma la diferencia Sonnet vs MiniMax
Stage 8 rechazo 13.6% de MiniMax vs solo 0.1% de Sonnet. `is_sonnet` es el 4o feature mas importante (14.8%). La calidad de Sonnet como teacher es significativamente superior.

### Features de calidad correlacionan con contenido clinico
Las respuestas de alta calidad tienen: mayor ratio respuesta/pregunta, mayor diversidad lexica, mayor longitud, y mas citaciones de evidencia. Esto valida los criterios clinicos del scoring LLM.

### Dataset final: 5x el original con calidad verificada
De 30K pares sin filtro (dataset original llama8b-onco) a 142K train con filtrado de 8 stages. Mejora tanto en volumen como en calidad verificada.

---

## 10. Archivos generados

```
data/clean/
  train.jsonl          (142,030 ejemplos)
  valid.jsonl          (7,885 ejemplos)
  test.jsonl           (7,901 ejemplos)
  cleanup_report.json  (reporte JSON completo)
  llm_scores.json      (12,967 scores de stage 7)
```

---

## 11. Proximos pasos

1. **Fine-tune 1**: seleccion de modelo base (gpt-oss-20b, Nemotron, Gemma 4)
2. **Fine-tune 2-3**: modelos adicionales segun resultados
3. **Arena Final**: evaluacion comparativa de tiers fine-tuned vs base+RAG vs API
4. **Evaluacion humana**: panel de oncologos para tiers con score > 2.5
