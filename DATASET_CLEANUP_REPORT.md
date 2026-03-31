# Dataset Quality Funnel Report

**Fecha**: 2026-03-31
**Script**: `02b_clean_dataset.py`
**Duración**: ~67 minutos (stages 1-4: <1 min, stage 5: ~53 min, stage 6: ~11 min)

---

## 1. Datasets de entrada

| Dataset | Archivo | Ejemplos | Tamaño | Modelo teacher |
|---------|---------|----------|--------|----------------|
| Sonnet | `data/all_examples.jsonl` | 34,728 | 42 MB | Claude Sonnet 4.6 |
| MiniMax Batch 1 | `data/batch1/batch_examples.jsonl` | 129,135 | 130 MB | MiniMax M2.7 |
| MiniMax Batch 2 | `data/batch2/batch_examples.jsonl` | 128,999 | 136 MB | MiniMax M2.7 |
| **Total** | | **292,862** | **308 MB** | |

Composicion: Sonnet 11.9% / MiniMax 88.1%

---

## 2. Pipeline de filtrado (7 stages)

| Stage | Filtro | Tipo | Costo |
|-------|--------|------|-------|
| 1 | Validacion encoding/formato (UTF-8, JSON) | Deterministico | Gratis |
| 2 | Longitud minima respuesta >= 300 chars | Deterministico | Gratis |
| 3 | Deteccion respuestas evasivas (regex) | Regex | Gratis |
| 4 | Calidad de pregunta >= 30 chars | Deterministico | Gratis |
| 5 | Dedup semantica (cosine > 0.95, all-MiniLM-L6-v2) | Embedding | Gratis (local) |
| 6 | Dedup cross-dataset (Sonnet vs MiniMax) | Embedding | Gratis (local) |
| 7 | LLM-as-filter (5% muestra, Sonnet, escala 0-5) | Model-based | ~$5-10 USD |

Stages 1-6 ejecutados. Stage 7 pendiente (1-abr).

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
| 7. LLM filter | PENDIENTE | — | — |
| Balance tematico | 82,549 | 178,508 | 60.9% |

### Por fuente

| Stage | Sonnet rechazados | Sonnet % | MiniMax rechazados | MiniMax % |
|-------|------------------|----------|-------------------|-----------|
| Inicial | — | — | — | — |
| 1. Encoding | 0 | 0.0% | 0 | 0.0% |
| 2. Longitud >= 300 | 816 | 2.4% | 27,409 | 10.6% |
| 3. Evasivas | 9 | 0.03% | 27 | 0.01% |
| 4. Pregunta >= 30 | 0 | 0.0% | 1 | 0.0% |
| 5. Dedup semantica | 497 | 1.5% | 3,046 | 1.3% |
| 6. Cross-dataset dedup | 0 | 0.0% | 0 | 0.0% |

### Retencion final por fuente

| Fuente | Inicial | Final | % retenido |
|--------|---------|-------|------------|
| Sonnet | 34,728 | 26,891 | 77.4% |
| MiniMax | 258,134 | 151,617 | 58.7% |
| **Total** | **292,862** | **178,508** | **60.9%** |

---

## 4. Distribucion tematica

### Antes del balance

| Estrato | Ejemplos | % |
|---------|----------|---|
| Tratamiento | 153,957 | 59.0% |
| Diagnostico | 51,816 | 19.9% |
| Farmacologia | 18,107 | 6.9% |
| Otros | 16,672 | 6.4% |
| Seguimiento | 16,011 | 6.1% |
| Soporte | 4,494 | 1.7% |

### Despues del balance (cap 40%)

| Estrato | Ejemplos | % |
|---------|----------|---|
| Tratamiento | 71,408 | 40.0% |
| Diagnostico | 51,816 | 29.0% |
| Farmacologia | 18,107 | 10.1% |
| Otros | 16,672 | 9.3% |
| Seguimiento | 16,011 | 9.0% |
| Soporte | 4,494 | 2.5% |

82,549 ejemplos removidos por balance (solo de "tratamiento").

---

## 5. Split final (90/5/5)

| Split | Ejemplos |
|-------|----------|
| Train | 160,652 |
| Valid | 8,918 |
| Test | 8,938 |
| **Total** | **178,508** |

Split estratificado por fuente (sonnet/minimax) y estrato clinico.

### Distribucion de fuente en dataset final

| Fuente | Ejemplos | % |
|--------|----------|---|
| MiniMax | 151,617 | 84.9% |
| Sonnet | 26,891 | 15.1% |

---

## 6. Estadisticas de longitud de respuesta

| Metrica | Valor |
|---------|-------|
| Minimo | 300 chars |
| Maximo | 6,333 chars |
| Media | 511.8 chars |
| Mediana | 492 chars |

---

## 7. Hallazgos clave

### MiniMax genera respuestas mas cortas

MiniMax M2.7 produjo 10.6% de respuestas con <300 chars, vs solo 2.4% de Sonnet. Esto confirma la tendencia de MiniMax a ser mas conciso, lo cual fue un factor en los scores bajos de Arena v2b (mediana ~180 tokens).

### Cross-dataset dedup es nulo

Sonnet y MiniMax generaron desde chunks diferentes del corpus ChromaDB (11,592 vs ~258K chunks respectivamente), por lo que no hubo overlap tematico detectable (cosine > 0.95). Esto valida que los datasets son complementarios, no redundantes.

### Dedup semantica minima

Solo 1.3-1.5% de duplicados semanticos dentro de cada fuente. El proceso de generacion (1 pregunta por chunk) produce diversidad natural suficiente.

### Balance tematico es el ajuste mas grande

"Tratamiento" domina el corpus oncologico (59%), lo cual refleja la distribucion real del contenido clinico pero sesga el entrenamiento. El cap a 40% removio 82K ejemplos (28% del total post-funnel).

### Sonnet produce mayor calidad proporcional

77.4% de los ejemplos de Sonnet sobrevivieron el funnel vs 58.7% de MiniMax. La diferencia principal esta en longitud de respuesta (stage 2), donde MiniMax pierde 4x mas proporcionalmente.

---

## 8. Archivos generados

```
data/clean/
  train.jsonl       (160,652 ejemplos)
  valid.jsonl       (8,918 ejemplos)
  test.jsonl        (8,938 ejemplos)
  cleanup_report.json  (reporte JSON completo)
```

---

## 9. Proximos pasos

1. **Stage 7 (LLM filter)**: 1-abr, con limites frescos de Anthropic
2. **Fine-tune gpt-oss-20b-onco**: primer modelo, ~8-10h
3. **Fine-tune MedGemma-onco**: segundo modelo, ~12-15h
4. **Fine-tune Nemotron-onco**: tercer modelo, ~10-12h
5. **Arena Final**: 7 tiers (3 base+RAG + 3 fine-tuned + Sonnet), 105 evaluaciones
