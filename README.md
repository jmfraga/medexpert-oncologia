# MedExpert Oncologia — Fine-tuning & Arena

Pipeline de fine-tuning de modelos LLM para oncologia clinica y evaluacion comparativa (Arena).

## Modelos entrenados

| Modelo | Base | Parametros | Metodo | Dataset | Val loss | Hardware |
|--------|------|-----------|--------|---------|----------|----------|
| **llama8b-onco** | Meta-Llama-3.1-8B-Instruct-4bit | 8B | LoRA (rank 64) | 34,728 examples | 0.881 (iter 3000), **0.874** (iter 2600) | Mac Mini M4, 64 GB |
| **gpt-oss-20b-onco** | gpt-oss-20b-MLX-4bit | 20B (MoE) | LoRA (rank 64) | 34,728 examples | 0.885 (iter 3000), **0.863** (iter 2600) | Mac Mini M4, 64 GB |

Ambos modelos usan el mismo dataset de oncologia clinica generado con Sonnet 4.6 como teacher a partir de 134,478 chunks de guias clinicas (NCCN, ESMO, IMSS, CMCM, ASCO).

## Pipeline

```
01_sample_chromadb.py     # Muestreo estratificado de ChromaDB (11,592 chunks)
02_generate_dataset.py    # Teacher (Sonnet 4.6) genera 34,728 pares Q&A
03_finetune_mlx.sh        # LoRA training en MLX (Apple Silicon nativo)
04_fuse_model.sh          # Fusion de adapters al modelo base
05_evaluate_model.py      # Evaluacion base vs fine-tuned
```

## Arena v2 — Evaluacion comparativa

Compara 6 estrategias de respuesta clinica oncologica contra 15 casos clinicos con gold standard:

| # | Tier | Modelo | Fine-tune | RAG | Tipo |
|---|------|--------|-----------|-----|------|
| 1 | llama8b base + RAG | Meta-Llama-3.1-8B-Instruct-4bit | No | Si | Local (denso) |
| 2 | llama8b-onco | llama8b-onco LoRA | Si | No | Local (denso) |
| 3 | gpt-oss-20b base + RAG | gpt-oss-20b-MLX-4bit | No | Si | Local (MoE) |
| 4 | gpt-oss-20b-onco | gpt-oss-20b-onco LoRA | Si | No | Local (MoE) |
| 5 | MiniMax-M1-80K + RAG | MiniMax-Text-01 | No | Si | API |
| 6 | Sonnet 4 + RAG | claude-sonnet-4 | No | Si | API |

**Juez**: Claude Opus 4.6 + RAG (no compite como candidato)

### Comparaciones clave

- **RAG vs fine-tune**: tiers 1 vs 2 (llama8b), tiers 3 vs 4 (gpt-oss)
- **Denso vs MoE**: tiers 2 vs 4 (fine-tuned, mismo dataset)
- **Local vs API**: tiers 4 vs 5 vs 6

## Resultados de entrenamiento

### llama8b-onco (completado 2026-03-26)

- 3,000 iteraciones, ~11 horas
- Peak memory: 11.2 GB
- Val loss: 6.031 (iter 1) -> **0.874** (iter 2600) -> 0.881 (iter 3000)
- Sin overfitting (train ~0.82-0.92, val ~0.87-0.91)

### gpt-oss-20b-onco (completado 2026-03-26)

- 3,000 iteraciones, ~8.5 horas
- Peak memory: 30.8 GB
- Val loss: 6.031 (iter 1) -> **0.863** (iter 2600) -> 0.885 (iter 3000)
- Sin overfitting (train ~0.77-0.93, val ~0.86-0.95)
- Mejor val loss que llama8b (0.863 vs 0.874)

### Configuracion de LoRA (compartida)

| Parametro | Valor |
|-----------|-------|
| Rank | 64 |
| Dropout | 0.05 |
| Scale | 32.0 |
| Batch size | 4 |
| Learning rate | 2e-5 (llama8b), 1e-5 (gpt-oss) |
| Max seq length | 2,048 |
| Grad checkpoint | Si |
| Mask prompt | Si |
| Capas | Todas (-1) |

## Base de conocimiento (ChromaDB)

134,478 chunks de 663 fuentes:

| Fuente | Chunks | Idioma |
|--------|--------|--------|
| NCCN Guidelines | 62,004 | Ingles |
| ESMO Guidelines | 40,414 | Ingles |
| IMSS GPC | 15,066 | Espanol |
| Fichas Tecnicas (397 medicamentos) | 11,292 | Espanol |
| CMCM 2025 | 2,717 | Espanol |
| ASCO + NCI | 250 | Ingles |

## Estructura del repositorio

```
medexpert-oncologia/
├── 01_sample_chromadb.py          # Muestreo estratificado
├── 02_generate_dataset.py         # Generacion de dataset con teacher
├── 03_finetune_mlx.sh             # Training script
├── 04_fuse_model.sh               # Fusion de adapters
├── 05_evaluate_model.py           # Evaluacion
├── lora-config.yaml               # Configuracion de LoRA
├── METODOLOGIA.md                 # Metodologia completa del estudio
├── clinical_cases_gold_standard.md # 15 casos clinicos con gold standard
├── data/                          # Dataset de entrenamiento
├── adapters/                      # LoRA adapters (no en git, backup local)
│   ├── lora-full/                 # llama8b-onco (6 checkpoints, 640 MB c/u)
│   └── gpt-oss-full/             # gpt-oss-20b-onco (6 checkpoints, 3.5 GB c/u)
├── logs/                          # Logs de entrenamiento
├── results/                       # Resultados de Arena
└── paper_npj_digital_medicine.md  # Draft del paper
```

## Requisitos

- macOS con Apple Silicon (M4 recomendado, 64 GB RAM)
- Python 3.12+
- mlx-lm >= 0.31
- ChromaDB con corpus oncologico indexado (en M1 MedExpert)
- API keys: Anthropic, MiniMax

