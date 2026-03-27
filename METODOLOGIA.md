# MedExpert OncoLight — Metodología Completa

## 1. Construcción de la Base de Conocimiento (ChromaDB)

### 1.1 Fuentes de datos

| Fuente | Chunks | Idioma | Contenido |
|--------|--------|--------|-----------|
| NCCN Guidelines | 62,004 | Inglés | Guías por tipo de cáncer (mama, pulmón, colon, próstata, melanoma, etc.) |
| ESMO Guidelines | 40,414 | Inglés | Guías europeas de práctica clínica |
| IMSS GPC | 15,066 | Español | Guías de Práctica Clínica del sistema de salud mexicano |
| PHARMA (Fichas Técnicas) | 11,292 | Español | 397 fichas técnicas de medicamentos oncológicos |
| CMCM (Consenso Mexicano) | 2,717 | Español | Consenso Mexicano de Cáncer de Mama 2025 |
| ASCO | 225 | Inglés | Guías de la American Society of Clinical Oncology |
| NCI PDQ | 25 | Inglés | Patient Data Queries del National Cancer Institute |
| **Total** | **134,478** | Bilingüe | **663 fuentes únicas** |

### 1.2 Pipeline de indexación

```
Documento (PDF/DOCX/TXT)
  → Extracción de texto (python-docx, PyPDF)
  → Chunking clínico (500 chars, overlap 100, section-aware)
  → Extracción de metadata automática:
      - source: nombre del documento
      - society: NCCN, ESMO, IMSS, PHARMA, CMCM, ASCO, NCI
      - category: tipo de contenido
      - section_path: jerarquía de secciones del documento
      - doc_type: guideline, consensus, review, article
  → Embedding (sentence-transformers)
  → Almacenamiento en ChromaDB (cosine similarity)
```

### 1.3 Capacidades de búsqueda

- **Búsqueda bilingüe**: queries en español se traducen a inglés para buscar en NCCN/ESMO; queries en inglés se buscan directamente
- **Expansión de sinónimos**: 137 pares marca↔genérico (ej. Keytruda → pembrolizumab)
- **Filtrado por sociedad**: permite buscar solo en fuentes seleccionadas
- **Diversificación de resultados**: máximo 3 hits por fuente para evitar sesgo

---

## 2. Entrenamiento de Modelos Locales

### 2.1 Modelos base

Se entrenan dos modelos con el mismo dataset para comparar arquitecturas:

| Modelo | Tipo | Parámetros | Fuente | Justificación |
|--------|------|-----------|--------|---------------|
| **llama8b-onco** | Denso | 8B | mlx-community/Meta-Llama-3.1-8B-Instruct-4bit | Modelo eficiente, bajo consumo de memoria (~11 GB) |
| **gpt-oss-20b-onco** | MoE | 20B | InferenceIllusionist/gpt-oss-20b-MLX-4bit | Mayor capacidad, compara denso vs MoE (~31 GB) |

- **Hardware**: Mac Mini M4, 64 GB RAM, Apple Silicon
- **Framework**: mlx-lm 0.31.1 (nativo Apple Silicon)

### 2.2 Muestreo estratificado del corpus

Del total de 134,478 chunks, se seleccionan ~12,000 chunks representativos mediante muestreo estratificado:

| Estrato | % | Target | Contenido |
|---------|---|--------|-----------|
| Tratamiento | 30% | 3,600 | Protocolos de quimioterapia, radioterapia, cirugía, inmunoterapia |
| Farmacología | 25% | 3,000 | Dosis, mecanismos de acción, toxicidades, interacciones |
| Diagnóstico | 20% | 2,400 | Estadificación TNM, biomarcadores, patología molecular |
| Soporte | 15% | 1,800 | Manejo de efectos adversos, cuidado paliativo |
| Seguimiento | 10% | 1,200 | Vigilancia post-tratamiento, supervivencia |

**Criterios de calidad del muestreo**:
- Filtro de chunks de baja calidad (bibliografía, conflictos de interés, copyright, chunks <80 caracteres)
- Deduplicación por hash de contenido (primeros 150 caracteres)
- Diversidad de fuentes: round-robin entre documentos para evitar sobre-representación
- Resultado: 11,592 chunks de 568 fuentes distintas

### 2.3 Generación del dataset de entrenamiento

**Modelo teacher**: Claude Sonnet 4.6 (`claude-sonnet-4-6`, Anthropic API)

Por cada chunk seleccionado, el teacher genera 3 pares pregunta-respuesta:

1. **Conocimiento directo**: Pregunta factual sobre el contenido del chunk
2. **Caso clínico**: Caso breve con edad, género, estadio que requiere razonamiento
3. **Decisión terapéutica**: Comparación entre opciones o decisión de manejo

**Formato de salida** (JSONL chat para mlx-lm):
```json
{"messages": [
  {"role": "system", "content": "Eres MedExpert Onco, un asistente clínico especializado en oncología médica..."},
  {"role": "user", "content": "<pregunta clínica en español>"},
  {"role": "assistant", "content": "<respuesta basada estrictamente en el chunk, con fuente>"}
]}
```

**Controles de calidad**:
- Respuestas basadas estrictamente en el chunk original (no inventar datos)
- Respuestas en español (aunque la fuente esté en inglés)
- Filtro de longitud (>30 tokens respuesta)
- Inclusión de fuente al final de cada respuesta

**Volumen estimado**: ~35,000 pares de entrenamiento
**Split**: 80% train / 10% validation / 10% test

### 2.4 Configuración de fine-tuning

**Método**: QLoRA (LoRA sobre modelo 4-bit quantized)

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Rank | 64 | Dominio médico denso requiere más capacidad que rank 8-16 estándar |
| Capas | Todas (-1) | Conocimiento médico distribuido en todas las capas del transformer |
| Batch size | 4 | Balance entre velocidad y estabilidad |
| Learning rate | 2e-5 (llama8b), 1e-5 (gpt-oss) | Estándar para LoRA; reducido para modelo más grande |
| Iteraciones | 3,000 (full) | Calibrado para ~35K examples |
| Max seq length | 2,048 | Suficiente para Q&A clínico |
| Dropout | 0.05 | Regularización ligera para dataset especializado |
| Scale (lora_alpha) | 32.0 | 0.5× rank, recomendado para dominio estrecho |
| Grad checkpoint | Sí | Reduce memoria (~15 GB usado de 64 GB disponibles) |
| Mask prompt | Sí | Solo calcula loss en la respuesta, no en la pregunta |

**Framework**: mlx-lm 0.31.1 (nativo Apple Silicon)

### 2.5 Resultados de entrenamiento

#### llama8b-onco (completado 2026-03-26, ~11 horas)

| Iter | Val loss | Train loss | Observación |
|------|----------|------------|-------------|
| 1 | 6.031 | — | Baseline |
| 1000 | 0.933 | 0.882 | Convergencia rápida |
| 2000 | 0.923 | 0.855 | Estable |
| 2600 | **0.874** | 0.846 | **Mínimo** |
| 3000 | 0.881 | 0.860 | Final |

Peak memory: 11.2 GB | Velocidad: ~0.075 it/sec, ~53 tok/sec | Tokens entrenados: 2,112,717

#### gpt-oss-20b-onco (completado 2026-03-26, ~8.5 horas)

| Iter | Val loss | Train loss | Observación |
|------|----------|------------|-------------|
| 1 | 6.031 | — | Baseline |
| 1000 | 0.946 | 0.893 | Convergencia |
| 1800 | 0.867 | 0.856 | Inflexión |
| 2600 | **0.863** | 0.849 | **Mínimo** |
| 3000 | 0.885 | 0.844 | Final |

Peak memory: 30.8 GB | Velocidad: ~0.105 it/sec, ~65 tok/sec | Tokens entrenados: 1,880,841

**Observaciones**:
- gpt-oss-20b alcanza mejor val loss mínimo (0.863 vs 0.874) pese a tener más parámetros
- Ambos modelos muestran su mejor checkpoint en iter 2600, sugiriendo que 3000 iteraciones es ligeramente excesivo
- Sin señales de overfitting en ningún modelo

### 2.6 Post-entrenamiento

1. **Fusión**: LoRA adapters se fusionan al modelo base con `mlx_lm.fuse`
2. **Modelos finales**: `llama8b-onco` y `gpt-oss-20b-onco` — modelos autónomos sin necesidad de adapters
3. **Despliegue**: MLX server en M4 (puertos 8086 y 8092 respectivamente)

---

## 3. Metodología de Evaluación — MedExpert Arena

### 3.1 Diseño del estudio

**Evaluación comparativa de 6 candidatos** de respuesta clínica oncológica (Arena v2):

| # | Tier | Modelo | Provider | Fine-tune | RAG |
|---|------|--------|----------|-----------|-----|
| 1 | **llama8b base + RAG** | Meta-Llama-3.1-8B-Instruct-4bit | Meta (local) | No | Sí |
| 2 | **llama8b-onco** | llama8b-onco LoRA (local) | Meta (fine-tune local) | Sí | No |
| 3 | **gpt-oss-20b base + RAG** | gpt-oss-20b-MLX-4bit | Open-source (local) | No | Sí |
| 4 | **gpt-oss-20b-onco** | gpt-oss-20b-onco LoRA (local) | Open-source (fine-tune local) | Sí | No |
| 5 | **MiniMax-M1-80K + RAG** | MiniMax-Text-01 | MiniMax (API) | No | Sí |
| 6 | **Sonnet 4 + RAG** | Sonnet 4 | Anthropic (API) | No | Sí |

**Evaluador (juez)**: Claude Opus 4.6 + RAG (no compite como candidato, imparcial)

**Preguntas de investigación**:
1. **RAG vs fine-tune**: ¿Qué estrategia da mejor resultado, retrieval sobre modelo base o fine-tuning sin retrieval? (tiers 1 vs 2, tiers 3 vs 4)
2. **Denso vs MoE**: ¿Un modelo MoE 20B fine-tuned supera a un modelo denso 8B fine-tuned con el mismo dataset? (tier 2 vs 4)
3. **Local vs API**: ¿Los modelos locales fine-tuned alcanzan calidad comparable a modelos API generalistas + RAG? (tier 4 vs 5 vs 6)

### 3.2 Casos clínicos de prueba

15 casos clínicos oncológicos diseñados con distribución equilibrada:

**Por complejidad**:
- Simple (5): diagnóstico claro, tratamiento estándar bien definido
- Moderado (4): requiere integración de múltiples factores clínicos
- Complejo (6): biomarcadores específicos, decisiones con múltiples opciones, escenarios controversiales

**Por tipo de cáncer** (14 tipos distintos):
Mama (2), Pulmón (2), Colorrectal (2), Próstata, Cervicouterino, Gástrico, Linfoma de Hodgkin, Melanoma, Ovario, Tiroides, Leucemia mieloide crónica, Páncreas

**Cada caso incluye**:
- Presentación clínica realista (edad, género, síntomas, estudios)
- Estadificación completa
- Biomarcadores relevantes
- Gold standard: diagnóstico correcto + 6-8 recomendaciones clave basadas en guías + guías de referencia

### 3.3 Rúbrica de evaluación (compartida: Opus y humanos)

Escala 0-5 por criterio. La misma rúbrica se aplica tanto al juez automático (Opus) como a los evaluadores humanos.

| Criterio | Descripción | Peso |
|----------|-------------|------|
| **Precisión diagnóstica** | ¿El diagnóstico y estadificación coinciden con el gold standard? | 25% |
| **Apego a guías** | ¿Las recomendaciones son consistentes con NCCN/ESMO/IMSS? ¿Se citan correctamente? | 30% |
| **Completitud** | ¿Se cubren todos los aspectos relevantes? (tratamiento, alternativas, seguimiento, efectos adversos) | 25% |
| **Utilidad clínica** | ¿La respuesta es práctica y accionable para un oncólogo? | 20% |

**Score compuesto por evaluador**: promedio ponderado de los 4 criterios (0-5)

### 3.4 Evaluación automática — Juez Opus

```
Para cada caso clínico (15):
  Para cada candidato (6 tiers):
    1. Enviar caso → pipeline completo del candidato
    2. Registrar respuesta completa + métricas de rendimiento
    3. Enviar respuesta al juez (Opus 4.6 + RAG + gold standard)
    4. Juez califica con rúbrica estructurada + feedback textual
Total: 90 evaluaciones automáticas (15 casos × 6 candidatos)
```

**Ventajas**: reproducible, sin costo de coordinación, cubre los 15 casos × 6 candidatos.

### 3.5 Evaluación humana — Panel de oncólogos

#### 3.5.1 Diseño

Evaluación ciega por oncólogos clínicos. Las respuestas se presentan anonimizadas y en orden aleatorio (Candidato A-F) diferente por caso para evitar sesgo posicional. Los evaluadores no conocen qué modelo generó cada respuesta.

**Instrumento**: Google Forms con la rúbrica de 4 criterios (§3.3), escala 0-5 por criterio, más un campo de comentarios libres por respuesta.

#### 3.5.2 Escenarios de evaluadores

| Escenario | Doctores | Casos/doctor | Evaluadores/caso | Scores/doctor | Viabilidad |
|-----------|----------|-------------|-------------------|---------------|------------|
| A (mínimo) | 3 | 10 | 2 | 200 | Factible, carga alta |
| B (recomendado) | 4 | 8 | ~2 | 160 | Buen balance |
| C (ideal) | 5 | 6 | 2 | 120 | Carga ligera |

**Cálculo de carga**: casos × 6 respuestas × 4 criterios = scores por doctor.

La asignación de casos se hace con un **diseño de bloques incompletos balanceado** (BIBD): cada caso es evaluado por exactamente 2 doctores, y cada doctor evalúa una distribución proporcional de complejidades (simple/moderado/complejo).

#### 3.5.3 Distribución de casos por escenario

**Escenario A — 3 doctores, 10 casos cada uno:**

| Doctor | Casos asignados (IDs) | Simple | Moderado | Complejo |
|--------|----------------------|--------|----------|----------|
| D1 | 1, 3, 5, 7, 8, 9, 10, 11, 13, 15 | 3 | 3 | 4 |
| D2 | 1, 2, 4, 6, 7, 9, 10, 12, 14, 15 | 3 | 3 | 4 |
| D3 | 2, 3, 4, 5, 6, 8, 11, 12, 13, 14 | 4 | 2 | 4 |

**Escenario B — 4 doctores, 8 casos cada uno:**

| Doctor | Casos asignados (IDs) | Simple | Moderado | Complejo |
|--------|----------------------|--------|----------|----------|
| D1 | 1, 4, 5, 8, 9, 11, 13, 15 | 3 | 2 | 3 |
| D2 | 2, 3, 6, 7, 10, 12, 14, 15 | 2 | 2 | 4 |
| D3 | 1, 2, 5, 7, 8, 10, 11, 14 | 3 | 2 | 3 |
| D4 | 3, 4, 6, 9, 12, 13, 14, 15 | 2 | 2 | 4 |

**Escenario C — 5 doctores, 6 casos cada uno:**

| Doctor | Casos asignados (IDs) | Simple | Moderado | Complejo |
|--------|----------------------|--------|----------|----------|
| D1 | 1, 4, 7, 9, 11, 14 | 2 | 2 | 2 |
| D2 | 2, 5, 8, 10, 12, 15 | 2 | 2 | 2 |
| D3 | 3, 6, 7, 9, 13, 14 | 2 | 2 | 2 |
| D4 | 1, 4, 8, 10, 11, 15 | 2 | 2 | 2 |
| D5 | 2, 3, 5, 6, 12, 13 | 2 | 2 | 2 |

#### 3.5.4 Anonimización

Para cada caso, las 5 respuestas se asignan aleatoriamente a etiquetas (Candidato A/B/C/D/E). El mapeo se genera con semilla fija por caso (reproducible) pero es distinto por caso para evitar que el evaluador infiera patrones.

```
Ejemplo — Caso 1:
  Candidato A → Sonnet 4 + RAG
  Candidato B → llama8b-onco (fine-tuned)
  Candidato C → MiniMax-M1-80K + RAG
  Candidato D → gpt-oss-20b base + RAG
  Candidato E → llama8b base + RAG
  Candidato F → gpt-oss-20b-onco (fine-tuned)

Ejemplo — Caso 2:
  Candidato A → gpt-oss-20b-onco (fine-tuned)
  Candidato B → Sonnet 4 + RAG
  Candidato C → llama8b base + RAG
  Candidato D → MiniMax-M1-80K + RAG
  Candidato E → gpt-oss-20b base + RAG
  Candidato F → llama8b-onco (fine-tuned)
```

El mapeo completo se almacena en `arena/results/anonymization_map.json` y se revela solo al consolidar resultados.

#### 3.5.5 Concordancia inter-evaluador

- **Krippendorff's alpha (α)**: calculado sobre los pares de evaluadores que comparten casos
  - α > 0.80: concordancia excelente
  - 0.67 < α ≤ 0.80: concordancia aceptable
  - α ≤ 0.67: concordancia baja — investigar discrepancias
- Se reporta α global y por criterio (precisión, apego, completitud, utilidad)
- En caso de α bajo en un criterio, se revisan las respuestas discordantes y se documenta

### 3.6 Integración de evaluaciones

Los scores de Opus y humanos se reportan **por separado** (no se combinan en un score único). Esto permite:

1. **Validación cruzada**: ¿Opus evalúa de forma similar a los oncólogos?
   - Correlación de Pearson y Spearman entre score Opus y score humano promedio por caso × candidato
   - Si r > 0.80: Opus es un proxy confiable de evaluación clínica
   - Gráfico de dispersión Opus vs. humanos con IC 95%
2. **Detección de discrepancias**: identificar casos donde Opus y humanos difieren sustancialmente (>1 punto en score compuesto) — estos son los más interesantes para el paper
3. **Hallazgo publicable**: la correlación Opus-humanos valida (o invalida) el uso de LLMs como evaluadores clínicos

### 3.7 Métricas adicionales (automáticas)

| Métrica | Descripción |
|---------|-------------|
| TTFT (ms) | Tiempo al primer token — latencia percibida |
| Tokens/s | Velocidad de generación |
| Tiempo total (s) | Duración completa de la respuesta |
| Costo por consulta (USD) | Costo de API por respuesta |
| Tasa de alucinación | Respuestas con información no presente en guías |
| Seguridad farmacológica | Dosis correctas, interacciones mencionadas, contraindicaciones no omitidas |

### 3.8 Análisis estadístico

**Evaluación automática (Opus):**
1. **Score promedio por tier** con intervalos de confianza (bootstrap, n=15 casos)
2. **Score por complejidad**: ¿cómo se degradan los tiers en casos complejos?
3. **Score por criterio**: ¿dónde gana/pierde cada tier?
4. **Análisis costo-beneficio**: score/$ — ¿Light es suficiente para consultas simples?
5. **Casos de fallo**: identificar en qué tipo de caso cada tier falla

**Evaluación humana:**
6. **Score humano promedio por tier** (promedio de los evaluadores que calificaron cada caso)
7. **Concordancia inter-evaluador** (Krippendorff's α global y por criterio)
8. **Comparación de rankings**: ¿los doctores y Opus coinciden en cuál tier es mejor por caso?

**Análisis cruzado Opus vs. humanos:**
9. **Correlación Pearson/Spearman** del score compuesto (Opus vs. promedio humano)
10. **Bland-Altman plot**: diferencia Opus-humano vs. promedio — detectar sesgo sistemático
11. **Análisis de discrepancias**: casos con |Opus - humano| > 1 punto

### 3.9 Hipótesis a evaluar

1. **H1**: RAG sobre modelo base supera a fine-tune sin RAG en ambas arquitecturas (llama8b y gpt-oss)
2. **H2**: gpt-oss-20b-onco (MoE, fine-tuned) supera a llama8b-onco (denso, fine-tuned) con el mismo dataset
3. **H3**: Los modelos locales fine-tuned alcanzan ≥70% del score de Sonnet 4 + RAG en casos simples
4. **H4**: Los modelos locales tienen latencia <5s vs >10s de los API (mejor experiencia de usuario)
5. **H5**: El costo por consulta local es $0 vs ~$0.05-0.15 de MiniMax/Sonnet
6. **H6**: El score de Opus correlaciona r > 0.80 con el consenso humano (validación del juez LLM)
7. **H7**: MiniMax (generalista API) supera a los modelos locales base + RAG pero no a Sonnet

### 3.10 Arquitectura técnica de la Arena

```
medexpert-admin/arena/
├── clinical_cases.json         # 15 casos clínicos con gold standard
├── arena_runner.py             # CandidateRunner: ejecuta 6 tiers × 15 casos
│     ├── llama8b_rag:  llama8b base + RAG + ChromaDB
│     ├── llama8b_ft:   llama8b-onco (fine-tuned), sin RAG
│     ├── gptoss_rag:   gpt-oss-20b base + RAG + ChromaDB
│     ├── gptoss_ft:    gpt-oss-20b-onco (fine-tuned), sin RAG
│     ├── minimax:      MiniMax-M1-80K + RAG + ChromaDB
│     └── sonnet:       Sonnet 4 + RAG + ChromaDB
├── arena_judge.py              # JudgeRunner: Opus 4.6 + RAG + rúbrica
│     └── output: JSON con scores por criterio + feedback textual
├── arena_report.py             # Genera reportes + materiales para Google Forms
│     ├── JSON completo (scores Opus + humanos)
│     ├── CSV resumen (para análisis en R/Python)
│     ├── HTML interactivo (respuestas lado a lado)
│     └── Materiales anonimizados para evaluadores humanos
└── results/                    # Resultados de ejecuciones
      ├── responses_YYYYMMDD.json       # Respuestas de los candidatos
      ├── opus_scores_YYYYMMDD.json     # Calificaciones de Opus
      ├── human_scores.json             # Calificaciones humanas (importadas de Forms)
      └── anonymization_map.json        # Mapeo candidato→etiqueta por caso
```

### 3.11 Entregables

1. **Tabla comparativa**: score por candidato × criterio × complejidad (Opus y humanos por separado)
2. **Reporte por caso**: respuesta de cada candidato + evaluación Opus + evaluación humana
3. **Análisis de concordancia**: Krippendorff's α inter-evaluadores + correlación Opus-humanos
4. **Recomendación de tier**: qué candidato usar para qué tipo de consulta
5. **Paper**: evaluación formal con los 15 casos, doble jueceo (LLM + humanos)
6. **Configuración de producción**: reglas para asignar tier según tipo de consulta del usuario
7. **Anexo completo de la publicación**: documento con las 75 respuestas íntegras (15 casos × 5 candidatos), cada una con:
   - Caso clínico original
   - Respuesta completa del candidato (sin truncar)
   - Calificación de Opus por criterio (precisión, apego, completitud, utilidad)
   - Calificación humana promedio por criterio + rango inter-evaluador
   - Feedback textual de Opus y de evaluadores humanos
   - Métricas de rendimiento (TTFT, tokens/s, costo)
   - Comparación vs gold standard

**Formatos de exportación:**
- **JSON**: datos crudos para análisis programático
- **HTML**: reporte visual interactivo (respuestas lado a lado con scores)
- **CSV**: tablas para análisis estadístico en R/Python
- **PDF**: documento formal para anexo de publicación

---

## 4. Pipeline Completo (Scripts)

```
medexpert-oncologia/
├── 01_sample_chromadb.py          # Fase 1: Muestreo estratificado → sampled_chunks.json
├── 02_generate_dataset.py         # Fase 2: Teacher (Sonnet) → train/valid/test.jsonl
├── 03_finetune_mlx.sh             # Fase 3: LoRA training → adapters/
├── 04_fuse_model.sh               # Fase 4: Fusión → Llama8B-MedExpert-Oncologia
├── 05_evaluate_model.py           # Fase 5: Evaluación base vs fine-tuned
├── data/
│   ├── sampled_chunks.json        # 11,592 chunks muestreados
│   ├── train.jsonl                # ~28,000 pares (80%)
│   ├── valid.jsonl                # ~3,500 pares (10%)
│   └── test.jsonl                 # ~3,500 pares (10%)
├── adapters/                      # LoRA adapters por checkpoint
│   ├── lora-full/                 #   llama8b-onco (6 checkpoints, 640 MB c/u)
│   └── gpt-oss-full/             #   gpt-oss-20b-onco (6 checkpoints, 3.5 GB c/u)
├── models/
│   └── Llama8B-MedExpert-Oncologia/  # Modelo fusionado (llama8b-onco)
└── logs/                          # Logs de entrenamiento

medexpert-admin/arena/
├── clinical_cases.json            # 15 casos clínicos con gold standard
├── arena_runner.py                # Ejecuta 6 tiers × 15 casos
├── arena_judge.py                 # Opus 4.6 califica 90 respuestas
├── arena_report.py                # Reportes + materiales para Forms
└── results/                       # JSON/CSV con resultados
```

---

## 5. Cronograma Estimado

| Fase | Duración | Costo |
|------|----------|-------|
| Muestreo de ChromaDB | ✅ Completado | $0 |
| Generación de dataset (Sonnet 4.6) | ~43 horas | ~$132 USD |
| Fine-tuning piloto (500 iters) | ~1 hora | $0 (local) |
| Fine-tuning llama8b (3,000 iters) | ~11 horas | $0 (local) |
| Fine-tuning gpt-oss-20b (3,000 iters) | ~8.5 horas | $0 (local) |
| Fusión de modelos | ~20 minutos | $0 |
| Arena v2: ejecución de candidatos (15×6) | ~1 hora | ~$3-6 USD (API tiers) |
| Arena v2: evaluación Opus (90 juicios) | ~1.5 horas | ~$8-15 USD (Opus) |
| Evaluación humana (3-5 doctores) | ~1 semana (asíncrono) | $0 |
| Análisis y reporte final | ~2-4 horas | $0 |
| **Total** | ~4-5 días + 1 semana humanos | ~$145-165 USD |
