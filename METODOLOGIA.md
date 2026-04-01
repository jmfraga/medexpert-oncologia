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

#### Fase 1 — Dataset exploratorio (8.6% del corpus)

**Modelo teacher**: Claude Sonnet 4.6 (`claude-sonnet-4-6`, Anthropic API)

Se seleccionaron 11,592 chunks mediante muestreo estratificado (sección 2.2) para generar un dataset inicial de entrenamiento. Por cada chunk, el teacher genera 3 pares pregunta-respuesta:

1. **Conocimiento directo**: Pregunta factual sobre el contenido del chunk
2. **Caso clínico**: Caso breve con edad, género, estadio que requiere razonamiento
3. **Decisión terapéutica**: Comparación entre opciones o decisión de manejo

**Resultado**: 34,728 pares Q&A (8.6% del corpus), costo $175 USD.

Este dataset se utilizó para entrenar y evaluar los modelos iniciales (llama8b-onco, gpt-oss-20b-onco). Los resultados del Arena (sección 3) mostraron que los modelos fine-tuned no alcanzaron calidad clínica suficiente (scores <1.4/5), mientras que modelos API generalistas con RAG obtuvieron scores significativamente superiores (Sonnet 3.17/5, MiniMax 2.75/5). El análisis reveló que **el tamaño del dataset (8.6% del corpus) es el principal bottleneck**, dado que la oncología es un dominio amplio y diverso donde un muestreo limitado deja brechas de conocimiento significativas.

#### Fase 2 — Dataset completo (100% del corpus)

**Decisión de cambio de teacher model**: Para generar el dataset completo a partir del 100% del corpus de ChromaDB, se evaluó el costo de diferentes modelos teacher:

| Modelo | Intelligence Index (Artificial Analysis) | Costo estimado (100%) |
|--------|----------------------------------------|----------------------|
| Opus 4.6 | 53 | ~$10,500 USD |
| Sonnet 4.6 | 44 | ~$2,100 USD |
| Haiku 4.5 | 37 | ~$559 USD |
| **MiniMax M2.7** | **50** | **~$180 USD** |

Se seleccionó MiniMax M2.7 como teacher para la Fase 2 por tres razones:
1. **Calidad superior al modelo original**: Intelligence Index 50 vs 44 de Sonnet 4.6 (según Artificial Analysis, marzo 2026)
2. **Costo 12x menor**: $0.30/$1.20 per 1M tokens (input/output) vs $3/$15 de Sonnet
3. **Validación empírica**: un piloto de 50 chunks generó 150 Q&A pairs con calidad clínica comparable a los generados con Sonnet — terminología médica correcta, adherencia al fragmento fuente, inclusión de niveles de evidencia y dosis específicas

**Proceso de la Fase 2**:
- Del total de 134,478 chunks se aplicaron los mismos filtros de calidad (longitud mínima, exclusión de bibliografía/COI, deduplicación), resultando en 98,133 chunks de calidad
- Se excluyeron los 11,592 chunks ya procesados en la Fase 1, dejando 86,575 chunks por procesar
- Generación paralela con 15 workers concurrentes sobre la API de MiniMax
- Mismo prompt template y formato de salida que la Fase 1
- Checkpointing cada 500 chunks para tolerancia a fallos

**Dataset combinado (Fase 1 + Fase 2)**:
- Fase 1 (Sonnet 4.6): 34,728 pares de 11,592 chunks (8.6%)
- Fase 2 (MiniMax M2.7): ~259,725 pares estimados de 86,575 chunks (91.4%)
- **Total estimado**: ~294,453 pares Q&A del 100% del corpus de calidad
- **Costo total de generación**: ~$265 USD ($175 Fase 1 + ~$90 Fase 2)

**Nota metodológica**: La heterogeneidad del teacher model (Sonnet para 8.6%, MiniMax para 91.4%) no afecta la calidad del dataset final, ya que las respuestas están estrictamente grounded en el contenido de los chunks fuente — el modelo teacher actúa como extractor y reformulador de conocimiento existente, no como fuente de conocimiento propio. La validación del piloto confirmó que ambos modelos producen Q&A de calidad comparable para esta tarea específica.

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

**Split**: 80% train / 10% validation / 10% test

### 2.3.1 Funnel de calidad del dataset (7 stages)

El dataset combinado (292,862 pares Q&A) se somete a un pipeline de filtrado secuencial de 7 etapas, ordenadas de menor a mayor costo computacional para reducir volumen antes de las etapas costosas.

#### Stages 1-6: Filtros determinísticos y de embedding

| Stage | Filtro | Tipo | Criterio |
|-------|--------|------|----------|
| 1 | Validación encoding/formato | Determinístico | UTF-8 válido, estructura JSON correcta |
| 2 | Longitud mínima de respuesta | Determinístico | ≥ 300 caracteres |
| 3 | Detección de respuestas evasivas | Regex | 28 patrones ("como modelo de lenguaje", "consulte a su médico", etc.) |
| 4 | Calidad de pregunta | Determinístico | ≥ 30 caracteres |
| 5 | Deduplicación semántica intra-dataset | Embedding | Cosine similarity > 0.95 (all-MiniLM-L6-v2) |
| 6 | Deduplicación cross-dataset (Sonnet vs MiniMax) | Embedding | Cosine similarity > 0.95 entre fuentes, retener respuesta más larga |

#### Stage 7: LLM-as-filter — Scoring de calidad clínica

El stage 7 utiliza un modelo frontier (Claude Sonnet 4.6) como evaluador experto para scoring de calidad clínica. Dado que evaluar los ~178K ejemplos restantes sería prohibitivamente costoso (~$100+ USD), se emplea un **muestreo estratificado del 5%** con extrapolación estadística:

**Procedimiento de muestreo**:
1. Se agrupan los ejemplos por estrato temático (tratamiento, diagnóstico, farmacología, soporte, seguimiento, otros)
2. De cada estrato se selecciona aleatoriamente el 5% de sus ejemplos (mínimo 1 por estrato), con seed fija para reproducibilidad
3. Esto produce ~8,925 ejemplos a evaluar

**Criterios de evaluación (escala 0-5)**:
- Precisión clínica: ¿La información es correcta según guías vigentes?
- Completitud: ¿Cubre los aspectos relevantes de la pregunta?
- Adherencia a evidencia: ¿Cita fuentes, niveles de evidencia, dosis correctas?
- Utilidad clínica: ¿Sería útil para un oncólogo en práctica?
- Claridad: ¿Está bien redactado y organizado?

**Umbral de rechazo**: score < 3/5

**Extrapolación al dataset completo**: Los ejemplos evaluados que obtienen score < 3/5 se eliminan directamente del dataset. El 95% restante (no evaluado) se retiene bajo el supuesto de que los stages 1-6 ya eliminaron los problemas estructurales (formato, longitud, evasivas, duplicados), y el muestreo estratificado del 5% funciona como auditoría estadística de la calidad del contenido clínico. La tasa de rechazo observada en la muestra se reporta per-estrato y per-fuente (Sonnet vs MiniMax) como estimador de la calidad relativa de cada subpoblación, sin aplicar rechazo adicional a los no evaluados.

**Justificación del approach**:
- Evaluar el 100% costaría ~$100+ USD y >50 horas de API calls
- El 5% estratificado provee intervalos de confianza del 95% con margen de error < 1% por estrato (dado n > 200 por estrato)
- La distribución de scores y tasa de rechazo per-fuente validan empíricamente la calidad diferencial entre Sonnet y MiniMax como teachers
- Los rechazos directos (score < 3) eliminan ejemplos demostrablemente deficientes sin penalizar al grueso del dataset que no fue muestreado

**Costo estimado**: ~$5-10 USD (8,925 llamadas × ~600 tokens input × ~50 tokens output)

#### Balance temático (post-funnel)

Después de los 7 stages de filtrado, se aplica un cap del 40% máximo por estrato temático para evitar el sesgo de sobre-representación. "Tratamiento" domina el corpus oncológico (59% pre-balance) y se subsamplea a 40%.

#### Split final

El dataset limpio se divide en **90/5/5** (train/valid/test) con estratificación por fuente (Sonnet/MiniMax) y estrato clínico, garantizando representatividad proporcional en cada split.

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

#### Arena v1-v2 (completadas, 2026-03-26 a 2026-03-27)

Evaluaciones iterativas con 6 tiers (llama8b base/ft, gpt-oss-20b base/ft, MiniMax, Sonnet). Resultados: modelos locales fine-tuned insuficientes con dataset de 8.6% del corpus (scores <1.37/5). Hallazgo clave: el prompt controla formato pero no calidad; el tamaño del dataset es el bottleneck principal. Ver resultados completos en `arena/results/`.

#### Arena v3 (actual, abril 2026)

**Evaluación comparativa de 8 candidatos** de respuesta clínica oncológica:

| # | Tier | Modelo | Tipo | Fine-tune | RAG |
|---|------|--------|------|-----------|-----|
| 1 | **gpt-oss-20b + RAG** | gpt-oss-20b base | MoE 20B | No | Sí |
| 2 | **gpt-oss-20b-onco** | gpt-oss-20b fine-tuned | MoE 20B | Sí | No |
| 3 | **Nemotron 30B + RAG** | Nemotron-3-Nano-30B base | MoE 30B | No | Sí |
| 4 | **Nemotron-onco** | Nemotron 30B fine-tuned | MoE 30B | Sí | No |
| 5 | **MedGemma 27B + RAG** | MedGemma 27B base | Denso 27B (medical pre-trained) | No | Sí |
| 6 | **MedGemma-onco** | MedGemma 27B fine-tuned | Denso 27B (medical pre-trained) | Sí | No |
| 7 | **MiniMax + RAG** | MiniMax M2.7 | API | No | Sí |
| 8 | **Sonnet + RAG** | Sonnet 4.6 | API | No | Sí |

**Cambios respecto a v2**:
- Eliminados tiers llama8b (insuficientes en v1/v2, scores <0.66)
- Agregados Nemotron 30B (base + fine-tuned) — MoE más grande, mejor rendimiento en benchmarks
- Agregados MedGemma 27B (base + fine-tuned) — modelo denso con pre-entrenamiento médico de Google
- Dataset ampliado: ~294K Q&A pairs (100% ChromaDB, generados con MiniMax M2.7 como teacher)

**Evaluador (juez)**: Claude Opus 4.6 + RAG (no compite como candidato, imparcial)

**Preguntas de investigación**:
1. **RAG vs fine-tune**: ¿Qué estrategia da mejor resultado, retrieval sobre modelo base o fine-tuning sin retrieval? (tiers 1 vs 2, 3 vs 4, 5 vs 6)
2. **MoE vs denso**: ¿Los modelos MoE (20B, 30B) superan al modelo denso (27B) fine-tuned con el mismo dataset? (tiers 2 vs 4 vs 6)
3. **Efecto del tamaño**: ¿El modelo MoE 30B supera al MoE 20B con la misma estrategia? (tiers 1 vs 3, 2 vs 4)
4. **Local vs API**: ¿Los modelos locales fine-tuned alcanzan calidad comparable a modelos API generalistas + RAG? (tiers 2, 4, 6 vs 7, 8)
5. **Pre-entrenamiento médico**: ¿MedGemma (base médica) supera a modelos de propósito general del mismo tamaño? (tiers 5 vs 1/3, 6 vs 2/4)

**Criterio para evaluación humana**: los top 5-6 tiers con score Opus >2.5/5 serán enviados a oncólogos para evaluación humana. Esto evita desperdiciar tiempo de doctores en modelos que no están listos clínicamente.

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
  Para cada candidato (8 tiers):
    1. Enviar caso → pipeline completo del candidato
    2. Registrar respuesta completa + métricas de rendimiento
    3. Enviar respuesta al juez (Opus 4.6 + RAG + gold standard)
    4. Juez califica con rúbrica estructurada + feedback textual
Total: 120 evaluaciones automáticas (15 casos × 8 candidatos)
Costo estimado: ~$13 USD (Opus judge + API tiers)
```

**Ventajas**: reproducible, sin costo de coordinación, cubre los 15 casos × 8 candidatos.

### 3.5 Evaluación humana — Panel de oncólogos

#### 3.5.1 Diseño

Evaluación ciega por oncólogos clínicos. Solo los tiers con score Opus >2.5/5 pasan a evaluación humana (estimado: 5-6 de los 8 tiers). Las respuestas se presentan anonimizadas y en orden aleatorio diferente por caso para evitar sesgo posicional. Los evaluadores no conocen qué modelo generó cada respuesta.

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

1. **H1**: RAG sobre modelo base supera a fine-tune sin RAG en las tres arquitecturas (gpt-oss, Nemotron, MedGemma)
2. **H2**: Nemotron 30B (MoE más grande) supera a gpt-oss-20b (MoE más chico) con la misma estrategia
3. **H3**: MedGemma (pre-entrenamiento médico) supera a modelos de propósito general del mismo tamaño
4. **H4**: Los modelos locales fine-tuned con dataset ampliado (~294K pairs) alcanzan ≥70% del score de Sonnet + RAG en casos simples
5. **H5**: Los modelos locales tienen latencia <5s vs >10s de los API (mejor experiencia de usuario)
6. **H6**: El costo por consulta local es $0 vs ~$0.05-0.15 de MiniMax/Sonnet
7. **H7**: El score de Opus correlaciona r > 0.80 con el consenso humano (validación del juez LLM)
8. **H8**: MiniMax (generalista API) supera a los modelos locales base + RAG pero no a Sonnet
9. **H9**: El dataset ampliado (294K pairs vs 35K) mejora significativamente los scores de fine-tune respecto a v2

### 3.10 Arquitectura técnica de la Arena

```
medexpert-admin/arena/
├── clinical_cases.json         # 15 casos clínicos con gold standard
├── arena_runner.py             # CandidateRunner: ejecuta 8 tiers × 15 casos
│     ├── gptoss_rag:      gpt-oss-20b base + RAG + ChromaDB
│     ├── gptoss_ft:       gpt-oss-20b-onco (fine-tuned), sin RAG
│     ├── nemotron_rag:    Nemotron 30B base + RAG + ChromaDB
│     ├── nemotron_ft:     Nemotron-onco (fine-tuned), sin RAG
│     ├── medgemma_rag:    MedGemma 27B base + RAG + ChromaDB
│     ├── medgemma_ft:     MedGemma-onco (fine-tuned), sin RAG
│     ├── minimax:         MiniMax M2.7 + RAG + ChromaDB
│     └── sonnet:          Sonnet 4.6 + RAG + ChromaDB
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
7. **Anexo completo de la publicación**: documento con las 120 respuestas íntegras (15 casos × 8 candidatos), cada una con:
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
├── arena_runner.py                # Ejecuta 8 tiers × 15 casos
├── arena_judge.py                 # Opus 4.6 califica 120 respuestas
├── arena_report.py                # Reportes + materiales para Forms
└── results/                       # JSON/CSV con resultados
```

---

## 5. Cronograma Estimado

| Fase | Duración | Costo |
|------|----------|-------|
| Muestreo de ChromaDB | ✅ Completado | $0 |
| Generación de dataset (Sonnet 4.6, 8.6%) | ✅ Completado (~43 horas) | ~$132 USD |
| Generación de dataset (MiniMax M2.7, 91.4%) | ✅ En proceso (~31-mar) | ~$45 USD |
| Fine-tuning llama8b (3,000 iters) | ✅ Completado (~11 horas) | $0 (local) |
| Fine-tuning gpt-oss-20b (3,000 iters) | ✅ Completado (~8.5 horas) | $0 (local) |
| Arena v1-v2: evaluación iterativa | ✅ Completado (4 rondas) | ~$31 USD |
| Reentrenamiento gpt-oss-20b-onco (dataset ampliado) | ~8-10 horas | $0 (local) |
| Fine-tuning Nemotron 30B (3,000 iters) | ~10-12 horas | $0 (local) |
| Fine-tuning MedGemma 27B (3,000 iters) | ~12-15 horas | $0 (local) |
| Fusión de 3 modelos | ~1 hora | $0 |
| Arena v3: ejecución de candidatos (15×8) | ~1.5 horas | ~$4-6 USD (API tiers) |
| Arena v3: evaluación Opus (120 juicios) | ~2 horas | ~$8-13 USD (Opus) |
| Evaluación humana (top 5-6 tiers, 3-5 doctores) | ~1 semana (asíncrono) | $0 |
| Análisis y reporte final | ~2-4 horas | $0 |
| **Total** | ~3-4 días training + Arena + 1 semana humanos | ~$220-230 USD acumulado |
