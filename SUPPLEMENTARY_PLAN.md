# Plan de Material Suplementario — npj Digital Medicine

**Paper:** Base Language Models Outperform RAG and Fine-Tuning for Clinical Oncology…
**Fecha:** 2026-07-20

## 1. Reglas de npj Digital Medicine (extraídas del Guide to Authors)

| Regla | Especificación |
|---|---|
| Tamaño por archivo suplementario | **≤ 30 MB** |
| Tamaño total combinado | **≤ 150 MB** |
| Formato preferido | Un **PDF único combinado** para Supplementary Figures/Tables/Notes |
| Datos grandes / tablas Excel | Como **Supplementary Data XX** (archivo aparte: .xlsx, .txt, .csv-en-txt) |
| Etiquetado | "Supplementary Table/Figure/Data/Note 1…", numeración separada del main |
| Citar en texto | Cada ítem suplementario debe referenciarse ≥1 vez en el manuscrito |
| Supplementary **Methods** | **YA NO se permiten** — todo Methods va en el cuerpo |
| Datasets grandes / código / modelos | **Repositorio público obligatorio** (no como archivo suplementario). DOI persistente (Zenodo/Figshare); accession/DOI citado en el statement |
| Data & Code Availability | **Statements obligatorios**, sección aparte tras Methods |

## 2. El problema de fondo: tamaño + copyright

Ninguno de los tres artefactos "pesados" puede ir como archivo suplementario:

| Artefacto | Tamaño real | ¿Cabe en SI? | Copyright | Destino |
|---|---|---|---|---|
| **Adapters fine-tuned (LoRA)** | dir 100 GB (con checkpoints); adapter final ~cientos MB c/u | ❌ (>30 MB) | Propio (OK compartir) | **Hugging Face** + Zenodo DOI |
| **RAG / ChromaDB** | 768 MB (índice + embeddings) | ❌ | ⚠️ **Deriva de NCCN/ESMO — NO redistribuible** | Solo **código + descripción**; NO el índice |
| **Dataset sintético Q&A** | 179 MB (limpio) / 42 MB (crudo) | ❌ | ⚠️ Derivado de texto con copyright | **Muestra** redistribuible + acceso bajo solicitud |

> **Punto crítico de derechos de autor.** Las guías **NCCN y ESMO son material con copyright**. El índice ChromaDB y los embeddings *reconstituyen* ese texto, y el dataset sintético se *generó a partir* de él. Publicarlos abiertamente es muy probablemente una violación de licencia. La solución estándar (y lo que hace el paper): **describir la construcción del RAG en Methods con todo detalle** (fuentes, chunking, indexación, retrieval) y **liberar el código del pipeline**, pero **no el corpus ni los embeddings**. Las guías fuente son públicas en sus organizaciones; el lector las reconstruye con nuestro código.

## 3. Qué SÍ enviar como Supplementary (pequeño y sin copyright)

Todo esto es sintético/propio → seguro. Cabe holgado en los límites:

| Ítem | Contenido | Formato | Tamaño aprox |
|---|---|---|---|
| **Supplementary Data 1** | 15 casos clínicos + gold standard | .xlsx / .txt | <1 MB |
| **Supplementary Data 2** | Arena completa (480 evals: caso×tier×criterio, 5 rondas) | .xlsx | <2 MB |
| **Supplementary Data 3** | Evaluación humana de-identificada (104 ratings, 6 evaluadores) | .csv→.txt | <50 KB |
| **Supplementary Data 4** | Llave de anonimización (seed 42) | .txt/.json | <10 KB |
| **Supplementary Table 1** | Rúbrica completa con descriptores ancla | en PDF combinado | — |
| **Supplementary Table 2** | Salidas del modelo mixto por criterio (medias, IC, p) + Krippendorff α | en PDF combinado | — |
| **Supplementary Note 1** | Plantillas de prompt (SAER) generación + evaluación | en PDF combinado | — |
| **Supplementary Note 2** | Hiperparámetros QLoRA + curvas de convergencia | en PDF combinado | — |
| **Supplementary Note 3** | Especificación estadística + robustez | en PDF combinado | — |

→ **Un PDF combinado** (Tables + Notes + Figures suplementarias) **+ 4 archivos Supplementary Data**. Total < 10 MB. Sin problema.

## 4. Cómo enviar modelos y código (repositorios externos)

1. **GitHub** — `github.com/jmfraga/medexpert-oncologia` (ya existe): pipeline completo (sampling, generación, quality funnel, fine-tune, capa RAG/ChromaDB, arena+judge, análisis estadístico). Limpiar de PII/IPs internas antes de hacerlo público.
2. **Zenodo** — snapshot del repo GitHub → **DOI persistente** (Nature lo prefiere sobre solo-GitHub porque GitHub no es permanente). Zenodo tiene integración directa con GitHub (release → DOI automático).
3. **Hugging Face** — subir los **adapters LoRA finales** (no los 100 GB de checkpoints; solo el mejor checkpoint por modelo, ~cientos MB): `gemma4-31b-onco`, `gemma4-26b-moe-onco`, `gpt-oss-20b-onco`, `llama8b-onco`. Tarjeta de modelo que aclare: se liberan por reproducibilidad y como **resultado negativo documentado** (el FT degradó calidad), no como artefacto de producción.
4. **Dataset sintético** — muestra redistribuible (p. ej. 1–2 k ejemplos que no reproduzcan guías completas) en el repo; dataset completo bajo **solicitud controlada** por la restricción de copyright.

## 5. Acciones concretas para JMF

- [ ] Decidir alcance de release público (Data Availability tiene `[AUTHOR DECISION]`).
- [ ] Identificar el **checkpoint final** de cada adapter (evitar subir los 100 GB) y subir a Hugging Face.
- [ ] Crear release en GitHub → conectar Zenodo → obtener DOI; pegarlo en Data/Code Availability.
- [ ] Auditar el repo por IPs/hostnames internos y PII antes de hacerlo público (regla de la flota).
- [ ] Exportar los 4 Supplementary Data desde los datos que ya tenemos (los tengo listos para generar si quieres).
- [ ] Confirmar con asesoría legal/institucional la postura sobre el dataset derivado de NCCN/ESMO.

## 6. Recomendación en una línea

**No mandes el RAG ni los modelos como material suplementario.** El suplementario de npj es para los datos pequeños y sintéticos (casos, scores, rúbrica, prompts). Los modelos van a Hugging Face, el código a GitHub+Zenodo (DOI), y el RAG **solo como código + descripción** — nunca el corpus, por copyright de NCCN/ESMO.
