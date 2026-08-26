# Google Forms — Instrucciones para Crear el Formulario de Evaluación

## Concepto

Un solo formulario para todos los evaluadores. Cada evaluador recibe una lista de asignaciones (ej. "Caso 3 Respuesta B, Caso 7 Respuesta D..."). Para cada asignación, llenan una entrada del formulario con el número de caso y la letra de respuesta que están evaluando, más sus calificaciones.

---

## Estructura del formulario

### Sección 1 — Identificación del evaluador

| Campo | Tipo | Opciones | Requerido |
|-------|------|----------|-----------|
| Nombre del evaluador | Respuesta corta | — | Sí |
| Correo electrónico | Respuesta corta | — | Sí |

### Sección 2 — Identificación de la evaluación

| Campo | Tipo | Opciones | Requerido |
|-------|------|----------|-----------|
| Número de caso | Desplegable | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | Sí |
| Letra de respuesta | Desplegable | A, B, C, D, E | Sí |

### Sección 3 — Evaluación con rúbrica (4 criterios)

Cada criterio se califica en escala **0–5**:

| Puntaje | Significado |
|---------|-------------|
| 0 | Completamente incorrecto / ausente |
| 1 | Mayormente incorrecto, errores graves |
| 2 | Parcialmente correcto, errores significativos |
| 3 | Aceptable, algunos errores menores |
| 4 | Bueno, errores mínimos |
| 5 | Excelente, sin errores relevantes |

#### Criterio 1 — Precisión diagnóstica (peso 25%)
> ¿El diagnóstico y la estadificación coinciden con el gold standard?

| Campo | Tipo | Opciones | Requerido |
|-------|------|----------|-----------|
| Precisión diagnóstica | Escala lineal | 0–5 | Sí |

#### Criterio 2 — Apego a guías clínicas (peso 30%)
> ¿Las recomendaciones son consistentes con guías NCCN/ESMO/IMSS? ¿Cita correctamente?

| Campo | Tipo | Opciones | Requerido |
|-------|------|----------|-----------|
| Apego a guías | Escala lineal | 0–5 | Sí |

#### Criterio 3 — Completitud (peso 25%)
> ¿Cubre todos los aspectos relevantes: tratamiento, alternativas, seguimiento, efectos adversos?

| Campo | Tipo | Opciones | Requerido |
|-------|------|----------|-----------|
| Completitud | Escala lineal | 0–5 | Sí |

#### Criterio 4 — Utilidad clínica (peso 20%)
> ¿La respuesta es práctica y accionable para un oncólogo en consulta?

| Campo | Tipo | Opciones | Requerido |
|-------|------|----------|-----------|
| Utilidad clínica | Escala lineal | 0–5 | Sí |

### Sección 4 — Comentarios

| Campo | Tipo | Opciones | Requerido |
|-------|------|----------|-----------|
| Comentarios libres | Párrafo | "Errores específicos, observaciones, aspectos destacados" | No |

---

## Configuración del formulario

1. **Título**: "MedExpert Arena — Evaluación de Respuestas Clínicas en Oncología"
2. **Descripción**:
   > Usted recibirá un documento con respuestas generadas por diferentes sistemas de IA para casos clínicos oncológicos. Cada respuesta está identificada por un número de caso (1–15) y una letra (A–E). Para cada respuesta asignada, complete una entrada de este formulario.
   >
   > Las respuestas están anonimizadas — usted no sabe qué sistema generó cada una. Evalúe cada respuesta de forma independiente usando la rúbrica proporcionada.

3. **Permitir editar respuestas**: Sí
4. **Limitar a 1 respuesta**: NO (cada evaluador envía múltiples entradas)
5. **Recopilar correos**: Sí
6. **Orden de preguntas**: Fijo (no aleatorizar)

---

## Flujo del evaluador

```
1. Recibe por correo:
   - Link al formulario
   - PDF con sus casos asignados (ej. Casos 1, 3, 5, 7, 9, 11)
   - Cada caso tiene 5 respuestas (A–E) impresas

2. Lee el caso clínico

3. Lee la Respuesta A → llena el formulario:
   Caso: 1, Respuesta: A, scores: [4, 3, 4, 5], comentarios: "..."

4. Lee la Respuesta B → llena el formulario de nuevo:
   Caso: 1, Respuesta: B, scores: [2, 2, 3, 2], comentarios: "..."

5. Repite para C, D, E del mismo caso

6. Pasa al siguiente caso asignado
```

---

## Escenarios de asignación

| Escenario | Oncólogos | Casos/evaluador | Respuestas/caso | Entradas/evaluador | Total entradas |
|-----------|-----------|-----------------|------------------|--------------------|----------------|
| A (mínimo) | 3 | 10 | 5 (A–E) | 50 | 150 |
| B (recomendado) | 4 | 8 | 5 (A–E) | 40 | 150* |
| C (ideal) | 5 | 6 | 5 (A–E) | 30 | 150 |

*En escenario B, algunos casos tendrán 2 evaluadores y otros 3, para cubrir los 15 casos × 2 evaluadores mínimo.

Cada caso es evaluado por al menos 2 oncólogos para calcular concordancia inter-evaluador (Krippendorff's alpha).

---

## Material que se entrega a cada evaluador

1. **Documento PDF personalizado** con:
   - Instrucciones y rúbrica
   - Sus N casos asignados
   - Para cada caso: presentación clínica + 5 respuestas anonimizadas (A–E)
   - **NO incluir el gold standard** (evitar sesgo)

2. **Link al formulario Google**

3. **Tabla de asignación** (qué casos evalúa cada doctor)

---

## Mapeo interno (no compartir con evaluadores)

La asignación de letras a tiers se aleatoriza por caso con seed fijo para reproducibilidad. Ejemplo:

| Caso | A | B | C | D | E |
|------|---|---|---|---|---|
| 1 | Light | Premium | Básico A | Light+RAG | Básico B |
| 2 | Premium | Básico B | Light+RAG | Light | Básico A |
| ... | (aleatorio por caso) | | | | |

Este mapeo se genera automáticamente por `arena_report.py` y se guarda en `results/letter_mapping.json`. Solo el investigador principal tiene acceso.
