# Demo visual del pipeline Re-ID - OpenCows2020

## Acceso a las visuales

Todos los artefactos están en: `reports/reid/demo_state_today/`

---

## 1. Distribución de scores (score_distribution.png)

**Qué muestra:**
- Histograma de similitudes top-1
- Verde: cuando acertamos (predicción correcta)
- Azul: cuando nos equivocamos (predicción incorrecta)

**Qué observamos:**
- Verde está concentrado en 0.9975-1.0000 (muy alta confianza)
- Azul también está alto (~0.9974 media) pero con distribución más ancha
- Separation clara: hay espacio para threshold

**Interpretación:**
- El modelo emite scores ALTOS siempre (FAISS con cosine similarity normalizada)
- Cuando acierta, scores son más altos
- Cuando se equivoca, aún son altos (similares entre ellos)
- → OpenCows2020 dataset es fácil; todas las vacas se ven similares

---

## 2. Matriz de confusión (confusion_matrix.png)

**Qué muestra:**
- Grid 46×46 (una celda por par de IDs)
- Diagonal = aciertos (ID true == ID predicho) = oscuro
- Fuera diagonal = confusiones (predijo mal) = claro

**Qué observamos:**
- Diagonal muy marcada → modelo discrimina bien
- Algunas confusiones recurrentes:
  - ID 012 y 003 a veces se confunden
  - ID 001 y 016 generan algunos errores
- Patrón: confusiones son simétricas (si A→B se confunde, B→A también)

**Interpretación:**
- Hay pares de vacas que el modelo ve como similares
- 84% Top-1 accuracy = diagonal es más fuerte que fuera-diagonal

---

## 3. Ejemplos concretos (top5_examples.csv)

**Qué incluye:**
- 30 ejemplos reales de queries
- Columnas: ruta imagen, true_id, predicción, score, top-5 candidatos

**Tipos de ejemplos:**

A) **Correct high confidence** (10 ejemplos)
```
query_path: data/.../test/001/000001.jpg
true_id: 001
pred_top1: 001
similarity: 0.9956
top5: 001|001|001|001|001
→ Acertó con mucha confianza (todos top-5 son ID 001)
```

B) **Incorrect high score** (10 ejemplos)
```
query_path: data/.../test/002/000008.jpg
true_id: 002
pred_top1: 013
similarity: 0.9988 (¡muy alto!)
top5: 013|013|013|013|013
→ Se equivocó pero con ALTA confianza
→ Esto es un problema para open-set; rechaza falsos negativamente
```

C) **Correct low confidence** (5 ejemplos)
```
query_path: data/.../test/001/000003.jpg
true_id: 001
pred_top1: 012 (¡¡¡ se equivocó top-1)
topk: 012|012|001|001|001 (pero está en top-3)
similarity: 0.9957
top5: 012|012|001|001|001
→ No acertó top-1 pero el verdadero está en top-5
→ Esto es el 94.76% Top-5 accuracy
```

---

## 4. Estadísticas consolidadas (demo_summary.json)

```json
{
  "total_queries": 496,
  "correct_predictions": 418,
  "incorrect_predictions": 78,
  "top1_accuracy": 0.8427,
  "mean_score_when_correct": 0.9990,
  "mean_score_when_incorrect": 0.9974,
  "min_score_when_correct": 0.9730,
  "min_score_when_incorrect": 0.9488,
  "max_score_when_incorrect": 0.9999
}
```

**Lectura:**
- 418/496 queries acertadas (84.27%)
- Cuando acertamos: scores ~0.9990
- Cuando nos equivocamos: scores ~0.9974 (apenas 0.0016 menos)
- Rango incorrecto: 0.9488-0.9999 (ANCHO)
- Rango correcto: 0.9730-1.0000 (más estrecho)

---

## Conclusiones de las visuales

### ✅ Lo que funciona bien

1. **Discriminación base:** El modelo crea embeddings que separar vacas
2. **Confidence:** Cuando acierta, emite score alto (+0.998)
3. **Estabilidad:** Top-5 es 94.76% (muy confiable si permitimos top-5)

### ⚠️ Desafíos observados

1. **Scores muy altos siempre:** OpenCows2020 es un dataset fácil
   - En campo, esperamos scores más bajos
   - Thresholds aquí pueden no servir en producción

2. **Confusiones correlacionadas:** Si 012→003, también 003→012
   - Las vacas 012 y 003 estructuralmente se ven parecidas
   - No es ruido del modelo; es datos del dataset

3. **No hay margen de safety para unknowns:**
   - Max score cuando incorrecto: 0.9999
   - Min score cuando correcto: 0.9730
   - Overlap grande → threshold es muy frágil

### 🔄 Lo que necesitamos en campo

1. Datos reales de Matilda con variación (ángles, luz, distancia)
2. Fotos de vacas desconocidas para validar rechazo
3. Reentrenamiento si scores en campo son significativamente más bajos

---

## Archivos generados

```
reports/reid/demo_state_today/
├── score_distribution.png      (histograma)
├── confusion_matrix.png         (heatmap)
├── top5_examples.csv           (ejemplos concretos)
├── demo_summary.json           (stats)
└── (metadata + annotations)
```

**Usar para:**
- Presentación ejecutiva: muestra qué funciona hoy
- Debugging: entiende dónde falla el modelo
- Baseline: compara estos números contra datos reales de Matilda

---

## Siguiente paso

Una vez tengas fotos de Matilda, corre el mismo script:
```bash
python scripts/generate_demo_visuals.py \
  --query_csv "reports/reid/matilda_real_queries.csv" \
  --out_dir "reports/reid/demo_matilda_real"
```

Compara visuales de OpenCows2020 vs Matilda. Si scores bajan significativamente, sabrás que necesitas reentrenamiento.
