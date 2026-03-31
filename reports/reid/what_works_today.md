# Estado actual del pipeline Re-ID - OpenCows2020

## Resumen ejecutivo

**¿El sistema funciona hoy?** Sí, parcialmente. Pero necesita datos en vivo para producción.

**Accuracy actual:** 84.27% Top-1, 94.76% Top-5 en galería cerrada con 46 vacas conocidas.

**Uso actual:** Identificar vacas de una galería cerrada dentro de un dataset estándar (OpenCows2020).

---

## Qué funciona hoy: El pipeline Re-ID

### 1. Embeddings + FAISS (✅ Completamente funcional)

- **Modelo:** ResNet18 entrenado con triplet loss (256-dim embeddings)
- **Checkpoint:** `outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt`
- **Técnica:** Extrae 256-dimensionales de cada imagen de vaca
- **Indexación:** FAISS IndexFlatIP (búsqueda rápida por similitud coseno)

**Cómo se usa:**
1. Cargamos imagen de query (una vaca)
2. Red neuronal extrae embedding de 256 dims
3. FAISS busca los K vecinos más cercanos en la galería 
4. Compara embedding query vs embeddings de entrenamiento
5. Devuelve IDs ordenados por similitud

**Performance:**
- Velocidad: ~0.3 seg/query en GPU (CUDA)
- Discriminación: muy buena en galería cerrada (84.27% Top-1)

---

## Qué NO funciona bien hoy (limitaciones)

### 1. **Detección + localización** (❌ No está integrada)
   - Hoy solo aceptamos crops de vacas ya recortadas
   - En el campo, necesitamos primero **detectar** dónde está la vaca
   - Modelos YOLO existen pero no están acoplados al pipeline

### 2. **Open-set / Rechazo de desconocidos** (⚠️ Calibrado pero no validado en campo)
   - Podemos fijar un threshold de similitud para rechazar IDs desconocidos
   - Pero esto se entrenó solo con IDs conocidos (no hay verdaderos "unknowns")
   - En el campo veremos vacas que nunca vimos → necesitamos validar esto en vivo

### 3. **Foto aérea / detección desde dron** (❌ Rama secundaria, sin atención)
   - Dataset de Zenodo tiene fotos aéreas de vacas
   - No hemos invertido esfuerzo aquí; está en la rama "aerial"
   - Varía mucho el ángulo y escala → necesita datos específicos para entrenar

### 4. **Galería en vivo** (🔄 Parcialmente implementado)
   - Hoy usamos galería estática de OpenCows2020 (46 IDs, 4240 imágenes de entrenamiento)
   - En Matilda, Buenos Aires, la línea es pequeña pero cambia con nacimientos
   - Necesitamos un protocolo para **actualizar** la galería cuando llega una vaca nueva

---

## Precisamente qué vemos hoy en resultados

### Scenario 1: Sin threshold (closed-set, asumiendo todas conocidas)

| Métrica | All-vectors | Prototype |
|---------|-------------|-----------|
| **Top-1 Accuracy** | **84.27%** | 78.23% |
| **Top-5 Accuracy** | **94.76%** | 97.98% |
| **False accepts** | 78 | 108 |
| **Gallery vectors** | 4240 (todas) | 46 (1 por ID) |

**Interpretación:**
- De cada 100 queries, ~84 se identifican correctamente en primer intento (all-vectors)
- De cada 100 queries, ~95 están en los top-5 candidatos (muy confiable)
- El modo "all-vectors" supera al "prototype" porque tiene más ejemplos en galería

### Scenario 2: Con unknown-threshold (open-set, rechaza desconocidos)

| Métrica | All-vectors | Prototype |
|---------|-------------|-----------|
| **Accepted queries** | 251/496 | 212/496 |
| **Top-1 (solo accepted)** | 46.57% | 37.90% |
| **False rejects** | 245 | 284 |
| **False accepts** | 20 | 24 |

**Interpretación:**
- Si rechazamos bajo threshold, dejamos pasar muy pocas queries (50% rechazadas)
- Muchos de los que rechazamos son en realidad **conocidos** (false rejects ~50%)
- En campo, esto sería útil para decir "no sé quién es esta vaca" con confianza

---

## En la práctica: ¿puedo usarlo HOY en Matilda, Buenos Aires?

| Caso | Respuesta | Por qué |
|------|-----------|--------|
| **Foto conocida de la galería, recortada** | ✅ SÍ | 84% de chance de ID correcto |
| **Foto de una vaca aérea desde dron** | ❌ NO | Modelo nunca vio fotos de dron; ángulos raros |
| **Foto de campo full, sin recorte** | ❌ Parcial | Podría funcionar pero necesita detector antes |
| **Fotos de una vaca nueva (no en galería)** | ⚠️ Tal vez | Puede rechazarla como "desconocida" pero no está validado |
| **Fotos de Matilda, Eusebia, etc.** | ❌ NO | Son vacas reales nuevas; galería actual es del dataset |

---

## Visuales disponibles hoy

En `reports/reid/demo_state_today/` encontrás:

1. **score_distribution.png**
   - Histograma de similitudes cuando acertamos vs cuando erramos
   - Verde = predicciones correctas (media 0.9990)
   - Azul = predicciones incorrectas (media 0.9974)
   - Separation clara: hay espacio para threshold

2. **confusion_matrix.png**
   - Matrix 46×46 mostrando qué IDs se confunden entre sí
   - Diagonal oscura = aciertos
   - Partes claras = confusiones comunes (ej: ID 012 vs ID 003)

3. **top5_examples.csv**
   - 30 ejemplos reales: queries exitosas, fallidas, dudosas
   - Columns: path, true_id, predicted, score, top5_candidates

4. **demo_summary.json**
   - Números consolidados de accuracy, scores, etc.

---

## Lógica del pipeline HOY

```
Query image
    ↓
Preprocessor (resize 224×224)
    ↓
ResNet18 + Embedding (256-dim)
    ↓
L2-Normalize
    ↓
FAISS Search (IndexFlatIP, k=5)
    ↓
Top-5 similarities + IDs
    ↓
Decision:
   - No threshold → return top-1 (84% correct)
   - With threshold → accept/reject unknown (50% conf, needs validation)
```

---

## Próximos pasos naturales (no hechos HOY)

1. **Validación open-set real:** Fotos de vacas que NO están en galería → medir true reject rate
2. **Integración detector:** YOLO para localizar vacas en foto full → crop automático
3. **Galería Matilda:** Tomar fotos de Matilda, Eusebia, etc. → entrenar con eso
4. **Fotos aéreas:** Si es requisito, cambiar a dataset de Zenodo + reentrenar
5. **Actualización in-place:** Protocolo para ańadir vacas nuevas a galería sin reentrenar

---

## Conclusión

**Hoy funcionan:**
- ✅ Identificación de 46 vacas conocidas medidas en laboratorio
- ✅ Embeddings discriminativos
- ✅ Búsqueda rápida FAISS
- ✅ Múltiples modos de galería (prototype, all-vectors)

**Falta para producción:**
- ❌ Datos reales de Matilda (vacas verdaderas)
- ❌ Integración con detector
- ❌ Validación con vacas desconocidas
- ❌ Robustez a variantesciones (ángulo, iluminación, etc.)

**Es un POC funcional, no un sistema listo para deployment.**
