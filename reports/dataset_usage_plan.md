# Plan de uso de datasets - Timeline del proyecto

## Resumen ejecutivo

Tenemos **4 datasets + 1 repositorio real**. Cada uno sirve para una etapa distinta. Hoy usamos OpenCows2020 para validar arquitectura. Mañana necesitamos datos reales de Matilda.

---

## Dataset 1: OpenCows2020 (En uso HOY ✅)

### Qué es
- **Source:** https://doi.org/10.34148/1/2020.12
- **Contenido:** 46 vacas de raza Frisona blanco-negro, idendificadas
- **Imágenes:** 
  - Training: 4240 imágenes, 46 IDs
  - Test: 496 imágenes, 46 IDs (mismos IDs)
- **Contexto:** Fotos estándar de laboratorio, ángulos controlados, buena iluminación

### Para qué lo usamos HOY
1. **Validar arquitectura:** ¿Funciona el modelo de embeddings en general?
2. **Baseline cerrado:** ¿Qué accuracy máxima podemos lograr con datos limpios?
3. **Tuning de thresholds:** Calibrar unknown-threshold sin ruido

### Resultados con OpenCows2020
- **All-vectors mode:** Top-1 84.27%, Top-5 94.76%
- **Learnings:** La arquitectura funciona; discriminación es fuerte

### Cuándo dejamos de usarlo
Una vez validemos con **datos reales de Matilda**. OpenCows2020 es un dataset de laboratorio limpio; Matilda serén condiciones reales (ángulos, iluminación variables).

---

## Dataset 2: MultiCamCows2024 (Próximo → ROBUSTECER)

### Qué es
- **Source:** https://zenodo.org/records/12776530
- **Contenido:** Vacas identificadas, múltiples cámaras, ángulos variados
- **Context:** Más cercano a campo que OpenCows2020, pero aún controlar
- **Imágenes:** 100+ hours de video de 8 vacas (Frisona + Jersey + Montbéliarde)

### Para qué lo necesitamos
1. **Robustez a variación:** Ángulos, distancias, iluminación no controlada
2. **Finer embeddings:** El modelo de OpenCows2020 probablemente no generalice bien a Matilda
3. **Validación abierta:** Algunas cámaras mezclan vacas; buen test para open-set real

### Cuándo lo usamos
- **Etapa 2:** Después de validar demo con OpenCows2020
- **Acción:** Reentrenar ResNet18 con MultiCamCows2024 de entrenamiento
- **Esperado:** Top-1 ~70-80% (será más difícil pero más realista)

### Cómo integrar
```
1. Descargar + procesar videos (convertirlos a frames)
2. Entrenar modelo nuevamente: train_reid.py con MultiCamCows2024
3. Generar galería con embeddings nuevos
4. Evaluar Top-1/Top-5 en test split
5. Subir checkpoint de ese modelo a production
```

---

## Dataset 3: Zenodo Aerial (~100 fotos aéreas + drone)

### Qué es
- **Source:** Zenodo, imágenes aéreas de vacas desde dron/altura
- **Contenido:** Vacas vistas desde arriba (ortogonal, 45°, etc.)
- **Problem:** Ángulos radicalmente distintos a fotos de frente/lado

### Para qué sirve
- **Detección aérea:** Si el proyecto escala a "monitoreo aéreo de rebaño"
- **Localización:** Primero detectar dónde está la vaca, luego identificar

### Por qué NO es el foco AHORA
1. **Prioridad baja:** OpenCows2020 + MultiCamCows2024 cierran el loop de Re-ID a ras de suelo
2. **Ángulos extremos:** Requería modelo separado o data augmentation especial
3. **Sin captura actual:** No estamos usando captura aérea hoy en Matilda

### Cuándo lo usamos
- **Etapa 3+:** Si el scope incluye "monitoreo desde dron"
- **Acción:** Entrenar **detector + identificador aéreo** (YOLO + Re-ID aerial-specific)
- **Timeline:** 6+ meses después de cerrar identificación en campo

---

## Dataset 4: Zenodo Benchmark (validación general)

### Qué es
- **Source:** Zenodo, benchmark multi-especie (bovinos, ovejas, cabras)
- **Contenido:** datasets públicos compilados para validación

### Para qué sirve
- **Generalización:** ¿El modelo funciona en otras razas/especies?
- **Robustez:** Edge cases, variación extrema

### Por qué NO es prioritario
- Primero validamos en una especie/raza (Frisona)
- Depois generalizamos

---

## Dataset 5: Datos REALES de Matilda (SIN DESCARGA, pero ES EL OBJETIVO)

### Qué es
- Vacas reales de la línea de Matilda, Buenos Aires
- Fotos tomadas en campo (smartphone, célula, cámara fija)
- Luz natural, ángulos reales, ruido de fondo

### Para qué
- **Producción:** Esto es lo que importa realmente
- De ahí generamos galería final con verdaderas Matilda, Eusebia, etc.

### Cuándo lo necesitamos
- **Etapa 2b:** Paralelo a validación con MultiCamCows2024
- **Acción:**
  1. Tomar 50-100 fotos de cada vaca (5+ ángulos), 2-3 sesiones
  2. Extraer embeddings con modelo entrenado en OpenCows2020 (o MultiCamCows2024)
  3. Generar galería ("Matilda_prod_gallery_v1.index")
  4. Test contra fotos nuevas de campo
  5. Si Top-1 < 60%, reentrenamos con fotos de Matilda

### Costo
- Tiempo: 1-2 horas de captura por vaca
- Labeling: 30 min por vaca (confirmar IDs)
- Procesamiento: ~1 hora

---

## Timeline propuesto

```
HOY (Marzo 2026):
├─ OpenCows2020 ✅ (demo, baseline)
│
1-2 SEMANAS:
├─ Validar Con MultiCamCows2024 dataset (descargar, procesar)
├─ Entrenar modelo ResNet18 v2
├─ Generar métricas comparativas
│
2-4 SEMANAS:  
├─ Capturar fotos de Matilda en campo (5 vacas, múltiples ángulos, sesiones)
├─ Generar galería "Matilda_v1"
├─ Evaluar Top-1 vs expectativa
├─ Iterar si es necesario
│
1-3 MESES:
├─ Integrar detector YOLO (fotos full → crops)
├─ Validación closed-set con fotos capturadas
├─ Umbrales de unknown-threshold calibrados en datos reales
│
3-6 MESES:
├─ Expansión: más vacas, más datos
├─ Robustez: edge cases, iluminación variable
├─ Integración: sistema end-to-end campo-to-ID
│
6+ MESES:
├─ Exploración aérea (si es scope)
├─ Escalabilidad: 100+ vacas
```

---

## Matriz decisión: cuál dataset usar cuándo

| Etapa | Usar | Métrica | Objetivo | 
|-------|------|---------|----------|
| **Validar arquitectura** | OpenCows2020 | Top-1 >80% | POC funcional |
| **Robustecer embeddings** | MultiCamCows2024 | Top-1 >70% | Generalización |
| **Producción Matilda** | Fotos reales Matilda | Top-1 >60% | Identificación real |
| **Aerial (si scope)** | Zenodo Aerial | Top-1 >50% (aéreo) | Monitoreo dron |
| **Validar robustecer general** | Zenodo benchmark | Top-1 >40% multi-especie | Edge cases |

---

## Resumen decisiones

### Hoy usamos OpenCows2020 porque:
1. **Limpio:** Datos de laboratorio, sin ruido
2. **Rápido:** 496 queries de test es pequeño
3. **Baseline:** Saber qué es el máximo posible

### Después cambiaremos a MultiCamCows2024/Matilda porque:
1. **Realista:** Condiciones variadas (ángulos, luz, distancia)
2. **Validación:** Scores más bajos = expectativa correcta
3. **Producción:** Son los datos que importan

### No tiramos a Zenodo Aerial ahora porque:
1. **Ángulos extremos:** Requiere modelo separado
2. **Baja prioridad:** Hoy no hay captura aérea operacional
3. **Timing:** Después de cerrar identificación en suelo

---

## Conclusión

**Hoy:** Validar con OpenCows2020 (limpio)  
**Mañana:** Robustecer con MultiCamCows2024 (variado)  
**Después:** Producción con Matilda real (real)  
**Futuro:** Aéreo si es scope (dron)  

Cada dataset aporta algo específico. No se solapan; son escalones.
