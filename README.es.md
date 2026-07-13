# CowTrack - Re-ID Pipeline para identificación de vacas

[English](README.md)

Proyecto capstone de equipo para identificación individual de vacas usando embeddings de red neuronal + búsqueda FAISS.

## Objetivo

Sistema de identificación por apariencia (Re-ID) para:

- Identificar vacas individuales dentro de una galería conocida.
- Rechazar vacas desconocidas con threshold de similitud.
- Buscar rápidamente sobre galería grande usando FAISS.

**Status actual:** MVP funcional en laboratorio. Top-1 accuracy 84.27% en OpenCows2020.

## Estado actual del proyecto

- ✅ Pipeline Re-ID completo (embeddings + FAISS + métricas)
- ✅ Checkpoint optimizado: `outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt`
- ✅ Calibración open-set (unknown threshold)
- ⚠️ Validado solo en OpenCows2020 (dataset de laboratorio)
- ❌ No integrado con detector YOLO aún
- ❌ No validado con datos reales de campo

**Documentación de estado:** Ver `reports/ENTREGABLES_estado_actual.md`

## Estructura del proyecto

```text
cowtrack/
├── data/raw/
│   └── opencows2020/               # Dataset OpenCows2020 (46 vacas, ~4240 imágenes)
├── scripts/
│   ├── train_reid.py               # Entrenar modelo ResNet18 + triplet loss
│   ├── build_faiss_gallery.py      # Generar índice FAISS
│   ├── query_faiss_gallery.py      # Evaluar queries contra galería
│   ├── calibrate_open_set_thresholds.py
│   ├── generate_demo_visuals.py
│   └── ... (otros scripts)
├── src/
│   ├── preprocessing/
│   │   └── transforms.py           # Augmentación de imágenes
│   ├── reid/
│   │   ├── model.py               # ResNet18 + embedding head
│   │   ├── losses.py              # Triplet loss
│   │   ├── train.py               # Loop de entrenamiento
│   │   └── evaluate.py
│   └── utils/
├── outputs/
│   └── reid/
│       ├── checkpoints/            # Checkpoints del modelo
│       ├── faiss/                  # Índices FAISS + metadatos
│       └── embeddings/
├── reports/
│   ├── reid/
│   │   ├── what_works_today.md           # Explicación simple
│   │   ├── dataset_usage_plan.md         # Timeline datasets
│   │   ├── demo_visual_explanation.md    # Guía de visuales
│   │   └── demo_state_today/
│   │       ├── score_distribution.png    # Histograma
│   │       ├── confusion_matrix.png      # Heatmap
│   │       ├── top5_examples.csv         # Ejemplos reales
│   │       └── demo_summary.json
│   └── ENTREGABLES_estado_actual.md
├── .gitignore
├── README.md (este archivo)
├── README.es.md
├── requirements.txt
└── .venv/ (virtual environment)
```

## Quickstart

### 1. Configurar ambiente

```bash
# Clonar repo
git clone <repo-url>
cd cowtrack

# Crear virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Descargar dataset OpenCows2020

```bash
# Descargar desde Zenodo: https://doi.org/10.34148/1/2020.12
# Descomprimir en data/raw/opencows2020/
```

### 3. Generar galería FAISS

```bash
# Con checkpoint preentrenado
python scripts/build_faiss_gallery.py \
  --train_dir data/raw/opencows2020/10m32.../identification/images/train \
  --weights outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt \
  --gallery_mode all_vectors \
  --out_index outputs/reid/faiss/opencows2020_gallery_allvectors.index
```

### 4. Evaluar contra test set

```bash
python scripts/query_faiss_gallery.py \
  --query_dir data/raw/opencows2020/.../test \
  --weights outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt \
  --index_path outputs/reid/faiss/opencows2020_gallery_allvectors_bestckpt.index \
  --meta_path outputs/reid/faiss/opencows2020_gallery_allvectors_bestckpt_meta.json \
  --out_json reports/reid/results.json
```

### 5. Generar visuales

```bash
python scripts/generate_demo_visuals.py \
  --query_csv reports/reid/opencows2020_allvectors_nothr_eval_queries.csv \
  --out_dir reports/reid/demo_state_today
```

## Resultados actuales

### OpenCows2020 - Closed-set (sin threshold)

| Métrica | All-vectors | Prototype |
|---------|-------------|-----------|
| **Top-1 Accuracy** | **84.27%** | 78.23% |
| **Top-5 Accuracy** | **94.76%** | 97.98% |
| Gallery vectors | 4240 | 46 |
| Query count | 496 | 496 |
| False accepts | 78 | 108 |

### OpenCows2020 - Open-set (con threshold calibrado)

| Config | Top-1 thresholded | Rejection rate | False accepts |
|--------|------------------|-----------------|---------------|
| All-vectors + thr | 46.57% | 49.40% | 20 |
| Prototype + thr | 37.90% | 57.26% | 24 |

**Nota:** Thresholds calibrados rechazan ~50% de queries. Validación con verdaderos unknowns aún pendiente.

## Visuales disponibles

En `reports/reid/demo_state_today/`:

- `score_distribution.png` - Histograma de scores cuando acierta vs falla
- `confusion_matrix.png` - Heatmap 46×46 mostrando confusiones entre IDs
- `top5_examples.csv` - 30 ejemplos reales: aciertos, fallos, casos dudosos
- `demo_summary.json` - Estadísticas consolidadas

## Capacidades actuales

- Entrenamiento de embeddings (ResNet18 + triplet loss)
- Indexación FAISS rápida
- Identificación en galería cerrada (84% Top-1)
- Calibración de unknown-threshold
- Evaluación de métricas closed/open-set
- Visualización de resultados

## Experimentos posteriores

- Integrar detector YOLO para imágenes completas
- Validar con datos reales de campo
- Probar verdaderos unknowns
- Estudiar robustez ante variaciones extremas de ángulo e iluminación

## Documentación principal

Leer en este orden:

1. `reports/ENTREGABLES_estado_actual.md` - Resumen ejecutivo (START HERE)
2. `reports/reid/what_works_today.md` - Explicación simple de capacidades
3. `reports/reid/dataset_usage_plan.md` - Timeline y roadmap
4. `reports/reid/demo_visual_explanation.md` - Guía de las visuales

## Tecnologías

- **Framework:** PyTorch 2.x
- **Búsqueda:** FAISS (IndexFlatIP)
- **Modelo base:** ResNet18 pretrained ImageNet
- **Loss:** Triplet loss (margin=0.3)
- **Augmentación:** albumentations
- **GPU:** CUDA 11.8+
- **Python:** 3.10+

## Requisitos

Ver `requirements.txt`. Principales:

- torch
- torchvision
- faiss-cpu (o faiss-gpu)
- opencv-python
- numpy
- albumentations
- Pillow

## Entrenamiento propio

Si querés reentrenar con nuevo dataset:

```bash
python scripts/train_reid.py \
  --train_dir data/raw/my_dataset/train \
  --val_split 0.2 \
  --epochs 20 \
  --batch_size 32 \
  --embedding_dim 256 \
  --lr 1e-4 \
  --margin 0.3
```

## Roadmap

**Corto plazo (1-2 sem):**

- Validar con MultiCamCows2024 dataset
- Comparar scores vs OpenCows2020

**Mediano (2-4 sem):**

- Capturar fotos de Matilda (datos reales)
- Generar galería de evaluación repetible para experimentos controlados
- Evaluar Top-1 en condiciones reales

**Largo (1-3 meses):**

- Integrar detector YOLO
- Pipeline end-to-end (foto completa → ID)
- Validación con unknowns

**Futuro:**

- Escalabilidad (100+ vacas)
- Detección aérea (drone)

Ver `reports/dataset_usage_plan.md` para detalles completos.

## Limitaciones

- MVP funcional de laboratorio.
- Validado solo en OpenCows2020.
- No integrado con YOLO.
- No validado con datos reales de campo.
- La validación con verdaderos unknowns sigue pendiente.
- La interfaz de usuario y el deployment no están implementados.

## Equipo

- Proyecto capstone de equipo desarrollado de forma colaborativa.

## Contacto

ramiroottonevillar@gmail.com

---

**Última actualización:** 31 de Marzo 2026  
**Versión:** 1.0 (MVP)  
**Status:** MVP funcional de laboratorio