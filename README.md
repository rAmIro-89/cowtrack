# CowTrack - Re-ID Pipeline para identificación de vacas

Identificación individual de vacas usando embeddings de red neuronal + búsqueda FAISS.

## 🎯 Objetivo

Sistema de identificación por apariencia (Re-ID) para:
- Identificar vacas individuales dentro de una galería conocida
- Rechazar vacas desconocidas con threshold de similitud
- Buscar rápidamente sobre galería grande usando FAISS

**Status actual:** MVP funcional en laboratorio. Top-1 accuracy 84.27% en OpenCows2020.

## 📊 Estado actual del proyecto

- ✅ Pipeline Re-ID completo (embeddings + FAISS + métricas)
- ✅ Checkpoint optimizado: `outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt`
- ✅ Calibración open-set (unknown threshold)
- ⚠️ Validado solo en OpenCows2020 (dataset de laboratorio)
- ❌ No integrado con detector YOLO aún
- ❌ No validado con datos reales de campo

**Documentación de estado:** Ver `reports/ENTREGABLES_estado_actual.md`

## 📁 Estructura del proyecto

```
cowtrack/
├── data/raw/
│   └── opencows2020/               # Dataset OpenCows2020 (46 vacas, ~4240 imágenes)
│
├── scripts/
│   ├── train_reid.py               # Entrenar modelo ResNet18 + triplet loss
│   ├── build_faiss_gallery.py      # Generar índice FAISS
│   ├── query_faiss_gallery.py      # Evaluar queries contra galería
│   ├── calibrate_open_set_thresholds.py
│   ├── generate_demo_visuals.py
│   └── ... (otros scripts)
│
├── src/
│   ├── preprocessing/
│   │   └── transforms.py           # Augmentación de imágenes
│   ├── reid/
│   │   ├── model.py               # ResNet18 + embedding head
│   │   ├── losses.py              # Triplet loss
│   │   ├── train.py               # Loop de entrenamiento
│   │   └── evaluate.py
│   └── utils/
│
├── outputs/
│   └── reid/
│       ├── checkpoints/            # Checkpoints del modelo
│       ├── faiss/                  # Índices FAISS + metadatos
│       └── embeddings/
│
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
│
├── .gitignore
├── README.md (este archivo)
├── requirements.txt
└── .venv/ (virtual environment)
```

## 🚀 Quickstart

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

## 📊 Resultados actuales

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

## 📈 Visuales disponibles

En `reports/reid/demo_state_today/`:

- **score_distribution.png** - Histograma de scores cuando acierta vs falla
- **confusion_matrix.png** - Heatmap 46×46 mostrando confusiones entre IDs
- **top5_examples.csv** - 30 ejemplos reales: aciertos, fallos, casos dudosos
- **demo_summary.json** - Estadísticas consolidadas

## 🔄 Pipeline actual

```
Query image
    ↓
Preprocessor (resize 224×224)
    ↓
ResNet18 encoder → 256-dim embedding
    ↓
L2-normalize
    ↓
FAISS search (IndexFlatIP, k=5)
    ↓
Top-5 IDs + similitudes
    ↓
Decision:
  - Sin threshold → return top-1 (84% correct)
  - Con threshold → accept/reject unknown
```

## ✅ Qué funciona HOY

- ✅ Entrenamiento de embeddings (ResNet18 + triplet loss)
- ✅ Indexación FAISS rápida
- ✅ Identificación en galería cerrada (84% Top-1)
- ✅ Calibración de unknown-threshold
- ✅ Evaluación de métricas closed/open-set
- ✅ Visualización de resultados

## ❌ Qué falta

- ❌ Detector YOLO integrado (fotos completas sin recorte)
- ❌ Datos reales de Matilda (Buenos Aires)
- ❌ Validación con verdaderos unknowns
- ❌ Robustez a variación extrema (ángulos, iluminación)
- ❌ Interfaz de usuario / deployment

## 📚 Documentación principal

Leer en este orden:

1. **`reports/ENTREGABLES_estado_actual.md`** - Resumen ejecutivo (START HERE)
2. **`reports/reid/what_works_today.md`** - Explicación simple de capacidades
3. **`reports/reid/dataset_usage_plan.md`** - Timeline y roadmap
4. **`reports/reid/demo_visual_explanation.md`** - Guía de las visuales

## 🛠️ Tecnologías

- **Framework:** PyTorch 2.x
- **Búsqueda:** FAISS (IndexFlatIP)
- **Modelo base:** ResNet18 pretrained ImageNet
- **Loss:** Triplet loss (margin=0.3)
- **Augmentación:** albumentations
- **GPU:** CUDA 11.8+
- **Python:** 3.10+

## 📋 Requisitos

Ver `requirements.txt`. Principales:
- torch
- torchvision
- faiss-cpu (o faiss-gpu)
- opencv-python
- numpy
- albumentations
- Pillow

## 🏗️ Entrenamiento propio

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

## 🔄 Roadmap

**Corto plazo (1-2 sem):**
- Validar con MultiCamCows2024 dataset
- Comparar scores vs OpenCows2020

**Mediano (2-4 sem):**
- Capturar fotos de Matilda (datos reales)
- Generar galería de producción
- Evaluar Top-1 en condiciones reales

**Largo (1-3 meses):**
- Integrar detector YOLO
- Pipeline end-to-end (foto completa → ID)
- Validación con unknowns

**Futuro:**
- Escalabilidad (100+ vacas)
- Detección aérea (drone)

Ver `reports/dataset_usage_plan.md` para detalles completos.

## 👥 Equipo

- Ramiro (desarrollo principal)

## 📞 Contacto

ramiro@matilda.local

---

**Última actualización:** 31 de Marzo 2026  
**Versión:** 1.0 (MVP)  
**Status:** En desarrollo - NO listo para producción

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso rapido (CLI)

Extraer frames:

```bash
python scripts/extract_frames.py --video data/raw/video.mp4 --out_dir data/interim/frames --stride 10
```

Entrenar ReID:

```bash
python scripts/train_reid.py --train_dir data/processed/reid_train --output_model outputs/reid/reid_model.pt --epochs 10
```

Construir galeria:

```bash
python scripts/build_gallery.py --dataset_dir data/processed/reid_gallery --weights outputs/reid/reid_model.pt --out_npz outputs/reid/gallery_embeddings.npz
```

Inferencia end-to-end:

```bash
python scripts/run_inference.py --video data/raw/video.mp4 --detector yolov8n.pt --reid_weights outputs/reid/reid_model.pt --gallery_npz outputs/reid/gallery_embeddings.npz --out_video outputs/inference/annotated.mp4 --out_csv outputs/inference/tracks.csv --out_json outputs/inference/tracks.json
```

Sprint 1 (secuencia de imagenes): tracking + guardado de crops por track

```bash
python scripts/sprint1_track_crops.py --input_dir data/raw/aerialcattle2017/extracted --output_dir data/interim/sprint1 --max_frames 300
```

Evaluacion de ventanas largas sobre secuencia ganadora (100/200/400 + frames anotados):

```bash
python scripts/evaluate_winner_windows.py --sequence_dir data/raw/zenodo_grazing_cows/extracted/Cattle_drone_images/Derval/JPGImages/DJI_202308091442_012 --windows 100 200 400 --reports_dir reports --output_root data/interim/winner_windows --tracker_cfg configs/bytetrack_relaxed_no_gmc.yaml --conf 0.08 --imgsz 256
```

Fine-tuning baseline de detector aereo (YOLO):

```bash
python scripts/train_detector_aerial_baseline.py --model yolov8n.pt --data configs/yolo_aerial_baseline.yaml --epochs 60 --imgsz 640 --batch 8 --device auto
```

Formato esperado para fine-tuning detector:

- Imagenes: `data/processed/yolo_aerial_baseline/images/{train,val}`
- Labels YOLO: `data/processed/yolo_aerial_baseline/labels/{train,val}`
- Clase unica: `cow` (id 0)
- Archivo de config: `configs/yolo_aerial_baseline.yaml`

## Evaluacion (baseline)

- Deteccion: precision, recall y mAP con herramientas de YOLO.
- ReID: top-1 / top-k accuracy y matriz de confusion en evaluacion offline.
- Negocio: conteo detectado, identificadas, faltantes, desconocidas, porcentaje reconocido.

## Datasets puente recomendados

Para no frenar el desarrollo cuando aun no hay dataset propio completo, se recomienda este uso por etapa:

1. Deteccion y conteo desde dron: Zenodo grazing cows (principal) y AerialCattle2017 (complementario).
2. Logica de identificacion individual (embeddings + galeria): OpenCows2020.
3. ReID multi-vista y evaluacion mas exigente: MultiCamCows2024.

Orden practico sugerido:

1. Entrenar YOLO para deteccion/conteo con dataset aereo.
2. Validar tracking y guardado de crops por track sobre video.
3. Entrenar embeddings de ReID con dataset por individuo.
4. Construir galeria FAISS y decidir identidad por track.
5. Adaptar/fine-tuning con fotos reales del productor.

Nota: los datasets publicos sirven para pipeline y baseline, pero no reemplazan la adaptacion al dominio real del rodeo objetivo.

La asignacion de datasets por etapa se puede mantener en `configs/datasets_bridge.yaml`.

Links de descarga y referencia:

1. AerialCattle2017: https://data.bris.ac.uk/data/dataset/3owflku95bxsx24643cybxu3qh
2. Drone images and their annotations of grazing cows (Zenodo): https://zenodo.org/records/11048412
3. OpenCows2020: https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17
4. MultiCamCows2024: https://data.bris.ac.uk/data/dataset/2inu67jru7a6821kkgehxg3cv2
5. Research page AerialCattle2017: https://research-information.bris.ac.uk/en/datasets/aerialcattle2017/
6. Research page OpenCows2020: https://research-information.bris.ac.uk/en/datasets/opencows2020/

## Limitaciones actuales

1. No incluye aun evaluador completo de mAP y top-k en script dedicado.
2. No incluye calibracion automatica de umbral con ROC/PR.
3. `train.py` usa un minado de tripletas basico dentro de batch (baseline).
4. La calidad de reconocimiento depende fuertemente de la calidad del dataset por individuo.


## Nota sobre el notebook original

Se elimino la simulacion de embeddings aleatorios y las dependencias de Colab/Drive. El flujo principal se migro a modulos y scripts locales en VS Code.


## The project was subsequently extended by team member Emiliano Orlando:
https://github.com/emilianorlando-collab/cow-tracker-mvp
