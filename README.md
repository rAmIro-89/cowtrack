# CowTrack - Re-ID Pipeline for Cow Identification

[Español](README.es.md)

Team capstone project for individual cow identification using neural embeddings and FAISS search.

## Objective

This project is an appearance-based Re-ID system designed to:

- Identify individual cows within a known gallery.
- Reject unknown cows with a similarity threshold.
- Search quickly over a large gallery using FAISS.

**Current status:** functional laboratory MVP. Top-1 accuracy is 84.27% on OpenCows2020.

## Current Project Status

- Completed Re-ID pipeline with embeddings, FAISS, and metrics.
- Optimised checkpoint: `outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt`
- Open-set calibration implemented with an unknown threshold.
- Validated only on OpenCows2020, a laboratory dataset.
- Not yet integrated with YOLO.
- Not validated with real field data.

**Status documentation:** see `reports/ENTREGABLES_estado_actual.md`

## Project Structure

```text
cowtrack/
├── data/raw/
│   └── opencows2020/               # OpenCows2020 dataset (46 cows, ~4240 images)
├── scripts/
│   ├── train_reid.py               # Train ResNet18 + triplet loss model
│   ├── build_faiss_gallery.py      # Build FAISS index
│   ├── query_faiss_gallery.py      # Evaluate queries against the gallery
│   ├── calibrate_open_set_thresholds.py
│   ├── generate_demo_visuals.py
│   └── ... (other scripts)
├── src/
│   ├── preprocessing/
│   │   └── transforms.py           # Image augmentation
│   ├── reid/
│   │   ├── model.py                # ResNet18 + embedding head
│   │   ├── losses.py               # Triplet loss
│   │   ├── train.py                # Training loop
│   │   └── evaluate.py
│   └── utils/
├── outputs/
│   └── reid/
│       ├── checkpoints/            # Model checkpoints
│       ├── faiss/                  # FAISS indices + metadata
│       └── embeddings/
├── reports/
│   ├── reid/
│   │   ├── what_works_today.md           # Plain-language explanation
│   │   ├── dataset_usage_plan.md         # Dataset timeline
│   │   ├── demo_visual_explanation.md    # Visual guide
│   │   └── demo_state_today/
│   │       ├── score_distribution.png    # Histogram
│   │       ├── confusion_matrix.png      # Heatmap
│   │       ├── top5_examples.csv         # Real examples
│   │       └── demo_summary.json
│   └── ENTREGABLES_estado_actual.md
├── .gitignore
├── README.md (this file)
├── README.es.md
├── requirements.txt
└── .venv/ (virtual environment)
```

## Quickstart

### 1. Set up the environment

```bash
# Clone the repo
git clone <repo-url>
cd cowtrack

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the OpenCows2020 dataset

```bash
# Download from Zenodo: https://doi.org/10.34148/1/2020.12
# Extract into data/raw/opencows2020/
```

### 3. Build the FAISS gallery

```bash
# Using the pretrained checkpoint
python scripts/build_faiss_gallery.py \
  --train_dir data/raw/opencows2020/10m32.../identification/images/train \
  --weights outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt \
  --gallery_mode all_vectors \
  --out_index outputs/reid/faiss/opencows2020_gallery_allvectors.index
```

### 4. Evaluate against the test set

```bash
python scripts/query_faiss_gallery.py \
  --query_dir data/raw/opencows2020/.../test \
  --weights outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt \
  --index_path outputs/reid/faiss/opencows2020_gallery_allvectors_bestckpt.index \
  --meta_path outputs/reid/faiss/opencows2020_gallery_allvectors_bestckpt_meta.json \
  --out_json reports/reid/results.json
```

### 5. Generate visuals

```bash
python scripts/generate_demo_visuals.py \
  --query_csv reports/reid/opencows2020_allvectors_nothr_eval_queries.csv \
  --out_dir reports/reid/demo_state_today
```

## Current Results

### OpenCows2020 - Closed-set (no threshold)

| Metric | All-vectors | Prototype |
|--------|-------------|-----------|
| **Top-1 Accuracy** | **84.27%** | 78.23% |
| **Top-5 Accuracy** | **94.76%** | 97.98% |
| Gallery vectors | 4240 | 46 |
| Query count | 496 | 496 |
| False accepts | 78 | 108 |

### OpenCows2020 - Open-set (calibrated threshold)

| Config | Top-1 thresholded | Rejection rate | False accepts |
|--------|-------------------|----------------|---------------|
| All-vectors + thr | 46.57% | 49.40% | 20 |
| Prototype + thr | 37.90% | 57.26% | 24 |

**Note:** calibrated thresholds reject about 50% of queries. Validation with real unknowns is still pending.

## Available Visuals

In `reports/reid/demo_state_today/`:

- `score_distribution.png` - score histogram for correct versus incorrect matches
- `confusion_matrix.png` - 46×46 heatmap showing ID confusions
- `top5_examples.csv` - 30 real examples: hits, misses, and ambiguous cases
- `demo_summary.json` - consolidated statistics

## Current Capabilities

- Training embeddings (ResNet18 + triplet loss)
- Fast FAISS indexing
- Closed-gallery identification (84% Top-1)
- Unknown-threshold calibration
- Closed-set and open-set metric evaluation
- Result visualisation

## Later Experiments

- Integrate YOLO detector for full-frame images
- Validate with real field data
- Test true unknowns
- Study robustness to extreme variation in angle and lighting

## Main Documentation

Read in this order:

1. `reports/ENTREGABLES_estado_actual.md` - executive summary (START HERE)
2. `reports/reid/what_works_today.md` - plain-language capability summary
3. `reports/reid/dataset_usage_plan.md` - timeline and roadmap
4. `reports/reid/demo_visual_explanation.md` - visual guide

## Technology Stack

- Framework: PyTorch 2.x
- Search: FAISS (IndexFlatIP)
- Base model: ResNet18 pretrained on ImageNet
- Loss: Triplet loss (margin=0.3)
- Augmentation: albumentations
- GPU: CUDA 11.8+
- Python: 3.10+

## Requirements

See `requirements.txt`. Main packages:

- torch
- torchvision
- faiss-cpu (or faiss-gpu)
- opencv-python
- numpy
- albumentations
- Pillow

## Own Training

If you want to retrain with a new dataset:

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

**Short term (1-2 weeks):**

- Validate with the MultiCamCows2024 dataset
- Compare scores against OpenCows2020

**Medium term (2-4 weeks):**

- Capture photos of Matilda (real data)
- Build a repeatable evaluation gallery for controlled experiments
- Evaluate Top-1 in real conditions

**Long term (1-3 months):**

- Integrate YOLO detector
- End-to-end pipeline (full image → ID)
- Unknowns validation

**Future:**

- Scale to 100+ cows
- Aerial detection (drone)

See `reports/dataset_usage_plan.md` for the full details.

## Limitations

- Functional laboratory MVP.
- Validated only on OpenCows2020.
- Not yet integrated with YOLO.
- Not validated with real field data.
- True unknown validation is still pending.
- User interface and deployment are not implemented.

## Team

- Team capstone project developed collaboratively.

## Contact

ramiroottonevillar@gmail.com

---

**Last updated:** 31 March 2026
**Version:** 1.0 (MVP)
**Status:** Functional laboratory MVP