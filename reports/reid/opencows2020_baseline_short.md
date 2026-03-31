# OpenCows2020 baseline short (real run)

## Run status
- Training completed: yes
- Device used: CUDA
- Model saved: `outputs/reid/opencows2020_baseline_short.pt`

## Data used
- Train split: `data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/train`
- Train IDs: 46
- Train images: 4240
- Validation split used for quick retrieval check: `.../identification/images/test`
- Validation IDs: 46
- Validation images evaluated: 496

## Training metrics
- Epochs: 3
- Final loss: 0.0550
- Loss progression:
  - epoch 1: 0.0962
  - epoch 2: 0.0636
  - epoch 3: 0.0550
- Artifacts:
  - log: `reports/reid/opencows2020_baseline_short_train.log`
  - metrics json: `reports/reid/opencows2020_baseline_short_train_metrics.json`
  - metrics csv: `reports/reid/opencows2020_baseline_short_train_metrics.csv`

## Minimal useful evaluation
- Retrieval setup: gallery prototypes from train IDs, queries from test split.
- Top-1 accuracy: 0.61895 (46 IDs, 496 queries)
- Eval file: `reports/reid/opencows2020_baseline_short_eval.json`

## Embedding diagnostics
- Embeddings generated correctly: yes
- subset_debug used real images: yes
- subset_debug embeddings shape: (48, 256)
- Unique IDs in subset_debug diagnostics: 12
- NN@1 same-ID rate on subset_debug: 0.6875
- Visual artifacts:
  - similarity matrix: `reports/reid/opencows2020_baseline_short_visuals/subset_similarity_matrix.png`
  - nearest neighbors: `reports/reid/opencows2020_baseline_short_visuals/subset_nearest_neighbors.csv`

## Interpretation
- The baseline is functional end-to-end and embeddings are informative.
- Loss decreased consistently and retrieval signals are above chance for 46-way identity matching.
- Separation looks promising but still early-stage due to short training schedule.

## Limitations of this short run
- Only 3 epochs.
- Evaluation is a quick baseline (train-prototype vs test query), not a full Re-ID benchmark protocol.
- No threshold calibration for unknown identities yet.
- No FAISS gallery persistence/inference latency benchmarking in this run.

## Exact next technical step
1. Train a medium run (10-15 epochs) with same setup and keep per-epoch metrics.
2. Build FAISS gallery from train split using trained weights.
3. Evaluate top-1/top-5 and unknown-threshold sweep over test queries.

Exact next command (medium run):

```powershell
& "c:/Users/Ramiro/Documents/Trabajo Integrador/.venv/Scripts/python.exe" scripts/train_reid.py --train_dir "data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/train" --output_model "outputs/reid/opencows2020_baseline_e12.pt" --metrics_json "reports/reid/opencows2020_baseline_e12_train_metrics.json" --metrics_csv "reports/reid/opencows2020_baseline_e12_train_metrics.csv" --embedding_dim 256 --batch_size 16 --epochs 12 --lr 3e-4 --margin 0.3
```
