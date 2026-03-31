# OpenCows2020 intermediate iteration final report

## 1) Training status
- Intermediate training run completed successfully on CUDA.
- Model: outputs/reid/opencows2020_baseline_e12.pt
- Train split: 46 IDs, 4240 images.
- Configuration: embedding_dim=256, batch_size=16, epochs=12, lr=3e-4, margin=0.3.
- Final train loss: 0.0369.

## 2) Retrieval performance (non-FAISS evaluator)
- Query split: 46 IDs, 496 queries.
- Top-1: 0.5867
- Top-5: 0.9617
- Intra-ID distance mean: 0.0690
- Inter-ID distance mean: 0.6799

## 3) Improvement vs baseline short
Comparison baseline (3 epochs):
- Top-1: 0.6190
- Top-5: 0.9738

Intermediate (12 epochs):
- Top-1: 0.5867
- Top-5: 0.9617

Delta (intermediate - short):
- Top-1: -0.0323
- Top-5: -0.0121

Interpretation:
- This e12 run did not improve rank-based retrieval on the current evaluation protocol.
- Embeddings are still operational and structured, but rank-1 quality regressed relative to baseline short.
- Subset diagnostics still look promising (NN@1 same-ID rate 0.7917 on subset_debug), suggesting possible overfitting or protocol sensitivity rather than collapse.

## 4) Visual diagnostics
Generated visuals in reports/reid/opencows2020_intermediate_visuals:
- subset_similarity_matrix.png
- subset_pca_projection.png
- subset_nn_examples.csv
- nearest_neighbor_stats.json

## 5) FAISS status
- Gallery/FAISS is operational with real IDs.
- Index: outputs/reid/faiss/opencows2020_gallery_prototype_e12.index
- Metadata: outputs/reid/faiss/opencows2020_gallery_prototype_e12_meta.json
- Closed-set query results (FAISS):
  - Top-1: 0.5867
  - Top-5: 0.9617
- FAISS acts as search backend; metrics match non-FAISS evaluator for the same embeddings/protocol.

## 6) Limitations
- No scheduler/early stopping in this run.
- Prototype-gallery evaluation can be sensitive to class imbalance and per-ID image variability.
- No unknown-threshold calibration yet.

## 7) Next exact step
Run a controlled follow-up with early checkpointing and validation tracking to find best epoch (instead of fixed last epoch), then rebuild FAISS from best checkpoint:

powershell
& "c:/Users/Ramiro/Documents/Trabajo Integrador/.venv/Scripts/python.exe" scripts/train_reid.py --train_dir "data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/train" --output_model "outputs/reid/opencows2020_baseline_e15.pt" --metrics_json "reports/reid/opencows2020_baseline_e15_train_metrics.json" --metrics_csv "reports/reid/opencows2020_baseline_e15_train_metrics.csv" --embedding_dim 256 --batch_size 16 --epochs 15 --lr 3e-4 --margin 0.3

Then evaluate and compare checkpoints using scripts/evaluate_reid_baseline.py before selecting the model for FAISS deployment.
