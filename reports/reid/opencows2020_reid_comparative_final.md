# OpenCows2020 Re-ID comparative final

## Selected run objective
- Train with checkpoints and select by real retrieval performance (Top-1), not by default last epoch.

## Best checkpoint selection outcome
- Candidates evaluated: 15 checkpoints (epoch_001..epoch_015)
- Selection metric: Top-1 retrieval on test split (train prototypes as gallery)
- Best epoch/checkpoint: epoch 015
- Best checkpoint path: outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt
- Best Top-1: 0.7822580645
- Best Top-5: 0.9798387097

## Comparison across stages
- Baseline short (3 epochs):
  - Top-1: 0.6189516129
  - Top-5: 0.9737903226
- Previous e12 model:
  - Top-1: 0.5866935484
  - Top-5: 0.9616935484
- New best checkpoint (e15 run, selected):
  - Top-1: 0.7822580645
  - Top-5: 0.9798387097

## Improvement summary
- Best checkpoint vs previous e12:
  - Top-1 delta: +0.1955645161
  - Top-5 delta: +0.0181451613
- Best checkpoint vs baseline short:
  - Top-1 delta: +0.1633064516
  - Top-5 delta: +0.0060483871

## FAISS consistency
- Rebuilt gallery/index with best checkpoint:
  - outputs/reid/faiss/opencows2020_gallery_bestckpt.index
  - outputs/reid/faiss/opencows2020_gallery_bestckpt_meta.json
- Closed-set FAISS eval:
  - Top-1: 0.7822580645
  - Top-5: 0.9798387097
- Consistency: FAISS metrics match retrieval evaluation for the selected checkpoint under the same gallery/query protocol.

## Best-checkpoint visuals
- reports/reid/opencows2020_bestckpt_visuals/subset_similarity_matrix.png
- reports/reid/opencows2020_bestckpt_visuals/subset_nn_examples.csv
- reports/reid/opencows2020_bestckpt_visuals/subset_pca_projection.png
- reports/reid/opencows2020_bestckpt_visuals/nearest_neighbor_stats.json (NN@1 same-ID rate: 0.875)

## Next exact step
1. Keep this best checkpoint as current production candidate for identity branch.
2. Add threshold calibration for unknown rejection (open-set boundary) on top of current closed-set pipeline.
3. Evaluate Top-1/Top-5 under two gallery modes (prototype vs all-vectors) and lock one for deployment.
