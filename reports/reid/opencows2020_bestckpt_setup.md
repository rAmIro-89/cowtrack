# OpenCows2020 best-checkpoint setup

## Why this iteration
- Previous e12 run showed that lower train loss did not guarantee better Top-1 retrieval.
- New requirement: select by real retrieval metric (Top-1), not by last epoch by default.

## Implemented strategy
1. Training now supports per-epoch checkpoint saving:
- Added `checkpoint_dir` to train config/CLI.
- Saves `epoch_XXX.pt` each epoch.

2. Added automatic checkpoint selection script:
- `scripts/select_best_reid_checkpoint.py`
- Evaluates all epoch checkpoints on train-gallery vs test-query protocol.
- Ranking key: Top-1, tie-breaker: Top-5.
- Produces:
  - `reports/reid/opencows2020_best_checkpoint_eval.json`
  - `reports/reid/opencows2020_best_checkpoint_eval.md`

3. Evaluation protocol used per checkpoint:
- Gallery: train split, one prototype embedding per ID.
- Query: full test split.
- Metrics: Top-1, Top-5, query count, gallery ID count.

4. FAISS rebuild requirement:
- Rebuild gallery/index strictly from selected best checkpoint.
- Re-evaluate closed-set with FAISS and compare consistency against retrieval evaluation.

## Output isolation
- New training run and artifacts were written to new paths (no overwrite of previous runs):
  - Model final: outputs/reid/opencows2020_e15_ckptsel_final.pt
  - Checkpoints: outputs/reid/checkpoints/opencows2020_e15_ckptsel/
  - Logs/metrics: reports/reid/opencows2020_e15_ckptsel_*
