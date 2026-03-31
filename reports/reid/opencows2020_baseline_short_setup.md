# OpenCows2020 baseline short setup

## Pre-run validation
- Checked `scripts/train_reid.py`: runnable CLI, now extended with metrics outputs.
- Checked loader `src/reid/dataset.py`: expects `root/ID/image` structure.
- Verified train directory exists:
  - `data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/train`
  - IDs found: 46
  - train images found: 4240
- No loader format fix was required for OpenCows train split.

## Adjustments made before training
1. `src/reid/train.py`
- Added optional metrics outputs:
  - JSON summary (`metrics_json_path`)
  - CSV per-epoch metrics (`metrics_csv_path`)
- Added persisted epoch metrics (`epoch`, `avg_loss`, `steps_with_triplets`).
- Added saved training summary with device, sample count, identity count and final loss.

2. `scripts/train_reid.py`
- Added CLI args:
  - `--metrics_json`
  - `--metrics_csv`
- Wired both args into `TrainConfig`.

3. Runtime command adjustment
- Added `PYTHONPATH=.` when launching scripts from terminal to resolve local `src` package imports.

## Training artifacts target
- Model: `outputs/reid/opencows2020_baseline_short.pt`
- Log: `reports/reid/opencows2020_baseline_short_train.log`
- Metrics JSON: `reports/reid/opencows2020_baseline_short_train_metrics.json`
- Metrics CSV: `reports/reid/opencows2020_baseline_short_train_metrics.csv`
