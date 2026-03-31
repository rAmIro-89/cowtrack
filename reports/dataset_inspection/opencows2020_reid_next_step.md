# OpenCows2020 Re-ID next step (short baseline)

## Ready components
- Dataset loader: `src/reid/dataset.py` (folder-per-ID format).
- Train script: `scripts/train_reid.py`.
- Baseline val evaluator: `scripts/evaluate_reid_baseline.py`.
- Real dataset root for identity: `data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images`.
- Train split: `.../identification/images/train`.
- Val split (used as validation for first baseline): `.../identification/images/test`.

## Minimal short experiment config
- embedding_dim: 256
- batch_size: 16
- epochs: 3
- lr: 3e-4
- margin: 0.3

## Exact command: short baseline train
```powershell
& "c:/Users/Ramiro/Documents/Trabajo Integrador/.venv/Scripts/python.exe" scripts/train_reid.py --train_dir "data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/train" --output_model "outputs/reid/opencows2020_baseline_short.pt" --embedding_dim 256 --batch_size 16 --epochs 3 --lr 3e-4 --margin 0.3
```

## Exact command: quick val check after training
```powershell
& "c:/Users/Ramiro/Documents/Trabajo Integrador/.venv/Scripts/python.exe" scripts/evaluate_reid_baseline.py --train_dir "data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/train" --val_dir "data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/test" --weights "outputs/reid/opencows2020_baseline_short.pt" --embedding_dim 256 --image_size 224 --out_json "reports/dataset_inspection/opencows2020_reid_val_eval.json"
```

## Expected outputs
- Model weights: `outputs/reid/opencows2020_baseline_short.pt`
- Validation summary: `reports/dataset_inspection/opencows2020_reid_val_eval.json`
