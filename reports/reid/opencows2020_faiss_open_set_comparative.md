# Open-set FAISS comparative report

Checkpoint: outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt

## Recommended thresholds
- prototype: 0.9927988052368164
- all_vectors: 0.9995235204696655

## Results table

| mode | setting | threshold | top1 | top5 | top1_thresholded | top5_thresholded | reject_rate | false_rejects | false_accepts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prototype | no_threshold | - | 0.782258 | 0.979839 | 0.782258 | 0.979839 | 0.000000 | 0 | 108 |
| prototype | with_threshold | 0.992798805 | 0.782258 | 0.979839 | 0.379032 | 0.419355 | 0.572581 | 284 | 24 |
| all_vectors | no_threshold | - | 0.842742 | 0.947581 | 0.842742 | 0.947581 | 0.000000 | 0 | 78 |
| all_vectors | with_threshold | 0.999523520 | 0.842742 | 0.947581 | 0.465726 | 0.495968 | 0.493952 | 245 | 20 |

## Best configuration
- mode: all_vectors
- setting: no_threshold
- threshold: None
- top1_thresholded: 0.842741935483871
- top5_thresholded: 0.9475806451612904
- reject_rate: 0.0
- false_accepts: 78
- false_rejects: 0

Notes:
- Dataset used for query evaluation contains known IDs only.
- Rejections in thresholded runs correspond to false rejects under this protocol.