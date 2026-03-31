# Project Branching: Aerial Detection vs Individual Re-ID

## Branch A: Aerial Detection (Drone Domain)
- Problem solved: detect/count cows from aerial frames and produce reliable trackable detections.
- Datasets: `data/raw/zenodo_grazing_cows`, `data/raw/aerialcattle2017`.
- Input type: drone imagery (small objects, sparse positives, many empty frames).
- Main metrics:
  - detection precision/recall/mAP
  - detections per frame
  - % frames with detections
  - valid tracks and crops exported
- Current status: weak signal for tiny boxes; not yet stable enough as identity source.

## Branch B: Individual Identity / Re-ID
- Problem solved: learn discriminative embeddings and gallery search per cow identity.
- Datasets:
  - first stage: `OpenCows2020` (when extracted)
  - second stage: `MultiCamCows2024` (when extracted)
- Working area prepared: `data/reid/opencows2020`.
- Main metrics:
  - Top-1 / Top-k retrieval
  - CMC / mAP for retrieval
  - unknown rejection threshold behavior
  - gallery/query consistency by ID

## Why these branches must stay separate
- Aerial detection validates localization under drone constraints; Re-ID validates identity discrimination.
- Strong Re-ID on clean identity datasets does not prove aerial detector quality.
- Strong aerial detection does not prove individual identity discrimination.
- Mixing both validations can hide the real bottleneck and produce misleading conclusions.

## Interface between branches (only safe coupling)
- Branch A exports detector crops with metadata (track id, frame id, confidence).
- Branch B consumes crops and returns identity predictions + confidence.
- Integration KPI is computed after both are independently validated.

## Practical rule set
- Do not train Re-ID using aerial detection labels that lack per-cow identity.
- Do not justify aerial detector progress using identity retrieval metrics.
- Keep reports and artifacts in separate folders:
  - aerial: `reports/tiling*`, `data/interim/*model_comparison*`
  - re-id: `reports/dataset_inspection`, `data/reid/*`
