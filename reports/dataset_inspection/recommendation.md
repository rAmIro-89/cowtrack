# Dataset Recommendation (Identity vs Aerial Detection)

## Evidence from local inspection
- `data/raw/opencows2020` currently has 1 file: `BITAFE4.tmp` (~2.12 GB), and no extracted images/labels.
- `data/raw/multicamcows2024` currently has 1 file: `BITAFF4.tmp` (~36.57 GB), and no extracted images/labels.
- No train/val/test folders, no per-cow IDs, no videos/tracklets were detectable from extracted content because content is not extracted yet.

## Decision by objective
### 1) Which to use first for individual identity / Re-ID
- Use `OpenCows2020` first.
- Reason: project bridge config and README position it as the primary bridge dataset for identity embedding logic and gallery construction.
- Practical reason: smaller expected scope than MultiCam, lower friction to validate training/inference loop.

### 2) Which to leave for second stage
- Use `MultiCamCows2024` second.
- Reason: it is better suited for harder multi-view / scaling scenarios after baseline Re-ID pipeline is stable.

### 3) Which not to use to justify aerial drone detection
- Do not use `OpenCows2020` nor `MultiCamCows2024` to justify aerial detection from drone.
- Reason: this branch needs drone-domain detection evidence (Zenodo/Aerial datasets), not identity-focused datasets.

## Utility diagnosis per dataset
### OpenCows2020
- Generic detection utility: low for current local state (not extracted).
- Re-ID utility: high as target intent (once extracted), currently blocked by incomplete local files.
- Similarity to final mixed pipeline: medium, useful for identity stage but not for aerial detection proof.
- Recommended role: primary bridge dataset for Re-ID.

### MultiCamCows2024
- Generic detection utility: low for current local state (not extracted).
- Re-ID utility: medium-high for advanced multi-view validation (once extracted).
- Similarity to final mixed pipeline: medium for Re-ID robustness, low for drone detection claim.
- Recommended role: secondary benchmark and robustness stage for Re-ID.

## Integration plan without mixing objectives
1. Keep aerial detection experiments isolated in the aerial branch (`data/raw/zenodo_grazing_cows`, `data/raw/aerialcattle2017`).
2. Keep identity/Re-ID experiments isolated in the Re-ID branch (`data/reid/opencows2020`, then `data/reid/multicamcows2024`).
3. Exchange only detector crops from branch A into branch B at the interface point; do not mix training sets or metrics.
4. Compare branch outputs only at end-to-end business KPIs, not by mixing intermediate model metrics.

## Exact next technical step
1. Complete extraction of `OpenCows2020` (replace `.tmp` with actual dataset structure).
2. Re-run:
	- `python scripts/inspect_bridge_datasets.py`
	- `python scripts/prepare_reid_workspace.py --dataset_name opencows2020 --dataset_root data/raw/opencows2020 --output_root data/reid/opencows2020`
3. Validate the light Re-ID pipeline on `data/reid/opencows2020/subset_debug` with a short smoke pass (no heavy training yet).

