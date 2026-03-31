# OpenCows2020 open-set setup

## Base checkpoint
- Primary checkpoint for this iteration:
  - outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt

## Pipeline review
- Existing scripts already supported:
  - prototype gallery build (`mode=prototype`)
  - all-vectors gallery build (`mode=all`)
  - closed-set FAISS query metrics
- Added open-set support and calibration tooling for unknown rejection.

## Changes applied
1. `scripts/query_faiss_gallery.py`
- Added `--unknown_threshold` argument.
- Added threshold-aware metrics:
  - `top1_accuracy_thresholded`
  - `topk_accuracy_thresholded`
  - `unknown_rejection_rate`
  - `false_rejects`
  - `false_accepts`
  - `accepted_queries`
- Added per-query fields in CSV:
  - `rejected_unknown`
  - `top1_correct_thresholded`
  - `topk_correct_thresholded`

2. `scripts/calibrate_open_set_thresholds.py` (new)
- Sweeps candidate thresholds on top-1 similarity scores.
- Uses a balanced criterion between:
  - accepting correct matches
  - rejecting incorrect matches
- Produces recommended thresholds and top candidate list for:
  - prototype gallery
  - all-vectors gallery

3. `scripts/visualize_open_set_scores.py` (new)
- Generates score histograms (correct vs incorrect top-1 scores).
- Generates compact examples CSV for accepted/rejected behavior at chosen thresholds.

## Strategy notes
- Open-set threshold is calibrated on known-ID test queries by separating score distributions of correct vs incorrect predictions.
- This gives a traceable first threshold baseline; final deployment threshold should later include true unknown identities.
