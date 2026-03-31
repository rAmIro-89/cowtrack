# OpenCows2020 intermediate setup

## Pipeline checks
- Verified train split path: data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/train
- Verified test split path: data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/test
- Confirmed folder-per-ID structure compatible with loader in src/reid/dataset.py
- Current split stats:
  - train IDs: 46, train images: 4240
  - test IDs: 46, test images: 496

## Adjustments made for this iteration
1. scripts/train_reid.py
- Added robust local import resolution (project root inserted in sys.path).

2. scripts/evaluate_reid_baseline.py
- Added robust local import resolution.
- Added Top-k support (configurable, default k=5).
- Added query/gallery counts.
- Added Top-1 and Top-k metrics.
- Added intra-ID vs inter-ID mean distance estimates.

3. scripts/export_reid_embeddings.py (new)
- Exports embeddings by split to NPZ (+ metadata JSON).
- Supports downstream analysis and reproducible split-level embedding dumps.

4. scripts/build_faiss_gallery.py (new)
- Builds FAISS IndexFlatIP gallery from train split.
- Supports prototype mode (1 vector per ID) and all-vectors mode.
- Saves index + metadata with real ID labels.

5. scripts/query_faiss_gallery.py (new)
- Queries test split against FAISS gallery.
- Reports Top-1 and Top-k closed-set accuracy.
- Saves per-query predictions CSV and global JSON metrics.

6. scripts/visualize_reid_diagnostics.py
- Added robust local import resolution.
- Added PCA 2D projection image.
- Added nearest-neighbor examples CSV with correctness flags.
