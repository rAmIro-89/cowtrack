# OpenCows2020 intermediate evaluation

## Run context
- Model: outputs/reid/opencows2020_baseline_e12.pt
- Device: CUDA
- Train IDs: 46
- Query IDs: 46
- Gallery vectors: 46 (prototype per identity)
- Query images: 496

## Retrieval metrics
- Top-1: 0.5866935484
- Top-5: 0.9616935484

## Distance statistics
- Mean intra-ID distance: 0.0689617995
- Mean inter-ID distance: 0.6798994576
- Separation gap (inter - intra): 0.6109376581

## Embedding integrity
- Embeddings generated: yes
- Embedding dimension: 256

## Diagnostic interpretation
- The model remains functional for identity retrieval with high Top-5.
- Top-1 is lower than the short baseline, so this intermediate run is not yet a net improvement for strict rank-1 identification.
- Intra/inter distances still show separation, but less favorable than baseline short in this split-level protocol.
