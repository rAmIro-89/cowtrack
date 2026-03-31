# OpenCows2020 FAISS evaluation (best checkpoint)

## Setup
- Checkpoint: outputs/reid/checkpoints/opencows2020_e15_ckptsel/epoch_015.pt
- Gallery index: outputs/reid/faiss/opencows2020_gallery_bestckpt.index
- Gallery metadata: outputs/reid/faiss/opencows2020_gallery_bestckpt_meta.json
- Query split: data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/test
- Retrieval mode: closed-set, prototype gallery (1 vector per ID), cosine via FAISS IndexFlatIP.

## Results
- Gallery vectors: 46
- Gallery unique IDs: 46
- Query count: 496
- Query unique IDs: 46
- Top-1: 0.7822580645
- Top-5: 0.9798387097
- Per-query details: reports/reid/opencows2020_faiss_bestckpt_eval_queries.csv

## Consistency note
- These FAISS metrics match best-checkpoint retrieval evaluation from checkpoint selection.
- FAISS is acting as search backend over the same embeddings and gallery protocol, so metric parity is expected.
