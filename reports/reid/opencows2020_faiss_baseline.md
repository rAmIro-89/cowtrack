# OpenCows2020 FAISS baseline

## Gallery build
- Script: scripts/build_faiss_gallery.py
- Mode: prototype (1 vector per ID)
- Gallery source: data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/train
- Output index: outputs/reid/faiss/opencows2020_gallery_prototype_e12.index
- Output metadata: outputs/reid/faiss/opencows2020_gallery_prototype_e12_meta.json
- Gallery vectors: 46
- Gallery IDs: 46

## Closed-set query evaluation
- Script: scripts/query_faiss_gallery.py
- Query source: data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17/identification/images/test
- Query count: 496
- Query IDs: 46
- Top-1: 0.5866935484
- Top-5: 0.9616935484
- Per-query details: reports/reid/opencows2020_faiss_eval_queries.csv
- Summary JSON: reports/reid/opencows2020_faiss_eval.json

## Interpretation
- FAISS is operational with real OpenCows IDs and real query traffic.
- In this setup FAISS is the search backend over the same embedding vectors/prototypes, so retrieval metrics match the non-FAISS evaluation protocol.
- FAISS adds deployment-ready indexing/search mechanics, not a metric boost by itself.
