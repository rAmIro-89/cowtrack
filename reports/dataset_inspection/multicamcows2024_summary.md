# multicamcows2024 inspection summary

## Local scan
- Path: data\raw\multicamcows2024
- Exists: True
- Total files: 102595
- Total directories: 562
- Total size (GB): 73.2708
- Image files: 101329
- Video files: 137
- Annotation-like files: 2
- Split hints: train=False, val=False, test=False, query=False, gallery=False
- Candidate ID folders: 528
- Likely has per-cow IDs: True
- Likely has tracklets: False
- Likely has multi-view setup: True
- Temporary download files: 1
- Top extensions: .jpg:101329, .html:1124, .mp4:137, .tmp:1, .json:1, .rdf:1, <no_ext>:1, .txt:1
- Top-level dirs: 2inu67jru7a6821kkgehxg3cv2

## Utility diagnosis
- Generic cow detection utility: medium (2/3)
- Individual ID / Re-ID utility: high (3/3)
- Similarity to final project target: medium (2/3)

## Recommendation role
- Bridge dataset for Re-ID: yes
- Benchmarking dataset: yes
- Gallery-by-identity source: yes
- Secondary reference only: no

## Findings and caveats
- Temporary download files detected (.tmp/.part/.crdownload), but extracted dataset content is available.
- No explicit train/val/test split directories detected.
