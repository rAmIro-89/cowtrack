# opencows2020 inspection summary

## Local scan
- Path: data\raw\opencows2020
- Exists: True
- Total files: 26083
- Total directories: 102
- Total size (GB): 4.2782
- Image files: 11779
- Video files: 0
- Annotation-like files: 14098
- Split hints: train=True, val=False, test=True, query=False, gallery=False
- Candidate ID folders: 93
- Likely has per-cow IDs: True
- Likely has tracklets: False
- Likely has multi-view setup: False
- Temporary download files: 1
- Top extensions: .jpg:11779, .txt:7044, .xml:7043, .html:204, .json:11, .tmp:1, .rdf:1
- Top-level dirs: 10m32xl88x2b61zlkkgz3fml17

## Utility diagnosis
- Generic cow detection utility: medium (2/3)
- Individual ID / Re-ID utility: high (3/3)
- Similarity to final project target: low (1/3)

## Recommendation role
- Bridge dataset for Re-ID: yes
- Benchmarking dataset: yes
- Gallery-by-identity source: yes
- Secondary reference only: no

## Findings and caveats
- Temporary download files detected (.tmp/.part/.crdownload), but extracted dataset content is available.
- No video files detected.
