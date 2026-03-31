# Extraction status (OpenCows2020 and MultiCamCows2024)

## OpenCows2020
- Archive-like file detected: data/raw/opencows2020/BITAFE4.tmp
- Signature check: ZIP header (`50 4B 03 04`)
- Extraction command executed: `tar -xf data/raw/opencows2020/BITAFE4.tmp -C data/raw/opencows2020`
- Extracted root found: `data/raw/opencows2020/10m32xl88x2b61zlkkgz3fml17`
- Evidence after extraction:
  - files: 26082 (inside extracted root)
  - images: 11779 (inside extracted root)
  - key folders: `identification`, `detection_and_localisation`
- Conclusion: extracted and usable for real Re-ID preparation.

## MultiCamCows2024 (verification/extraction only)
- Archive-like file detected: data/raw/multicamcows2024/BITAFF4.tmp
- Signature check: ZIP header (`50 4B 03 04`)
- Extraction command executed: `tar -xf data/raw/multicamcows2024/BITAFF4.tmp -C data/raw/multicamcows2024`
- Extracted root found: `data/raw/multicamcows2024/2inu67jru7a6821kkgehxg3cv2`
- Evidence after extraction:
  - files: 102594 (inside extracted root)
  - images: 101329 (inside extracted root)
  - videos: present (`.mp4` found in inventory)
- Conclusion: extracted at basic level. No deeper Re-ID preparation was performed for this dataset.

## Notes
- Temporary `.tmp` files are still present in both dataset roots after extraction.
- Presence of `.tmp` does not block current OpenCows Re-ID work because real extracted structure is already available.
