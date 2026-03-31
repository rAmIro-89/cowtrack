# Zenodo Sprint 1 Comparative Report

## Top 3 sequences selected automatically

| sequence_name | frames_probed | detections_total | detections_avg_per_frame | frames_with_detection_pct | continuity_score |
| --- | --- | --- | --- | --- | --- |
| DJI_202308091442_012 | 40 | 4 | 0.1 | 10.0 | 0.07692307692307693 |
| DJI_202306271323_032 | 40 | 0 | 0.0 | 0.0 | 0.0 |
| DJI_202306071624_028 | 40 | 0 | 0.0 | 0.0 | 0.0 |

## Sprint 1 comparative results (2 configs per sequence)

| sequence_name | config_name | frames_processed | detections_total | detections_avg_per_frame | valid_track_ids | crops_saved | frames_with_detection_pct | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DJI_202308091442_012 | lowconf_relaxed | 20 | 4 | 0.2 | 0 | 4 | 20.0 | ok |
| DJI_202308091442_012 | current | 20 | 3 | 0.15 | 0 | 3 | 15.0 | ok |
| DJI_202306271323_032 | current | 20 | 0 | 0.0 | 0 | 0 | 0.0 | ok |
| DJI_202306271323_032 | lowconf_relaxed | 20 | 0 | 0.0 | 0 | 0 | 0.0 | ok |
| DJI_202306071624_028 | current | 20 | 0 | 0.0 | 0 | 0 | 0.0 | ok |
| DJI_202306071624_028 | lowconf_relaxed | 20 | 0 | 0.0 | 0 | 0 | 0.0 | ok |

## Recommendation

Best base sequence for continuing tracking and crop generation:

- Sequence: DJI_202308091442_012
- Recommended config: lowconf_relaxed
- Reason: highest detections/frame, highest frame coverage with detections, and highest number of useful crops among tested runs.

## Important note

Valid track IDs are still 0 in these short CPU runs, indicating unstable assignment in the tested windows. For stronger temporal stability, continue with longer contiguous windows and a detector fine-tuned on aerial cattle imagery.
