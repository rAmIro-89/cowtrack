# Small-Object Tiled vs Baseline Detector Comparison

| modelo | tile size | overlap | train imgsz | epochs | detecciones totales | det/frame | %frames con deteccion | tracks validos | crops | delta detecciones vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_prev_ft | n/a | n/a | n/a | n/a | 0 | 0.0000 | 0.00 | 0 | 0 | 0.00 |
| smallobj_yolov8n_img1024_e24 | 512 | 0.4 | 1024 | 24 | 0 | 0.0000 | 0.00 | 0 | 0 | 0.00 |
| smallobj_yolov8s_img1280_e30 | 512 | 0.4 | 1280 | 30 | 0 | 0.0000 | 0.00 | 0 | 0 | 0.00 |

## Conclusion

- Best model: baseline_prev_ft
- Best model path: C:\Users\Ramiro\Documents\Trabajo Integrador\cowtrack\runs\detect\outputs\detector_aerial\yolov8n_aerial_baseline\weights\best.pt
- Assessment: Esta iteracion small-object aun no supera baseline; siguiente cuello: etiquetas y hard negative mining.