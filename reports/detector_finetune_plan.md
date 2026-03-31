# Plan minimo de fine-tuning del detector (dominio aereo)

## Objetivo

Adaptar YOLO preentrenado a vistas de dron para mejorar deteccion de bovinos y estabilizar tracking en secuencias largas.

## Hipotesis a validar

1. Si al aumentar longitud de ventana no mejora tracking, el problema principal es detector no adaptado al dominio aereo.
2. Un fine-tuning simple en subset aereo debe subir detecciones/frame y cobertura temporal con deteccion.

## Subset recomendado (baseline)

1. Fuente: secuencias Zenodo con mayor deteccion inicial (comenzar por DJI_202308091442_012 y otras con vacas visibles).
2. Tamano inicial: 300 a 800 frames anotados manualmente o semiautomaticos revisados.
3. Split sugerido: 70% train, 20% val, 10% test.
4. Balance: incluir variedad de altura, iluminacion, densidad de rodeo y fondo.

## Formato esperado (YOLO)

Estructura:

- data/processed/yolo_aerial_baseline/images/train
- data/processed/yolo_aerial_baseline/images/val
- data/processed/yolo_aerial_baseline/labels/train
- data/processed/yolo_aerial_baseline/labels/val

Cada imagen debe tener su .txt con lineas:

- class x_center y_center width height

Valores normalizados entre 0 y 1, clase unica cow=0.

## Configuracion baseline

Archivo de datos:

- configs/yolo_aerial_baseline.yaml

Script de armado de subset inicial:

- scripts/prepare_yolo_aerial_subset.py

Comando ejemplo (subset inicial):

python scripts/prepare_yolo_aerial_subset.py --zenodo_root data/raw/zenodo_grazing_cows/extracted/Cattle_drone_images --out_root data/processed/yolo_aerial_baseline --non_empty_max 220 --empty_max 80 --val_ratio 0.2

Script de entrenamiento:

- scripts/train_detector_aerial_baseline.py

Comando ejemplo:

python scripts/train_detector_aerial_baseline.py --model yolov8n.pt --data configs/yolo_aerial_baseline.yaml --epochs 60 --imgsz 640 --batch 8 --device cpu

Evaluacion antes/despues recomendada (misma secuencia y mismo esquema):

python scripts/evaluate_winner_windows.py --model yolov8n.pt --windows 100 200 400 --reports_dir reports --output_root data/interim/winner_windows_before

python scripts/evaluate_winner_windows.py --model outputs/detector_aerial/yolov8n_aerial_baseline/weights/best.pt --windows 100 200 400 --reports_dir reports --output_root data/interim/winner_windows_after

## Criterios de exito baseline

1. Aumentar detecciones promedio/frame en secuencia ganadora.
2. Aumentar porcentaje de frames con al menos una deteccion.
3. Obtener track_id validos en ventanas largas con la misma configuracion de tracking.

## Siguiente paso recomendado

Repetir evaluacion de ventanas 100/200/400 con el detector fine-tuned y comparar contra el detector base (yolov8n.pt).
