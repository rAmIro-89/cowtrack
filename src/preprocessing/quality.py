from __future__ import annotations

import cv2
import numpy as np


def blur_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_score(image: np.ndarray, blur_threshold: float = 80.0) -> float:
    blur = blur_variance(image)
    h, w = image.shape[:2]
    area_component = min(100.0, (h * w) / 3000.0)
    blur_component = min(100.0, (blur / max(blur_threshold, 1.0)) * 100.0)
    return float(0.4 * area_component + 0.6 * blur_component)
