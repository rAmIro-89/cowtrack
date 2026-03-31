from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

from src.preprocessing.transforms import build_eval_transform
from src.reid.faiss_index import FaissGallery, SearchResult
from src.reid.model import EmbeddingNet


class ReIDInferencer:
    def __init__(
        self,
        model: EmbeddingNet,
        gallery: FaissGallery,
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.gallery = gallery
        self.transform = build_eval_transform(image_size=image_size)

    @torch.no_grad()
    def embed_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=rgb)["image"].unsqueeze(0).to(self.device)
        embedding = self.model(tensor).cpu().numpy().astype(np.float32)
        return embedding

    @torch.no_grad()
    def identify(self, crop_bgr: np.ndarray, unknown_threshold: float = 0.68) -> SearchResult:
        embedding = self.embed_crop(crop_bgr)
        return self.gallery.search_best(embedding, unknown_threshold=unknown_threshold)
