from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import faiss
import numpy as np


@dataclass
class SearchResult:
    identity: str
    score: float


class FaissGallery:
    """Closed-set identity gallery over normalized embeddings."""

    def __init__(self, embedding_dim: int = 256) -> None:
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.identities: List[str] = []

    @staticmethod
    def normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-9, None)

    def add(self, identity: str, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embeddings shape must be [N, embedding_dim]")
        normed = self.normalize(embeddings.astype(np.float32))
        self.index.add(normed)
        self.identities.extend([identity] * len(normed))

    def search(self, query: np.ndarray, top_k: int = 1) -> List[SearchResult]:
        if self.index.ntotal == 0:
            return []
        if query.ndim != 2:
            raise ValueError("Query embedding must be [1, embedding_dim] or [N, embedding_dim]")

        query = self.normalize(query.astype(np.float32))
        scores, indices = self.index.search(query, top_k)

        output: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            output.append(SearchResult(identity=self.identities[idx], score=float(score)))
        return output

    def search_best(self, query: np.ndarray, unknown_threshold: float = 0.68) -> SearchResult:
        results = self.search(query=query, top_k=1)
        if not results:
            return SearchResult(identity="unknown", score=0.0)
        best = results[0]
        if best.score < unknown_threshold:
            return SearchResult(identity="unknown", score=best.score)
        return best

    def aggregate_gallery_from_embeddings(self, identity_embeddings: Dict[str, List[np.ndarray]]) -> None:
        for identity, vectors in identity_embeddings.items():
            stack = np.vstack(vectors)
            self.add(identity, stack)
