from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class IdentityDecision:
    track_id: int
    identity: str
    confidence: float
    n_observations: int


class TrackIdentityManager:
    """Accumulates per-track evidence and decides identity robustly."""

    def __init__(self, min_votes: int = 3, aggregation: str = "average") -> None:
        self.min_votes = min_votes
        self.aggregation = aggregation
        self.track_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.track_labels: Dict[int, List[str]] = defaultdict(list)
        self.track_scores: Dict[int, List[float]] = defaultdict(list)

    def update(self, track_id: int, embedding: np.ndarray, predicted_label: str, similarity: float) -> None:
        self.track_embeddings[track_id].append(embedding)
        self.track_labels[track_id].append(predicted_label)
        self.track_scores[track_id].append(float(similarity))

    def can_decide(self, track_id: int) -> bool:
        return len(self.track_labels.get(track_id, [])) >= self.min_votes

    def get_track_embedding(self, track_id: int) -> Optional[np.ndarray]:
        embeddings = self.track_embeddings.get(track_id)
        if not embeddings:
            return None
        matrix = np.vstack(embeddings)
        mean_embedding = matrix.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(mean_embedding, axis=1, keepdims=True)
        return mean_embedding / np.clip(norm, 1e-9, None)

    def decide(self, track_id: int) -> Optional[IdentityDecision]:
        labels = self.track_labels.get(track_id)
        scores = self.track_scores.get(track_id)
        if not labels or not scores:
            return None

        if self.aggregation == "vote":
            winner, _ = Counter(labels).most_common(1)[0]
            winner_scores = [s for l, s in zip(labels, scores) if l == winner]
            conf = float(np.mean(winner_scores)) if winner_scores else float(np.mean(scores))
        else:
            combined: Dict[str, List[float]] = defaultdict(list)
            for label, score in zip(labels, scores):
                combined[label].append(score)
            winner = max(combined.items(), key=lambda item: np.mean(item[1]))[0]
            conf = float(np.mean(combined[winner]))

        return IdentityDecision(
            track_id=track_id,
            identity=winner,
            confidence=conf,
            n_observations=len(labels),
        )

    def decide_all(self) -> List[IdentityDecision]:
        decisions: List[IdentityDecision] = []
        for track_id in self.track_labels:
            decision = self.decide(track_id)
            if decision is not None:
                decisions.append(decision)
        return decisions
