from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Set


@dataclass
class StockReport:
    expected_count: int
    detected_tracks: int
    identified_count: int
    missing_ids: List[str]
    unknown_tracks: int
    recognized_stock_pct: float
    anomaly_alert: bool

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_stock(
    expected_stock: int,
    official_ids: List[str],
    observed_identities: List[str],
) -> StockReport:
    official_set: Set[str] = set(official_ids)
    observed_known = [x for x in observed_identities if x != "unknown"]
    observed_known_set = set(observed_known)

    missing = sorted(list(official_set - observed_known_set))
    unknown_tracks = len([x for x in observed_identities if x == "unknown"])
    identified_count = len(observed_known)
    detected_tracks = len(observed_identities)

    recognized_stock_pct = 0.0
    if expected_stock > 0:
        recognized_stock_pct = (len(observed_known_set) / expected_stock) * 100.0

    anomaly_alert = bool(missing) or detected_tracks < expected_stock

    return StockReport(
        expected_count=expected_stock,
        detected_tracks=detected_tracks,
        identified_count=identified_count,
        missing_ids=missing,
        unknown_tracks=unknown_tracks,
        recognized_stock_pct=recognized_stock_pct,
        anomaly_alert=anomaly_alert,
    )
