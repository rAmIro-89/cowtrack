from __future__ import annotations

import cv2


def draw_box_and_label(
    frame,
    bbox: tuple[int, int, int, int],
    track_id: int | None,
    identity: str,
    score: float,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 80), 2)
    prefix = f"T{track_id}" if track_id is not None else "T?"
    label = f"{prefix} | {identity} | {score:.2f}"
    y_text = max(15, y1 - 8)
    cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 2)
