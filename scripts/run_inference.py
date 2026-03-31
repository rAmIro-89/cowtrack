from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.detection.detector import CowDetector
from src.preprocessing.crops import crop_bbox
from src.preprocessing.quality import quality_score
from src.reid.faiss_index import FaissGallery
from src.reid.infer import ReIDInferencer
from src.reid.model import EmbeddingNet
from src.tracking.tracker import TrackIdentityManager
from src.utils.visualization import draw_box_and_label


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    return device


def load_gallery(npz_path: str, embedding_dim: int = 256) -> FaissGallery:
    data = np.load(npz_path)
    gallery = FaissGallery(embedding_dim=embedding_dim)
    for identity in data.files:
        gallery.add(identity, data[identity])
    return gallery


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end video inference")
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--reid_weights", required=True)
    parser.add_argument("--gallery_npz", required=True)
    parser.add_argument("--out_video", default="outputs/inference/annotated.mp4")
    parser.add_argument("--out_csv", default="outputs/inference/tracks.csv")
    parser.add_argument("--out_json", default="outputs/inference/tracks.json")
    parser.add_argument("--quality_threshold", type=float, default=80.0)
    parser.add_argument("--unknown_threshold", type=float, default=0.68)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda device id (e.g. 0)")
    args = parser.parse_args()

    output_video = Path(args.out_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    detector = CowDetector(model_path=args.detector, class_id=19, device=args.device)
    gallery = load_gallery(args.gallery_npz, embedding_dim=256)

    reid_model = EmbeddingNet(embedding_dim=256, pretrained=False).to(device)
    reid_model.load_state_dict(torch.load(args.reid_weights, map_location=device))
    inferencer = ReIDInferencer(model=reid_model, gallery=gallery, image_size=224, device=device)

    identity_manager = TrackIdentityManager(min_votes=3, aggregation="average")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    track_log: dict[int, dict] = {}

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.track(frame)
        for det in detections:
            crop = crop_bbox(frame, det.bbox)
            if crop is None:
                continue

            if quality_score(crop) < args.quality_threshold:
                continue

            result = inferencer.identify(crop, unknown_threshold=args.unknown_threshold)
            track_id = det.track_id if det.track_id is not None else -1
            embedding = inferencer.embed_crop(crop)
            identity_manager.update(track_id, embedding, result.identity, result.score)

            decision = identity_manager.decide(track_id)
            shown_identity = decision.identity if decision else result.identity
            shown_conf = decision.confidence if decision else result.score

            draw_box_and_label(frame, det.bbox, det.track_id, shown_identity, shown_conf)

            entry = track_log.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "identity_final": shown_identity,
                    "confidence": float(shown_conf),
                    "timestamps_seen": [],
                },
            )
            entry["identity_final"] = shown_identity
            entry["confidence"] = float(shown_conf)
            entry["timestamps_seen"].append(round(frame_idx / fps, 2))

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows = list(track_log.values())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["track_id", "identity_final", "confidence", "timestamps_seen"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for row in rows:
            writer_csv.writerow(
                {
                    "track_id": row["track_id"],
                    "identity_final": row["identity_final"],
                    "confidence": row["confidence"],
                    "timestamps_seen": ";".join(map(str, row["timestamps_seen"])),
                }
            )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved annotated video: {output_video}")
    print(f"Saved CSV report: {out_csv}")
    print(f"Saved JSON report: {out_json}")


if __name__ == "__main__":
    main()
