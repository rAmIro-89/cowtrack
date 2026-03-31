from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import CowDetector
from src.preprocessing.crops import crop_bbox
from src.preprocessing.quality import blur_variance, quality_score


def collect_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    images.sort()
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 1: tracking + crops from image sequences")
    parser.add_argument("--input_dir", required=True, help="Root directory with image frames")
    parser.add_argument("--output_dir", default="data/interim/sprint1", help="Output folder for crops and reports")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda device id (e.g. 0)")
    parser.add_argument("--class_id", type=int, default=19, help="COCO class id for cow")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--max_frames", type=int, default=300, help="Limit frames for quick baseline run")
    parser.add_argument("--min_quality", type=float, default=70.0, help="Minimum quality score for saving crop")
    parser.add_argument("--save_annotated_frames", action="store_true", help="Save annotated frames with bbox, score and track_id")
    parser.add_argument("--annotated_every_n", type=int, default=1, help="Save one annotated frame every N frames")
    parser.add_argument(
        "--tracker_cfg",
        default="configs/bytetrack_no_gmc.yaml",
        help="Ultralytics tracker yaml; default disables GMC for stability",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops_by_track"
    report_dir = output_dir / "reports"
    annotated_dir = output_dir / "annotated_frames"
    crops_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.save_annotated_frames:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    frames = collect_images(input_dir)
    if not frames:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    detector = CowDetector(model_path=args.model, class_id=args.class_id, device=args.device)

    csv_path = report_dir / "detections_tracks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_index",
                "frame_name",
                "track_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "det_confidence",
                "quality_score",
                "blur_variance",
                "crop_path",
            ]
        )

        saved = 0
        total_dets = 0
        frames_with_detection = 0
        valid_track_ids: set[int] = set()
        for idx, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            detections = detector.track(
                frame,
                conf=args.conf,
                iou=args.iou,
                persist=True,
                tracker=args.tracker_cfg,
                imgsz=args.imgsz,
            )
            total_dets += len(detections)
            if detections:
                frames_with_detection += 1

            if args.save_annotated_frames and args.annotated_every_n > 0 and idx % args.annotated_every_n == 0:
                annotated = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    t_id = det.track_id if det.track_id is not None else -1
                    label = f"tid={t_id} conf={det.confidence:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 220, 80), 2)
                    cv2.putText(annotated, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 2)
                out_ann = annotated_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_ann), annotated)

            for det in detections:
                crop = crop_bbox(frame, det.bbox)
                if crop is None:
                    continue

                q_score = quality_score(crop)
                blur = blur_variance(crop)
                if q_score < args.min_quality:
                    continue

                track_id = det.track_id if det.track_id is not None else -1
                if track_id >= 0:
                    valid_track_ids.add(track_id)
                track_dir = crops_dir / f"track_{track_id:05d}"
                track_dir.mkdir(parents=True, exist_ok=True)

                crop_name = f"frame_{idx:06d}.jpg"
                crop_path = track_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                x1, y1, x2, y2 = det.bbox
                writer.writerow(
                    [
                        idx,
                        frame_path.name,
                        track_id,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"{det.confidence:.4f}",
                        f"{q_score:.2f}",
                        f"{blur:.2f}",
                        str(crop_path),
                    ]
                )
                saved += 1

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(frames)} frames | detections={total_dets} | crops_saved={saved}")

    frames_processed = len(frames)
    det_avg = (total_dets / frames_processed) if frames_processed else 0.0
    frames_with_det_pct = (frames_with_detection / frames_processed) * 100.0 if frames_processed else 0.0

    summary = {
        "frames_processed": frames_processed,
        "detections_total": total_dets,
        "detections_avg_per_frame": round(det_avg, 6),
        "valid_track_ids": len(valid_track_ids),
        "crops_saved": saved,
        "frames_with_detection_pct": round(frames_with_det_pct, 4),
        "tracker_cfg": args.tracker_cfg,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
    }

    summary_csv = report_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))

    summary_json = report_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Sprint 1 run completed")
    print(f"Frames processed: {frames_processed}")
    print(f"Total detections: {total_dets}")
    print(f"Detections avg/frame: {det_avg:.4f}")
    print(f"Valid track IDs: {len(valid_track_ids)}")
    print(f"Saved crops: {saved}")
    print(f"Frames with >=1 detection (%): {frames_with_det_pct:.2f}")
    print(f"CSV report: {csv_path}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
