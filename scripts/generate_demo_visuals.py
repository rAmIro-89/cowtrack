"""
Generate visual demo of current Re-ID pipeline state:
- similarity matrix heatmap
- correct/incorrect examples with top-5
- score distribution histogram
- confusion analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _to_int(v: str) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _create_score_histogram(rows: list[dict[str, str]], out_png: Path) -> None:
    """Create histogram of top1 similarity scores, split by correct/incorrect."""
    w, h = 1200, 600
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    scores_correct = [_to_float(r["top1_similarity"]) for r in rows if _to_int(r["top1_correct"]) == 1]
    scores_incorrect = [_to_float(r["top1_similarity"]) for r in rows if _to_int(r["top1_correct"]) == 0]

    bins = np.linspace(0.98, 1.0, 21)
    hc, _ = np.histogram(scores_correct, bins=bins)
    hi, _ = np.histogram(scores_incorrect, bins=bins)
    maxv = max(hc.max() if len(hc) else 1, hi.max() if len(hi) else 1, 1)

    x0, y0 = 80, h - 60
    pw = w - 140
    ph = h - 120
    nb = len(bins) - 1
    bw = max(1, int(pw / nb))

    # Draw axes
    cv2.rectangle(canvas, (x0, y0 - ph), (x0 + pw, y0), (50, 50, 50), 2)

    # Draw bars
    for i in range(nb):
        ch = int((hc[i] / maxv) * ph) if maxv > 0 else 0
        ih = int((hi[i] / maxv) * ph) if maxv > 0 else 0
        x = x0 + i * bw
        cv2.rectangle(canvas, (x, y0 - ch), (x + bw // 2, y0), (50, 200, 50), -1)  # Green correct
        cv2.rectangle(canvas, (x + bw // 2, y0 - ih), (x + bw, y0), (50, 50, 200), -1)  # Blue incorrect

    # Labels
    cv2.putText(canvas, "Top-1 Similarity Score Distribution", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
    cv2.putText(canvas, f"Green = Correct predictions (n={len(scores_correct)})", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 1)
    cv2.putText(canvas, f"Blue = Wrong predictions (n={len(scores_incorrect)})", (600, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 200), 1)

    # X-axis labels
    for i in range(0, nb, 2):
        x = x0 + i * bw
        label = f"{bins[i]:.3f}"
        cv2.putText(canvas, label, (x, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)


def _create_confusion_matrix(rows: list[dict[str, str]], out_png: Path, max_ids: int = 46) -> None:
    """Create confusion matrix: true_id vs pred_top1."""
    # Build matrix
    matrix = defaultdict(lambda: defaultdict(int))
    id_set = set()

    for r in rows:
        true_id = r["true_id"]
        pred_id = r["pred_top1"]
        correct = _to_int(r["top1_correct"])
        id_set.add(true_id)
        id_set.add(pred_id)
        matrix[true_id][pred_id] += 1

    id_list = sorted(id_set)[:max_ids]  # Limit display
    n = len(id_list)
    cell_size = max(10, 800 // n)
    h = cell_size * n + 100
    w = cell_size * n + 100
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # Normalize values for color mapping
    max_count = max((matrix[i][j] for i in id_list for j in id_list), default=1)

    for i, true_id in enumerate(id_list):
        for j, pred_id in enumerate(id_list):
            count = matrix[true_id][pred_id]
            intensity = int(255 * (1 - count / max_count)) if count > 0 else 255
            color = (50, 50, 50) if true_id == pred_id else (intensity, intensity, intensity)
            x = 50 + j * cell_size
            y = 50 + i * cell_size
            cv2.rectangle(canvas, (x, y), (x + cell_size, y + cell_size), color, -1)
            if cell_size > 15 and count > 0:
                cv2.putText(canvas, str(count), (x + 3, y + cell_size - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.putText(canvas, "Confusion Matrix (True ID vs Predicted Top-1)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    cv2.putText(canvas, "True ID (rows) vs Predicted ID (cols) | Dark = Correct", (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)


def _create_top5_examples(rows: list[dict[str, str]], out_csv: Path, split_by_correct: bool = True) -> None:
    """Create CSV with concrete examples: query image, true ID, top-5, score, correctness."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Separate correct vs incorrect
    correct_rows = [r for r in rows if _to_int(r["top1_correct"]) == 1]
    incorrect_rows = [r for r in rows if _to_int(r["top1_correct"]) == 0]

    # Pick best/worst examples
    correct_rows.sort(key=lambda r: _to_float(r["top1_similarity"]), reverse=True)
    incorrect_rows.sort(key=lambda r: _to_float(r["top1_similarity"]), reverse=True)

    examples = []
    examples.extend([(r, "correct_high_confidence") for r in correct_rows[:10]])
    examples.extend([(r, "incorrect_high_score") for r in incorrect_rows[:10]])
    examples.extend([(r, "correct_low_confidence") for r in correct_rows[-5:]])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example_type",
                "query_path",
                "true_id",
                "pred_top1",
                "top1_similarity",
                "top1_correct",
                "top_5_candidates",
            ],
        )
        writer.writeheader()
        for r, etype in examples:
            writer.writerow(
                {
                    "example_type": etype,
                    "query_path": r["query_path"],
                    "true_id": r["true_id"],
                    "pred_top1": r["pred_top1"],
                    "top1_similarity": r["top1_similarity"],
                    "top1_correct": r["top1_correct"],
                    "top_5_candidates": r["pred_topk"],
                }
            )


def _create_summary_stats(rows: list[dict[str, str]], out_json: Path) -> None:
    """Create summary statistics JSON."""
    correct_count = sum(1 for r in rows if _to_int(r["top1_correct"]) == 1)
    incorrect_count = len(rows) - correct_count
    scores_correct = [_to_float(r["top1_similarity"]) for r in rows if _to_int(r["top1_correct"]) == 1]
    scores_incorrect = [_to_float(r["top1_similarity"]) for r in rows if _to_int(r["top1_correct"]) == 0]

    summary = {
        "total_queries": len(rows),
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "top1_accuracy": correct_count / len(rows),
        "mean_score_when_correct": np.mean(scores_correct) if scores_correct else None,
        "mean_score_when_incorrect": np.mean(scores_incorrect) if scores_incorrect else None,
        "min_score_when_correct": np.min(scores_correct) if scores_correct else None,
        "min_score_when_incorrect": np.min(scores_incorrect) if scores_incorrect else None,
        "max_score_when_incorrect": np.max(scores_incorrect) if scores_incorrect else None,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual demo of Re-ID pipeline")
    parser.add_argument("--query_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    rows = _load_rows(Path(args.query_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_visuals] Processing {len(rows)} queries...")
    _create_score_histogram(rows, out_dir / "score_distribution.png")
    print("  -> score_distribution.png")

    _create_confusion_matrix(rows, out_dir / "confusion_matrix.png")
    print("  -> confusion_matrix.png")

    _create_top5_examples(rows, out_dir / "top5_examples.csv")
    print("  -> top5_examples.csv")

    _create_summary_stats(rows, out_dir / "demo_summary.json")
    print("  -> demo_summary.json")

    print(json.dumps(json.loads((out_dir / "demo_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
