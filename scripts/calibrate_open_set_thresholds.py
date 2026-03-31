from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def calibrate_for_mode(rows: list[dict[str, str]], mode_name: str) -> dict[str, object]:
    scores_correct = [_to_float(r["top1_similarity"]) for r in rows if int(r["top1_correct"]) == 1]
    scores_incorrect = [_to_float(r["top1_similarity"]) for r in rows if int(r["top1_correct"]) == 0]
    all_scores = sorted({_to_float(r["top1_similarity"]) for r in rows})
    if not all_scores:
        raise ValueError(f"No scores for mode={mode_name}")

    # Candidate thresholds from quantiles and score grid.
    q_points = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    by_q = []
    n = len(all_scores)
    for q in q_points:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        by_q.append(all_scores[idx])
    candidates = sorted(set(all_scores[:: max(1, n // 120)] + by_q))

    total = len(rows)
    total_correct = len(scores_correct)
    total_incorrect = len(scores_incorrect)
    eval_rows: list[dict[str, float]] = []

    best = None
    best_score = -1.0
    for thr in candidates:
        accepted = [r for r in rows if _to_float(r["top1_similarity"]) >= thr]
        rejected = [r for r in rows if _to_float(r["top1_similarity"]) < thr]
        accepted_correct = sum(1 for r in accepted if int(r["top1_correct"]) == 1)
        accepted_incorrect = sum(1 for r in accepted if int(r["top1_correct"]) == 0)
        rejected_correct = sum(1 for r in rejected if int(r["top1_correct"]) == 1)
        rejected_incorrect = sum(1 for r in rejected if int(r["top1_correct"]) == 0)

        tpr = (accepted_correct / total_correct) if total_correct else 0.0
        tnr = (rejected_incorrect / total_incorrect) if total_incorrect else 0.0
        balanced = 0.5 * (tpr + tnr)

        row = {
            "threshold": thr,
            "balanced_score": balanced,
            "accepted_rate": len(accepted) / total,
            "rejection_rate": len(rejected) / total,
            "accepted_correct_rate": accepted_correct / total,
            "accepted_incorrect_rate": accepted_incorrect / total,
            "false_reject_rate": rejected_correct / total,
            "false_accept_rate": accepted_incorrect / total,
        }
        eval_rows.append(row)

        # tie-breakers: higher accepted_correct_rate, then lower false_accept_rate
        score_tuple = (balanced, row["accepted_correct_rate"], -row["false_accept_rate"])
        if best is None or score_tuple > best_score:
            best = row
            best_score = score_tuple

    assert best is not None
    top_candidates = sorted(eval_rows, key=lambda x: (x["balanced_score"], x["accepted_correct_rate"]), reverse=True)[:10]

    return {
        "mode": mode_name,
        "total_queries": total,
        "correct_without_threshold": total_correct,
        "incorrect_without_threshold": total_incorrect,
        "score_correct_mean": (sum(scores_correct) / len(scores_correct)) if scores_correct else None,
        "score_incorrect_mean": (sum(scores_incorrect) / len(scores_incorrect)) if scores_incorrect else None,
        "recommended_threshold": best["threshold"],
        "recommended_metrics": best,
        "top_threshold_candidates": top_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate unknown/open-set thresholds from query score distributions")
    parser.add_argument("--prototype_csv", required=True)
    parser.add_argument("--allvectors_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    proto_rows = _load_rows(Path(args.prototype_csv))
    all_rows = _load_rows(Path(args.allvectors_csv))

    proto = calibrate_for_mode(proto_rows, mode_name="prototype")
    allv = calibrate_for_mode(all_rows, mode_name="all_vectors")

    payload = {
        "method": "threshold sweep on top1 similarity; objective=balanced acceptance/rejection for correct vs incorrect predictions",
        "prototype": proto,
        "all_vectors": allv,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Open-set threshold calibration",
        "",
        "Method: sweep similarity thresholds over FAISS top-1 score and maximize balanced score between accepting correct matches and rejecting incorrect matches.",
        "",
    ]
    for item in [proto, allv]:
        md_lines.extend(
            [
                f"## {item['mode']}",
                f"- Total queries: {item['total_queries']}",
                f"- Correct (no threshold): {item['correct_without_threshold']}",
                f"- Incorrect (no threshold): {item['incorrect_without_threshold']}",
                f"- Mean score correct: {item['score_correct_mean']}",
                f"- Mean score incorrect: {item['score_incorrect_mean']}",
                f"- Recommended threshold: {item['recommended_threshold']}",
                f"- Balanced score: {item['recommended_metrics']['balanced_score']}",
                f"- Accepted correct rate: {item['recommended_metrics']['accepted_correct_rate']}",
                f"- False reject rate: {item['recommended_metrics']['false_reject_rate']}",
                f"- False accept rate: {item['recommended_metrics']['false_accept_rate']}",
                "",
            ]
        )
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
