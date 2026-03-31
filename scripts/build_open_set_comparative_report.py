from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_best(rows: list[dict[str, object]]) -> dict[str, object]:
    # Primary objective: maximize thresholded top1. Tie-break by lower false accepts, then lower rejection.
    return sorted(
        rows,
        key=lambda r: (
            float(r["top1_accuracy_thresholded"]),
            -int(r["false_accepts"]),
            -float(r["unknown_rejection_rate"]),
        ),
        reverse=True,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build open-set comparative report artifacts")
    parser.add_argument("--thresholds_json", required=True)
    parser.add_argument("--prototype_nothr_json", required=True)
    parser.add_argument("--prototype_thr_json", required=True)
    parser.add_argument("--allvectors_nothr_json", required=True)
    parser.add_argument("--allvectors_thr_json", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    thresholds = _load_json(Path(args.thresholds_json))
    p_n = _load_json(Path(args.prototype_nothr_json))
    p_t = _load_json(Path(args.prototype_thr_json))
    a_n = _load_json(Path(args.allvectors_nothr_json))
    a_t = _load_json(Path(args.allvectors_thr_json))

    rows = [
        {
            "mode": "prototype",
            "setting": "no_threshold",
            "threshold": None,
            "gallery_vectors": int(p_n["gallery_vectors"]),
            "top1_accuracy": float(p_n["top1_accuracy"]),
            "top5_accuracy": float(p_n["top5_accuracy"]),
            "top1_accuracy_thresholded": float(p_n["top1_accuracy_thresholded"]),
            "top5_accuracy_thresholded": float(p_n["top5_accuracy_thresholded"]),
            "unknown_rejection_rate": float(p_n["unknown_rejection_rate"]),
            "false_rejects": int(p_n["false_rejects"]),
            "false_accepts": int(p_n["false_accepts"]),
            "accepted_queries": int(p_n["accepted_queries"]),
            "query_count": int(p_n["query_count"]),
        },
        {
            "mode": "prototype",
            "setting": "with_threshold",
            "threshold": float(p_t["unknown_threshold"]),
            "gallery_vectors": int(p_t["gallery_vectors"]),
            "top1_accuracy": float(p_t["top1_accuracy"]),
            "top5_accuracy": float(p_t["top5_accuracy"]),
            "top1_accuracy_thresholded": float(p_t["top1_accuracy_thresholded"]),
            "top5_accuracy_thresholded": float(p_t["top5_accuracy_thresholded"]),
            "unknown_rejection_rate": float(p_t["unknown_rejection_rate"]),
            "false_rejects": int(p_t["false_rejects"]),
            "false_accepts": int(p_t["false_accepts"]),
            "accepted_queries": int(p_t["accepted_queries"]),
            "query_count": int(p_t["query_count"]),
        },
        {
            "mode": "all_vectors",
            "setting": "no_threshold",
            "threshold": None,
            "gallery_vectors": int(a_n["gallery_vectors"]),
            "top1_accuracy": float(a_n["top1_accuracy"]),
            "top5_accuracy": float(a_n["top5_accuracy"]),
            "top1_accuracy_thresholded": float(a_n["top1_accuracy_thresholded"]),
            "top5_accuracy_thresholded": float(a_n["top5_accuracy_thresholded"]),
            "unknown_rejection_rate": float(a_n["unknown_rejection_rate"]),
            "false_rejects": int(a_n["false_rejects"]),
            "false_accepts": int(a_n["false_accepts"]),
            "accepted_queries": int(a_n["accepted_queries"]),
            "query_count": int(a_n["query_count"]),
        },
        {
            "mode": "all_vectors",
            "setting": "with_threshold",
            "threshold": float(a_t["unknown_threshold"]),
            "gallery_vectors": int(a_t["gallery_vectors"]),
            "top1_accuracy": float(a_t["top1_accuracy"]),
            "top5_accuracy": float(a_t["top5_accuracy"]),
            "top1_accuracy_thresholded": float(a_t["top1_accuracy_thresholded"]),
            "top5_accuracy_thresholded": float(a_t["top5_accuracy_thresholded"]),
            "unknown_rejection_rate": float(a_t["unknown_rejection_rate"]),
            "false_rejects": int(a_t["false_rejects"]),
            "false_accepts": int(a_t["false_accepts"]),
            "accepted_queries": int(a_t["accepted_queries"]),
            "query_count": int(a_t["query_count"]),
        },
    ]

    best = _pick_best(rows)

    payload = {
        "checkpoint": str(p_n["weights"]),
        "threshold_calibration_method": thresholds["method"],
        "recommended_thresholds": {
            "prototype": thresholds["prototype"]["recommended_threshold"],
            "all_vectors": thresholds["all_vectors"]["recommended_threshold"],
        },
        "rows": rows,
        "best_configuration": best,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "mode",
        "setting",
        "threshold",
        "gallery_vectors",
        "top1_accuracy",
        "top5_accuracy",
        "top1_accuracy_thresholded",
        "top5_accuracy_thresholded",
        "unknown_rejection_rate",
        "false_rejects",
        "false_accepts",
        "accepted_queries",
        "query_count",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Open-set FAISS comparative report",
        "",
        f"Checkpoint: {p_n['weights']}",
        "",
        "## Recommended thresholds",
        f"- prototype: {thresholds['prototype']['recommended_threshold']}",
        f"- all_vectors: {thresholds['all_vectors']['recommended_threshold']}",
        "",
        "## Results table",
        "",
        "| mode | setting | threshold | top1 | top5 | top1_thresholded | top5_thresholded | reject_rate | false_rejects | false_accepts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        thr = "-" if r["threshold"] is None else f"{float(r['threshold']):.9f}"
        md_lines.append(
            "| "
            + f"{r['mode']} | {r['setting']} | {thr} | "
            + f"{float(r['top1_accuracy']):.6f} | {float(r['top5_accuracy']):.6f} | "
            + f"{float(r['top1_accuracy_thresholded']):.6f} | {float(r['top5_accuracy_thresholded']):.6f} | "
            + f"{float(r['unknown_rejection_rate']):.6f} | {int(r['false_rejects'])} | {int(r['false_accepts'])} |"
        )

    md_lines.extend(
        [
            "",
            "## Best configuration",
            f"- mode: {best['mode']}",
            f"- setting: {best['setting']}",
            f"- threshold: {best['threshold']}",
            f"- top1_thresholded: {best['top1_accuracy_thresholded']}",
            f"- top5_thresholded: {best['top5_accuracy_thresholded']}",
            f"- reject_rate: {best['unknown_rejection_rate']}",
            f"- false_accepts: {best['false_accepts']}",
            f"- false_rejects: {best['false_rejects']}",
            "",
            "Notes:",
            "- Dataset used for query evaluation contains known IDs only.",
            "- Rejections in thresholded runs correspond to false rejects under this protocol.",
        ]
    )

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
