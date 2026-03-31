from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def _draw_hist(correct: list[float], incorrect: list[float], out_png: Path, title: str) -> None:
    w, h = 900, 500
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    bins = np.linspace(0.0, 1.0, 21)
    hc, _ = np.histogram(correct, bins=bins)
    hi, _ = np.histogram(incorrect, bins=bins)
    maxv = max(hc.max() if len(hc) else 1, hi.max() if len(hi) else 1, 1)

    x0, y0 = 60, h - 50
    pw = w - 100
    ph = h - 110
    nb = len(bins) - 1
    bw = max(1, int(pw / nb))

    cv2.rectangle(canvas, (x0, y0 - ph), (x0 + pw, y0), (50, 50, 50), 1)
    for i in range(nb):
        ch = int((hc[i] / maxv) * ph)
        ih = int((hi[i] / maxv) * ph)
        x = x0 + i * bw
        cv2.rectangle(canvas, (x, y0 - ch), (x + bw // 2, y0), (40, 120, 40), -1)
        cv2.rectangle(canvas, (x + bw // 2, y0 - ih), (x + bw, y0), (40, 40, 180), -1)

    cv2.putText(canvas, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(canvas, "Green=correct top1 score, Red=incorrect top1 score", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)


def _write_examples(rows: list[dict[str, str]], threshold: float, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched = []
    for r in rows:
        sc = _to_float(r["top1_similarity"])
        ok = int(r["top1_correct"])
        rej = int(sc < threshold)
        enriched.append(
            {
                "query_path": r["query_path"],
                "true_id": r["true_id"],
                "pred_top1": r["pred_top1"],
                "top1_similarity": sc,
                "top1_correct": ok,
                "rejected_at_threshold": rej,
            }
        )

    enriched.sort(key=lambda x: x["top1_similarity"], reverse=True)
    accepted = [r for r in enriched if r["rejected_at_threshold"] == 0]
    rejected = [r for r in enriched if r["rejected_at_threshold"] == 1]
    selected = []
    selected.extend([r for r in accepted if r["top1_correct"] == 1][:10])
    selected.extend([r for r in accepted if r["top1_correct"] == 0][:10])
    selected.extend(rejected[:10])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_path", "true_id", "pred_top1", "top1_similarity", "top1_correct", "rejected_at_threshold"],
        )
        writer.writeheader()
        writer.writerows(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create open-set score visual diagnostics")
    parser.add_argument("--prototype_csv", required=True)
    parser.add_argument("--allvectors_csv", required=True)
    parser.add_argument("--prototype_threshold", type=float, required=True)
    parser.add_argument("--allvectors_threshold", type=float, required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_rows = _load_rows(Path(args.prototype_csv))
    a_rows = _load_rows(Path(args.allvectors_csv))

    p_corr = [_to_float(r["top1_similarity"]) for r in p_rows if int(r["top1_correct"]) == 1]
    p_inc = [_to_float(r["top1_similarity"]) for r in p_rows if int(r["top1_correct"]) == 0]
    a_corr = [_to_float(r["top1_similarity"]) for r in a_rows if int(r["top1_correct"]) == 1]
    a_inc = [_to_float(r["top1_similarity"]) for r in a_rows if int(r["top1_correct"]) == 0]

    p_hist = out_dir / "prototype_score_hist.png"
    a_hist = out_dir / "allvectors_score_hist.png"
    _draw_hist(p_corr, p_inc, p_hist, "Prototype gallery score distribution")
    _draw_hist(a_corr, a_inc, a_hist, "All-vectors gallery score distribution")

    _write_examples(p_rows, threshold=args.prototype_threshold, out_csv=out_dir / "prototype_examples.csv")
    _write_examples(a_rows, threshold=args.allvectors_threshold, out_csv=out_dir / "allvectors_examples.csv")

    summary = {
        "prototype_hist": str(p_hist),
        "allvectors_hist": str(a_hist),
        "prototype_examples": str(out_dir / "prototype_examples.csv"),
        "allvectors_examples": str(out_dir / "allvectors_examples.csv"),
    }
    (out_dir / "visuals_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
