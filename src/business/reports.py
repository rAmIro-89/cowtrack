from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.business.stock_logic import StockReport


def export_daily_report(report: StockReport, output_path: str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".csv":
        pd.DataFrame([report.to_dict()]).to_csv(out, index=False)
    elif out.suffix.lower() == ".json":
        pd.DataFrame([report.to_dict()]).to_json(out, orient="records", indent=2)
    else:
        raise ValueError("Output path must end with .csv or .json")

    return out
