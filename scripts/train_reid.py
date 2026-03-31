from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reid.train import TrainConfig, train_reid


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ReID embedding model")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--output_model", default="outputs/reid/reid_model.pt")
    parser.add_argument("--metrics_json", default=None)
    parser.add_argument("--metrics_csv", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    args = parser.parse_args()

    cfg = TrainConfig(
        train_dir=args.train_dir,
        output_model_path=args.output_model,
        metrics_json_path=args.metrics_json,
        metrics_csv_path=args.metrics_csv,
        checkpoint_dir=args.checkpoint_dir,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
    )
    train_reid(cfg)


if __name__ == "__main__":
    main()
