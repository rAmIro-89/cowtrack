from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a video every N frames")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    frame_idx = 0
    saved = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride == 0:
            path = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
