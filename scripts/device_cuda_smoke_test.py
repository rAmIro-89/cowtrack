from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


def main() -> None:
    out_dir = Path("reports/device_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    info: dict[str, object] = {
        "pid": os.getpid(),
        "torch_cuda_is_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_device_0_name": None,
        "tensor_device": None,
        "matmul_ok": False,
        "elapsed_sec": 0.0,
        "max_memory_allocated_bytes": 0,
    }

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        info["torch_cuda_device_0_name"] = torch.cuda.get_device_name(0)
        device = torch.device("cuda:0")

        a = torch.randn((4096, 4096), device=device)
        b = torch.randn((4096, 4096), device=device)
        torch.cuda.synchronize()

        t0 = time.time()
        c = a @ b
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        info["tensor_device"] = str(c.device)
        info["matmul_ok"] = True
        info["elapsed_sec"] = round(elapsed, 6)
        info["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    else:
        device = torch.device("cpu")
        x = torch.randn((2048, 2048), device=device)
        y = torch.randn((2048, 2048), device=device)
        t0 = time.time()
        z = x @ y
        _ = float(z.mean())
        elapsed = time.time() - t0
        info["tensor_device"] = str(device)
        info["elapsed_sec"] = round(elapsed, 6)

    print(f"torch.cuda.is_available(): {info['torch_cuda_is_available']}")
    print(f"torch.cuda.device_count(): {info['torch_cuda_device_count']}")
    print(f"torch.cuda.get_device_name(0): {info['torch_cuda_device_0_name']}")
    print(f"tensor device: {info['tensor_device']}")
    print(f"matmul ok: {info['matmul_ok']}")
    print(f"elapsed_sec: {info['elapsed_sec']}")

    (out_dir / "device_cuda_smoke_test.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
