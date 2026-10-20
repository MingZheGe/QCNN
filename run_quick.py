"""Run every input/GAF combination and test both CNN and QCNN models."""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "hybrid_qcnn_small" / "train.py"
INPUT_MODES = ("conv1d", "conv2d", "fusion")
GAF_MODES = ("gasf", "gadf", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="依次测试所有 input-mode、gaf-mode，并为每组训练 CNN 和 QCNN。"
    )
    parser.add_argument("--horizon", choices=("1", "5", "both"), default="both")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[59])
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combinations = list(itertools.product(INPUT_MODES, GAF_MODES))
    horizon_count = 2 if args.horizon == "both" else 1
    training_count = len(combinations) * 2 * horizon_count * len(args.seeds)
    started = time.perf_counter()

    print(
        f"将运行 {len(combinations)} 组配置；每组均测试 CNN 和 QCNN，"
        f"共 {training_count} 次模型训练。",
        flush=True,
    )

    for index, (input_mode, gaf_mode) in enumerate(combinations, start=1):
        print(
            f"\n{'=' * 72}\n"
            f"[{index}/{len(combinations)}] input-mode={input_mode}, "
            f"gaf-mode={gaf_mode}, model=both\n"
            f"{'=' * 72}",
            flush=True,
        )
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--input-mode", input_mode,
            "--gaf-mode", gaf_mode,
            "--model", "both",
            "--horizon", args.horizon,
            "--stride", str(args.stride),
            "--epochs", str(args.epochs),
            "--seeds", *map(str, args.seeds),
            "--patience", str(args.patience),
        ]
        subprocess.run(command, cwd=ROOT, check=True)

    elapsed = time.perf_counter() - started
    print(f"\n全部组合测试完成，总耗时 {elapsed / 60:.1f} 分钟。", flush=True)


if __name__ == "__main__":
    main()
