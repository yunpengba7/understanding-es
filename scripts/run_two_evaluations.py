from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationTask:
    label: str
    model: str


def idle_gpus(*, maximum_used_memory_mib: int) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    available = []
    for line in result.stdout.splitlines():
        index_text, memory_text = (part.strip() for part in line.split(",", maxsplit=1))
        if int(memory_text) <= maximum_used_memory_mib:
            available.append(int(index_text))
    return available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Base and ES evaluation tasks independently on any idle GPUs"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--es-model", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    pending = [
        EvaluationTask("base", args.base_model),
        EvaluationTask("es_step_234", args.es_model),
    ]
    running: dict[int, tuple[EvaluationTask, subprocess.Popen[str]]] = {}
    failures: list[str] = []

    while pending or running:
        for gpu, (task, process) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del running[gpu]
            if return_code != 0:
                failures.append(f"{task.label} exited with status {return_code}")
            else:
                print(f"completed task={task.label} gpu={gpu}", flush=True)

        free = [gpu for gpu in idle_gpus(
            maximum_used_memory_mib=args.maximum_used_memory_mib
        ) if gpu not in running]
        while pending and free:
            task = pending.pop(0)
            gpu = free.pop(0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                "uv",
                "run",
                "es-evaluate",
                "--model",
                task.model,
                "--label",
                task.label,
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(output_root / task.label),
            ]
            process = subprocess.Popen(command, env=environment, text=True)
            running[gpu] = (task, process)
            print(f"started task={task.label} gpu={gpu}", flush=True)

        if pending or running:
            time.sleep(max(1.0, args.poll_seconds))

    if failures:
        raise RuntimeError("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
