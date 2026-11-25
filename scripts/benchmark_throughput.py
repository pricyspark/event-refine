"""Run isolated Event-Flow throughput experiments and summarize repetitions."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time
from typing import cast


@dataclass(frozen=True)
class Case:
    batch_size: int
    threads: int
    mode: str
    repeat: int

    @property
    def name(self) -> str:
        return f"b{self.batch_size}-t{self.threads}-{self.mode}-r{self.repeat}"


def build_cases(
    suite: str, batches: list[int], threads: list[int], repeats: int
) -> list[Case]:
    settings: list[tuple[int, int, str]] = []
    if suite in ("threads", "all"):
        settings.extend((1, thread, "eager") for thread in threads)
    if suite in ("batching", "all"):
        settings.extend((batch, threads[0], "eager") for batch in batches)
    if suite in ("checks", "all"):
        settings.extend(
            (batch, threads[0], mode)
            for batch in batches
            for mode in ("eager", "trusted-inputs")
        )
    if suite in ("compile", "all"):
        settings.extend(
            (batch, threads[0], mode)
            for batch in batches
            for mode in ("trusted-inputs", "compile")
        )
    settings = list(dict.fromkeys(settings))
    return [
        Case(batch, thread, mode, repeat)
        for repeat in range(1, repeats + 1)
        for batch, thread, mode in settings
    ]


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, str], list[float]] = {}
    for row in rows:
        if row["status"] == "ok":
            key = (
                cast(int, row["batch_size"]),
                cast(int, row["threads"]),
                cast(str, row["mode"]),
            )
            groups.setdefault(key, []).append(cast(float, row["clips_per_second"]))
    return [
        {
            "batch_size": batch,
            "threads": threads,
            "mode": mode,
            "runs": len(values),
            "median_clips_per_second": statistics.median(values),
            "min_clips_per_second": min(values),
            "max_clips_per_second": max(values),
        }
        for (batch, threads, mode), values in groups.items()
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _wait_for_case(
    process: subprocess.Popen[bytes], *, timeout: float, log: Path
) -> None:
    started = time.monotonic()
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            try:
                process.wait(timeout=min(30, remaining))
                return
            except subprocess.TimeoutExpired:
                print(
                    f"  Still running ({time.monotonic()-started:.0f}s); log: {log}",
                    flush=True,
                )
    except BaseException:
        # Include compiler subprocesses when timing out or cancelling a case.
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("batching", "threads", "checks", "compile", "all"),
        default="batching",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--threads", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--architecture", choices=("dfine", "rtdetrv2"), default="dfine"
    )
    parser.add_argument("--detector-model")
    parser.add_argument(
        "--compile-mode", choices=("default", "reduce-overhead"), default="default"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    batches = args.batch_sizes or ([1, 2] if args.suite in ("batching", "all") else [1])
    threads = args.threads or (
        [2, 1, 4, 8] if args.suite in ("threads", "all") else [2]
    )
    if (
        min(
            *batches,
            *threads,
            args.repeats,
            args.steps,
            args.warmup,
            args.pool_size,
            args.timeout,
        )
        <= 0
    ):
        parser.error(
            "Counts, thread counts, batch sizes, timeout and warmup must be positive."
        )
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    cases = build_cases(args.suite, batches, threads, args.repeats)
    commands = []
    for case in cases:
        command = [
            sys.executable,
            "-m",
            "event_flow.benchmark",
            "--manifest",
            str(manifest),
            "--output",
            str(output / case.name),
            "--batch-size",
            str(case.batch_size),
            "--threads",
            str(case.threads),
            "--mode",
            case.mode,
            "--compile-mode",
            args.compile_mode,
            "--device",
            args.device,
            "--architecture",
            args.architecture,
            "--image-size",
            str(args.image_size),
            "--clip-length",
            str(args.clip_length),
            "--hidden-dim",
            str(args.hidden_dim),
            "--warmup",
            str(args.warmup),
            "--steps",
            str(args.steps),
            "--pool-size",
            str(args.pool_size),
            "--seed",
            str(args.seed),
        ]
        if args.detector_model:
            source = Path(args.detector_model)
            command.extend(
                [
                    "--detector-model",
                    str(source.resolve()) if source.exists() else args.detector_model,
                ]
            )
        commands.append(command)
    if args.dry_run:
        for case, command in zip(cases, commands, strict=True):
            print(
                f"OMP_NUM_THREADS={case.threads} MKL_NUM_THREADS={case.threads} {shlex.join(command)}"
            )
        return
    if output.exists():
        parser.error(
            "Output directory exists; choose a fresh directory to avoid mixing measurements."
        )
    output.mkdir(parents=True)
    plan = {
        "suite": args.suite,
        "manifest": str(manifest),
        "warmup": args.warmup,
        "steps": args.steps,
        "image_size": args.image_size,
        "clip_length": args.clip_length,
        "compile_mode": args.compile_mode,
        "cases": [asdict(case) for case in cases],
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for number, (case, command) in enumerate(zip(cases, commands, strict=True), 1):
        print(f"[{number}/{len(cases)}] {case.name}", flush=True)
        directory = output / case.name
        directory.mkdir()
        log = directory / "console.log"
        environment = dict(
            os.environ,
            OMP_NUM_THREADS=str(case.threads),
            MKL_NUM_THREADS=str(case.threads),
        )
        if case.mode == "compile":
            environment["TORCH_LOGS"] = "graph_breaks,recompiles"
        timed_out = False
        with log.open("w") as handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                _wait_for_case(process, timeout=args.timeout, log=log)
            except subprocess.TimeoutExpired:
                timed_out = True
        result_path = directory / "result.json"
        if timed_out:
            result = {"status": "timeout", "error": f"Case exceeded {args.timeout}s"}
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        elif result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {
                "status": "error",
                "error": f"Worker exited {process.returncode} without results",
            }
        row = {
            **asdict(case),
            "status": result["status"],
            "clips_per_second": result.get("clips_per_second"),
            "frames_per_second": result.get("frames_per_second"),
            "peak_allocated_gib": result.get("peak_allocated_gib"),
            "warmup_seconds": result.get("warmup_seconds"),
            "error": result.get("error", ""),
        }
        rows.append(row)
        _write_csv(output / "results.csv", rows)
        _write_csv(output / "summary.csv", summarize(rows))
        print(json.dumps(row), flush=True)
    print(
        f'Results: {output / "results.csv"}\nMedians: {output / "summary.csv"}',
        flush=True,
    )
    if any(row["status"] in ("error", "timeout") for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
