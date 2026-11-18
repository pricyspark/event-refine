"""Controlled overnight temporal experiments, prepared with Codex assistance.

Reuses a completed overnight run. Does not evaluate test data or alter checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import TypedDict

from flowsis.artifacts import atomic_write_text
from experiments.overnight_support import inspect_dataset, check_temporal_checkpoint
from experiments.temporal_experiment_report import write_report

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    "current": [],
    "fallback": ["--base-fallback"],
    "history": ["--base-fallback", "--skip-initial-refinement"],
}


class Job(TypedDict):
    name: str
    kind: str
    module: str
    arguments: list[str]
    output: str
    candidate: str
    policy: str
    length: str
    refinement: str


def make_plan(
    args: argparse.Namespace,
    base: Path,
    temporal: Path,
    strong_base: Path,
    validation_length: int,
    validation_videos: int,
) -> list[Job]:
    jobs: list[Job] = []
    shared = [
        "--dataset-path",
        str(args.dataset_path),
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
    ]
    limits = []
    for flag, value in (
        ("--max-videos", args.max_videos),
        ("--max-frames-per-video", args.max_frames_per_video),
    ):
        if value is not None:
            limits += [flag, str(value)]

    def evaluation(
        candidate: str,
        checkpoint: Path,
        policy: str,
        length: str,
        identity: bool,
        mode: str = "runtime",
        model: Path = base,
    ) -> None:
        refinement = "identity" if identity else "learned"
        name = f"{candidate}-{policy}-{refinement}-{length}"
        output = args.output_dir / name
        flags = ["--identity-refinement"] if identity else []
        arguments = (
            shared
            + [
                "--base-model",
                str(model),
                "--temporal-model",
                str(checkpoint),
                "--flow-checkpoint",
                args.flow_checkpoint,
                "--output-dir",
                str(output),
                "--split",
                "validation",
                "--mode",
                mode,
                "--clip-length",
                "0" if length == "full" else length,
                "--all-frames",
            ]
            + POLICIES[policy]
            + flags
            + limits
        )
        jobs.append(
            {
                "name": name,
                "kind": "eval",
                "module": "flowsis.cli.eval.evaluate_temporal",
                "arguments": arguments,
                "output": str(output),
                "candidate": candidate,
                "policy": policy,
                "length": length,
                "refinement": refinement,
            }
        )

    for length in args.lengths:
        for policy in POLICIES:
            for identity in (True, False):
                evaluation("saved", temporal, policy, length, identity)
    if not args.skip_oracle:
        for identity in (True, False):
            evaluation("oracle", temporal, "current", "full", identity, mode="oracle")
    evaluation(
        "strong-base",
        temporal,
        "current",
        "full",
        False,
        mode="base",
        model=strong_base,
    )
    if args.train_candidates:
        batches = max(args.training_batches, validation_videos, 40)
        for max_length in (8, 16):
            candidate = f"train-max{max_length}"
            output = args.output_dir / candidate
            arguments = shared + [
                "--base-model",
                str(base),
                "--output-dir",
                str(output),
                "--flow-model",
                "raft_small",
                "--flow-checkpoint",
                args.flow_checkpoint,
                "--batch-size",
                "1",
                "--epochs",
                str(args.training_epochs),
                "--batches-per-epoch",
                str(batches),
                "--validation-batches",
                str(max(40, validation_videos)),
                "--min-clip-length",
                "4",
                "--max-clip-length",
                str(max_length),
                "--validation-clip-length",
                str(validation_length),
                "--strides",
                "1",
                "2",
                "3",
                "--stride-weights",
                "0.7",
                "0.2",
                "0.1",
                "--learning-rate",
                "1e-4",
                "--gradient-clip",
                "1",
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
            ]
            jobs.append(
                {
                    "name": candidate,
                    "kind": "train",
                    "module": "flowsis.cli.train.train_temporal",
                    "arguments": arguments,
                    "output": str(output),
                    "candidate": candidate,
                    "policy": "",
                    "length": "",
                    "refinement": "",
                }
            )
            for policy in POLICIES:
                for identity in (True, False):
                    evaluation(
                        candidate, output / "temporal-best.pt", policy, "full", identity
                    )
    return jobs


def source_fingerprint() -> dict[str, str]:
    files = sorted((ROOT / "flowsis").rglob("*.py")) + [
        Path(__file__),
        ROOT / "experiments/temporal_experiment_report.py",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("outputs/overnight/dev-v4")
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/temporal/dev-v4-experiments")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--flow-checkpoint", default="things")
    parser.add_argument(
        "--lengths", nargs="+", default=["4", "8", "16", "32", "64", "full"]
    )
    parser.add_argument(
        "--train-candidates",
        action="store_true",
        help="Also train matched-budget max-8/max-16 candidates from fresh refiner initialization.",
    )
    parser.add_argument("--training-epochs", type=int, default=16)
    parser.add_argument("--training-batches", type=int, default=75)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument(
        "--max-videos", type=int, help="Smoke subset only; omitted for scientific runs."
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        help="Smoke subset only; omitted for scientific runs.",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    if len(set(args.lengths)) != len(args.lengths) or any(
        value != "full" and (not value.isdigit() or int(value) < 2)
        for value in args.lengths
    ):
        parser.error("Use unique lengths >=2 or full.")
    if (
        min(args.image_size, args.training_epochs, args.training_batches) < 1
        or args.num_workers < 0
    ):
        parser.error("Sizes and budgets must be positive; workers nonnegative.")
    if any(
        value is not None and value < 1
        for value in (args.max_videos, args.max_frames_per_video)
    ):
        parser.error("Smoke subset limits must be positive.")
    if args.train_candidates and (args.max_videos or args.max_frames_per_video):
        parser.error("Do not mix training candidates with smoke subset flags.")
    args.run_dir, args.output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    saved = json.loads((args.run_dir / "summary.json").read_text())
    provenance = json.loads((args.run_dir / "provenance.json").read_text())
    args.dataset_path = (
        args.dataset_path or Path(provenance["args"]["dataset_path"])
    ).resolve()
    base, temporal = Path(saved["base"]), Path(saved["temporal"])
    strong = Path(saved.get("standalone_base", saved["base"]))
    for path in (base / "base.pt", strong / "base.pt", temporal):
        if not path.is_file():
            raise FileNotFoundError(path)
    check_temporal_checkpoint(temporal)
    audit = inspect_dataset(args.dataset_path)
    if not audit["temporal_length"]:
        raise ValueError("No usable train/validation temporal clips.")
    if args.train_candidates and audit["temporal_length"] < 8:
        raise ValueError(
            "Training comparisons require common 8-frame validation clips."
        )
    jobs = make_plan(
        args,
        base,
        temporal,
        strong,
        audit["temporal_length"],
        audit["splits"]["validation"]["temporal_videos"][str(audit["temporal_length"])],
    )
    print(
        f"Planned {len(jobs)} stages; validation only; all policies share scored frames.",
        flush=True,
    )
    if args.max_videos or args.max_frames_per_video:
        print("SMOKE SUBSET: results are not the full benchmark.", flush=True)
    if args.dry_run:
        for job in jobs:
            print(
                f"[{job['name']}] {shlex.join([sys.executable, '-m', job['module'], *job['arguments']])}"
            )
        return
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate flowsis or use --device cpu.")
    # Hash the model directories, not just head bundles: detector weights matter.
    models = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in {base, strong}
        for p in root.iterdir()
        if p.is_file() and p.name != "training_state.pt"
    }
    models[str(temporal.resolve())] = hashlib.sha256(temporal.read_bytes()).hexdigest()
    flow_path = Path(args.flow_checkpoint).expanduser()
    flow_files = (
        [flow_path]
        if flow_path.is_file()
        else sorted(
            (Path.home() / ".cache/torch/hub/checkpoints").glob(
                f"raft_small-{args.flow_checkpoint}*.ckpt"
            )
        )
    )
    models.update(
        {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in flow_files
        }
    )
    from flowsis.data.loaders import load_local_dataset, temporal_dataset_view

    evaluated = temporal_dataset_view(
        load_local_dataset(args.dataset_path)["validation"]
    )
    assets = sorted(
        {
            str(Path(value).resolve())
            for row in evaluated
            for value in [
                row.get("image_path", ""),
                row.get("video_path", ""),
                row.get("mask_bundle_path", ""),
                *[obj.get("mask_path", "") for obj in row["objects"]],
            ]
        }
        - {str(Path("").resolve())}
    )
    asset_stats = {
        path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
        for path in assets
    }
    config = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"resume", "dry_run", "fail_fast"}
        },
        "dataset_fingerprint": audit["fingerprint"],
        "sources": source_fingerprint(),
        "models": models,
        "asset_stats": asset_stats,
        "python": sys.executable,
        "jobs": jobs,
    }
    serialized = json.dumps(config, sort_keys=True, indent=2) + "\n"
    config_file = args.output_dir / "experiment_config.json"
    state_file = args.output_dir / "state.json"
    completed: set[str] = set()
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(
                f"Use a fresh output directory or --resume: {args.output_dir}"
            )
        if config_file.read_text() != serialized:
            raise ValueError(
                "Experiment configuration, inputs, model/code fingerprints or recipe changed; use a new output directory instead of resume."
            )
        completed = (
            set(json.loads(state_file.read_text()).get("completed", []))
            if state_file.exists()
            else set()
        )
    else:
        args.output_dir.mkdir(parents=True)
        atomic_write_text(config_file, serialized)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "matplotlib-cache"))
    failures: dict[str, str] = {}

    def save(status: str) -> None:
        atomic_write_text(
            state_file,
            json.dumps(
                {
                    "status": status,
                    "completed": sorted(completed),
                    "failures": failures,
                },
                indent=2,
            )
            + "\n",
        )
        write_report(args.output_dir, jobs, completed, failures, status)

    save("running")
    try:
        for job in jobs:
            output = Path(job["output"])
            expected = (
                [output / "summary.json", output / "frames.csv"]
                if job["kind"] == "eval"
                else [output / "temporal-best.pt"]
            )
            if job["name"] in completed and all(p.is_file() for p in expected):
                print(f"[skip complete] {job['name']}", flush=True)
                continue
            completed.discard(job["name"])
            command = [sys.executable, "-m", job["module"], *job["arguments"]]
            print(f"[{job['name']}] {shlex.join(command)}", flush=True)
            with (args.output_dir / "commands.jsonl").open("a") as handle:
                handle.write(
                    json.dumps({"name": job["name"], "command": command}) + "\n"
                )
            started = time.monotonic()
            try:
                if job["kind"] == "eval":
                    model = Path(command[command.index("--temporal-model") + 1])
                    if not model.is_file():
                        raise FileNotFoundError(
                            f"Training prerequisite missing: {model}"
                        )
                for artifact in expected:
                    artifact.unlink(missing_ok=True)
                with (args.output_dir / f"{job['name']}.log").open("w") as handle:
                    result = subprocess.run(
                        command,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg"),
                    )
                if result.returncode or not all(p.is_file() for p in expected):
                    raise RuntimeError(
                        f"Exit {result.returncode}; inspect {job['name']}.log"
                    )
                if job["kind"] == "train":
                    check_temporal_checkpoint(expected[0])
                completed.add(job["name"])
                write_report(args.output_dir, jobs, completed, failures, "running")
                print(
                    f"[complete] {job['name']} ({time.monotonic()-started:.0f}s)",
                    flush=True,
                )
            except Exception as error:
                completed.discard(job["name"])
                failures[job["name"]] = str(error)
                print(f"[failed] {job['name']}: {error}", flush=True)
                if args.fail_fast:
                    raise
            save("running")
        if any(
            (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) != stats
            for path, stats in asset_stats.items()
        ):
            raise RuntimeError(
                "Evaluated source assets changed during the suite; comparisons are not valid."
            )
        if source_fingerprint() != config["sources"]:
            raise RuntimeError(
                "Source code changed during the suite; comparisons are not valid."
            )
        if any(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            for path, digest in models.items()
        ):
            raise RuntimeError(
                "Input model weights/config changed during the suite; comparisons are not valid."
            )
        if source_fingerprint() != config["sources"]:
            raise RuntimeError(
                "Source code changed during the suite; comparisons are not valid."
            )
        if any(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            for path, digest in models.items()
        ):
            raise RuntimeError(
                "Input model weights changed during the suite; comparisons are not valid."
            )
        save("completed_with_failures" if failures else "complete")
    except BaseException as error:
        save(f"interrupted/failed: {error}")
        raise
    if failures:
        raise SystemExit(1)
    print(f"Results: {args.output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
