"""Train and evaluate base/temporal candidates, with Codex assistance.

Run from the repository: python -m experiments.run_overnight --help
A fresh export trains a fresh detector by default; existing runs are opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from experiments.overnight_report import RunState, represented_ap, write_report

ROOT = Path(__file__).resolve().parents[1]


def detector_history(root: Path) -> list[tuple[int, int, float]]:
    path = root / "training_log.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue  # The active writer may be halfway through its final line.
        event = json.loads(line)
        if event.get("event") == "validation_epoch":
            loss = float(event["loss"])
            if not math.isfinite(loss):
                raise ValueError("Detector validation has a non-finite loss.")
            row = (int(event["epoch"]) + 1, int(event["global_step"]), loss)
            if rows and (row[0] <= rows[-1][0] or row[1] <= rows[-1][1]):
                raise ValueError(
                    "Detector log contains restarted/reused epochs; use a fresh run directory."
                )
            rows.append(row)
    return rows


def detector_finished(root: Path, epochs: int) -> bool:
    rows = detector_history(root)
    # This metadata file is atomically written AFTER weights and preprocessing.
    return bool(
        rows
        and max(row[0] for row in rows) >= epochs
        and (root / "final/flowsis_detector.json").is_file()
    )


def detector_candidates(root: Path) -> list[Path]:
    history = detector_history(root)
    ranked = sorted(history, key=lambda row: row[2])
    if not ranked:
        raise ValueError("No completed detector validation epochs.")
    _, step, _ = ranked[0]
    candidates = [root / f"checkpoint-{step:06d}"]
    if step != history[-1][1]:
        candidates.append(root / "final")
    elif len(ranked) > 1:
        # Final repeats the last epoch's weights. Compare another epoch instead.
        candidates.append(root / f"checkpoint-{ranked[1][1]:06d}")
    for path in candidates:
        if not (path / "flowsis_detector.json").is_file():
            raise FileNotFoundError(f"Incomplete detector checkpoint: {path}")
    return candidates


def data_smoke(dataset_path: Path, image_size: int, clip_length: int = 8) -> None:
    from flowsis.data.loaders import load_local_dataset, temporal_dataset_view
    from flowsis.data.temporal import (
        DynamicClipBatchSampler,
        TemporalClipDataset,
        ClipSpec,
    )

    data = load_local_dataset(dataset_path)
    for split_name in ("train", "validation"):
        dataset = TemporalClipDataset(
            temporal_dataset_view(data[split_name]),
            image_size=image_size,
            training=False,
        )
        for length, stride in sorted({(min(4, clip_length), 1), (clip_length, 1)}):
            sampler = DynamicClipBatchSampler(
                dataset.frame_indices,
                batch_size=1,
                batches_per_epoch=1,
                min_length=length,
                max_length=length,
                strides=(stride,),
                stride_weights=(1,),
                positive_indices=dataset.positive_indices,
                sequence_ids=dataset.sequence_ids,
            )
            eligible = sampler.geometries[0][3]
            for video, starts in sorted(eligible.items()):
                sample = dataset[ClipSpec(video, starts[0], length, stride)]
                if sample["frames"].shape != (length, 3, image_size, image_size):
                    raise ValueError("Unexpected decoded clip shape.")
                if not sample["mask_valid"].any():
                    raise ValueError("Decoded clip has no supervised masks.")
            print(
                f"Loader OK: {split_name} length={length} stride={stride}, {len(eligible)} videos",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--detector-run",
        type=Path,
        help="Explicitly reuse/wait for an existing detector run instead of training a fresh detector.",
    )
    parser.add_argument("--detector", choices=("dfine", "rtdetrv2"), default="dfine")
    parser.add_argument(
        "--detector-model",
        help="Pretrained initialization; defaults to the detector backend's registered model.",
    )
    parser.add_argument("--detector-epochs", type=int, default=20)
    parser.add_argument("--detector-batch-size", type=int, default=2)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        help="Optional existing bank; vocabulary/shapes must match the export.",
    )
    parser.add_argument(
        "--prompt-suite",
        type=Path,
        help="Optional reviewed three-prompt JSON. Otherwise bootstrap label/template embeddings without an LLM/API call.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--temporal-epochs", type=int, default=8)
    parser.add_argument("--temporal-batches", type=int, default=75)
    parser.add_argument("--flow-checkpoint", default="things")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait-hours", type=float, default=4)
    parser.add_argument(
        "--optional-after-hours",
        type=float,
        default=8,
        help="Skip starting optional control after this elapsed time; not a total runtime limit.",
    )
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument(
        "--require-temporal",
        action="store_true",
        help="Fail preflight if supervised train/validation clips are unavailable.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate final frozen models on available test cohorts. Never used for selection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit data and print commands without downloads, training, or output writes.",
    )
    parser.add_argument("--data-smoke-only", action="store_true")
    parser.add_argument("--smoke-image-size", type=int, default=128)
    args = parser.parse_args()
    os.chdir(ROOT)
    if (
        min(
            args.detector_epochs,
            args.detector_batch_size,
            args.base_epochs,
            args.batch_size,
            args.temporal_epochs,
            args.temporal_batches,
            args.image_size,
            args.smoke_image_size,
        )
        < 1
        or args.num_workers < 0
    ):
        parser.error("Epochs, batches and sizes must be positive; workers nonnegative.")
    if args.wait_hours <= 0 or args.optional_after_hours <= 0:
        parser.error("Time budgets must be positive.")
    if args.skip_temporal and args.require_temporal:
        parser.error("--skip-temporal and --require-temporal conflict.")
    if args.embeddings_dir and args.prompt_suite:
        parser.error("Choose --embeddings-dir or --prompt-suite, not both.")
    from experiments.overnight_support import (
        inspect_dataset,
        prompt_suite,
        check_temporal_checkpoint,
        identity_checkpoint,
        validation_views,
    )
    from experiments.overnight_report import model_manifest, learning_curves
    from flowsis.data.prompts import load_prompt_bank, validate_prompt_suite
    import torch

    args.dataset_path = args.dataset_path.expanduser().resolve()
    audit = inspect_dataset(args.dataset_path)
    labels = tuple(audit["labels"])
    print(
        f"Dataset: {len(audit['labels'])} classes; train={audit['splits']['train']['rows']}, validation={audit['splits']['validation']['rows']}",
        flush=True,
    )
    for warning in audit["warnings"]:
        print(f"Data warning: {warning}", flush=True)
    length = audit["temporal_length"]
    temporal_enabled = bool(length) and not args.skip_temporal
    if args.require_temporal and not length:
        raise ValueError(
            "Temporal training requires usable clips in both train and validation; see dataset warnings."
        )
    if args.data_smoke_only:
        if not length:
            raise ValueError("No shared temporal geometry in train/validation.")
        data_smoke(args.dataset_path, args.smoke_image_size, length)
        return
    if (
        not args.dry_run
        and args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable. Activate the training environment or select --device cpu."
        )
    if args.output_dir is None:
        args.output_dir = Path("outputs/overnight") / time.strftime("run-%Y%m%d-%H%M%S")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"Choose a fresh output directory: {args.output_dir}")
    external_detector = args.detector_run is not None
    args.detector_run = (
        args.detector_run or args.output_dir / "detector-training"
    ).resolve()
    expected_epochs = args.detector_epochs
    if external_detector:
        config = json.loads((args.detector_run / "run_config.json").read_text())[
            "training_args"
        ]
        if Path(config["dataset_path"]).resolve() != args.dataset_path:
            raise ValueError("The supplied detector run uses a different dataset path.")
        if (
            config.get("max_steps") is not None
            or config.get("resume_from") is not None
            or config.get("save_every_epochs", 1) != 1
        ):
            raise ValueError(
                "Expected full detector epochs with per-epoch checkpoints."
            )
        expected_epochs = int(config["epochs"])
        if config.get("detector_architecture", args.detector) != args.detector:
            raise ValueError("--detector does not match the supplied detector run.")
        audit["warnings"].append(
            "Reused detector: verify its training provenance before making unseen-tool claims; matching paths do not detect replaced datasets."
        )
    existing_embeddings = args.embeddings_dir is not None
    args.embeddings_dir = (
        args.embeddings_dir or args.output_dir / "prompts/embeddings"
    ).resolve()
    if existing_embeddings:
        bank = load_prompt_bank(labels, args.embeddings_dir)
        if not torch.isfinite(bank).all():
            raise ValueError("Nonfinite prompt embeddings.")
    supplied_suite = args.prompt_suite is not None
    suite = (
        json.loads(args.prompt_suite.read_text())
        if supplied_suite
        else prompt_suite(list(labels))
    )
    validate_prompt_suite(suite, dict.fromkeys(labels, ""))
    state: RunState = {"status": "preflight", "outcomes": [], "dataset_audit": audit}
    started = time.monotonic()

    def run(stage: str, module: str, *arguments: object) -> None:
        command = [sys.executable, "-m", module, *(str(arg) for arg in arguments)]
        print(f"[{stage}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            return
        state["status"] = stage
        write_report(args.output_dir, state)
        with (args.output_dir / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": stage, "command": command}) + "\n")
        env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        stage_started = time.monotonic()
        with (args.output_dir / f"{stage}.log").open("w") as handle:
            completed = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, env=env
            )
        if completed.returncode:
            raise RuntimeError(
                f"{stage} failed ({completed.returncode}); see {args.output_dir / (stage + '.log')}"
            )
        state["outcomes"].append(
            f"{stage}: complete ({time.monotonic() - stage_started:.0f}s)"
        )
        write_report(args.output_dir, state)

    if not args.dry_run:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "dataset_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        for name in ("build_report.json", "split_assignments.json"):
            if (args.dataset_path / name).is_file():
                shutil.copyfile(
                    args.dataset_path / name, args.output_dir / f"dataset_{name}"
                )
        (args.output_dir / "prompts").mkdir()
        (args.output_dir / "prompts/text_prompts.json").write_text(
            json.dumps(suite, indent=2) + "\n"
        )
        provenance = {
            "args": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "python": sys.version,
            "dataset_fingerprint": audit["fingerprint"],
            "prompt_types": (
                "existing embeddings"
                if existing_embeddings
                else (
                    "reviewed three prompts"
                    if supplied_suite
                    else "label/template bootstrap"
                )
            ),
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / "flowsis").rglob("*.py"))
                + sorted((ROOT / "experiments").glob("*.py"))
            },
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
    try:
        if not existing_embeddings:
            types = (
                []
                if supplied_suite
                else ["--prompt-type", "label", "--prompt-type", "template"]
            )
            run(
                "prompt-embeddings",
                "flowsis.cli.data.embed_text_prompts",
                "--dataset-path",
                args.dataset_path,
                "--prompt-suite",
                args.output_dir / "prompts/text_prompts.json",
                "--output-dir",
                args.embeddings_dir,
                "--device",
                args.device,
                *types,
            )
            if (
                not args.dry_run
                and not torch.isfinite(
                    load_prompt_bank(labels, args.embeddings_dir)
                ).all()
            ):
                raise ValueError("Generated prompt embeddings are nonfinite.")
        if temporal_enabled and not args.dry_run:
            data_smoke(args.dataset_path, args.smoke_image_size, length)
        if not external_detector:
            initialization = (
                ["--model", args.detector_model] if args.detector_model else []
            )
            run(
                "detector-train",
                "flowsis.cli.train.train_detector",
                "--dataset-path",
                args.dataset_path,
                "--detector",
                args.detector,
                *initialization,
                "--output-dir",
                args.detector_run,
                "--epochs",
                expected_epochs,
                "--batch-size",
                args.detector_batch_size,
                "--num-workers",
                args.num_workers,
                "--image-size",
                args.image_size,
                "--lr",
                "1e-4",
                "--backbone-lr",
                "1e-5",
                "--warmup-steps",
                min(
                    100,
                    round(
                        0.05
                        * math.ceil(
                            audit["splits"]["train"]["base_rows"]
                            / args.detector_batch_size
                        )
                        * expected_epochs
                    ),
                ),
                "--max-grad-norm",
                1,
                "--save-every-epochs",
                1,
                "--device",
                args.device,
                "--rotation-min-degrees",
                -22.5,
                "--rotation-max-degrees",
                22.5,
                "--seed",
                args.seed,
            )
        while not args.dry_run and not detector_finished(
            args.detector_run, expected_epochs
        ):
            if not external_detector:
                raise RuntimeError(
                    "Detector training returned without its completion marker."
                )
            state["detector_history"] = detector_history(args.detector_run)
            write_report(args.output_dir, state)
            if time.monotonic() - started > args.wait_hours * 3600:
                raise TimeoutError(
                    "Supplied detector did not finish within --wait-hours."
                )
            time.sleep(30)
        state["detector_history"] = detector_history(args.detector_run)
        candidates = (
            detector_candidates(args.detector_run)
            if not args.dry_run
            else [
                args.detector_run / "BEST-LOSS-CHECKPOINT",
                args.detector_run / "final",
            ]
        )
        for index, candidate in enumerate(candidates):
            for iou in (0.5, 0.75):
                run(
                    f"detector-{index}-ap{iou}",
                    "flowsis.cli.eval.evaluate_detector",
                    "--detector",
                    args.detector,
                    "--model",
                    candidate,
                    "--dataset-path",
                    args.dataset_path,
                    "--split",
                    "validation",
                    "--device",
                    args.device,
                    "--iou-threshold",
                    iou,
                    "--output-dir",
                    args.output_dir / f"detector/{index}-ap{iou}",
                )
        selected = (
            0
            if args.dry_run
            else max(
                range(len(candidates)),
                key=lambda i: represented_ap(
                    args.output_dir / f"detector/{i}-ap0.5/metrics.json"
                ),
            )
        )
        detector = candidates[selected]
        if not args.dry_run:
            mapping = json.loads((detector / "config.json").read_text())["id2label"]
            if tuple(mapping[str(i)] for i in range(len(mapping))) != labels:
                raise ValueError(
                    "Selected detector vocabulary differs from the export."
                )
        state["detector"] = str(detector)
        common = [
            "--detector",
            args.detector,
            "--detector-model",
            detector,
            "--dataset-path",
            args.dataset_path,
            "--text-embeddings-dir",
            args.embeddings_dir,
            "--device",
            args.device,
            "--image-size",
            args.image_size,
            "--batch-size",
            args.batch_size,
            "--num-workers",
            args.num_workers,
            "--head-lr",
            "1e-4",
            "--scheduler-type",
            "constant_with_warmup",
            "--warmup-ratio",
            0.05,
            "--gradient-clip",
            1,
            "--no-detector-amp",
            "--rotation-min-degrees",
            -22.5,
            "--rotation-max-degrees",
            22.5,
            "--overlap-min-overlays",
            0,
            "--overlap-max-overlays",
            1,
            "--overlap-p",
            0.25,
            "--video-fraction",
            0.5,
            "--seed",
            args.seed,
        ]
        smoke_base = args.output_dir / "base/smoke"
        run(
            "base-smoke",
            "flowsis.cli.train.train_base",
            *common,
            "--output-dir",
            smoke_base,
            "--train-stages",
            "head:1",
            "--max-steps",
            8,
            "--max-validation-images",
            8,
            "--num-workers",
            0,
            "--batch-size",
            1,
            "--no-amp",
        )
        temporal_common = [
            "--dataset-path",
            args.dataset_path,
            "--device",
            args.device,
            "--flow-model",
            "raft_small",
            "--flow-checkpoint",
            args.flow_checkpoint,
            "--batch-size",
            1,
            "--image-size",
            args.image_size,
            "--seed",
            args.seed,
        ]
        val_videos = audit["splits"]["validation"]["temporal_videos"].get(
            str(length), 0
        )
        validation_batches = max(val_videos, min(40, args.temporal_batches))
        temporal_batches = max(args.temporal_batches, validation_batches)
        if temporal_enabled:
            for smoke_length in sorted({min(4, length), length}):
                run(
                    f"temporal-smoke{smoke_length}",
                    "flowsis.cli.train.train_temporal",
                    *temporal_common,
                    "--base-model",
                    smoke_base / "best",
                    "--output-dir",
                    args.output_dir / f"temporal/smoke{smoke_length}",
                    "--epochs",
                    1,
                    "--batches-per-epoch",
                    max(4, val_videos),
                    "--validation-batches",
                    max(4, val_videos),
                    "--min-clip-length",
                    smoke_length,
                    "--max-clip-length",
                    smoke_length,
                    "--strides",
                    1,
                    "--stride-weights",
                    1,
                    "--num-workers",
                    0,
                )
        base_run = args.output_dir / "base/overlap25"
        run(
            "base-main",
            "flowsis.cli.train.train_base",
            *common,
            "--output-dir",
            base_run,
            "--train-stages",
            f"head:{args.base_epochs}",
            "--amp",
        )
        base = base_run / "best"
        if not args.dry_run and not (base / "base.pt").is_file():
            raise RuntimeError(
                "No guardrail-qualified base checkpoint; temporal training stopped."
            )
        state["base"] = state["standalone_base"] = str(base)

        def evaluate_base(
            stage: str,
            model: Path,
            split: str,
            output: Path,
            dataset: Path | None = None,
        ) -> None:
            run(
                stage,
                "flowsis.cli.eval.evaluate_base",
                "--model-path",
                model,
                "--dataset-path",
                dataset or args.dataset_path,
                "--split",
                split,
                "--device",
                args.device,
                "--output-path",
                output,
                "--pr-output-dir",
                output.with_suffix(""),
                "--no-detector-amp",
            )

        evaluate_base(
            "base-validation", base, "validation", base_run / "validation.json"
        )
        if temporal_enabled:
            run(
                "base-temporal-parity",
                "experiments.overnight_support",
                "--dataset-path",
                args.dataset_path,
                "--base-model",
                base,
                "--output-dir",
                args.output_dir / "temporal/preprocessing-check",
                "--device",
                args.device,
                "--image-size",
                args.image_size,
            )
            temporal_run = args.output_dir / "temporal/main"
            run(
                "temporal-main",
                "flowsis.cli.train.train_temporal",
                *temporal_common,
                "--base-model",
                base,
                "--output-dir",
                temporal_run,
                "--epochs",
                args.temporal_epochs,
                "--batches-per-epoch",
                temporal_batches,
                "--validation-batches",
                validation_batches,
                "--min-clip-length",
                min(4, length),
                "--max-clip-length",
                length,
                "--strides",
                1,
                2,
                3,
                "--stride-weights",
                0.7,
                0.2,
                0.1,
                "--num-workers",
                args.num_workers,
                "--learning-rate",
                "1e-4",
                "--gradient-clip",
                1,
            )
            temporal = temporal_run / "temporal-best.pt"
            if not args.dry_run:
                check_temporal_checkpoint(temporal)
            state["temporal"] = str(temporal)
            identity = temporal_run / "identity-refinement.pt"
            if not args.dry_run:
                identity_checkpoint(temporal, identity)
            for name, clip_length, checkpoint in (
                ("clips16", 16, temporal),
                ("intervals", 256, temporal),
                ("identity", 256, identity),
            ):
                run(
                    f"temporal-eval-{name}",
                    "flowsis.cli.eval.evaluate_temporal",
                    "--dataset-path",
                    args.dataset_path,
                    "--base-model",
                    base,
                    "--temporal-model",
                    checkpoint,
                    "--flow-checkpoint",
                    args.flow_checkpoint,
                    "--device",
                    args.device,
                    "--image-size",
                    args.image_size,
                    "--clip-length",
                    clip_length,
                    "--clips-per-video",
                    6,
                    "--mode",
                    "both",
                    "--output-dir",
                    args.output_dir / f"temporal/eval-{name}",
                )
        else:
            state["outcomes"].append(
                "Temporal skipped: explicitly disabled or no usable train/validation clips."
            )
        views = (
            {
                name: args.output_dir / "validation-views" / name
                for name in ("stills", "videos")
                if (
                    audit["splits"]["validation"].get("still_rows", 0)
                    if name == "stills"
                    else audit["splits"]["validation"]["video_ids"]
                )
            }
            if args.dry_run
            else validation_views(
                args.dataset_path, args.output_dir / "validation-views"
            )
        )
        for source, view in views.items():
            evaluate_base(
                f"base-validation-{source}",
                base,
                "validation",
                base_run / f"validation-{source}.json",
                view,
            )
        if (
            not args.skip_control
            and time.monotonic() - started < args.optional_after_hours * 3600
        ):
            try:
                control = args.output_dir / "base/overlap-off"
                run(
                    "base-overlap-control",
                    "flowsis.cli.train.train_base",
                    *common,
                    "--output-dir",
                    control,
                    "--train-stages",
                    f"head:{args.base_epochs}",
                    "--no-use-overlap-augment",
                    "--amp",
                )
                evaluate_base(
                    "control-validation",
                    control / "best",
                    "validation",
                    control / "validation.json",
                )
                for source, view in views.items():
                    evaluate_base(
                        f"control-validation-{source}",
                        control / "best",
                        "validation",
                        control / f"validation-{source}.json",
                        view,
                    )
                if not args.dry_run:
                    control_metrics = json.loads(
                        (control / "validation.json").read_text()
                    )
                    main_metrics = json.loads(
                        (base_run / "validation.json").read_text()
                    )
                    if control_metrics["matched_dice"] > main_metrics["matched_dice"]:
                        state["standalone_base"] = str(control / "best")
            except RuntimeError as error:
                state["outcomes"].append(
                    f"Optional overlap control failed; main models retained: {error}"
                )
        else:
            state["outcomes"].append(
                "Optional overlap control skipped by time budget or --skip-control."
            )
        if args.evaluate_test:
            for split in ("test", "test_known", "test_unseen"):
                if (
                    split not in audit["splits"]
                    or not audit["splits"][split]["base_rows"]
                ):
                    continue
                evaluate_base(
                    f"base-{split}",
                    Path(state["standalone_base"]),
                    split,
                    args.output_dir / f"evaluation/{split}/base.json",
                )
                run(
                    f"detector-{split}",
                    "flowsis.cli.eval.evaluate_detector",
                    "--detector",
                    args.detector,
                    "--model",
                    detector,
                    "--dataset-path",
                    args.dataset_path,
                    "--split",
                    split,
                    "--device",
                    args.device,
                    "--output-dir",
                    args.output_dir / f"evaluation/{split}/detector",
                )
                if temporal_enabled and any(
                    audit["splits"][split]["temporal_videos"].values()
                ):
                    run(
                        f"temporal-{split}",
                        "flowsis.cli.eval.evaluate_temporal",
                        "--dataset-path",
                        args.dataset_path,
                        "--split",
                        split,
                        "--base-model",
                        base,
                        "--temporal-model",
                        state["temporal"],
                        "--flow-checkpoint",
                        args.flow_checkpoint,
                        "--device",
                        args.device,
                        "--image-size",
                        args.image_size,
                        "--mode",
                        "both",
                        "--clip-length",
                        256,
                        "--output-dir",
                        args.output_dir / f"evaluation/{split}/temporal",
                    )
        state["status"] = "complete"
    except Exception as error:
        state["status"] = f"failed: {error}"
        raise
    finally:
        if not args.dry_run:
            model_manifest(args.output_dir, state, args.flow_checkpoint)
            learning_curves(args.output_dir, args.detector_run)
            (args.output_dir / "PROMPT_EMBEDDINGS.json").write_text(
                json.dumps(
                    {
                        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(args.embeddings_dir.glob("*.pt"))
                    },
                    indent=2,
                )
                + "\n"
            )
            write_report(args.output_dir, state)
    print(f"Report: {args.output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
