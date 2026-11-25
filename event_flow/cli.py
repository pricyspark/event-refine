"""Event-Flow data preparation, training, and streaming evaluation."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from flowsis.pretrained.detector import BaseDetector, load_detector

from .checkpoint import load_checkpoint, save_checkpoint
from .data import ClipDataset, FrameSample, import_fred, load_manifest, read_frame
from .download import download_fred
from .events import accumulate_events, save_event_map
from .metrics import DetectionRecord, detection_metrics
from .model import EventFlow
from .profiling import profile_region, training_profiler
from .training import batch_clip_loss


def _clip_loss(
    model: EventFlow,
    clip: list[FrameSample],
    device: str,
    base_weight: float,
    *,
    profile: bool = False,
) -> torch.Tensor:
    return batch_clip_loss(
        model, [clip], device, base_weight=base_weight, profile=profile
    )


def _training_event(output: Path, **record: object) -> None:
    line = json.dumps(record)
    print(line, flush=True)
    with (output / "progress.jsonl").open("a") as handle:
        handle.write(line + "\n")


def _phase_progress(
    output: Path,
    *,
    phase: str,
    epoch: int,
    steps: int,
    processed: int,
    total: int,
    started: float,
    loss_sum: float,
    load_seconds: float,
    compute_seconds: float,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = processed / elapsed
    _training_event(
        output,
        event="progress",
        phase=phase,
        epoch=epoch,
        steps=steps,
        completed_clips=processed,
        total_clips=total,
        mean_loss=loss_sum / processed,
        elapsed_seconds=round(elapsed, 2),
        eta_seconds=round((total - processed) / rate, 2),
        clips_per_second=round(rate, 4),
        load_seconds=round(load_seconds, 2),
        compute_seconds=round(compute_seconds, 2),
    )


def train(args: argparse.Namespace) -> None:
    args.init_from = getattr(args, "init_from", None)
    if args.init_from is not None:
        if args.detector_model is not None:
            raise ValueError("Use either --init-from or --detector-model.")
        if Path(args.init_from).resolve().is_relative_to(args.output.resolve()):
            raise ValueError(
                "Choose a new output directory outside the source checkpoint."
            )
    args.batch_size = getattr(args, "batch_size", 1)
    args.profile = getattr(args, "profile", False)
    args.profile_warmup = getattr(args, "profile_warmup", 10)
    args.profile_steps = getattr(args, "profile_steps", 5)
    if args.profile_warmup < 1 or args.profile_steps < 1:
        raise ValueError("Profiler warmup and active steps must be positive.")
    args.log_every = getattr(args, "log_every", 10)
    args.max_validation_clips = getattr(args, "max_validation_clips", None)
    if (
        args.batch_size < 1
        or args.epochs < 1
        or args.max_steps is not None
        and args.max_steps < 1
        or min(args.learning_rate, args.temporal_learning_rate) <= 0
        or args.base_loss_weight < 0
        or args.log_every < 1
        or args.max_validation_clips is not None
        and args.max_validation_clips < 1
    ):
        raise ValueError(
            "Invalid batch size, epochs, step limits, learning rates, loss weight, or logging interval."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    _training_event(
        args.output,
        event="startup",
        device=args.device,
        architecture=args.architecture,
        image_size=args.image_size,
        clip_length=args.clip_length,
        batch_size=args.batch_size,
    )
    dataset = ClipDataset(
        args.manifest,
        split=args.split,
        clip_length=args.clip_length,
        max_gap=args.max_gap,
    )
    validation = (
        ClipDataset(
            args.manifest,
            split=args.validation_split,
            clip_length=args.clip_length,
            max_gap=args.max_gap,
        )
        if args.validation_split
        else None
    )
    if args.validation_split == args.split:
        raise ValueError("Training and validation splits must differ.")
    validation_total = (
        min(len(validation), args.max_validation_clips or len(validation))
        if validation is not None
        else 0
    )
    _training_event(
        args.output,
        event="data_ready",
        training_clips=len(dataset),
        validation_clips=validation_total,
        classes=dataset.classes,
    )
    if args.profile:
        batches = (len(dataset) + args.batch_size - 1) // args.batch_size
        available = min(batches, args.max_steps or batches)
        required = args.profile_warmup + args.profile_steps
        if available < required:
            raise ValueError(
                f"Profiling needs at least {required} training steps in the first epoch; increase --max-steps or reduce the profile window."
            )
    if args.init_from is not None:
        model, classes = load_checkpoint(args.init_from, device=args.device)
        if classes != dataset.classes:
            raise ValueError("Manifest/checkpoint class names and order must match.")
        if (
            args.architecture is not None
            and args.architecture != model.detector.architecture
        ):
            raise ValueError("Architecture does not match the source checkpoint.")
        if args.image_size != model.detector.image_size:
            raise ValueError("Image size does not match the source checkpoint.")
        args.architecture = model.detector.architecture
        args.hidden_dim = model.temporal.hidden_dim
        _training_event(
            args.output,
            event="checkpoint_loaded",
            checkpoint=str(args.init_from),
            optimizer_restored=False,
            hidden_dim=args.hidden_dim,
        )
    else:
        id2label = dict(enumerate(dataset.classes))
        detector = load_detector(
            args.detector_model,
            architecture=args.architecture,
            num_labels=len(id2label),
            id2label=id2label,
            label2id={name: index for index, name in id2label.items()},
            image_size=args.image_size,
            device=args.device,
        )
        if not isinstance(detector, BaseDetector):
            raise TypeError("Expected a native detector adapter.")
        model = EventFlow(detector, hidden_dim=args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.detector.parameters(), "lr": args.learning_rate},
            {"params": model.temporal.parameters(), "lr": args.temporal_learning_rate},
        ],
        weight_decay=1e-4,
    )
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "run"
    }
    (args.output / "run_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )
    _training_event(args.output, event="model_ready", device=str(model.detector.device))
    best = float("inf")
    steps = 0
    with training_profiler(
        args.output,
        enabled=args.profile,
        device=model.detector.device,
        warmup_steps=args.profile_warmup,
        active_steps=args.profile_steps,
    ) as profiler:
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(dataset)).tolist()
            training_total = (
                min(len(dataset), (args.max_steps - steps) * args.batch_size)
                if args.max_steps
                else len(dataset)
            )
            total = 0.0
            processed = 0
            started = time.perf_counter()
            load_seconds = compute_seconds = 0.0
            _training_event(
                args.output,
                event="phase_start",
                phase="train",
                epoch=epoch + 1,
                total_clips=training_total,
            )
            for batch_index, start in enumerate(
                range(0, training_total, args.batch_size)
            ):
                before_load = time.perf_counter()
                with profile_region("data_load", args.profile):
                    clips = [
                        dataset[index]
                        for index in order[start : start + args.batch_size]
                    ]
                load_seconds += time.perf_counter() - before_load
                if processed == 0:
                    _training_event(
                        args.output,
                        event="first_clip_loaded",
                        phase="train",
                        epoch=epoch + 1,
                        load_seconds=round(load_seconds, 2),
                    )
                before_compute = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = batch_clip_loss(
                    model,
                    clips,
                    args.device,
                    base_weight=args.base_loss_weight,
                    profile=args.profile,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss.")
                with profile_region("backward", args.profile):
                    loss.backward()
                with profile_region("gradient_clip", args.profile):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                with profile_region("optimizer_step", args.profile):
                    optimizer.step()
                total += float(loss.detach()) * len(clips)
                compute_seconds += time.perf_counter() - before_compute
                steps += 1
                processed += len(clips)
                if profiler is not None:
                    profiler.step()
                if (
                    batch_index == 0
                    or (batch_index + 1) % args.log_every == 0
                    or processed == training_total
                ):
                    _phase_progress(
                        args.output,
                        phase="train",
                        epoch=epoch + 1,
                        steps=steps,
                        processed=processed,
                        total=training_total,
                        started=started,
                        loss_sum=total,
                        load_seconds=load_seconds,
                        compute_seconds=compute_seconds,
                    )
                if args.max_steps and steps >= args.max_steps:
                    break
            record = {
                "epoch": epoch + 1,
                "steps": steps,
                "training_loss": total / processed,
                "training_clips": processed,
            }
            if validation is not None:
                model.eval()
                started = time.perf_counter()
                validation_sum = 0.0
                load_seconds = compute_seconds = 0.0
                _training_event(
                    args.output,
                    event="phase_start",
                    phase="validation",
                    epoch=epoch + 1,
                    total_clips=validation_total,
                )
                with torch.inference_mode():
                    for batch_index, start in enumerate(
                        range(0, validation_total, args.batch_size)
                    ):
                        before_load = time.perf_counter()
                        end = min(start + args.batch_size, validation_total)
                        clips = [validation[index] for index in range(start, end)]
                        load_seconds += time.perf_counter() - before_load
                        before_compute = time.perf_counter()
                        value = float(
                            batch_clip_loss(
                                model,
                                clips,
                                args.device,
                                base_weight=args.base_loss_weight,
                            )
                        )
                        compute_seconds += time.perf_counter() - before_compute
                        if not np.isfinite(value):
                            raise FloatingPointError("Non-finite validation loss.")
                        validation_sum += value * len(clips)
                        if (
                            batch_index == 0
                            or (batch_index + 1) % args.log_every == 0
                            or end == validation_total
                        ):
                            _phase_progress(
                                args.output,
                                phase="validation",
                                epoch=epoch + 1,
                                steps=steps,
                                processed=end,
                                total=validation_total,
                                started=started,
                                loss_sum=validation_sum,
                                load_seconds=load_seconds,
                                compute_seconds=compute_seconds,
                            )
                value = validation_sum / validation_total
                record["validation_loss"] = value
                if value < best:
                    best = value
                    _training_event(
                        args.output,
                        event="checkpoint_start",
                        checkpoint="best",
                        epoch=epoch + 1,
                    )
                    save_checkpoint(
                        model, args.output / "best", classes=dataset.classes
                    )
                    _training_event(
                        args.output,
                        event="checkpoint_saved",
                        checkpoint="best",
                        epoch=epoch + 1,
                    )
            _training_event(
                args.output,
                event="checkpoint_start",
                checkpoint="last",
                epoch=epoch + 1,
            )
            save_checkpoint(model, args.output / "last", classes=dataset.classes)
            _training_event(
                args.output,
                event="checkpoint_saved",
                checkpoint="last",
                epoch=epoch + 1,
            )
            with (args.output / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if args.max_steps and steps >= args.max_steps:
                break


def predict(args: argparse.Namespace) -> None:
    if args.max_gap <= 0 or args.max_detections < 1:
        raise ValueError("max_gap and max_detections must be positive.")
    model, classes = load_checkpoint(args.model, device=args.device)
    manifest_classes, sequences = load_manifest(args.manifest)
    if manifest_classes != classes:
        raise ValueError("Manifest/checkpoint class names and order must match.")
    selected = [sequence for sequence in sequences if sequence.split == args.split]
    if not selected:
        raise ValueError(f"No sequences in split {args.split!r}.")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with (args.output / "predictions.jsonl").open(
        "w"
    ) as handle, torch.inference_mode():
        for sequence in selected:
            state = None
            previous_timestamp = None
            previous_size = None
            for index, frame in enumerate(sequence.frames):
                sample = read_frame(frame)
                size = tuple(sample.rgb.shape[-2:])
                if previous_timestamp is not None and (
                    frame.timestamp - previous_timestamp > args.max_gap
                    or size != previous_size
                ):
                    state = None
                pixels, mask = model.prepare(
                    sample.rgb[None].to(args.device),
                    sample.events[None].to(args.device),
                )
                output = model(pixels, pixel_mask=mask, state=state)
                state = output.state.detach()
                result = model.detections(
                    output,
                    image_size=model.detector.image_size,
                    original_size=size,
                    threshold=args.threshold,
                )[0]
                result = {
                    key: value[: args.max_detections].cpu()
                    for key, value in result.items()
                }
                handle.write(
                    json.dumps(
                        {
                            "sequence": sequence.name,
                            "frame": index,
                            "timestamp": frame.timestamp,
                            "annotated": frame.annotated,
                            **{key: value.tolist() for key, value in result.items()},
                        }
                    )
                    + "\n"
                )
                if frame.annotated and args.command == "evaluate":
                    records.append(
                        DetectionRecord(
                            **result,
                            target_boxes=sample.boxes,
                            target_labels=sample.labels,
                        )
                    )
                previous_timestamp, previous_size = frame.timestamp, size
    if args.command == "evaluate":
        if not records:
            raise ValueError("Evaluation split has no annotated frames.")
        metrics = detection_metrics(records, classes)
        metrics.update(
            {"score_threshold": args.threshold, "max_detections": args.max_detections}
        )
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)


def prepare_events(args: argparse.Namespace) -> None:
    with np.load(args.events, allow_pickle=False) as events:
        x, y, timestamps = events["x"], events["y"], events["t"]
    ends = np.load(args.timestamps, allow_pickle=False)
    if (
        timestamps.ndim != 1
        or not np.isfinite(timestamps).all()
        or bool((np.diff(timestamps.astype(np.float64)) < 0).any())
    ):
        raise ValueError("Raw event timestamps must be finite and sorted.")
    if x.shape != timestamps.shape or y.shape != timestamps.shape:
        raise ValueError("Raw event arrays must align.")
    if (
        ends.ndim != 1
        or not np.isfinite(ends).all()
        or bool((np.diff(ends.astype(np.float64)) <= 0).any())
        or args.window <= 0
    ):
        raise ValueError(
            "Frame timestamps must be finite and increasing; window must be positive."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for index, end in enumerate(ends):
        start = float(end) - args.window
        left, right = np.searchsorted(timestamps, [start, end], side="right")
        occupancy = accumulate_events(
            x[left:right],
            y[left:right],
            timestamps[left:right],
            start=start,
            end=float(end),
            height=args.height,
            width=args.width,
        )
        suffix = ".npz" if args.compress_events else ".npy"
        save_event_map(args.output / f"{index:06d}{suffix}", occupancy)
    print(f"Wrote {len(ends)} binary occupancy maps to {args.output}")


def download_fred_command(args: argparse.Namespace) -> None:
    output = args.output if args.output is not None else Path("data") / args.config.stem
    result = download_fred(
        args.config,
        output,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        compress_events=args.compress_events,
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Event-Flow RGB/event temporal detection"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    downloader = commands.add_parser(
        "download-fred", help="Download and import a pinned FRED subset"
    )
    downloader.add_argument("--config", type=Path, required=True)
    downloader.add_argument("--output", type=Path)
    downloader.add_argument("--cache-dir", type=Path, default=Path("data/fred-cache"))
    downloader.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected files and total bytes without downloading",
    )
    downloader.add_argument(
        "--compress-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store events as bit-packed compressed NPZ (default); --no-compress-events saves NPY",
    )
    downloader.set_defaults(run=download_fred_command)
    importer = commands.add_parser(
        "import-fred", help="Import extracted official FRED folders"
    )
    importer.add_argument("--root", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.add_argument(
        "--event-background", type=int, nargs=3, required=True, metavar=("R", "G", "B")
    )
    importer.add_argument("--frame-period", type=float, default=0.033333)
    importer.set_defaults(
        run=lambda args: import_fred(
            args.root,
            args.output,
            event_background=tuple(args.event_background),
            frame_period=args.frame_period,
        )
    )
    events = commands.add_parser(
        "prepare-events",
        help="Accumulate registered raw NPZ x/y/t into causal binary maps",
    )
    events.add_argument("--events", type=Path, required=True)
    events.add_argument("--timestamps", type=Path, required=True)
    events.add_argument("--window", type=float, required=True)
    events.add_argument("--height", type=int, required=True)
    events.add_argument("--width", type=int, required=True)
    events.add_argument("--output", type=Path, required=True)
    events.add_argument(
        "--compress-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store bit-packed compressed NPZ (default); --no-compress-events saves NPY",
    )
    events.set_defaults(run=prepare_events)
    training = commands.add_parser(
        "train", help="Train four-channel detector and temporal box/class head"
    )
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--detector-model")
    training.add_argument(
        "--init-from",
        type=Path,
        help="Warm start the complete Event-Flow checkpoint with a fresh optimizer; epochs are additional",
    )
    training.add_argument("--architecture", choices=("dfine", "rtdetrv2"))
    training.add_argument("--split", default="train")
    training.add_argument("--validation-split")
    training.add_argument("--image-size", type=int, default=640)
    training.add_argument("--hidden-dim", type=int, default=128)
    training.add_argument("--clip-length", type=int, default=4)
    training.add_argument("--max-gap", type=float, default=0.05)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Independent clips per optimizer update (default: 1)",
    )
    training.add_argument(
        "--max-steps", type=int, help="Maximum optimizer updates across all epochs"
    )
    training.add_argument(
        "--profile",
        action="store_true",
        help="Capture one short CPU/CUDA trace under OUTPUT/profile",
    )
    training.add_argument(
        "--profile-warmup",
        type=int,
        default=10,
        help="Unrecorded warmup steps before the trace (default: 10)",
    )
    training.add_argument(
        "--profile-steps",
        type=int,
        default=5,
        help="Training steps recorded in the trace (default: 5)",
    )
    training.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print and save progress after the first batch and every N batches",
    )
    training.add_argument(
        "--max-validation-clips",
        type=int,
        help="Limit validation work for diagnostic runs; default validates all clips",
    )
    training.add_argument("--learning-rate", type=float, default=1e-5)
    training.add_argument("--temporal-learning-rate", type=float, default=1e-4)
    training.add_argument("--base-loss-weight", type=float, default=0.5)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    training.set_defaults(run=train)
    for name in ("predict", "evaluate"):
        prediction = commands.add_parser(name)
        prediction.add_argument("--manifest", type=Path, required=True)
        prediction.add_argument("--model", type=Path, required=True)
        prediction.add_argument("--split", default="test")
        prediction.add_argument("--output", type=Path, required=True)
        prediction.add_argument(
            "--threshold", type=float, default=0.001 if name == "evaluate" else 0.25
        )
        prediction.add_argument("--max-detections", type=int, default=100)
        prediction.add_argument("--max-gap", type=float, default=0.05)
        prediction.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
        prediction.set_defaults(run=predict)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
