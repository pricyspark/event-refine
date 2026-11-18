"""Build research figures and portable inference artifacts with Codex assistance."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tarfile
import textwrap
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from experiments.benchmark_generations import CHECKPOINTS

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/overnight/coarse-v2"
NAVY, BLUE, TEAL, ORANGE, GREY = "#172b4d", "#3375ad", "#008b83", "#c67b29", "#697588"
MODEL_NAMES = {
    "prior_head10": "Oct 29 · frozen base",
    "prior_gentle": "Oct 29 · gentle unfreezing",
    "current_overlap25": "Nov 13 · overlap 25%",
    "current_overlap_off": "Nov 13 · overlap off",
}
COLORS = {
    "prior_head10": GREY,
    "prior_gentle": ORANGE,
    "current_overlap25": BLUE,
    "current_overlap_off": TEAL,
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def canvas(title: str, subtitle: str):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.055, 0.94, title, fontsize=27, weight="bold", color=NAVY)
    fig.text(0.055, 0.885, subtitle, fontsize=13, color=GREY)
    return fig


def footer(fig, text: str) -> None:
    fig.text(0.055, 0.065, text, fontsize=10.5, color=GREY)
    fig.text(
        0.945,
        0.025,
        "Codex-assisted analysis · 13 Nov 2026",
        ha="right",
        fontsize=8,
        color=GREY,
    )


def style_axis(axis, label: str = "Score (%)") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e5e9ef", linewidth=0.8)
    axis.set_ylabel(label)
    axis.tick_params(labelsize=12)


def save(fig, output: Path, name: str, pdf: PdfPages) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200, facecolor="white")
    pdf.savefig(fig)
    plt.close(fig)


def overview(output: Path, pdf: PdfPages) -> None:
    detector = read(RUN / "detector/0-ap0.5/metrics.json")
    base = read(RUN / "base/overlap-off/validation.json")
    fig = canvas(
        "From detection to instance masks to temporal refinement",
        "Current development benchmark · 11 output labels, 7 represented in validation",
    )
    cards = [
        (
            0.06,
            "Detector",
            f"{100*detector['micro_ap']:.1f}%",
            "Pooled box AP50",
            "Epoch 20 · D-FINE Medium",
            BLUE,
        ),
        (
            0.37,
            "Base segmentation",
            f"{100*base['matched_dice']:.1f}%",
            "Matched Dice",
            "Frozen detector + trained mask head",
            TEAL,
        ),
        (
            0.68,
            "Temporal refinement",
            "Pending",
            "Corrected training required",
            "Input-range bug identified and fixed",
            ORANGE,
        ),
    ]
    for x, title, score, metric, note, color in cards:
        fig.add_artist(
            FancyBboxPatch(
                (x, 0.29),
                0.26,
                0.49,
                boxstyle="round,pad=.015",
                transform=fig.transFigure,
                facecolor="#f6f8fb",
                edgecolor=color,
                linewidth=1.5,
            )
        )
        fig.text(x + 0.02, 0.71, title, fontsize=19, weight="bold", color=NAVY)
        fig.text(x + 0.02, 0.54, score, fontsize=43, weight="bold", color=color)
        fig.text(x + 0.02, 0.45, metric, fontsize=15, color=NAVY)
        fig.text(x + 0.02, 0.34, textwrap.fill(note, 30), fontsize=12, color=GREY)
    for x in (0.33, 0.64):
        fig.add_artist(
            FancyArrowPatch(
                (x, 0.535),
                (x + 0.025, 0.535),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=20,
                color=GREY,
            )
        )
    fig.text(
        0.07,
        0.18,
        "The temporal pipeline contains the full base model, pretrained optical flow, and a recurrent refiner.",
        fontsize=15,
        color=NAVY,
    )
    footer(
        fig,
        "370 validation frames · section split shares recordings with training · 0 real multi-tool / empty validation frames",
    )
    save(fig, output, "01_model_effectiveness", pdf)


def same_image_benchmark(
    output: Path, benchmark: dict, cohort: dict, pdf: PdfPages
) -> None:
    old = benchmark["prior_head10"]["metrics"]["matched_dice"]
    new = benchmark["current_overlap_off"]["metrics"]["matched_dice"]
    fig = canvas(
        "Previous and current models on the same held-out images",
        f"Current base improves matched Dice by {100*(new-old):.1f} percentage points versus the stronger prior frozen checkpoint",
    )
    axis = fig.add_axes([0.09, 0.23, 0.85, 0.57])
    keys = ("matched_dice", "oracle_dice", "mask_ap50")
    x = np.arange(3)
    for offset, (name, result) in enumerate(benchmark.items()):
        values = [100 * result["metrics"][key] for key in keys]
        bars = axis.bar(
            x + (offset - 1.5) * 0.19,
            values,
            0.18,
            label=MODEL_NAMES[name],
            color=COLORS[name],
        )
        axis.bar_label(
            bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=12
        )
    axis.set_xticks(
        x, ["Class-aware matched Dice", "GT-query-matched oracle Dice", "Mask AP50"]
    )
    axis.set_ylim(0, 108)
    style_axis(axis)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    footer(
        fig,
        f"NEW FP32 benchmark: {cohort['images']} hash-verified shared validation stills · {cohort['capture_groups']} capture groups · 4 classes\nIdentical current annotations, 640 px, confidence/mask thresholds 0.5; class IDs remapped by name. Development data, one seed.",
    )
    save(fig, output, "02_same_image_generation_benchmark", pdf)


def base_sources(output: Path, pdf: PdfPages) -> None:
    fig = canvas(
        "Current base: 89.6% matched Dice on the full validation set",
        "The no-overlap control has a small overall advantage; overlap effectiveness on real multi-tool scenes is unmeasured",
    )
    axis = fig.add_axes([0.09, 0.23, 0.85, 0.56])
    x = np.arange(3)
    for offset, (name, color, label) in enumerate(
        (
            ("overlap25", BLUE, "25% synthetic overlap"),
            ("overlap-off", TEAL, "No synthetic base overlap"),
        )
    ):
        values = [
            100 * read(RUN / f"base/{name}/{filename}.json")["matched_dice"]
            for filename in ("validation", "validation-stills", "validation-videos")
        ]
        bars = axis.bar(
            x + (offset - 0.5) * 0.29, values, 0.27, color=color, label=label
        )
        axis.bar_label(
            bars, labels=[f"{v:.2f}" for v in values], padding=5, fontsize=15
        )
    axis.set_xticks(
        x,
        [
            "All validation\n370 frames",
            "Still images\n119 frames",
            "Video frames\n251 frames",
        ],
    )
    axis.set_ylim(0, 105)
    style_axis(axis, "Matched Dice (%)")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    footer(
        fig,
        "Both runs: same detector, head-only training, 10-epoch budget, seed 42; validation selects epochs 10 and 5 respectively.\nValidation has no real multi-tool images. Both bases inherit a detector trained with synthetic overlap.",
    )
    save(fig, output, "03_current_base_by_source", pdf)


def class_profile(output: Path, pdf: PdfPages) -> None:
    metrics = read(RUN / "base/overlap-off/validation.json")["per_class"]
    support = read(RUN / "dataset_build_report.json")["splits"]["validation"]["classes"]
    ordered = sorted(metrics, key=lambda label: metrics[label]["matched_dice"])
    fig = canvas(
        "Most remaining class-aware errors concentrate in a few classes",
        "Current no-overlap base · bonepunch and gouge are the clearest priorities for error inspection",
    )
    axis = fig.add_axes([0.09, 0.24, 0.85, 0.55])
    values = [100 * metrics[label]["matched_dice"] for label in ordered]
    bars = axis.bar(
        np.arange(len(ordered)),
        values,
        color=[
            ORANGE if label in ("bonepunch", "gouge") else TEAL for label in ordered
        ],
        width=0.65,
    )
    axis.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=5, fontsize=14)
    axis.set_xticks(
        np.arange(len(ordered)), [f"{label}\nn={support[label]}" for label in ordered]
    )
    axis.set_ylim(0, 106)
    style_axis(axis, "Matched Dice (%)")
    fig.text(
        0.09,
        0.14,
        "No validation positives: needleholder, pliers, rongeur, tube.",
        fontsize=13,
        color=NAVY,
    )
    footer(
        fig,
        "n counts annotated objects; repeated views/frames are correlated. These are development results, not new-tool generalization.",
    )
    save(fig, output, "04_current_base_by_class", pdf)


def learning_curves(output: Path, pdf: PdfPages) -> None:
    fig = canvas(
        "Checkpoint selection depends on the quality metric",
        "Dice and mask AP can select different epochs; loss alone is not enough",
    )
    axes = [
        fig.add_axes([0.08, 0.23, 0.39, 0.54]),
        fig.add_axes([0.56, 0.23, 0.39, 0.54]),
    ]
    for name, color, label in (
        ("overlap25", BLUE, "Overlap 25%"),
        ("overlap-off", TEAL, "Overlap off"),
    ):
        rows = [
            json.loads(line)
            for line in (RUN / f"base/{name}/training_log.jsonl")
            .read_text()
            .splitlines()
        ]
        rows = [r for r in rows if r["event"] == "epoch"]
        x = [r["epoch"] for r in rows]
        for axis, key in zip(
            axes, ("validation_matched_dice", "validation_mask_ap50"), strict=True
        ):
            y = [100 * r[key] for r in rows]
            axis.plot(x, y, "o-", color=color, label=label, linewidth=2)
            best = int(np.argmax(y))
            axis.scatter(
                x[best],
                y[best],
                s=140,
                facecolors="none",
                edgecolors=color,
                linewidth=2,
            )
    for axis, title in zip(axes, ("Matched Dice", "Mask AP50"), strict=True):
        axis.set(xlabel="Training epoch", title=title, xlim=(0.6, 10.4), ylim=(80, 95))
        axis.set_xticks(range(1, 11))
        style_axis(axis)
        axis.legend(frameon=False, fontsize=11)
    footer(
        fig,
        "Axes zoomed to 80–95%; circles mark the best logged epoch for each metric.\nAll detector guardrails passed; isolated AMP skipped steps occurred. One seed per recipe.",
    )
    save(fig, output, "05_learning_curves", pdf)


def temporal_status(output: Path, pdf: PdfPages) -> None:
    diagnostic = read(RUN / "preprocessing_diagnostic.json")["frames"][0]
    fig = canvas(
        "Temporal refinement: corrected training is still pending",
        "The original temporal training and oracle evaluation used incorrectly scaled detector inputs",
    )
    axis = fig.add_axes([0.09, 0.28, 0.4, 0.48])
    values = [0.00018165304268846503 * 100, diagnostic["fixed_oracle_dice"] * 100]
    bars = axis.bar([0, 1], values, color=[ORANGE, TEAL], width=0.55)
    axis.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=6, fontsize=18)
    axis.set_xticks([0, 1], ["Old input path", "Corrected input path"])
    axis.set_ylim(0, 108)
    style_axis(axis, "Base mask Dice on one diagnostic frame (%)")
    fig.text(0.56, 0.70, "Root cause", fontsize=19, weight="bold", color=NAVY)
    fig.text(
        0.56,
        0.59,
        "Clip floats [0,1] were rescaled again,\nand preprocessing modified flow inputs.",
        fontsize=16,
        color=NAVY,
    )
    fig.text(0.56, 0.45, "Fix verified", fontsize=19, weight="bold", color=TEAL)
    fig.text(
        0.56,
        0.33,
        "Separate uint8 detector inputs; preserve\nflow frames. Four sampled video frames\nnow match normal runtime predictions.",
        fontsize=15,
        color=NAVY,
    )
    fig.text(
        0.09,
        0.16,
        "This is a preprocessing diagnostic, NOT a learned temporal gain. Corrected temporal training remains pending.",
        fontsize=14,
        weight="bold",
        color=ORANGE,
    )
    footer(
        fig,
        "Same saved base and same diagnostic frame, CPU verification. Full temporal effectiveness is deliberately excluded.",
    )
    save(fig, output, "06_temporal_validation_status", pdf)


def qualitative_examples(output: Path, cohort: dict, pdf: PdfPages) -> None:
    import torch
    from types import SimpleNamespace
    from datasets import disable_progress_bars
    from flowsis.base_checkpoint import load_base
    from flowsis.cli.train.train_base import build_dataset
    from flowsis.data.loaders import load_local_dataset
    from flowsis.cli.eval.evaluate_base import _mask_metrics

    torch.set_num_threads(2)
    disable_progress_bars()
    data = load_local_dataset(
        ROOT / "../flowsis-data/exports/combined-sections-coarse"
    )["validation"]
    ids = {row["image_id"]: index for index, row in enumerate(data)}
    examples = []
    for label in ("gouge", "osteotome"):
        record = min(
            (row for row in cohort["frames"] if row["labels"] == [label]),
            key=lambda row: row["image_id"],
        )
        example = build_dataset(
            data.select([ids[record["image_id"]]]),
            SimpleNamespace(image_size=640),
            training=False,
        )[0]
        examples.append((label, record, example))
    fig = canvas(
        "Top-score predictions still show class confusions",
        "Two illustrative images selected deterministically by image ID: one per class, with identical crops for both models",
    )
    for model_index, key in enumerate(("prior_head10", "current_overlap_off")):
        model = load_base(
            ROOT / CHECKPOINTS[key], device="cpu", use_amp=False, detector_amp=False
        )
        for example_index, (label, record, example) in enumerate(examples):
            column = example_index * 2 + model_index
            axis = fig.add_axes([0.055 + column * 0.235, 0.20, 0.215, 0.56])
            image = np.asarray(example["image"])
            truth = example["objects"][0]["mask"]
            prediction = model.infer([example["image"]])[0].instances
            mask = np.zeros_like(truth, dtype=bool)
            predicted_label = "none"
            dice = 0.0
            if prediction.scores.numel():
                chosen = int(prediction.scores.argmax())
                predicted_label = model.label_names[int(prediction.labels[chosen])]
                mask = (prediction.mask_logits[chosen].sigmoid() >= 0.5).cpu().numpy()
                if predicted_label == label:
                    dice = _mask_metrics(
                        torch.from_numpy(mask), torch.from_numpy(truth)
                    )[0]
            axis.imshow(image)
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = [0.2, 0.46, 0.76, 0.3]
            axis.imshow(overlay)
            axis.contour(
                truth.astype(float), levels=[0.5], colors=[TEAL], linewidths=1.2
            )
            y, x = np.nonzero(truth)
            margin = 0.2 * max(x.max() - x.min(), y.max() - y.min())
            axis.set_xlim(
                max(0, x.min() - margin), min(image.shape[1], x.max() + margin)
            )
            axis.set_ylim(
                min(image.shape[0], y.max() + margin), max(0, y.min() - margin)
            )
            axis.axis("off")
            title = "Prior frozen base" if model_index == 0 else "Current base"
            axis.set_title(
                f"{title}\nPredicted: {predicted_label} · Dice {dice*100:.1f}%",
                fontsize=12,
            )
        del model
    for index, (label, record, _) in enumerate(examples):
        fig.text(
            0.275 + index * 0.47,
            0.825,
            f"Ground truth: {label}",
            ha="center",
            fontsize=17,
            weight="bold",
            color=NAVY,
        )
        fig.text(
            0.275 + index * 0.47,
            0.15,
            Path(record["image_path"]).name,
            ha="center",
            fontsize=11,
            color=GREY,
        )
    footer(
        fig,
        "Green: manual mask · blue: top-score predicted mask · displayed Dice requires the correct class\nIllustrative selections; the aggregate benchmark uses GT-matched queries and all accepted instances.",
    )
    save(fig, output, "07_prediction_examples", pdf)


def detector_comparison(output: Path, pdf: PdfPages) -> None:
    results = read(output / "benchmark/detector_summary.json")
    old_best = max(
        results[name]["metrics"]["0.5"]["micro_ap"]
        for name in ("prior_detector", "prior_detector_final")
    )
    current = results["current_detector"]["metrics"]["0.5"]["micro_ap"]
    fig = canvas(
        "Detector comparison on the same 31 held-out still images",
        f"Current pooled AP50: {current*100:.1f}% · stronger prior pooled AP50: {old_best*100:.1f}% · difference: {(current-old_best)*100:.1f} percentage points",
    )
    axis = fig.add_axes([0.09, 0.23, 0.85, 0.56])
    x = np.arange(3)
    for index, (name, color, label) in enumerate(
        (
            ("prior_detector", GREY, "Oct 29 · epoch 18"),
            ("prior_detector_final", ORANGE, "Oct 29 · epoch 20"),
            ("current_detector", TEAL, "Nov 13 · epoch 20"),
        )
    ):
        metric = results[name]["metrics"]
        represented = [
            row["ap"]
            for row in metric["0.5"]["classes"].values()
            if row["ground_truth"] > 0
        ]
        values = [
            metric["0.5"]["micro_ap"] * 100,
            np.mean(represented) * 100,
            metric["0.75"]["micro_ap"] * 100,
        ]
        bars = axis.bar(x + (index - 1) * 0.23, values, 0.22, color=color, label=label)
        axis.bar_label(
            bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=13
        )
    axis.set_xticks(
        x, ["Pooled AP50", "Mean AP50 · 4 represented classes", "Pooled AP75"]
    )
    axis.set_ylim(0, 108)
    style_axis(axis)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    footer(
        fig,
        "NEW FP32 same-image benchmark · 31 stills / 4 capture groups / 4 classes · one seed; validation used for model selection\nFixed-IoU AP, not COCO AP50:95. Detector and base AP use different postprocessing and should be reported separately.",
    )
    save(fig, output, "08_same_image_detector_benchmark", pdf)


def package_models(output: Path) -> None:
    models = {
        "current_base": (
            ROOT / CHECKPOINTS["current_overlap_off"],
            "Standalone current base; strongest pooled validation Dice",
        ),
        "current_temporal_base": (
            ROOT / CHECKPOINTS["current_overlap25"],
            "Base used for the temporal experiment and controlled corrected rerun",
        ),
        "prior_base_frozen": (
            ROOT / CHECKPOINTS["prior_head10"],
            "Oct 29 frozen base; stronger prior on the common-image cohort",
        ),
        "prior_base_gentle": (
            ROOT / CHECKPOINTS["prior_gentle"],
            "Oct 29 latest gently unfrozen base candidate",
        ),
        "current_detector": (
            ROOT / "outputs/detectors/coarse-sections-v1/checkpoint-005600",
            "Standalone epoch-20 detector; already embedded in current base folders",
        ),
        "prior_detector_final": (
            ROOT / "outputs/detectors/dfine-tool-class-20epochs/final",
            "Prior standalone epoch-20 detector; pooled-AP alternative in the same-image comparison",
        ),
    }
    manifest = {
        "prepared_with": "Codex assistance",
        "models": {},
        "temporal_status": "Original temporal weights invalid; corrected retraining pending.",
    }
    files = []
    for name, (directory, role) in models.items():
        required = [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "flowsis_detector.json",
        ]
        if "base" in name:
            required.append("base.pt")
        records = []
        for filename in required:
            path = directory / filename
            data = path.read_bytes()
            archive = f"models/{name}/{filename}"
            records.append(
                {
                    "file": filename,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            files.append((path, archive))
        config = read(directory / "config.json")
        manifest["models"][name] = {
            "original_directory": str(directory),
            "bundle_directory": f"models/{name}",
            "role": role,
            "labels": [
                config["id2label"][str(i)] for i in range(len(config["id2label"]))
            ],
            "files": records,
        }
    flow = Path.home() / ".cache/torch/hub/checkpoints/raft_small-things-b7d9f997.ckpt"
    manifest["flow"] = {
        "original_file": str(flow),
        "bundle_file": f"flow/{flow.name}",
        "sha256": hashlib.sha256(flow.read_bytes()).hexdigest(),
    }
    files.append((flow, f"flow/{flow.name}"))
    (output / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    versions = {
        name: version(name)
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "datasets",
            "numpy",
            "pillow",
            "ptlflow",
            "timm",
            "opencv-python",
            "matplotlib",
        )
    }
    (output / "runtime_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    guide = """# Model preservation and loading

Prepared with Codex assistance. Preserve the WHOLE selected base directory.

Each base requires these five files together:
`config.json`, `model.safetensors`, `preprocessor_config.json`,
`flowsis_detector.json`, `base.pt`.

`base.pt` contains the mask head, labels, thresholds and prompt embedding bank;
the detector weights live in `model.safetensors`. No separate SigLIP weights or
training images/masks are needed for inference. `training_state.pt` is needed
only for resuming training, not for loading these inference checkpoints.

Original directories and all file checksums are in MODEL_MANIFEST.json.
The archive includes self-contained model folders, the cached flow weights,
runtime source code, environment specification and benchmark metadata.

After extracting model_bundle.tar.gz, create/activate the matching environment
and install the included checkout (`cd code; python -m pip install -e .`).
runtime_versions.json records the versions used for this presentation.

From the extracted archive root:

```python
import cv2
from flowsis.base_checkpoint import load_base
from flowsis.cli.deploy.common import center_square

base = load_base("models/current_base", device="cuda", use_amp=False,
                 detector_amp=False)
frame = center_square(cv2.imread("your_image.jpg"))  # BGR uint8, 0..255
prediction = base.infer_frame(frame)
```

For a standalone detector, use `flowsis.pretrained.load_detector` with
`models/current_detector`. The complete current base already embeds it.
The prior epoch-18 detector is embedded unchanged in `models/prior_base_frozen`
and can be loaded from that directory using the detector loader.

A temporal pipeline additionally needs the MATCHING base directory, a valid
`temporal-best.pt`, and the flow checkpoint. The original coarse-v2 temporal
checkpoint was trained with incorrect preprocessing and is EXCLUDED from this
inference archive. Its audit-only path is
`outputs/overnight/coarse-v2/temporal/main/temporal-best.pt`.

After corrected temporal retraining, preserve the new temporal checkpoint with
`models/current_temporal_base` and the included `flow/raft_small-things-b7d9f997.ckpt`.
If temporal training fine-tunes the base head, preserve that run's `base-best`
instead. A temporal `.pt` file does not include base or flow weights.

```python
from flowsis.pretrained.flow import load_flow_estimator
from flowsis.temporal_checkpoint import load_temporal_checkpoint
from flowsis.temporal_model import FlowSISTemporal

base = load_base("models/current_temporal_base", device="cuda", use_amp=False)
flow = load_flow_estimator("raft_small", "flow/raft_small-things-b7d9f997.ckpt").to("cuda").eval()
# Run only after obtaining the corrected checkpoint:
temporal = load_temporal_checkpoint("corrected-temporal-best.pt", device="cuda")
model = FlowSISTemporal(base, flow, temporal)
prediction = model.infer_frame(frame)
```
"""
    (output / "SAVE_AND_LOAD.md").write_text(guide)
    for path in sorted((ROOT / "flowsis").rglob("*.py")):
        if "models" not in path.relative_to(ROOT / "flowsis").parts:
            files.append((path, "code/" + str(path.relative_to(ROOT))))
    for name in ("pyproject.toml", "environment.yml", "README.md"):
        files.append((ROOT / name, "code/" + name))
    for path in sorted((ROOT / "docs").glob("*.md")):
        files.append((path, "code/" + str(path.relative_to(ROOT))))
    for path in sorted((ROOT / "experiments").glob("*.py")):
        files.append((path, "code/" + str(path.relative_to(ROOT))))
    for path in sorted(
        (ROOT / "configs/prompt_suites").glob("combined-coarse-v1*.json")
    ):
        files.append((path, "code/" + str(path.relative_to(ROOT))))
    for name in ("MODEL_MANIFEST.json", "runtime_versions.json", "SAVE_AND_LOAD.md"):
        files.append((output / name, name))
    for path in sorted((output / "benchmark").rglob("*.json")):
        files.append((path, str(path.relative_to(output))))
    archive = output / "model_bundle.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for path, name in files:
            tar.add(path, arcname=name, recursive=False)
    (output / "model_bundle.tar.gz.sha256").write_text(
        hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
        + "  model_bundle.tar.gz\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/presentation/coarse-v2"
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "text.color": NAVY,
        }
    )
    benchmark = read(output / "benchmark/summary.json")
    cohort = read(output / "benchmark/cohort.json")
    with PdfPages(output / "presentation_figures.pdf") as pdf:
        overview(output, pdf)
        same_image_benchmark(output, benchmark, cohort, pdf)
        base_sources(output, pdf)
        class_profile(output, pdf)
        learning_curves(output, pdf)
        temporal_status(output, pdf)
        qualitative_examples(output, cohort, pdf)
        detector_comparison(output, pdf)
    with PdfPages(output / "short_presentation.pdf") as pdf:
        overview(output, pdf)
        same_image_benchmark(output, benchmark, cohort, pdf)
        detector_comparison(output, pdf)
        class_profile(output, pdf)
        temporal_status(output, pdf)
    notes = f"""# Short presentation notes

Prepared with Codex assistance. Eight 16:9 figures are provided as PNG, SVG and PDF.
short_presentation.pdf is a five-slide selection. presentation_figures.pdf contains
all eight as ready-to-present pages; use the individual images to edit your own slides.

1. Pipeline: detector box AP50 is 99.46% pooled on current validation;
   strongest pooled base Dice is 89.63%. These measure different tasks.
2. Same-image comparison: {cohort['images']} hash-verified stills, {cohort['capture_groups']} capture groups,
   classes {', '.join(cohort['class_support'])}. Prior frozen base matched Dice
   {100*benchmark['prior_head10']['metrics']['matched_dice']:.2f}%; current overlap-off
   {100*benchmark['current_overlap_off']['metrics']['matched_dice']:.2f}%.
   Prior gently unfrozen and current overlap25 models are also shown, avoiding
   cherry-picking the weaker prior checkpoint. All use the same current annotations,
   canonical-name class remapping, FP32 and 0.5 thresholds. This is validation
   intersection data used for selection; it is not an untouched test benchmark.
3. On the larger current validation set, no-overlap base is slightly stronger;
   zero real multi-tool validation scenes prevent conclusions about real overlaps.
4. Bonepunch and gouge deserve inspection; four classes lack validation positives.
5. Checkpoint choice changes with Dice versus AP. One seed; some AMP skipped steps.
6. Temporal improvement is unproven: original training inputs were incorrect.
   The diagnostic figure shows correction of base preprocessing, not learned gain.
7. Two qualitative examples compare the stronger prior frozen checkpoint against
   the current no-overlap base. Images were selected by sorted image ID within
   two classes; the same ground-truth crop is used for both models. Class-aware
   Dice is zero when the top prediction has the wrong label, even if its shape
   resembles the manual mask. Full numerical results cover all 31 images.
8. Standalone detector comparison uses the same cohort and canonical-name mapping.
   Both prior epoch-18 and epoch-20 detectors are shown. Epoch 20 is the stronger
   prior pooled-AP candidate on this cohort; current pooled AP50 is 94.07% versus
   prior epoch-20 AP50 of 62.08%. These are four-class development results.

Do not compare old 115-image headline metrics directly against current 370-frame
headline metrics as a controlled generation improvement. Use figure 02 instead.
Original model paths, inference files and backup checksums: MODEL_MANIFEST.json.
Runtime loading instructions: SAVE_AND_LOAD.md.
"""
    (output / "SLIDE_NOTES.md").write_text(notes)
    sources = {
        "prepared_with": "Codex assistance",
        "current_run": str(RUN),
        "benchmark": str(output / "benchmark"),
        "figures": {p.stem: str(p) for p in sorted(output.glob("0*.png"))},
        "temporal_status": "Corrected retraining pending; no temporal effectiveness claim.",
    }
    (output / "figure_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    lines = [
        "# Same-image benchmark results",
        "",
        "Prepared with Codex assistance.",
        "",
        f"Cohort: {cohort['images']} images, {cohort['capture_groups']} capture groups; support {cohort['class_support']}.",
        "",
        "| Base checkpoint | Matched Dice | Oracle Dice | Mask AP50 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in benchmark.items():
        metric = result["metrics"]
        lines.append(
            f"| {MODEL_NAMES[name]} | {metric['matched_dice']*100:.2f}% | {metric['oracle_dice']*100:.2f}% | {metric['mask_ap50']*100:.2f}% |"
        )
    lines += ["", "## Method and limits", ""] + [
        "- " + note for note in cohort["notes"]
    ]
    (output / "BENCHMARK_RESULTS.md").write_text("\n".join(lines) + "\n")
    package_models(output)
    with ZipFile(output / "presentation_assets.zip", "w", ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.suffix in (".png", ".svg", ".pdf", ".md", ".json"):
                archive.write(path, path.name)
        for filename in ("summary.json", "detector_summary.json", "cohort.json"):
            archive.write(output / "benchmark" / filename, "benchmark/" + filename)
    print("Presentation figures, PDF, notes and model archive:", output)


if __name__ == "__main__":
    main()
