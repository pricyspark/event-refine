# Local throughput experiments

Activate the `event-flow` environment and run from the repository root. These
scripts exercise a separate benchmark training path. They do not change normal
`event-flow train`, overwrite trained checkpoints, or modify the dataset.
No validation or model-checkpoint serialization is included in timed regions.

## Start with clip batching

```bash
python scripts/benchmark_throughput.py \
  --manifest data/fred-5-videos/manifest.json \
  --suite batching --batch-sizes 1 2 --threads 2 \
  --warmup 10 --steps 50 --repeats 3 \
  --output outputs/bench-batching
```

This compares one versus two independent four-frame clips per optimizer step.
Frames within a clip remain ordered and carry recurrent state. Different clips
occupy different batch rows, and state resets between optimizer steps. Compare
**clips/second or frames/second**, not optimizer steps/second. Each measured case
processes `steps * batch_size` clips from the same deterministically selected
cycling pool; it is a throughput experiment, not an equal-training-budget quality
comparison. Batch normalization sees the larger batch in training, so these runs
are not expected to follow identical optimization trajectories.

Start with 1 and 2 on the 12 GB RTX 5070. Add `4` only if useful: the worker records
CUDA out-of-memory as `oom`, exits, releases its memory, and the runner proceeds.
Clips in a batch must have equal lengths. Different image sizes are resized and
padded separately before batching; the installed FRED data uses uniform sizes.
Normal training supports the same batching with `event-flow train --batch-size 2`
(default: 1). The existing detection objective is evaluated per
image and averaged across images and time. It is not a vectorized Hungarian
matcher; loss-side CPU/GPU transfers still exist.

## Repeated GPU input validation

```bash
python scripts/benchmark_throughput.py \
  --manifest data/fred-5-videos/manifest.json \
  --suite checks --batch-sizes 1 --threads 2 \
  --warmup 10 --steps 50 --repeats 3 \
  --output outputs/bench-input-checks
```

`eager` uses the existing GPU value checks. `trusted-inputs` skips repeated GPU
RGB-range/event-binary/normalization-value checks, using the canonical values
produced by the loader. File decoding, event-archive validation, box checks, tensor
shape/device checks, finite loss, and finite-gradient checks remain enabled.
This isolates one source of synchronization; it does not remove the CPU Hungarian
matching or all CPU/GPU synchronization. Normal training keeps its input checks.

## Optional compilation / fusion

```bash
python scripts/benchmark_throughput.py \
  --manifest data/fred-5-videos/manifest.json \
  --suite compile --batch-sizes 1 --threads 2 \
  --warmup 10 --steps 50 --repeats 1 --timeout 600 \
  --output outputs/bench-compile
```

This compares **trusted eager** with **trusted compiled** so input checking is
held constant. Compilation applies to `EventFlow.forward`, including detector
and temporal head, with `torch.nn.Module.compile(dynamic=False)`. The loss and
CPU matcher stay outside the compiled function. Mode `default` tests compilation
and fusion without explicitly requesting CUDA Graphs. After that works, optionally
try `--compile-mode reduce-overhead` in a new output directory to test PyTorch's
CUDA Graph-enabled mode. This is experimental, not a promise of whole-step graph
capture: the inherited detector uses hooks and some CPU-visible operations that
can cause graph breaks. Returned temporal state is used across frames.

Compilation may take minutes or be unsupported with a particular backend/driver.
Its initial cost belongs to `warmup_seconds`, not the measured rate. Graph-break
and recompilation diagnostics are captured in each case's `console.log`. A
recompile during measurement will still affect that measured result; inspect the
log. The runner limits each case to `--timeout` seconds and terminates its process
group on timeout or Ctrl+C. Failed or timed-out compilation remains a visible
result; it is not silently reported as eager execution. Partial graph breaks can
still fall back to eager regions and should be inspected in the log.

Finite losses/gradients are checked in all modes, but this alone is not a full
numerical or accuracy validation of compilation. Run a quality regression before
using an experimental setting for a research result.

## Other suites and dry runs

```bash
# Optional repeat of the earlier CPU-thread sweep.
python scripts/benchmark_throughput.py \
  --manifest data/fred-5-videos/manifest.json \
  --suite threads --threads 1 2 4 8 --repeats 3 \
  --output outputs/bench-threads

# Preview all commands without loading models or touching the GPU/output path.
python scripts/benchmark_throughput.py \
  --manifest data/fred-5-videos/manifest.json \
  --suite batching --batch-sizes 1 2 --dry-run \
  --output outputs/bench-preview
```

`--suite all` includes compilation and may take considerably longer. For a quick
harness smoke test use `--warmup 2 --steps 5 --repeats 1`; do not infer performance
wins from such a short run. `--image-size`, `--clip-length`, `--architecture`, and
`--detector-model` can be overridden, but hold them constant across comparisons.
`--device cpu` and a local tiny detector are supported for plumbing tests.

## Reading the results

Each case runs sequentially in a fresh process with its own OMP/MKL thread limits.
Run without competing GPU workloads. Measurement starts after warm-up and a CUDA
synchronization; another synchronization waits for the final GPU work before the
wall-clock timer stops. There is no profiler attached to throughput measurements.
The runner prints progress and a heartbeat while a case is running. Choose a fresh
output directory for every suite execution to avoid mixing measurements.

Outputs:

- `plan.json`: matrix settings and repetitions.
- `results.csv`: each repetition's status, clips/sec, frames/sec, peak allocated
  VRAM, and warm-up time.
- `summary.csv`: median/min/max successful clips/sec for each setting.
- `<case>/result.json`: full timing, actual PyTorch thread counts, device/software
  versions, manifest hash, selection-pool indices, loss, and peak allocated/reserved
  GPU memory.
- `<case>/case.json` and `console.log`: exact worker options and diagnostic output.

Peak allocated/reserved VRAM is PyTorch memory, not total board usage from desktop
applications or other processes. Loading timing is a coarse CPU measurement;
transfers and kernels can overlap it. CUDA synchronization at measurement
boundaries makes the aggregate end-to-end throughput the primary number.

Run functional regression tests after changes:

```bash
python -m pytest tests/test_throughput_benchmarks.py tests/test_event_flow.py -q
```

The benchmark tests check batch-loss scaling and temporal-head gradients against
independent clips, finite detector gradients, temporal batch-row isolation,
trusted-input value equivalence, worker execution, and matrix/dry-run behavior.
