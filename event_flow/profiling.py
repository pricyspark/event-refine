"""Short, scheduled CPU/CUDA traces with named training stages."""

from contextlib import AbstractContextManager, contextmanager, nullcontext
from collections.abc import Iterator
import json
from pathlib import Path

import torch


def profile_region(name: str, enabled: bool) -> AbstractContextManager[object]:
    return torch.profiler.record_function(name) if enabled else nullcontext()


@contextmanager
def training_profiler(
    output: Path,
    *,
    enabled: bool,
    device: torch.device,
    warmup_steps: int,
    active_steps: int,
) -> Iterator[torch.profiler.profile | None]:
    if not enabled:
        yield None
        return
    directory = output / "profile"
    directory.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda_requested = device.type == "cuda"
    if cuda_requested:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def export(profiler: torch.profiler.profile) -> None:
        trace = directory / "trace.json"
        profiler.export_chrome_trace(str(trace))
        averages = profiler.key_averages()
        cpu = averages.table(sort_by="self_cpu_time_total", row_limit=30)
        summary = "CPU operators (self time)\n" + cpu
        if cuda_requested:
            summary += "\n\nDevice operators (self time)\n" + averages.table(
                sort_by="self_device_time_total", row_limit=30
            )
        (directory / "operators.txt").write_text(summary + "\n")
        cuda_events = sum(
            event.device_type == torch.autograd.DeviceType.CUDA
            for event in profiler.events()
        )
        metadata = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device) if cuda_requested else "CPU"
            ),
            "device": str(device),
            "cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "warmup_steps": warmup_steps,
            "active_steps": active_steps,
            "completed_steps_at_export": profiler.step_num,
            "cuda_requested": cuda_requested,
            "cuda_event_count": cuda_events,
            "record_shapes": False,
            "with_stack": False,
            "profile_memory": False,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "event": "profile_saved",
                    "trace": str(trace),
                    "summary": str(directory / "operators.txt"),
                    "cuda_event_count": cuda_events,
                }
            ),
            flush=True,
        )
        if cuda_requested and not cuda_events:
            print(
                "CUDA tracing was requested but no device events were captured. Check the PyTorch/CUPTI/driver profiling support; the CPU trace remains available.",
                flush=True,
            )

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=warmup_steps - 1, warmup=1, active=active_steps, repeat=1
        ),
        on_trace_ready=export,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as profiler:
        yield profiler
