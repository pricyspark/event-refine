"""Batched temporal training objectives shared by training and benchmarks."""

import torch

from .data import FrameSample
from .loss import detection_loss, normalized_targets
from .model import EventFlow
from .profiling import profile_region


def batch_clip_loss(
    model: EventFlow,
    clips: list[list[FrameSample]],
    device: str,
    *,
    validate_values: bool = True,
    base_weight: float = 0.5,
    profile: bool = False,
) -> torch.Tensor:
    """Batch independent clips across B, preserving recurrence along T.

    Clip lengths must match; image sizes may differ between clips.
    Per-image detection objectives retain the existing mean-over-images scaling.
    Each invocation starts fresh memory; memory never crosses clip boundaries.
    """
    if not clips or not clips[0]:
        raise ValueError("Provide at least one nonempty clip.")
    length = len(clips[0])
    sizes = [clip[0].rgb.shape[-2:] if clip else () for clip in clips]
    if any(len(clip) != length for clip in clips):
        raise ValueError("Batched clips must have equal lengths.")
    if any(
        sample.rgb.shape[-2:] != size
        for clip, size in zip(clips, sizes)
        for sample in clip
    ):
        raise ValueError("Registered image dimensions changed within a clip.")
    state = None
    losses = []
    for frame_index in range(length):
        samples = [clip[frame_index] for clip in clips]
        with profile_region("preprocess_and_transfer", profile):
            if all(size == sizes[0] for size in sizes):
                rgb = torch.stack([sample.rgb for sample in samples]).to(device)
                events = torch.stack([sample.events for sample in samples]).to(device)
                pixels, mask = model.prepare(
                    rgb, events, validate_values=validate_values
                )
            else:
                prepared = [
                    model.prepare(
                        sample.rgb[None].to(device),
                        sample.events[None].to(device),
                        validate_values=validate_values,
                    )
                    for sample in samples
                ]
                pixels = torch.cat([pixels for pixels, _ in prepared])
                mask = torch.cat([mask for _, mask in prepared])
        with profile_region("forward", profile):
            output = model(pixels, pixel_mask=mask, state=state)
        with profile_region("matching_and_loss", profile):
            for image, sample in enumerate(samples):
                target_boxes = normalized_targets(
                    sample.boxes.to(device),
                    original_size=sizes[image],
                    image_size=model.detector.image_size,
                )
                labels = sample.labels.to(device)
                loss = detection_loss(
                    output.logits[image : image + 1],
                    output.boxes[image : image + 1],
                    target_boxes,
                    labels,
                )["loss"]
                if base_weight:
                    loss = (
                        loss
                        + base_weight
                        * detection_loss(
                            output.base.query_logits[image : image + 1],
                            output.base.query_boxes[image : image + 1],
                            target_boxes,
                            labels,
                        )["loss"]
                    )
                losses.append(loss)
        state = output.state
    return torch.stack(losses).mean()
