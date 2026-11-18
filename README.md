# Event-Flow

Event-Flow is a research project for temporally refined object detection using
RGB and event cameras, initially targeting FRED (the Florence RGB Event Drone
Dataset).

## Development status

This repository starts from FlowSIS's `end-to-end` branch, including its local
working changes. RGB/event fusion and temporal box/class refinement are planned;
the inherited implementation currently provides RGB detection, instance
segmentation, and temporal mask refinement.

The intended detector input is `[B, 4, H, W]`: RGB plus a binary event occupancy
channel indicating whether a pixel received an event during a causal accumulation
interval ending at the prediction timestamp. Camera alignment, synchronization,
and the accumulation interval must be specified by the dataset integration.
The initial optical-flow path will continue to use RGB.

## Starting code

The Python package and CLI names currently remain `flowsis` and `flowsis-*`.
See the [inherited FlowSIS README](docs/flowsis-readme.md) and existing `docs/`
for the original setup, training, and checkpoint documentation. Paths and commands
in that archived README describe the original project and may need adjustment.

Planned adaptation:

1. Load synchronized RGB, events, timestamps, and detection labels from FRED.
2. Add four-channel preprocessing and a pretrained RGB-compatible detector stem.
3. Add temporal refinement of boxes and class/objectness predictions.
4. Compare RGB and RGB/event baselines, each with and without temporal refinement.

## Provenance

Git history is preserved. See [the source snapshot record](docs/source_snapshot.json)
for the exact starting revision. The `upstream` remote points to FlowSIS;
this repository has no publishing remote configured yet.

Repository setup and this adaptation outline were prepared with Codex assistance.
