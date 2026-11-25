# Event-Flow data contract

Prepared with Codex assistance.

## Sequence manifest (version 1)

Paths are absolute or relative to the JSON file. Boxes use pixel `xyxy` in the
registered, unresized RGB/event plane. Labels index the ordered `classes` list.
Each sequence has exactly one split, preventing its clips from crossing splits.
Use globally unique sequence IDs, including across splits.

```json
{
  "format_version": 1,
  "classes": ["drone"],
  "sequences": [
    {
      "id": "flight-01",
      "split": "train",
      "frames": [
        {
          "timestamp": 0.033333,
          "rgb": "rgb/000000.jpg",
          "events": "events/000000.npz",
          "event_encoding": "binary",
          "annotated": true,
          "boxes": [[100, 50, 120, 70]],
          "labels": [0]
        },
        {
          "timestamp": 0.066666,
          "rgb": "rgb/000001.jpg",
          "events": "events/000001.npz",
          "event_encoding": "binary",
          "annotated": true,
          "boxes": [],
          "labels": []
        }
      ]
    }
  ]
}
```

`annotated: true` requires explicit `boxes` and `labels`, including `[]` for an
exhaustively reviewed empty image. `annotated: false` means supervision is unknown;
it can provide causal inference context, but contributes no training or metric
supervision. Training clips include only consecutive annotated frames with gaps
no larger than `--max-gap` (default 0.05 in manifest timestamp units).

Binary event maps default to bit-packed compressed NPZ: `format_version=1`,
`shape=[H,W]`, `bitorder="big"`, and a one-dimensional uint8 `packed` array with
`ceil(H*W/8)` bytes before compression. Every pixel is restored exactly,
including shapes whose pixel count is not divisible by eight. The writer uses
an atomic replacement so incomplete archives are not published.

The loader auto-detects this versioned NPZ format, legacy HxW boolean/0-1 NPY
arrays, or single-channel PNG images using 0/1/255. For rendered RGB PNG frames, set `event_encoding` to `rendered` and
supply `event_background: [R,G,B]` for every frame. A pixel differing from that
background in any channel is active. Polarity, count, and intra-window timing are
not encoded in the fourth channel.

Manifest timestamps must increase strictly and should be in seconds when using
the default max-gap. Raw event timestamps must use the same units as frame times.
The caller is responsible for registration, clock alignment, and a causal event
window. Native FRED HDF5 conversion is intentionally not guessed; export its raw
events into the documented x/y/t NPZ contract first.

Input RGB is converted to floating point [0,1], resized preserving aspect ratio,
then normalized with the detector's RGB processor statistics. Event occupancy is
max-pooled when shrinking and nearest-resized when enlarging. Both modalities
receive identical right/bottom padding to a square. Boxes are transformed into
that same padded plane for training and restored to original pixels on output.
No color, geometric, or synthetic overlap augmentations are enabled in this
initial detection pipeline.

## Training and state

Each optimizer step processes one clip. The detector and temporal head both
receive gradients through the whole clip; state is discarded between optimizer
steps. The temporal head attends over previous query memory using content,
spatial proximity, and confidence, then updates a GRU. It assumes neither stable
query indices nor persistent object IDs. Zero-initialized box/class correction
heads preserve the RGB detector's initial predictions. Event weights in the
expanded stem are zero-initialized but trainable.

Checkpoints contain `detector/` (Hugging Face model, processor, and adapter
metadata) plus `event_flow.pt` (format version, class names, head configuration,
and head weights). The detector configuration records four input channels,
including HGNetV2's stem channel configuration. Load these checkpoints with
`event_flow.checkpoint.load_checkpoint`, not inherited RGB inference helpers.
Training logs record loss and steps; `best` means lowest validation loss if a
validation split was explicitly supplied. Optimizer resume is not implemented.

## Initial experiment protocol

Reserve complete sequences for validation. Keep the official test set untouched
until model selection is complete. Compare RGB, RGB+events, RGB+temporal, and
RGB+events+temporal under the same sequence splits and training budget. For a
quick reliance check you can zero occupancy or reset memory every frame via the
Python API; retrain controls before making causal performance claims.

The implementation is a frame-synchronous research baseline. It does not recover
objects without any current-frame query, guarantee improved accuracy, implement
persistent track identities, or exploit event updates between RGB frames.
