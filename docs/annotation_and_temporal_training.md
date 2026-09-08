# Annotation and temporal training

`image_manifest.csv` remains the immutable capture-level source of truth. The
annotation command only appends `Image ID` and `Capture Session ID`; object
annotations are written to `image_object_manifest.csv`. Use one object manifest
for single-tool and future multi-tool captures.

The initial capture-session IDs are deterministic time/device suggestions, not
ground truth. Review them against the actual operator, camera, physical setup,
and background before building splits. A train/validation/test build fails
instead of leaking when fewer than three disconnected capture/tool groups are
available.

```bash
# Current single-tool images
flowsis-annotate images

# More specific alternative: e.g. "iris scissors" instead of "scissors"
flowsis-annotate images --image-prompt name-and-class

# Future cluttered images: broad discovery followed by canonical prompts
flowsis-annotate images \
  --discovery-prompt "surgical instruments" \
  --canonical-prompt "needle driver" \
  --canonical-prompt "forceps"

# A dense video segment; repeat per prompt when necessary
flowsis-annotate videos \
  --prompt "needle driver" \
  --video-id VIDEO_ID \
  --start-frame 0 --end-frame 600
```

Image annotation prompts SAM3 with the row's coarse `Tool Class` by default,
giving prompts such as `scissors`. The optional `--image-prompt
name-and-class` mode combines the canonical fine name and class, such as `iris
scissors`. Regardless of prompt style, accepted records retain the original
fine `Tool Name` and coarse `Tool Class` separately.

Review decisions are appended immediately to JSONL, so interruption and resume
are safe. `a` accepts selected candidates, `e` records an authoritative empty
frame, `i INDICES` marks unassignable fragments as ignore regions, `r` rejects,
`s` leaves work pending, `p TEXT` reruns a prompt, and `v STATE [INDICES]` sets
per-instance visibility. Video review additionally supports
`o INDICES`, which retains known track identities as fully occluded without a
positive mask. Accepted frames store one compressed, bit-packed NPZ bundle; its
first dimension is aligned by explicit instance/track IDs. An optional packed
ignore region represents visible but unassignable fragments. Legacy one-mask
NPZ archives remain readable.

Frames with ignore regions remain available to temporal mask training, whose
pixel loss masks those regions. They are excluded from base detector training
because the current DETR loss has no pixel-level ignore contract; this prevents
unknown tool fragments from silently becoming background targets.

The visibility contract is:

- `clear`: mask, class, and base losses are valid.
- `partial`: every visible fragment belongs to one instance mask; all losses
  remain valid.
- `severe`: the neighboring track establishes identity; temporal mask loss is
  valid, but current-frame base/class losses are disabled.
- `occluded`: keep the track with visibility false and no positive mask.

Do not label hidden geometry. Unknown visible fragments belong in the ignore
region, not background. Do not create a miscellaneous class for a known but
occluded tool.

Build separate Arrow datasets after review:

```bash
flowsis-build-annotated-dataset
flowsis-build-temporal-dataset
```

Both builders split connected physical-tool/capture-session groups as units.
The base loader samples physical tools, then samples, then rotations
hierarchically. The temporal loader dynamically emits clips of 4–8 frames with
default stride probabilities 0.7/0.2/0.1 for strides 1/2/3. It keeps masks as
external per-frame bundles, avoiding decompression of unrelated video frames.

Train the base first, then freeze the entire base and flow models for initial
temporal training:

```bash
flowsis-train-temporal \
  --dataset-path data/new/temporal-dataset \
  --base-model outputs/base/best \
  --output-dir outputs/temporal
```

If validation justifies mask-head adaptation, use the one-tenth-rate option
with mandatory still-image replay:

```bash
flowsis-train-temporal \
  --dataset-path data/new/temporal-dataset \
  --base-model outputs/base/best \
  --finetune-mask-head \
  --still-dataset-path data/new/segmentation-dataset
```

Preferred-track labels are stored but have weight 1 by default. A controlled
2× ablation uses `--preferred-track-weight 2`; it is not part of the initial
training recipe.

Measure the complete detector, mask head, flow, tracking, recurrence, and
selection path rather than extrapolating from component timings:

```bash
flowsis-profile-temporal \
  --base-model outputs/base/best \
  --temporal-model outputs/temporal/temporal-best.pt \
  --video-source data/new/videos/20260901_191034.mp4
```
