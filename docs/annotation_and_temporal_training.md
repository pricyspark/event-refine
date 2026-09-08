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
ignore region represents visible but unassignable fragments.

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
flowsis-author-text-prompts
# Review/edit data/new/text_prompts.json.
flowsis-embed-text-prompts
flowsis-build-temporal-dataset
```

The model label combines the manifest's descriptive `Tool Name` with its coarse
`Tool Class`: `iris` plus `scissors` becomes `iris scissors`, and an already
complete name such as `mayo scissors` is not duplicated. The original columns
remain separate and unchanged in the manifests and Arrow object metadata.

`flowsis-author-text-prompts` interactively asks for one visual description per
class and writes three prompts per class to `data/new/text_prompts.json`: the
full canonical label, the exact Transformers SigLIP2 pipeline template (`This
is a photo of {label}.`), and the manually authored description. Progress is
saved after every description, and rerunning the command resumes the existing
file. No external service or API key is required.

Automatic draft generation is optional rather than part of the reproducible
core environment:

```bash
pip install -e '.[prompt-generation]'
export OPENAI_API_KEY="..."
flowsis-generate-text-prompts
# Review/edit data/new/text_prompts.json before embedding it.
```

The generator sends only the canonical taxonomy, not images or annotations. It
uses a pinned model snapshot and strict structured output, requests that the
response is not stored, and records the model and exact instructions in
`text_prompts.provenance.json`. It refuses to overwrite an existing suite
unless `--force` is supplied. Generated descriptions remain drafts because an
API response is neither deterministic research data nor a substitute for
checking the actual physical instruments.

To edit JSON directly instead, first write a scaffold:

```bash
flowsis-author-text-prompts --scaffold
$EDITOR data/new/text_prompts.json
```

Fill the third string for every class. The embedding command rejects empty or
malformed entries, so a partially edited scaffold cannot silently enter
training. `--scaffold` refuses to replace an existing file unless `--force` is
explicitly supplied.

Review or edit each description (the third string) before running
`flowsis-embed-text-prompts`. Embedding is deliberately a separate,
deterministic step: it validates all three prompt types against the current
Arrow taxonomy, then stores their pooled SigLIP2 vectors in
`data/new/text-embeddings`. Use `--prompt-suite PATH` to embed a different
reviewed suite with the same contract.

Write the description as a compact noun phrase beginning with the complete
label. Prefer stable visible morphology that distinguishes sibling classes:
blade or jaw proportions, tip shape, curvature, serrations, teeth,
fenestrations, shaft length, and handle geometry. Check every statement against
the physical instrument. Avoid intended use, eponyms without the coarse class,
procedure or anatomy, background, occlusion state, temporary reflections,
variable color, and negation. Keep descriptions comparably specific across
classes so one category does not receive much richer language supervision than
another.

Treat the suite as a hypothesis rather than ground truth. On validation data,
compare label-only, template-only, description-only, and all-three prompt
banks. Select wording and prompt count using validation results, never the test
set. If more descriptions are tested later, add the same number and level of
detail for every class and retain each embedding separately for the existing
query-conditioned prompt attention.

Create single-type ablation banks from the same reviewed suite with, for
example:

```bash
flowsis-embed-text-prompts \
  --prompt-type description \
  --output-dir data/new/text-embeddings-description-only
```

Repeat `--prompt-type` to test a subset; omitting it embeds all three.

For a public experiment, commit the final reviewed prompt JSON used for the
reported run somewhere outside the ignored `data/` tree, such as
`configs/prompt_suites/<experiment>.json`, along with its generation provenance
when applicable. Reproduction should embed this checked-in artifact rather than
regenerating prompts from an external service.

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
