# End-to-end set-based architecture

## Tensor contract

For a preprocessed detector batch `[B,3,H,W]`, both supported adapters return a
`DetectorFeatureOutput` with:

| Tensor | Shape | Role |
|---|---:|---|
| detail features | `[B,C4,H/4,W/4]` | raw first backbone-stage detail |
| encoder pyramid | three `[B,C,H/s,W/s]` tensors | strides 8, 16, and 32 |
| query embeddings | `[B,Q,256]` | final detector decoder queries |
| class logits | `[B,Q,C]` | original focal/sigmoid detector predictions |
| normalized boxes | `[B,Q,4]` | center-x, center-y, width, height |

`Q` is never hard-coded; it comes from the detector checkpoint (300 in the
current production checkpoints). The adapter validates the Hugging Face
backbone path and all four spatial resolutions, failing with an explicit
contract error if a future Transformers layout changes.

The convolutional FPN projects C32/C16/C8 to `D=256`, upsamples and refines with
GroupNorm and GELU, then combines the result with projected C4 detail. This
shared, image-only tensor is computed once. The query branch produces `[B,Q,D]`
mask embeddings and the raw mask logits are:

```python
torch.einsum("bqd,bdhw->bqhw", mask_embeddings, pixel_features)
```

## Text conditioning

Training loads one ordered prompt bank `[C,P,768]`; the bank is a persistent,
non-trainable model buffer. Detector sigmoid/focal classification remains
unchanged. A temperature-controlled softmax is used only to mix class prompt
banks for mask conditioning. Each projected query then attends over the `P`
prompt variants.

Projected query and text features feed a gated residual text delta. Its final
projection starts at zero, so initialization is exactly query-only. The
`none` intervention bypasses the complete text-delta path; `learned` replaces
semantic prompts with learned class vectors while retaining the fusion shape.

Matched queries receive target-class text in `head`, anneal linearly to predicted
mixtures during `neck`, and use predicted mixtures in `all`.

## Box prior and losses

Normalized center boxes are converted to corners, expanded by 20%, and rendered
as sigmoid-edged gates on the stride-4 mask grid. The head adds
`alpha * log(clamp(gate))`; `alpha` is bounded, learnable, and initialized to
0.25. This path remains differentiable with respect to box coordinates.

Hungarian matching uses detector class/box/GIoU costs `(2,5,2)`. Only matched
queries are resized and receive BCE and soft-Dice losses. Losses average by
object and then by non-empty image. Unmatched queries remain governed by the
detector classification objective, and all-empty batches retain a valid
differentiable zero mask loss.

```text
total = detector_weight * detector_loss
      + bce_weight * mask_bce
      + dice_weight * mask_dice
```

The staged optimizer uses mask, detector non-backbone, and backbone learning
rates of `1e-4`, `1e-5`, and `1e-6`. `head` freezes and evaluates the detector;
`neck` enables the detector projections, hybrid encoder, decoder, and heads;
`all` also enables the backbone. The prompt bank never receives gradients.

## Inference and artifacts

Each query contributes only its highest-scoring sigmoid class, preventing the
same query from appearing multiple times. Thresholding and optional top-count
limiting select unique queries; only selected masks are resized. Independent
overlapping logits are retained, while binary consumers can use their pixelwise
maximum-logit union. An empty accepted set has `union_logits=None`.

A model directory contains the detector configuration/processor/weights and
`base.pt`. The latter embeds set-mask configuration and weights, prompts, label
order, thresholds, and format metadata. Resume checkpoints add optimizer,
scheduler, scaler, epoch, and step state. Artifacts must use the current
checkpoint format.

## Metrics and experiments

The evaluation command separates class-aware end-to-end matched metrics from
oracle-query mask metrics, and also reports union Dice/IoU, boundary F1, box and
mask AP50, precision/recall, empty-frame false positives, and per-class results.
Training records the initialized detector baseline and selects matched-Dice
checkpoints only when AP50 and empty-frame guardrails pass.

Run architectural ablations as independently retrained models with at least
three seeds and paired per-image confidence intervals. `flowsis-ablate-base`
only performs same-checkpoint interventions and labels them accordingly.

## Deferred work

- intermediate-decoder auxiliary mask supervision;
- query-text contrastive alignment;
- predicted mask-quality scores;
- mask-aware query refinement;
- segmentation-aware matching or selection;
- learned box strengths/margins and box-mask consistency;
- per-instance temporal association and recurrent state.

The generic `temporal.py` and optical-flow adapter remain isolated building
blocks for that last item. They are not wired into base inference.
