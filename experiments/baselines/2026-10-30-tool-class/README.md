# FlowSIS baseline: 2026-10-30

Prepared with Codex assistance from saved training/evaluation artifacts and the current exported dataset. No new training or inference evaluation was performed. This is a development/validation baseline, not a final test-set or deployment-performance claim.

## Checkpoints to retain

| Identifier | Exact checkpoint | Role |
| --- | --- | --- |
| `detector_epoch18` | `outputs/detectors/dfine-tool-class-20epochs/checkpoint-001170` | class-balanced detector candidate; initializer for all current base runs |
| `detector_epoch20` | `outputs/detectors/dfine-tool-class-20epochs/final` | highest standalone detector micro AP50 |
| `base_original_epoch5` | `outputs/base/dfine-tool-class-epoch18-v1/best` | historical base reference; original staged run selected frozen epoch |
| `base_head10_epoch10` | `outputs/base/dfine-tool-class-epoch18-head10-v1/best` | stable frozen-detector base reference |
| `base_gentle_epoch14` | `outputs/base/dfine-tool-class-epoch18-head10neck3all2-v1/best` | highest logged aggregate base Dice and AP among selected checkpoints |
| `base_gentle_epoch2_mask_ap` | `outputs/base/dfine-tool-class-epoch18-head10neck3all2-v1/checkpoint-000518` | highest logged mask AP50; lower matched Dice; requires standalone confirmation |

**Working choices:** use detector epoch 18 for class-balanced detector performance and to initialize base-model comparisons; retain detector epoch 20 as the pooled-AP alternative. Gentle base epoch 14 is the strongest logged aggregate base candidate; retain head-only epoch 10 as the stable frozen-detector reference. Retain original base epoch 5 as the historical reference, particularly because extra head epochs did not improve every metric. Also retain gentle-run epoch 2 as the mask-AP-specific candidate. These are useful alternatives, not interchangeable winners.

## Comparable detector experiments

All three broad-class results below are standalone validation evaluations of D-FINE Medium, pretrained on Objects365, on the same nine-label/115-image benchmark. Resolution 640, batch eight, AdamW main and backbone LR 1e-4, weight decay 1e-4, linear LR decay, no warmup, FP32, seed 42. Augmentation is rotation 0-45 degrees, ROI square cropping, photometric changes, and overlap compositing (0-1 overlays, p=0.25). The 20-epoch run uses a fresh longer schedule; its epoch 18 and 20 checkpoints share that trajectory.

| Detector | Micro AP50 | Macro AP50, all nine labels | Mean AP50, seven represented labels |
| --- | ---: | ---: | ---: |
| 5 epochs/final | 20.97% | 26.74% | 34.38% |
| 20-epoch schedule/epoch 18 | 68.08% | 57.41% | 73.81% |
| 20-epoch schedule/epoch 20 | 69.12% | 56.03% | 72.04% |

AP is the evaluator's interpolated PR-envelope area at box IoU 0.5, not full COCO mAP50:95. Evaluation score cutoff is zero, retaining 300 ranked query/class candidates per image (34,500 total); this count is not a count at an operational confidence threshold. The evaluator assigns AP zero to labels without positives, so mallet/rongeur reduce all-nine macro AP. Their absence is not measured failure on those classes.

| Class | Validation objects | 5-epoch AP50 | Epoch-18 AP50 | Epoch-20 AP50 |
| --- | ---: | ---: | ---: | ---: |
| bonepunch | 7 | 7.23% | 22.13% | 23.29% |
| forceps | 44 | 17.06% | 77.91% | 82.74% |
| gouge | 8 | 26.36% | 73.33% | 64.19% |
| mallet | 0 | 0.00% | 0.00% | 0.00% |
| osteotome | 16 | 59.53% | 88.50% | 94.77% |
| pliers | 8 | 73.28% | 83.11% | 85.83% |
| retractor | 8 | 26.90% | 92.81% | 75.38% |
| rongeur | 0 | 0.00% | 0.00% | 0.00% |
| scissors | 24 | 30.28% | 78.90% | 78.06% |

Longer training was the clearest detector improvement: pooled AP50 rose from 20.97% to 68-69%. Epoch 18 balances classes better; epoch 20 gains pooled AP, forceps, osteotome and pliers but loses gouge and retractor AP. No independent AP75 or test evaluation was found.

## Comparable segmentation experiments

All three base runs start from detector epoch 18 and use the same nine-class dataset and three-prompt frozen SigLIP2 bank. Common settings: resolution 640, batch two, four workers, mask dimension 256, stride-4 detail, box prior, head LR 1e-4, weight decay 1e-4, constant-with-warmup scheduler, warmup ratio 0.05, global gradient clip one, mask-head AMP, detector FP32, seed 42. Detection/mask thresholds are both 0.5; no max-detection cap. BCE/Dice/detector loss weights are one. Training samples capture groups with equal total Sample-ID weight rather than treating every view equally.

| Recipe | Detector / backbone LR | Selected epoch/stage | Matched Dice | Matched IoU | Box AP50 | Mask AP50 | Oracle Dice | Union Dice |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head:5,neck:3,all:2 | 1e-5 / 1e-6 | 5 / head | 68.43% | 65.19% | 54.31% | 46.45% | 94.61% | 92.93% |
| head:10 | frozen (recorded 1e-5 / 1e-6) | 10 / head | 68.82% | 65.88% | 54.31% | 44.28% | 95.15% | 93.43% |
| head:10,neck:3,all:2 | 1e-6 / 1e-7 | 14 / all | 68.89% | 66.05% | 56.18% | 47.12% | 95.17% | 94.30% |

**Metric-specific exception:** scanning every logged epoch, gentle-run **epoch 2** (still head-only) has the highest mask AP50, **48.78%**, with matched Dice **66.34%**, box AP50 **54.31%**, oracle Dice **91.89%**, and union Dice **90.22%**. Its detector guardrail passes. It was not saved as `best` because selection uses matched Dice. Retain `checkpoint-000518` and confirm its standalone evaluation before selecting on mask AP. This does not demonstrate a benefit from unfreezing: that epoch precedes unfreezing.

**Evidence status:** the table uses training-time validation for all three rows to keep the source consistent. Only the original selected model has a separate saved validation JSON (matched Dice 68.425%, mask AP50 46.497%), which closely agrees with its logged metrics. The two new best models still need the same standalone validation/visual inspection before calling the ranking confirmed.

Selection is highest matched Dice among epochs whose box AP50 stays within one percentage point of the base evaluator's initial detector baseline and whose empty-scene false positives satisfy the configured guardrail. Empty-scene guardrails have no supporting empty examples in this benchmark.

| Class | Original epoch-5 matched Dice | Head-only epoch-10 matched Dice | Gentle epoch-14 matched Dice |
| --- | ---: | ---: | ---: |
| bonepunch | 0.00% | 0.00% | 0.00% |
| forceps | 69.74% | 70.21% | 67.98% |
| gouge | 36.25% | 36.20% | 24.54% |
| osteotome | 90.77% | 91.06% | 97.53% |
| pliers | 84.85% | 85.20% | 97.48% |
| retractor | 11.82% | 11.87% | 11.86% |
| scissors | 95.20% | 95.90% | 95.83% |

## Insights and their limits

1. **Mask shapes learn quickly.** Oracle Dice improves from roughly 94.6% at five head epochs to 95.2% at ten. Class-aware Dice remains near 68-69%. Oracle evaluates the ground-truth-matched query mask without requiring correct top class/confidence; it does not imply ground-truth text was supplied. Query selection, predicted class, and acceptance remain important limitations. The gap alone does not identify a unique failure mechanism.
2. **More head epochs give modest overlap gains, not a universal improvement.** Selected matched Dice improves by about 0.39 percentage points over original epoch 5, but selected mask AP falls from 46.45% to 44.28%. Dice-based checkpoint selection and AP-based selection need not choose the same epoch.
3. **Original unfreezing hurt.** All five unfrozen epochs failed the detector guardrail. Original final matched Dice was 59.57%, versus 68.43% at its selected frozen epoch. Do not substitute final for best.
4. **Gentler unfreezing has a better peak but remains volatile.** Neck epochs 11-13 failed guardrails; epoch 14 passed and reached 68.89% matched Dice/47.12% mask AP; final epoch 15 failed again at 64.70%/41.43%. Compared with head-only best, its gains are 0.075 percentage points Dice, 1.87 box AP, and 2.84 mask AP. It trades lower forceps/gouge Dice for higher osteotome/pliers Dice.
5. **Joint optimization still deserves investigation.** The gentle run has an epoch-14 pre-clipping maximum gradient norm of 125,757 and mean 1,006. Gradients are clipped to one: these values are not the applied update norms. The train-loss jump at unfreezing also introduces detector loss and is not by itself numerical divergence. Each base run has three AMP-skipped steps in epoch one, with no further skips; no nonfinite-loss abort is recorded.
6. **Lower confidence is not an obvious fix.** For the original best model, saved PR curves retain the same box recall at score thresholds 0.5 through 0.2 while precision decreases. Misclassification/query selection needs inspection, not just permissive thresholds.
7. **Do not compare detector and base AP as the same metric implementation.** Standalone detector processing ranks query/class pairs; base evaluation uses one maximum-scoring class per query and a different preprocessing path. Its initial box AP is already 54.34% before mask training. That difference from standalone detector 68.08% does not establish detector degradation.
8. **Single-seed, selected-checkpoint results are provisional.** No uncertainty intervals or repeated-seed comparisons are available. The ten- versus fifteen-epoch base schedules also have different absolute warmup durations (about 130 versus 194 updates), so their frozen prefixes are not identical experiments. Selection over different numbers of epochs adds another source of optimism.
9. **Label/prompt ablations have not been established.** Switching from specific to broad labels also changed data coverage and splits. These runs do not prove broad labels intrinsically outperform specific labels, or that text conditioning outperforms learned/no-text alternatives.

## Dataset frozen for this baseline

The export contains **742 images, 747 object masks, 97 Sample IDs, and 20 physical Tool IDs**, using nine broad `tool_class` labels. Tool names, IDs and capture sessions remain metadata. Splits group by Sample ID, allow shared physical tools, and reserve training coverage for every exported class. The requested ratio is 70/15/15 and seed is 42.

| Split | Images | Masks | Sample IDs | Multi-tool images | Background images | Represented classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 518 | 523 | 68 | 5 | 0 | 9 / 9 |
| validation | 115 | 115 | 15 | 0 | 0 | 7 / 9 |
| test | 109 | 109 | 14 | 0 | 0 | 5 / 9 |

| Class | Train objects | Train samples | Train physical tools | Validation objects | Test objects |
| --- | ---: | ---: | ---: | ---: | ---: |
| bonepunch | 7 | 1 | 1 | 7 | 0 |
| forceps | 106 | 14 | 3 | 44 | 45 |
| gouge | 40 | 5 | 1 | 8 | 16 |
| mallet | 16 | 2 | 1 | 0 | 0 |
| osteotome | 72 | 9 | 3 | 16 | 8 |
| pliers | 35 | 6 | 2 | 8 | 8 |
| retractor | 56 | 7 | 1 | 8 | 0 |
| rongeur | 2 | 1 | 1 | 0 | 0 |
| scissors | 189 | 25 | 5 | 24 | 32 |

- Validation has no mallet or rongeur; test has no bonepunch, mallet, retractor or rongeur. Report those categories as unmeasured on the relevant split.
- Several validation categories have only one capture sample: their seven or eight images are related views, not seven or eight independent setups. Confidence in the estimates is limited.
- Physical Tool IDs and broader capture sessions are not held disjoint by this split policy. Performance does not establish generalization to unseen physical instruments or unseen photography sessions.
- Broad labels mask subtype gaps: training forceps include hunt_tumor, nonmagnetic_micro and rowes_maxillary; validation includes crile as well. Keep subtype coverage visible even when predicting only broad labels.
- There are five real multi-tool training images, all remaining validation/test images are single-tool, and there are zero empty images anywhere. Instance separation under overlap and empty-scene rejection are unmeasured.
- The builder excluded 141 images for missing Sample IDs and 70 without accepted review. Its full-source mask audit found 883 bundles/888 masks and no overlaps; those audit totals are not the exported training/evaluation totals.
- All positive mask instances in an included image must be exhaustively annotated. Audit class/instance consistency, thin structures, holes and visible occlusion boundaries when adding data.

## Highest-value data additions

Recover eligible images by assigning genuine missing capture groups; add independent setups for rongeur, bonepunch and mallet; ensure independent evaluation coverage for rare classes. Inspect retractor/gouge class/query errors before collecting more of their existing easy poses. Add real multi-tool scenes with touching/crossing tools, same-class instances and partial occlusion, plus reviewed target-free scenes. Vary lighting, glare, distance, out-of-plane pose and backgrounds using existing tools; acquire new physical instruments when feasible to expand physical diversity. More rotation views of an unchanged setup should not be counted as comparable independent diversity.

## How to compare after adding data

1. Keep these checkpoint directories intact. File hashes in `manifest.json` identify their current contents; weights are not duplicated into this archive.
2. Retain `frozen_dataset` as the legacy benchmark. It preserves current rows, schemas and split assignments with absolute external asset references. Images/masks are not copied: keep original assets immutable and verify `external_assets.sha256.json` before using the scores as comparable.
3. Do not move original held-out Sample IDs into training, including new images belonging to those samples. Keep an expanded, independently held-out benchmark as a separate version. If intentionally revising grouping or labels, declare a new benchmark rather than directly ranking raw scores against this table.
4. Evaluate old checkpoints and new models on the same legacy validation split and on the same expanded held-out split. A score change caused by a harder dataset is not a model-quality change. Keep an untouched final test set after development choices are settled; the present validation set has already been used for repeated model selection.
5. Keep evaluator code, label order, geometry, score handling, mask threshold and aggregation consistent. If fixing or changing evaluation, re-evaluate old and new checkpoints under the same updated implementation and retain both score versions.
6. Report supported-class macro AP, pooled AP, per-class quality, matched/oracle/union mask metrics, and results for empty/single/multi-tool scenes. Add AP75 if useful, repeated seeds, and uncertainty estimates that respect capture groups. Compare fixed training recipes to assess the effect of added data before simultaneously changing architecture, prompts, thresholds and optimization.
7. No new inference evaluations were run for this report. The archived prompts/embeddings and source are the current versions, not proof of historical identity: earlier training manifests did not content-hash their inputs. Treat the archive as a baseline captured now, not a reconstruction guaranteed byte-for-byte identical to every historical run.

Example: evaluate a selected base checkpoint on the frozen legacy split:

```bash
flowsis-evaluate-base \
  --model-path outputs/base/dfine-tool-class-epoch18-head10-v1/best \
  --dataset-path experiments/baselines/2026-10-30-tool-class/frozen_dataset \
  --split validation \
  --detection-threshold 0.5 \
  --mask-threshold 0.5 \
  --output-path outputs/base/dfine-tool-class-epoch18-head10-v1/legacy-baseline-validation.json \
  --device cuda \
  --no-detector-amp
```

## Archive contents and preservation

- `manifest.json`: recipes, selected-epoch metrics, standalone detector results, dataset counts, checkpoint hashes, input/source hashes and limitations.
- `dataset_build_report.json`, `frozen_dataset/`, `external_assets.sha256.json`: build policy, exported rows/splits and fingerprints of all 1,484 referenced image/mask files.
- `prompts/`: nine-class prompt JSON, Codex provenance and nine cached embedding tensors.
- `runs/`: saved run configurations, raw logs and normalized epoch metrics; the original base standalone validation JSON is included.
- `source/`, `workspace_changes.patch`: selected current source files and tracked workspace changes. This is not a complete environment/repository archive; configuration records the package versions and Git commit with dirty state.

Git retains the compact report and manifest (including all recipes, epoch metrics and standalone evaluation results), build report, external asset hashes, frozen dataset metadata, prompts and small embedding tensors. Explicit ignore exceptions include this curated snapshot's `.arrow` and prompt `.pt` files. Source duplicates, the workspace patch and raw `runs/` copies stay local; their useful metrics and recipes are already consolidated in `manifest.json`. The original detector log has two sequences and null bytes and remains preserved locally without repair. Nonfinite gradient-summary values are represented as strings in normalized JSON.

Back up the full local archive, referenced checkpoints and external media separately. Git does not back up model weights or media, and the frozen dataset still references external files. It is a reproducible split/annotation record, not a self-contained media dataset.
