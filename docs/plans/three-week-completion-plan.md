# Three-week project and paper plan

Prepared with Codex assistance, 2026-11-13, using the coarse-v2 training review and paired generation benchmark. Planning window: November 16–December 4, 2026; adjust working days for team availability. Assumes one shared training GPU and four team members with roughly consistent availability. Collection targets below are practical starting targets, not statistical power calculations.

## Intended finish

A reproducible detector → base → temporal inference pipeline, a frozen model bundle, an independent capture-held-out evaluation, and a complete internal paper draft with figures. This is a three-week completion plan, not a promise of publication acceptance or a positive temporal result. Keep coarse labels; defer fine-grained recognition, new architectures, deployment polish, and large parameter searches.

## Priorities and evidence

1. Evaluation credibility and real challenging data. Current validation shares source recordings with training, has zero real multi-tool scenes, and lacks positives for four of eleven labels. The old/new comparison covers only 31 stills from four capture groups/classes. Split by source recording/session before frame extraction; keep related stills, bursts, and derived synthetic images with their source split. Do not call previously examined test examples untouched. Reserve new capture sessions as the final test.
2. Correct temporal training and attribution. Original temporal training had incorrect pixel scaling and is not valid evidence of learned improvement. The fix exists; a successful corrected training/evaluation run is still needed. Include tracking without learned refinement to distinguish tracking benefits from learned mask refinement. Report cold starts as well as later frames.
3. Class/query selection and real overlaps. Bonepunch and gouge matched Dice are about 27.5% and 61% despite substantially better oracle masks. Diagnose annotation/class confusion, query selection, confidence, and detection misses before adding epochs. Synthetic overlap cannot replace real held-out overlaps. Prioritize real overlap in collection and annotation; compare its training inclusion with synthetic augmentation using matched budgets.
4. Reproducibility and writing. One seed and correlated frames limit confidence. Write methods and dataset sections while experiments run; reserve the final days for results and packaging.

## Roles

- A — programmer/data scientist: split/audit scripts, input correctness, training recipes, evaluation, aggregation, reproducibility. Owns scientific interpretation and code changes.
- B — collection lead: capture protocol, balanced classes and difficult sequences, session/group IDs, collection log. Later writes dataset collection section and supplies examples.
- C — annotation/QA lead: annotation conventions, mask/class corrections, difficult examples, cross-checking. Later writes annotation and quality-control sections.
- D — experiment operator/documentation lead: runs A's frozen scripts, records commands/checkpoints/logs, checks completion, maintains results sheet. Also annotates and independently reviews a sample of C's work; later assembles figures and paper references.

The whole team reviews claims and failures. D runs approved commands without changing splits, thresholds, or hyperparameters ad hoc; unexpected results go to A. B and C cross-review difficult labels. No one tunes on final test results.

## Week 1 — trustworthy data and a working temporal baseline

### Days 1–2

A: Audit existing split membership by recording/session and capture group, not frame count. Publish a coverage table by class, real overlap, tool count, empty scene, and source group. Decide primary metrics and checkpoint-selection rules before new runs. Run the corrected temporal recovery as a development diagnostic using `bash experiments/retrain_temporal_coarse_v2.sh`; verify checkpoint preprocessing marker, first-frame base/runtime agreement, unchanged flow inputs, finite gradients, and saved-model reload. A final-data temporal model will need retraining after the new split is frozen.

B: Start new independent sessions covering all claimed classes, with extra bonepunch/gouge and visually confusable tools. Include single-tool, two/three-tool overlaps at varied occlusion, crossings, entry/exit, reappearance, fast motion, and true empty scenes. Keep realistic camera/background variation; document session and source recording.

C: Review bonepunch/gouge labels and masks, define occlusion/mask conventions, and pilot annotation on difficult multi-tool clips. D: Set up run ledger and annotate/review the pilot; start paper outline and experiment table.

### Days 3–5

Team: Collect toward 40–60 genuinely separate short recordings/capture groups if feasible. Count independent groups, not thousands of adjacent frames. As an initial allocation aim for roughly 60/20/20 train/validation/test by group while deliberately checking class and difficult-scene coverage; adjust before freezing. Aim for at least 10 new held-out test recordings, including at least 5 with real overlaps, with negative/empty segments also represented. These are feasibility targets; record actual counts and narrow claims if coverage is inadequate. Maintain separate still-image capture groups where appropriate. Dense annotation of short evaluation subsequences is needed for frame-position analysis; sparsely annotate more diverse training frames. Preserve object identities in evaluation clips for reliable target correspondence.

A: Freeze dataset v3 manifest and group split by day 5, with hashes, class map, seed, annotation version, and split audit. Reserve the test manifest and avoid model inspection on it. Validate dataloader shapes, pixel ranges, mask alignment, class mapping, augmentation provenance, and rotations within the agreed 45-degree range; explicitly document whether that means ±45° or a 45° total span and implement the chosen convention consistently.

B/C/D: Complete annotation and independent review of all validation/test labels and difficult training examples; review a random sample of ordinary training annotations. Draft dataset and annotation sections.

Week-1 gate: no group leakage, usable validation/test coverage for each claimed class/condition, corrected temporal smoke passes and reload works. If a class has insufficient independent examples, collect more or explicitly exclude its performance claim. If behind schedule, reduce experiment count before weakening evaluation.

## Week 2 — small, controlled experiment matrix

A sets commands and selects checkpoints using validation only; D executes/records them. Use one strong detector and a frozen base initialization for paired temporal comparisons. Reuse existing checkpoints only when their training provenance is compatible with the new held-out sessions; otherwise retrain. Benchmark the historical checkpoint on supported shared labels, documenting its different training data and initialization.

Base experiments, all with the same splits, rotations, initialization, update budget, sampling policy, and selection metric:

- B0: real training data, synthetic overlap off.
- B1: same real data, synthetic overlap probability 0.25.
- B2, optional: B1 with extra sampling of real-overlap groups, using a stated multiplier and fixed total updates. Check this does not repeatedly sample only a few nearly identical frames. Real scenes remain the primary validation evidence.

Temporal experiments, paired with the same selected base and the same sequence reset/target selection policy:

- T0: stateless base inference.
- T1: tracker/association with identity refinement.
- T2: tracker/association plus corrected learned refinement.

Use oracle target matching only as a diagnostic, clearly separate from runtime results. If testing base fallback during tracker warm-up, compare policies explicitly and check false positives on empty scenes; do not quietly change policies between T1 and T2. Include the standalone base's actual cold-start behavior in end-to-end comparisons.

Repeat the final base control/augmentation pair and T1/T2 comparison with a second training seed if compute allows; prioritize this over a broad learning-rate sweep. Keep training budgets matched and report individual runs. B/C continue targeted training-only collection from validation failures early in the week, with any dataset revision frozen before the final runs; never add test-derived examples. D produces validation tables and logs runtime/memory on one documented device, with warm-up and batch size held constant.

By days 9–10: freeze final recipes/checkpoints and analysis code; complete methods and related-work drafts with verified citations. If learned refinement is still harmful or negligible, report that finding and use the stronger pipeline for the final demo. Do not spend week 3 inventing a new temporal architecture.

Week-2 gate: reproducible base and temporal inference, completed minimum B0/B1/T0/T1/T2 validation comparisons, selected checkpoints, and draft methods. Record incomplete optional runs rather than delaying the core result.

## Week 3 — final evaluation, paper, handoff

Days 11–12: Evaluate frozen models on the new test sessions after recipes are fixed. Treat this as the final evaluation campaign; reruns for verified software errors must be disclosed. Evaluate the historical model on supported shared classes and identical examples with name-based remapping; do not score nonexistent historical labels as learned capabilities. A analyzes results; D exports tables and figures; B/C independently check surprising failures without changing annotations based on which model won. Correct genuine annotation errors through a documented, model-blind review applied consistently.

Report:

- Detector box AP50 and AP75, and mask AP50 and matched Dice for the base; include misses under the documented matching rule.
- Per-class, per-recording, single/multi-tool, real overlap, empty-scene false positives, and difficult-condition results with sample/group counts. Separate frame-weighted averages from equal-recording averages.
- Paired temporal-minus-base and learned-minus-tracking-only changes, frame positions 0/1/2/etc., cold-start and later-frame summaries, reset rules, and counts at each position. Restrict long-position comparisons to common eligible sequences or show the changing cohort; otherwise apparent improvement may reflect which clips remain.
- Recording/group-level bootstrap confidence intervals for paired changes where there are enough independent groups; never treat adjacent frames as independent replicates. Show individual recording outcomes when counts are small. Report seed variability separately.
- Oracle quality versus runtime errors, throughput/latency and memory on stated hardware, and representative successes/failures chosen by a documented rule.

Days 13–14: Assemble 5–6 paper figures: pipeline/dataset/split overview; historical versus current on common labels; per-class and real-overlap results; augmentation control; temporal delta by position and recording; qualitative successes/failures. A writes results/discussion, B/C finish dataset/annotation sections and limitations, D formats methods tables, figures, references, and reproducibility appendix. Write the abstract after the results are fixed. Explicitly disclose assisted analysis/writing according to the target venue's policy and retain provenance.

Day 15: Everyone reviews the complete draft and checks that each claim has a supporting table or figure. D performs a clean-environment model restore and scripted inference on a small packaged example; A fixes blocking issues. Archive complete base folders, paired corrected temporal checkpoints, optical-flow weights, code revision, environment, configs, split manifests, raw result files, commands, licenses/provenance, and checksums. Do not include private raw data in a public bundle without an appropriate release decision.

Finish: internal paper draft, final figures and result tables, working base/temporal demo, documented limitations, model bundle and reproduction instructions. Submission formatting and further experiments can follow.

## Scope cuts if time runs short

Keep the corrected temporal result, independent evaluation, B0/B1 and T0/T1/T2 controls, honest negative findings, and a complete draft. Cut B2, additional detector candidates, third seeds, parameter sweeps, fine-grained classes, extra qualitative galleries, and UI/presentation polish first. A small credible evaluation is more useful than a large correlated one. If adequate held-out recordings cannot be obtained, frame the paper as a preliminary study and state that limitation plainly.

## Current source records

- outputs/overnight/coarse-v2/REVIEW.md
- outputs/presentation/coarse-v2/BENCHMARK_RESULTS.md
- outputs/presentation/coarse-v2/DEFERRED_WORK.md
- outputs/presentation/coarse-v2/SAVE_AND_LOAD.md
