# Physical-tool and capture holdouts

Prepared with Codex assistance on 2026-11-14. This proposal supersedes the earlier suggestion to normalize new labels to the old checkpoint spellings. New datasets should preserve underscore-separated labels such as needle_holder, bone_punch and suction_tube. Historical checkpoints/metrics retain their original taxonomy and need explicit compatible remapping for comparison.

## Current metadata check

Both image_manifest.csv and image_object_manifest.csv now have zero missing Sample IDs. Image Tool IDs and object Tool IDs are also populated. The image manifest has 144 blank Angle fields; these should be reviewed against capture conventions, since randomized multi-tool images do not necessarily have one meaningful angle. Do not fabricate angles solely to fill fields. One remaining video_manifest.csv class is needleholder; change it to needle_holder through the source-data workflow before regenerating. Existing frozen exports and model bundles are historical artifacts, not the new labeling authority.

Active detector/base/augmentation-preview defaults, the underlying rotation sampler, and current experiment commands now use -22.5 through +22.5 degrees. This is a 45-degree total range. Historical run records retain the actual 0–45-degree settings used then. Exporter augmentation recommendations live in the separate data repository and are not the source of trainer behavior.

## Why distinguish multi-tool and overlap?

Two tools can be present with no occlusion. Instance count alone cannot identify cases where one hides part of another. Add an evaluation tag such as tool_relation = separated / touching / occluding / uncertain (and optionally tool count). This is a small scene-level categorization, not a request to redo accepted masks. Use visible-surface masks consistently: actual opaque tools do not share the same visible image pixels, so intersecting instance masks is not a reliable overlap label. Neither intersecting bounding boxes nor a small gap proves physical occlusion. Only use an occlusion severity fraction if hidden full shapes are actually annotated; otherwise use a reviewed categorical judgment. Keep tags in an audit sidecar if the export schema has no field for them.

## Recommended evaluation cohorts

1. Train: known physical Tool IDs, selected capture Sample IDs and complete videos.
2. Validation: reserved captures of known tools for selection; add separate validation physical Tool IDs if the collection is large enough to tune novel-tool generalization without touching test identities.
3. Test-known-tools: entirely held-out captures/videos of tools whose other captures are in training. Tests robustness to new views/setup captures, with class labels unchanged.
4. Test-unseen-tools: physical Tool IDs absent from all project training, including detector pretraining/fine-tuning, synthetic cutout sources, base training and temporal replay. Their coarse categories should remain represented in training. Tests new physical instances within known categories; it does not establish recognition of novel classes.

Report the two tests separately, per class and per tool/capture group, including real overlaps and empty scenes. Use identical frozen checkpoints for both. Consider session generalization as an additional axis: a new Sample ID does not guarantee a new background/session. Record whether each test capture session was seen in training rather than conflating capture and session independence.

Implemented: the data builder now accepts --split-by tool-sample, --unseen-tool-size, --capture-grouping and --unseen-tool-ids. It exports train, validation, test_known and test_unseen plus split_assignments.json. See /home/pricyspark/flowsis-data/TOOL_SAMPLE_SPLITS.md. Existing legacy modes remain available. The second-stage GroupShuffleSplit draws include coverage constraints and row-count balancing; these use metadata only. Test-unseen metrics need target filtering by the reported unseen Tool IDs to exclude known companions from novel-target claims.

## Allocation procedure

- Inventory eligible physical Tool IDs by canonical class and their capture groups/videos. Select novel-tool test identities within each class where enough distinct tools exist; keep at least one train tool per evaluated class.
- Quarantine every image/frame containing a novel-test tool before assigning training. A mixed known/unseen-tool image must not enter training. Keep the entire capture group/video outside training; classify its targets as known versus unseen where relevant. Known companions are allowed in the novel-tool test but their metrics must not be silently pooled with unseen targets.
- Divide remaining capture groups/videos among training, validation and known-tool testing. Keep all angles, adjacent frames, derived crops and synthetic descendants with their parent split. Do not merge every tool across every mixed scene into one giant component by accident; the quarantine rule and cohort definitions should be explicit.
- If a capture group contains an unseen tool only in some images, withhold all images in that group. Such grouping constraints and uneven recording sizes alter the requested proportions.
- Your example gives 0.9 × 0.9 = 81% training before a separate validation allocation, only under equal data-per-tool assumptions. Ten percent of Tool IDs is not necessarily ten percent of images. Report both identity counts and row counts; reserve validation separately. Prefer coverage over exact percentages on this small dataset.
- Single-tool classes cannot support within-class unseen-tool testing while retaining that class in training. Mark them unmeasured or collect more physical tools. Consider several fixed, group-disjoint folds when few test identities make a single split unstable; report variability without tuning on test folds.
- Existing project checkpoints cannot establish an unseen-tool claim for physical tools they already encountered. Retrain from initialization without those project examples, or collect genuinely new physical tools. Foundation-model pretraining provenance is a separate limitation; define the claim as held out from project training rather than guaranteeing the object never appeared anywhere online.

## Prior art

The Washington RGB-D Object Dataset explicitly tests category recognition by leaving out one physical object instance per category, and distinguishes this from new views of known instances:
https://rgbd-dataset.cs.washington.edu/dataset.html
https://rse-lab.cs.washington.edu/papers/rgbd-dataset-icra-11.pdf

GroupShuffleSplit is a standard implementation primitive for holding out group identities. Its requested test_size refers to groups, not samples; grouping by tool and then capture requires explicit two-stage orchestration:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

These support the evaluation principle. The exact proposed multi-tool, two-cohort allocation above is a recommendation for this project, not a claim to reproduce either benchmark's complete protocol.


## Regeneration command

```bash
conda activate flowsis
cd /home/pricyspark/flowsis-data
python build_dataset.py \
  --output-path exports/combined-tool-sample-coarse-v3 \
  --label-fields tool_class \
  --split-by tool-sample --unseen-tool-size 0.1 \
  --capture-grouping sample \
  --validation-size 0.1 --test-size 0.1 \
  --video-sections 0 --seed 42
```

Requires scikit-learn in the builder environment (added to flowsis-data/environment.yml). Change capture-grouping to session for stricter acquisition separation. Requested fractions are group fractions; actual frame fractions can differ substantially when a few long videos must remain whole. The report warns when free groups cannot support condition coverage. Export regeneration was not run during implementation; validation used fixture exports and a live metadata simulation, not a new full mask audit.
