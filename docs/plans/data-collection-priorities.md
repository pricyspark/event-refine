# Collection priorities for the next 1–2 weeks

Prepared with Codex assistance on 2026-11-15 from the dev-v4 validation results and live source manifests. Numerical targets are practical collection goals, not statistical power calculations.

## What to prioritize

1. Independent weak-class examples and real multi-tool scenes. Gouge matched Dice is about 48%; needle_holder about 70%; suction_tube about 65–70%. Multi-tool Dice is 56–62%, versus about 89–90% for single-tool scenes. Multi-tool validation currently contains only two capture groups. Photograph side-by-side, touching and physically occluding tools separately; include same-class pairs and visually confusable classes. Label all visible instruments and retain their individual IDs. A correct foreground union does not establish correct instance/class separation.
2. Empty scenes, entry/exit and occlusion/reappearance. There are no exported negative/background scenes. Capture a scene empty, instrument entering, pausing/moving, becoming partly/fully occluded by a non-tool occluder, reappearing, leaving, then empty again. Include realistic hands/gloves, glare and ordinary nontool distractors. These test confirmation/fallback and false positives; data alone does not repair the current runtime warm-up policy.
3. More class/physical-instance/session diversity. Bone_punch, pliers and retractor have no validation positives. Accepted stills for bone_punch, gouge, mallet, pliers, retractor and suction_tube each use only one physical Tool ID. Borrow additional physical specimens where possible. Prioritize gouge, suction_tube, needle_holder, bone_punch, pliers and retractor; include mallet failure cases. Record genuinely separate setups/sessions, not just adjacent angles of an unchanged setup.
4. Broader usable video. Existing videos are four forceps, three scissors, one mallet and one needle_holder; the needle_holder has only one accepted frame and is unusable as a temporal clip. Start with contiguous reviewed needle_holder clips, then gouge and suction_tube, plus the classes missing validation. Revisit mallet video video-458f2ba49f1eabc2 frames 278–282 before recreating its difficult conditions; the cause of that failure remains undiagnosed.

## Capture checklist

- Record Tool ID(s), canonical class, Sample ID and true capture-session ID.
- Vary distance, out-of-plane pose, lighting/reflections, background and camera setup realistically. Preserve the original images; training rotations are already -22.5° to +22.5°.
- Multi-tool stills: include clear reference views, separated/touching/occluding configurations, and same-class pairs. Tag the relationship, not just object count. Annotate visible surfaces and distinguish instances; do not invent hidden mask regions.
- Video: capture short complete events, including initial and final empty portions. Maintain track identity through occlusion, and distinguish occluded from actually absent. Review contiguous event windows rather than isolated first frames. The builder currently exports only the first contiguous accepted run.
- Current video annotation/export supports one target instance per video and at most one mask. Collect multi-tool raw videos if useful, but do not treat them as fully annotated multi-instance training data yet. That requires extending the workflow. Use fully annotated multi-tool stills now, and single-tool video with nontool occlusion for immediate temporal experiments.

## Practical one-week targets

- 12–20 new multi-tool setups, roughly 4–6 selected stills each, across several sessions.
- 20–30 short videos, roughly 8–15 seconds each, emphasizing weak classes and complete entry/occlusion/exit events. Count distinct setups and tools, not raw adjacent frames.
- 30–50 explicitly reviewed empty/distractor stills across at least five setups, plus empty segments within videos.
- At least 3 separate capture setups for each class missing validation, preferably across different real sessions. Seek 2–3 physical specimens for single-specimen classes where available.

These goals overlap: a needle_holder recording with entry/occlusion/exit satisfies multiple needs. Annotation throughput determines the final usable total; prioritize correct complete evaluation events over large unreviewed collections.

## Use week two to fill coverage and protect evaluation

Assign new recordings/setups to train, validation or test before extracting adjacent frames. Freeze independent sessions for final evaluation; if claiming unseen physical tools, reserve their IDs across every training stage and synthetic source. Ensure the final test represents weak classes, genuine overlaps and negative scenes; current unseen-tool testing covers only scissors/forceps identities. Have another teammate check all evaluation masks, class IDs and difficult occlusions.

A workable division: one person collects and logs sessions, one annotates, one independently reviews and runs export checks; the programmer inspects model failures and confirms pipeline support. Do not spend most of the remaining time recording more easy forceps/scissors rotations or annotating thousands of nearly identical frames.

Evidence: outputs/overnight/dev-v4/REVIEW.md and base/*/validation.json; flowsis-data source manifests and dataset_video.py/video_data.py.
