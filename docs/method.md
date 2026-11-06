# Frozen verification and paper ablations

`configs/final.json` contains the active final parameters. There is no threshold search, learned fusion, peer-VLM feature, or calibration fitting in this release.

## Parameters

| Component | Setting |
| --- | --- |
| YOLO input | OpenCV BGR, size 960 |
| Extraction floors | Person 0.20; CHD objects 0.03; DINO box/text 0.20/0.20 |
| Rule 1 | Person 0.65; no-hardhat 0.04; head center in upper 40%; one vote per person; independent override ≥0.20, count 1 |
| Rule 2 | Same-VLM elevated-work context and Rule 2 claim; no required CV harness confirmation |
| Rule 3 | Same-VLM pit/edge context; person 0.55; guardrail 0.55; protection distance <1.0× person height |
| Rule 4 | VLM AND CV; person 0.20; machinery 0.08; distance <2.5× person height; machinery height ≥1.5× person height; bottom alignment ≤1.5× person height |
| Person filtering | Minimum height 0.01× image height |
| Machinery area | Strictly between 0.0001 and 0.64 of image area |
| Extra suppression | IoU 1.0, preserving source detections after source-model suppression |

No-hardhat detections are assigned to the smallest qualifying person box. The highest-confidence detection per person contributes one vote. Rule 4 likewise contributes at most one qualifying machinery pair per person. Distances are shortest box-edge distances in two-dimensional image coordinates.

The guardrail and harness GroundingDINO prompts are retained in the final configuration. Harness detections support CV-only and paper ablation calculations; their presence does not imply that the final Rule 2 acceptance decision requires them.

## Saved detector evidence

`reproduce --evidence` accepts JSONL with one object per image:

```json
{"image_id":"0000007","width":2048,"height":1536,"cv_detected_objects":{"people":[],"no_hardhats":[],"excavators":[],"safety_harnesses":[],"guardrails":[]}}
```

Each detection has `class`, `confidence` and an absolute-pixel `bbox` in `[x1,y1,x2,y2]` format. Coordinates are clipped to the image before geometric filtering. Detector outputs are model evidence, not human GT. `detect` writes this format.

## Ablations

`--ablations` enables the six final Qwen paper comparisons with unchanged per-rule geometry and thresholds:

| Condition | Intervention |
| --- | --- |
| `cv_only` | Raw CV rule claims; no VLM claims or scene gates |
| `all_and` | Intersection of VLM and raw CV claims, without semantic gates |
| `all_or` | Union of VLM and raw CV claims, without semantic gates |
| `vote` | Fixed original paper source/context policy, not a fitted or majority vote |
| `gate_off` | Disable only Rule 2/3 semantic scene gates |
| `r4_vlm` | Replace only Rule 4 with the same VLM's claim |

The Vote policy uses CV for Rules 1 and 4. Rule 2 accepts a VLM claim, or a CV claim with elevated-work context. Rule 3 requires a CV claim and either pit/edge context or a VLM Rule 3 claim.

All five evaluation subsets use identical parameters. The historical full-development-set parameter selection is disclosed in the root README; this release does not implement a new selection-validation study.
