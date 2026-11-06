# ConstructionSite-RB Benchmark

> Status: **placeholder**. The benchmark release, annotation files, and evaluation scripts will be published with the v0.4 milestone (see the Roadmap in the top-level `README.md`). Details below describe the benchmark as used in the manuscript and may change before release.

## Provenance

`ConstructionSite-RB` (Rule-Balanced) is a re-annotated evaluation subset derived from the publicly available **ConstructionSite10K** dataset. The framework is training-free and therefore not tied to the original train/test splits; images were sampled globally to obtain a rule-balanced evaluation set.

- **807 images**, covering the four explicit safety rules plus completely safe scenes.
- The original hazard/safe split is heavily skewed (12.76% hazard / 87.24% safe); the re-annotated subset is close to balanced (64.19% hazard / 35.81% safe).
- The original rule definitions include elements that are frequently occluded or imperceptible to cameras (e.g. safety glasses, footwear). The rules were refined into visually computable criteria, so that every violation can be backed by explicit geometric evidence.

### Hazard distribution (percentage of hazard images, multi-label)

| Category | ConstructionSite10K | ConstructionSite-RB |
| --- | :---: | :---: |
| Rule 1 — missing PPE | 78.33 | 40.93 |
| Rule 2 — height work risk | 6.57 | 18.92 |
| Rule 3 — edge protection risk | 13.46 | 39.77 |
| Rule 4 — machinery proximity | 5.48 | 22.39 |

## Rule definitions

| Rule | Hazard category | Visually computable criterion |
| --- | --- | --- |
| Rule 1 | Missing PPE | Personnel not wearing a hard hat. |
| Rule 2 | Height work risk | Workers at visually significant elevated heights lacking safety harnesses or lanyards. |
| Rule 3 | Edge protection risk | Unprotected deep pits, trenches, or steep edges lacking guardrails or fences, exposing nearby workers to fall hazards. |
| Rule 4 | Machinery proximity | Workers standing dangerously close to excavators, e.g. within blind spots or operation radii. |
| Safe | No violation | None of the above rules holds. |

## Annotation protocol

All 807 images were manually re-annotated at the rule level by a Ph.D. researcher specializing in Civil Engineering, using the refined Rule 1–4 criteria above. Ground truth is therefore rule-level and scene-level rather than object-level, which allows report-oriented metrics to be computed directly.

## Evaluation regimes

Let `Y_i` be the ground-truth set of rules present in image `i` and `Yhat_i` the set of verified risks predicted by the system. True positives are defined per image as:

| Regime | True positive condition | Intent |
| --- | --- | --- |
| Standard | `Yhat_i ∩ Y_i ≠ ∅` | Basic alarm behaviour: at least one correct risk found. |
| Strict | `Yhat_i = Y_i`, with `Y_i ≠ ∅` | Exact scene comprehension; penalizes omissions and spurious alerts. |
| Lenient | `Yhat_i ⊆ Y_i` and `Yhat_i ≠ ∅` | False-alarm suppression; tolerates missed detections. |

For safe images (`Y_i = ∅`), correctly producing no alert counts as a true negative under all regimes. Precision, recall, F1, and accuracy are computed from these image-level definitions, and independently at the rule level.

## Trustworthiness metrics

| Metric | Definition | Reading |
| --- | --- | --- |
| **i-FPR** | Share of ground-truth safe images that trigger an alert. | Operational false-alarm rate on safe scenes. |
| **iHR** | Share of image-level alerts that are unsupported by ground truth. | Reliability of the emitted alerts. |
| **rHR** | False-positive rules divided by all detected rules. | Rule-level false discovery rate. |
| **HER** | Share of VLM-generated unsupported hypotheses removed by verification. | Effectiveness of the verification hub. |

HER measures only the removal of hypotheses produced by the VLM, not the overall false-positive rate of the final system.

## Release and licensing

ConstructionSite10K is **not** redistributed with this repository. The re-annotated ConstructionSite-RB annotations and image list will be released subject to the terms of the source dataset; image files themselves will not be mirrored here. Download ConstructionSite10K from its official repository and follow the mapping file once published.
