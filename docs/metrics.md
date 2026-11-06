# Metric definitions

Let G be the set of ground-truth rules, P the predicted rule set, and V the same VLM's original rule set. Only Rule 1–Rule 4 enter evaluation. `Other` is ignored. Empty-denominator rates are zero, matching the original implementation.

| Image regime | TP | FP | FN | TN |
| --- | --- | --- | --- | --- |
| Standard | G nonempty and P intersects G | G empty and P nonempty | G nonempty and no overlap | Both empty |
| Strict | G nonempty and P=G | P contains any extra rule | G contains any missing rule | Both empty |
| Relaxed | P is a nonempty subset of G | P contains any extra rule | G nonempty and P empty | Both empty |

Strict FP and FN can both count the same image. Strict accuracy is consequently not exact-set accuracy. Exact-set accuracy is separately calculated as the fraction of images for which P=G.

Per-rule precision, recall, F1 and accuracy use binary rule-level confusion counts. Macro values average the four rules; micro precision/recall/F1 pool their counts.

| Rate | Definition |
| --- | --- |
| i-FPR | Safe images with any prediction / all safe images |
| iHR | Safe images with any prediction / all images with any prediction |
| rHR | Total false rule predictions / total rule predictions |
| HER | Count of wrong VLM rules removed, `(V − G) − P`, / count of wrong VLM rules, `V − G` |

iHR does not mean the fraction of reports containing any wrong rule. Standalone-VLM HER is zero by construction; CV-only HER is inapplicable. The historical GPT structural-zero HER test is retained only for parity with the earlier statistical protocol, not as evidence of a meaningful difference from a learned baseline.

Machine-readable values are fractions on 0–1. Multiply a rate by 100 to display it as a percentage. Pooled metrics use all selected images. Five-subset means and sample SDs are calculated independently of pooled metrics, with `ddof=1`; subset SD is not model-generation variability.
