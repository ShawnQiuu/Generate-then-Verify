"""Paired image statistics for the unchanged benchmark metric definitions."""
import numpy as np
from scipy.stats import binomtest

METRIC_NAMES = []
for group in ['standard', 'strict', 'relaxed'] + [f'Rule {index}' for index in range(1, 5)] + ['macro']:
    METRIC_NAMES.extend(f'{group}/{name}' for name in ['precision', 'recall', 'f1', 'accuracy'])
METRIC_NAMES += ['micro/precision', 'micro/recall', 'micro/f1', 'i_fpr', 'ihr', 'rhr', 'her', 'exact_set_accuracy']
PRIMARY_METRICS = ['standard/f1', 'strict/f1', 'relaxed/f1', 'micro/f1', 'rhr',
                   'Rule 1/f1', 'Rule 2/f1', 'Rule 3/f1', 'Rule 4/f1', 'exact_set_accuracy']


def features(prediction, truth, vlm):
    prediction, truth, vlm = [np.asarray(value, dtype=np.uint8) for value in (prediction, truth, vlm)]
    bits = np.array([1, 2, 4, 8], dtype=np.uint8)
    yes, target = (prediction[:, None] & bits) != 0, (truth[:, None] & bits) != 0
    per_rule = np.stack([yes & target, yes & ~target, ~yes & target, ~yes & ~target], axis=-1).reshape(-1, 16)
    positive, detected = truth != 0, prediction != 0
    overlap = (prediction & truth) != 0
    missing, extra = (truth & ~prediction) != 0, (prediction & ~truth) != 0
    tn = ~positive & ~detected
    standard = np.column_stack([positive & overlap, ~positive & detected, positive & ~overlap, tn])
    strict = np.column_stack([positive & ~missing & ~extra, extra, missing, tn])
    relaxed = np.column_stack([detected & ~extra, extra, positive & ~detected, tn])
    wrong_vlm = vlm & ~truth
    count_bits = np.array([int(value).bit_count() for value in range(16)])
    extra_features = np.column_stack([detected, count_bits[wrong_vlm & ~prediction],
                                      count_bits[wrong_vlm], prediction == truth])
    return np.column_stack([per_rule, standard, strict, relaxed, extra_features]).astype(float)


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def cm_metrics(cm):
    tp, fp, fn, tn = np.moveaxis(cm, -1, 0)
    return np.stack([divide(tp, tp + fp), divide(tp, tp + fn),
                     divide(2 * tp, 2 * tp + fp + fn), divide(tp + tn, tp + fp + fn + tn)], axis=-1)


def metrics(totals):
    totals = np.asarray(totals, dtype=float)
    rules = totals[..., :16].reshape(totals.shape[:-1] + (4, 4))
    per_rule = cm_metrics(rules)
    standard = cm_metrics(totals[..., 16:20])
    strict = cm_metrics(totals[..., 20:24])
    relaxed = cm_metrics(totals[..., 24:28])
    micro = cm_metrics(rules.sum(axis=-2))[..., :3]
    tp, fp = rules[..., :, 0].sum(axis=-1), rules[..., :, 1].sum(axis=-1)
    rates = np.stack([divide(totals[..., 17], totals[..., 17] + totals[..., 19]),
                      divide(totals[..., 17], totals[..., 28]), divide(fp, tp + fp),
                      divide(totals[..., 29], totals[..., 30]),
                      divide(totals[..., 31], totals[..., 16:20].sum(axis=-1))], axis=-1)
    return np.concatenate([standard, strict, relaxed, per_rule.reshape(totals.shape[:-1] + (16,)),
                           per_rule.mean(axis=-2), micro, rates], axis=-1)


def bootstrap(all_features, folds, count, seed):
    """Resample paired images within fixed outer folds; do not refit/reselect."""
    rng = np.random.default_rng(seed)
    size, methods, columns = all_features.shape
    flat = all_features.reshape(size, -1)
    output = []
    groups = [np.flatnonzero(folds == fold) for fold in sorted(set(folds))]
    for begin in range(0, count, 200):
        batch = min(200, count - begin)
        weights = np.zeros((batch, size))
        for indices in groups:
            weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=batch)
        totals = (weights @ flat).reshape(batch, methods, columns)
        output.append(metrics(totals))
    return np.concatenate(output, axis=0)


def paired_permutation(reference, candidate, count, seed, metric_names=None):
    """Swap complete rule predictions within images, via linear sufficient statistics."""
    rng = np.random.default_rng(seed)
    ref_total, new_total = reference.sum(axis=0), candidate.sum(axis=0)
    indices = [METRIC_NAMES.index(name) for name in (PRIMARY_METRICS if metric_names is None else metric_names)]
    observed = (metrics(new_total) - metrics(ref_total))[indices]
    difference = candidate - reference
    exceed = np.zeros(len(indices), dtype=int)
    for begin in range(0, count, 250):
        batch = min(250, count - begin)
        swaps = rng.integers(0, 2, size=(batch, len(reference))).astype(float)
        exchanged = swaps @ difference
        null_difference = (metrics(new_total - exchanged) - metrics(ref_total + exchanged))[:, indices]
        exceed += (np.abs(null_difference) >= np.abs(observed)[None, :] - 1e-12).sum(axis=0)
    return (exceed + 1) / (count + 1)


def mcnemar_rows(reference, candidate, truth):
    endpoints = []
    for index in range(4):
        target = (truth & (1 << index)) != 0
        ref, new = (reference & (1 << index)) != 0, (candidate & (1 << index)) != 0
        endpoints.append((f'Rule {index + 1}/accuracy', ref == target, new == target))
        endpoints.append((f'Rule {index + 1}/recall_on_GT_positive', ref[target], new[target]))
    safe = truth == 0
    endpoints += [('standard_correctness', ((reference & truth) != 0) | (safe & (reference == 0)),
                   ((candidate & truth) != 0) | (safe & (candidate == 0))),
                  ('exact_set_correctness', reference == truth, candidate == truth),
                  ('safe_image_specificity', reference[safe] == 0, candidate[safe] == 0)]
    rows = []
    for name, ref, new in endpoints:
        lost, gained = int((ref & ~new).sum()), int((~ref & new).sum())
        discordant = lost + gained
        p = float(binomtest(min(lost, gained), discordant, .5, alternative='two-sided').pvalue) if discordant else 1.
        rows.append(dict(endpoint=name, n=len(ref), reference_only_correct=lost, candidate_only_correct=gained,
                         discordant=discordant, accuracy_change=(gained-lost)/len(ref) if len(ref) else None,
                         p_value=p, identical_outcomes=bool(np.array_equal(ref, new))))
    return rows


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))))
    result = np.empty(len(values))
    result[order] = adjusted
    return result
