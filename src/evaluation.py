"""Pooled metrics, fixed-subset summaries and optional paired statistics."""
import json

import numpy as np

from data_io import write_csv, write_json
from fusion import RULES, ABLATIONS, mask, predict, vlm_rules
from paired_statistics import METRIC_NAMES, features, metrics, bootstrap, paired_permutation, mcnemar_rows, holm


def evaluate(ids, gt, logs, evidence, config, folds, model, output, ablations=False, statistics=False):
    predictions = {iid: predict(evidence[iid], logs[iid], config, ablations) if evidence is not None
                   else {'vlm': vlm_rules(logs[iid])} for iid in ids}
    names = list(predictions[ids[0]])
    arrays = {name: np.array([mask(predictions[iid][name]) for iid in ids], dtype=np.uint8) for name in names}
    truth = np.array([mask(gt[iid]) for iid in ids], dtype=np.uint8)
    folds = np.asarray(folds)
    values = np.stack([features(arrays[name], truth, arrays['vlm']) for name in names], axis=1)
    pooled = metrics(values.sum(axis=0))
    present = sorted(set(folds))
    by_fold = np.stack([metrics(values[folds == fold].sum(axis=0)) for fold in present])
    records = []
    for index, name in enumerate(names):
        for k, metric in enumerate(METRIC_NAMES):
            na = name == 'cv_only' and metric == 'her'
            records.append(dict(method=name, metric=metric, pooled=None if na else float(pooled[index, k]),
                                fold_mean=None if na else float(by_fold[:, index, k].mean()),
                                fold_sample_sd=None if na or len(present) < 2 else float(by_fold[:, index, k].std(ddof=1))))
    write_csv(output / 'metrics.csv', records)
    write_csv(output / 'fold_metrics.csv', [dict(method=name, metric=metric, fold=fold,
              images=int((folds == fold).sum()), value=None if name == 'cv_only' and metric == 'her' else float(by_fold[f, m, k]))
              for m, name in enumerate(names) for k, metric in enumerate(METRIC_NAMES) for f, fold in enumerate(present)])
    write_csv(output / 'fold_support.csv', [dict(fold=fold, images=int((folds == fold).sum()),
              safe_images=int(((truth == 0) & (folds == fold)).sum()),
              **{f'rule_{k + 1}_positives': int((((truth & (1 << k)) != 0) & (folds == fold)).sum()) for k in range(4)}) for fold in present])
    with (output / 'predictions.jsonl').open('w') as handle:
        for name in names:
            for iid in ids:
                handle.write(json.dumps(dict(image_id=iid, model=model.upper(), condition=name,
                            gt=sorted(gt[iid]), detected=sorted(predictions[iid][name]), vlm=sorted(vlm_rules(logs[iid])))) + '\n')
    report = dict(images=len(ids), methods=names, units='All metrics are fractions on 0-1.',
                  protocol='Descriptive fixed-subset evaluation; configuration previously selected on all 807 development labels.',
                  metrics=records,
                  confusion_matrices={name: dict(zip(RULES, values[:, m, :16].sum(axis=0).reshape(4, 4).astype(int).tolist())) for m, name in enumerate(names)},
                  confusion_order=['TP', 'FP', 'FN', 'TN'])
    if statistics:
        if len(ids) != 807 or present != list(range(5)) or 'fixed_chd960_f1' not in arrays:
            raise ValueError('Publication statistics require all 807 images and the final method')
        if model == 'qwen' and not ablations:
            raise ValueError('Qwen publication statistics require --ablations for the joint 355-test family')
        report['statistics'] = paired_analysis(names, arrays, truth, values, folds, model, output)
    write_json(output / 'metrics.json', report)
    return report


def paired_analysis(names, arrays, truth, values, folds, model, output):
    seed = 20261028 if model == 'qwen' else 20261027
    boot = bootstrap(values, folds, 20000, seed)
    points = metrics(values.sum(axis=0))
    ci = np.quantile(boot, [.025, .975], axis=0)
    write_csv(output / 'metric_intervals.csv', [dict(method=name, metric=metric,
              lower=None if name == 'cv_only' and metric == 'her' else float(ci[0, m, k]),
              upper=None if name == 'cv_only' and metric == 'her' else float(ci[1, m, k]))
              for m, name in enumerate(names) for k, metric in enumerate(METRIC_NAMES)])
    references = ['vlm'] + (ABLATIONS if model == 'qwen' else [])
    permutation, binary = [], []
    candidate = 'fixed_chd960_f1'
    c = names.index(candidate)
    for index, reference in enumerate(references):
        r = names.index(reference)
        p = paired_permutation(values[:, r], values[:, c], 100000, seed + index + 1, METRIC_NAMES)
        delta_ci = np.quantile(boot[:, c] - boot[:, r], [.025, .975], axis=0)
        for k, metric in enumerate(METRIC_NAMES):
            if metric == 'her' and model == 'qwen' and reference in {'vlm', 'cv_only'}:
                continue
            permutation.append(dict(reference=reference, candidate=candidate, metric=metric,
                                    delta=float(points[c, k] - points[r, k]), lower=float(delta_ci[0, k]),
                                    upper=float(delta_ci[1, k]), p_value=float(p[k])))
        binary.extend(dict(reference=reference, candidate=candidate, **row) for row in mcnemar_rows(arrays[reference], arrays[candidate], truth))
        print(f'Paired statistics: {index + 1}/{len(references)}', flush=True)
    tests = permutation + binary
    for row, adjusted in zip(tests, holm([row['p_value'] for row in tests])):
        row['p_holm'] = float(adjusted)
    write_csv(output / 'paired_metric_tests.csv', permutation)
    write_csv(output / 'mcnemar_tests.csv', binary)
    return dict(seed=seed, bootstrap_samples=20000, permutation_samples=100000, holm_family_size=len(tests),
                scope='Conditional on fixed predictions; pointwise confidence intervals. GPT structural-zero HER test retained only for archive parity.')
