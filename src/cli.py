"""Command-line entry points for the final paper release."""
import argparse
import json
import os
from pathlib import Path


from data_io import read_gt, read_logs, read_jsonl, read_folds, sha256, valid_id, write_json
from evaluation import evaluate
from project_paths import ROOT, resolve


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=['check', 'eval', 'reproduce', 'detect', 'validate', 'generate'])
    result.add_argument('--model', choices=['qwen', 'gpt'], default='qwen')
    result.add_argument('--logs', help='One VLM: JSONL file or directory of seven-digit JSON logs')
    result.add_argument('--evidence', help='Saved detector evidence JSONL for reproduce')
    result.add_argument('--images-dir', default='data/images')
    result.add_argument('--image_ids_file', help='Optional subset of the 807 benchmark IDs')
    result.add_argument('--ablations', action='store_true', help='Final Qwen paper ablations with unchanged thresholds')
    result.add_argument('--statistics', action='store_true', help='20,000 bootstrap samples and 100,000 paired permutations')
    result.add_argument('--model-path', help='Local Qwen2.5-VL model directory for generate')
    result.add_argument('--attention', default='sdpa', choices=['sdpa', 'flash_attention_2', 'eager'])
    result.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16'])
    result.add_argument('--device', default='cuda')
    result.add_argument('--run-name', default='final_method')
    result.add_argument('--run-note', default='Final frozen configuration.')
    result.add_argument('--baseline-run')
    return result




def main():
    arguments = parser().parse_args()
    config = json.loads((ROOT / 'configs/final.json').read_text())
    gt_path = ROOT / 'data/annotations/benchmark_labels_final.csv'
    gt = read_gt(gt_path)
    version = json.loads((ROOT / 'data/annotations/version.json').read_text())
    if sha256(gt_path) != version['csv_sha256'] or len(gt) != 807:
        raise ValueError('Authoritative final annotations do not match the release version')
    ids = sorted(gt)
    if arguments.image_ids_file:
        ids = resolve(arguments.image_ids_file).read_text().split()
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= gt.keys():
            raise ValueError('Invalid image subset')
        for iid in ids:
            valid_id(iid)
    folds_path = ROOT / 'data/splits/five_subsets.json'
    folds = read_folds(folds_path, ids)
    if arguments.command == 'check':
        canonical_path = ROOT / 'data/annotations/benchmark_labels_final_metadata.jsonl'
        if sha256(canonical_path) != version['canonical_sha256']:
            raise ValueError('Canonical annotation checksum mismatch')
        canonical = read_jsonl(canonical_path, ids)
        for iid in ids:
            categories = {value.replace('_violation', '').replace('rule_', 'Rule ') for value in canonical[iid]['risk_categories']}
            if categories != gt[iid]:
                raise ValueError(f'CSV/JSONL annotation disagreement: {iid}')
        if sorted((ROOT / 'data/image_ids.txt').read_text().split()) != sorted(gt):
            raise ValueError('Image list does not match the finalized annotations')
        print(f'Validated {len(ids)} annotations, matching CSV/JSONL versions, image list and five subsets.')
        print('Evaluation requires supplied VLM logs; fusion replay also requires saved detector evidence.')
        return
    if not os.environ.get('GTV_RUN_DIR'):
        raise RuntimeError('Invoke experiments with ./run.sh to preserve records')
    run = Path(os.environ['GTV_RUN_DIR'])
    output = run / 'outputs'
    write_json(run / 'parameters.json', vars(arguments))
    (run / 'image_ids.txt').write_text('\n'.join(ids) + '\n')
    if arguments.command in {'eval', 'reproduce', 'validate'} and not arguments.logs:
        raise ValueError('Provide --logs for the single VLM being evaluated')
    if arguments.command == 'reproduce' and not arguments.evidence:
        raise ValueError('Provide --evidence with saved detector JSONL')
    if arguments.command == 'eval' and (arguments.ablations or arguments.statistics):
        raise ValueError('Paper ablations/statistics require fusion replay or validation')
    inputs = {}

    def record(label, value):
        path = Path(value)
        if path.is_dir():
            items = {p.name: sha256(p) for p in sorted(path.glob('*.json'))}
            inputs[label] = dict(path=str(path), file_sha256=items)
        else:
            inputs[label] = dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha256(path))
        write_json(run / 'inputs.json', inputs)

    record('annotations', gt_path)
    record('subsets', folds_path)
    record('configuration', ROOT / 'configs/final.json')
    if arguments.ablations and arguments.model != 'qwen':
        raise ValueError('Published ablations use Qwen only')
    if arguments.command == 'generate':
        if arguments.model != 'qwen' or not arguments.model_path:
            raise ValueError('generate requires --model qwen and --model-path; GPT is saved-output evaluation only')
        from generation import generate
        generate(ids, resolve(arguments.images_dir), resolve(arguments.model_path), output, arguments)
        return
    evidence = None
    if arguments.command in {'detect', 'validate'}:
        from detection import detect
        evidence = detect(ids, resolve(arguments.images_dir), config, output, arguments.device)
        if arguments.command == 'detect':
            write_json(output / 'metrics.json', dict(images=len(ids), stage='fresh_detection', evaluation_metrics=None))
            return
    elif arguments.command == 'reproduce':
        evidence_path = resolve(arguments.evidence)
        record('detector_evidence', evidence_path)
        evidence = read_jsonl(evidence_path, ids)
    logs_path = resolve(arguments.logs)
    record('vlm_logs', logs_path)
    logs = read_logs(logs_path, ids)
    report = evaluate(ids, gt, logs, evidence, config, folds, arguments.model, output, arguments.ablations, arguments.statistics)
    print(f'Completed {len(ids)} images; {len(report["methods"])} conditions. See outputs/metrics.csv.')


if __name__ == '__main__':
    main()
