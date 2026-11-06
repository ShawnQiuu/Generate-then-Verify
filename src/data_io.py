"""Strict benchmark and saved-output readers."""
import csv
import hashlib
import json
import re
from pathlib import Path

from fusion import RULES


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{7}', value):
        raise ValueError(f'Expected a seven-digit image ID: {value!r}')
    return value


def read_gt(path):
    output = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            iid = valid_id(Path(row['image_id']).stem)
            if iid in output:
                raise ValueError(f'Duplicate annotation: {iid}')
            values = [row[f'rule{i}'] for i in range(1, 5)]
            if any(value not in {'0', '1'} for value in values):
                raise ValueError(f'Invalid binary annotations: {iid}')
            output[iid] = {rule for rule, value in zip(RULES, values) if value == '1'}
            if row['is_safe'] != str(int(not output[iid])):
                raise ValueError(f'Inconsistent is_safe: {iid}')
    return output


def read_jsonl(path, ids):
    output = {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = valid_id(row['image_id'])
            if iid in output:
                raise ValueError(f'Duplicate saved output: {iid}')
            output[iid] = row
    missing = set(ids) - output.keys()
    if missing:
        raise ValueError(f'Missing {len(missing)} images in {path}: {sorted(missing)[:5]}')
    return {iid: output[iid] for iid in ids}


def read_logs(path, ids):
    path = Path(path)
    if path.is_dir():
        logs = {iid: json.loads((path / f'{iid}.json').read_text()) for iid in ids}
    else:
        logs = read_jsonl(path, ids)
    for iid, row in logs.items():
        if not isinstance(row.get('anomalies'), list):
            raise ValueError(f'Missing or malformed anomalies for {iid}; generation failures cannot be scored as safe')
        if not isinstance(row.get('scene_context'), (list, str)):
            raise ValueError(f'Missing or malformed scene_context for {iid}')
        if any(not isinstance(item, dict) or not isinstance(item.get('rule_category'), str) for item in row['anomalies']):
            raise ValueError(f'Malformed anomaly for {iid}')
    return logs


def read_folds(path, ids):
    document = json.loads(Path(path).read_text())
    assignments = {}
    for subset in document['subsets']:
        for iid in subset['image_ids']:
            valid_id(iid)
            if iid in assignments:
                raise ValueError(f'Overlapping evaluation subsets: {iid}')
            assignments[iid] = subset['fold']
    if set(assignments.values()) != set(range(5)) or not set(ids) <= assignments.keys():
        raise ValueError('Invalid five-subset partition')
    return [assignments[iid] for iid in ids]
