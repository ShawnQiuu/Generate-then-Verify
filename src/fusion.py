"""Frozen single-VLM verification. Prediction never reads ground-truth labels."""
import math

RULES = ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4']
ABLATIONS = ['cv_only', 'all_and', 'all_or', 'vote', 'gate_off', 'r4_vlm']


def vlm_rules(log):
    return {a['rule_category'].replace('_', ' ') for a in log.get('anomalies', [])
            if a.get('rule_category') in {'Rule_1', 'Rule_2', 'Rule_3', 'Rule_4'}}


def contexts(log):
    value = log.get('scene_context', [])
    return set(value if isinstance(value, list) else [value])


def mask(rules):
    return sum(1 << index for index, rule in enumerate(RULES) if rule in rules)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def filtered(detections, threshold, row, nms):
    output = []
    for detection in sorted(detections, key=lambda item: -item['confidence']):
        if detection['confidence'] < threshold:
            continue
        box = [max(0.0, min(float(v), row['width'] if i % 2 == 0 else row['height']))
               for i, v in enumerate(detection['bbox'])]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(box_iou(box, old['bbox']) > nms for old in output):
            continue
        output.append(dict(detection, bbox=box))
    return output


def inside(item, person, fraction=1.0):
    x, y = (item[0] + item[2]) / 2, (item[1] + item[3]) / 2
    return person[0] <= x <= person[2] and person[1] <= y <= person[1] + fraction * (person[3] - person[1])


def distance(a, b):
    return math.hypot(max(0, a[0] - b[2], b[0] - a[2]), max(0, a[1] - b[3], b[1] - a[3]))


def rule_evidence(row, config):
    objects = row['cv_detected_objects']
    risks = {rule: [] for rule in RULES}
    for rule in RULES:
        setting = config['rules'][rule]
        people = filtered(objects['people'], setting['person_conf'], row, config['extra_nms_iou'])
        people = [p for p in people if p['bbox'][3] - p['bbox'][1] >= row['height'] * config['person_min_height']]
        if rule == 'Rule 1':
            heads = filtered(objects['no_hardhats'], setting['no_hardhat_conf'], row, config['extra_nms_iou'])
            assigned = {}
            for head in heads:
                matches = [i for i, p in enumerate(people) if inside(head['bbox'], p['bbox'], setting['head_fraction'])]
                if not matches:
                    continue
                index = min(matches, key=lambda i: (people[i]['bbox'][2] - people[i]['bbox'][0]) * (people[i]['bbox'][3] - people[i]['bbox'][1]))
                if index not in assigned or head['confidence'] > assigned[index]['confidence']:
                    assigned[index] = head
            risks[rule] = list(assigned.values())
        elif rule == 'Rule 2':
            harnesses = filtered(objects['safety_harnesses'], setting['harness_conf'], row, config['extra_nms_iou'])
            risks[rule] = [p for p in people if not any(inside(h['bbox'], p['bbox']) for h in harnesses)]
        elif rule == 'Rule 3':
            rails = filtered(objects['guardrails'], setting['guardrail_conf'], row, config['extra_nms_iou'])
            risks[rule] = [p for p in people if not any(distance(p['bbox'], r['bbox']) < setting['guardrail_distance'] * (p['bbox'][3] - p['bbox'][1]) for r in rails)]
        else:
            machines = filtered(objects['excavators'], setting['machinery_conf'], row, config['extra_nms_iou'])
            machines = [m for m in machines if setting['machinery_min_area'] <
                        (m['bbox'][2] - m['bbox'][0]) * (m['bbox'][3] - m['bbox'][1]) / (row['width'] * row['height']) < setting['machinery_max_area']]
            for person in people:
                box = person['bbox']
                height = box[3] - box[1]
                pairs = []
                for machine in machines:
                    other = machine['bbox']
                    if other[3] - other[1] < setting['machinery_min_height'] * height:
                        continue
                    if abs(other[3] - box[3]) > setting['bottom_tolerance'] * height:
                        continue
                    if distance(box, other) < setting['machinery_distance'] * height:
                        pairs.append(dict(machine, confidence=min(person['confidence'], machine['confidence'])))
                if pairs:
                    risks[rule].append(max(pairs, key=lambda item: item['confidence']))
    return risks


def predict(row, log, config, ablations=False):
    risks = rule_evidence(row, config)
    claims, scene = vlm_rules(log), contexts(log)
    cv = {rule for rule in RULES if risks[rule]}
    ungated = set()
    for rule in RULES:
        setting = config['rules'][rule]
        v, c = rule in claims, rule in cv
        mode = setting['mode']
        if mode == 'count':
            accepted = (v and c) or sum(d['confidence'] >= setting['override_conf'] for d in risks[rule]) >= setting['count']
        elif mode == 'vlm':
            accepted = v
        elif mode == 'cv':
            accepted = c
        elif mode == 'and':
            accepted = v and c
        else:
            raise ValueError(f'Unsupported frozen fusion mode: {mode}')
        if accepted:
            ungated.add(rule)
    final = {rule for rule in ungated if 'scene' not in config['rules'][rule] or config['rules'][rule]['scene'] in scene}
    output = {'vlm': claims, 'fixed_chd960_f1': final}
    if ablations:
        vote = cv & {'Rule 1', 'Rule 4'}
        if 'Rule 2' in claims or ('Rule 2' in cv and 'Elevated work' in scene):
            vote.add('Rule 2')
        if 'Rule 3' in cv and ('Rule 3' in claims or 'Deep pit/edge work' in scene):
            vote.add('Rule 3')
        output.update(cv_only=cv, all_and=claims & cv, all_or=claims | cv, vote=vote,
                      gate_off=ungated, r4_vlm=(final - {'Rule 4'}) | (claims & {'Rule 4'}))
    return output
