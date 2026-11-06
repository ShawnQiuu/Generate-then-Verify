"""CHD960 detector extraction; model boxes are evidence, never human GT."""
import json
import random
import time

import numpy as np

from data_io import write_json
from project_paths import resolve


def record_inference_assets(output, images, weights):
    inputs_path = output.parent / 'inputs.json'
    inputs = json.loads(inputs_path.read_text())

    def metadata(path):
        if path.is_dir():
            return dict(path=str(path), files=[dict(file=str(p.relative_to(path)), size_bytes=p.stat().st_size,
                        mtime_ns=p.stat().st_mtime_ns) for p in sorted(path.rglob('*')) if p.is_file()])
        return dict(path=str(path), size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)

    inputs['images'] = [metadata(path) for path in images]
    inputs['models'] = {name: metadata(path) for name, path in weights.items()}
    write_json(inputs_path, inputs)


def detect(ids, images_dir, config, output, device):
    import cv2
    import torch
    import groundingdino
    from ultralytics import YOLO
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util.inference import load_image, predict
    from pathlib import Path

    images = [images_dir / f'{iid}.jpg' for iid in ids]
    weights = {key: resolve(value) for key, value in config['models'].items()}
    for path in [*images, *weights.values()]:
        if not path.exists():
            raise FileNotFoundError(f'Required local inference asset: {path}')
    record_inference_assets(output, images, weights)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    extraction = config['extraction']
    person, chd = YOLO(str(weights['person_weights'])), YOLO(str(weights['chd_weights']))
    expected = {2: 'NO-Hardhat', 8: 'Machinery'}
    if person.names[0].lower() != 'person' or any(chd.names[k].lower() != name.lower() for k, name in expected.items()):
        raise ValueError('Checkpoint class maps differ from the paper models')
    dino_config = Path(groundingdino.__file__).parent / 'config/GroundingDINO_SwinB_cfg.py'
    cfg = SLConfig.fromfile(str(dino_config))
    cfg.device, cfg.text_encoder_type = device, str(weights['bert'])
    dino = build_model(cfg)
    checkpoint = torch.load(str(weights['dino_weights']), map_location='cpu')
    dino.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    dino.eval().to(device)

    def yolo(model, bgr, classes, names, confidence):
        groups = {key: [] for key in classes}
        for result in model.predict(source=bgr, conf=confidence, iou=extraction['yolo_nms_iou'],
                                    classes=classes, imgsz=extraction['imgsz'], device=device, verbose=False):
            for box in result.boxes:
                key = int(box.cls[0])
                groups[key].append(dict(class_name=names[key], confidence=float(box.conf[0]), bbox=box.xyxy[0].tolist()))
        return groups

    def protection(path, prompt):
        source, tensor = load_image(str(path))
        boxes, scores, phrases = predict(model=dino, image=tensor, caption=prompt,
                                        box_threshold=extraction['dino_floor'], text_threshold=extraction['text_floor'], device=device)
        height, width = source.shape[:2]
        result = []
        for box, score, phrase in zip(boxes, scores, phrases):
            cx, cy, w, h = box.tolist()
            result.append(dict(class_name=phrase.strip().lower(), confidence=float(score),
                               bbox=[(cx - w / 2) * width, (cy - h / 2) * height,
                                     (cx + w / 2) * width, (cy + h / 2) * height]))
        return result

    evidence, timings = {}, []
    with (output / 'evidence.jsonl').open('x') as handle:
        for iid, image in zip(ids, images):
            start = time.monotonic()
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise ValueError(f'Unreadable image: {image}')
            people = yolo(person, bgr, [0], {0: 'person'}, extraction['person_floor'])[0]
            objects = yolo(chd, bgr, [2, 8], expected, extraction['object_floor'])
            row = dict(image_id=iid, width=bgr.shape[1], height=bgr.shape[0], cv_detected_objects=dict(
                       people=people, no_hardhats=objects[2], excavators=objects[8],
                       safety_harnesses=protection(image, extraction['harness_prompt']),
                       guardrails=protection(image, extraction['guardrail_prompt'])))
            # Retain the saved-evidence schema used by the original pipeline.
            for detections in row['cv_detected_objects'].values():
                for item in detections:
                    item['class'] = item.pop('class_name')
            evidence[iid] = row
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            timings.append(dict(image_id=iid, seconds=time.monotonic() - start))
            print(f'Detection {len(evidence)}/{len(ids)}: {iid}', flush=True)
    write_json(output / 'evidence_metadata.json', dict(extraction=extraction, images=len(ids),
               model_evidence_not_gt=True, crops=False, detector_calls_per_image=dict(person_yolo=1, chd_yolo=1, groundingdino=2),
               timing_scope='Image loading and detector inference, excluding model load and VLM generation.', timings=timings))
    return evidence
