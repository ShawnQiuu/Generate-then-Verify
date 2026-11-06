"""Optional single-Qwen generation with the retained paper prompt."""
import json
import re
import time

from data_io import write_json
from detection import record_inference_assets
from project_paths import ROOT


def generate(ids, images_dir, model_path, output, args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    if not model_path.is_dir():
        raise FileNotFoundError(f'Local Qwen checkpoint directory required: {model_path}')
    images = [images_dir / f'{iid}.jpg' for iid in ids]
    record_inference_assets(output, images, {'qwen': model_path})
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),
            torch_dtype=getattr(torch, args.dtype), device_map='auto', attn_implementation=args.attention,
            local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(model_path), min_pixels=256 * 28 * 28,
                                              max_pixels=1024 * 28 * 28, local_files_only=True)
    prompt = (ROOT / 'prompts/safety_report.txt').read_text()
    raw_dir = output / 'raw'
    raw_dir.mkdir()
    timings = []
    with (output / 'logs.jsonl').open('x') as handle:
        for iid, path in zip(ids, images):
            start = time.monotonic()
            with Image.open(path) as source:
                image = source.convert('RGB')
            messages = [dict(role='user', content=[dict(type='image', image=image), dict(type='text', text=prompt)])]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                               return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            (raw_dir / f'{iid}.txt').write_text(raw)
            match = re.search(r'\{[\s\S]*\}', raw)
            if match is None:
                raise ValueError(f'Invalid generated JSON: {iid}; raw output retained')
            log = json.loads(match.group())
            if not isinstance(log.get('anomalies'), list) or not isinstance(log.get('scene_context'), (str, list)):
                raise ValueError(f'Incomplete generated schema: {iid}; raw output retained')
            handle.write(json.dumps(dict(log, image_id=iid), ensure_ascii=False) + '\n')
            handle.flush()
            timings.append(dict(image_id=iid, seconds=time.monotonic() - start))
            print(f'Generation {len(timings)}/{len(ids)}: {iid}', flush=True)
    write_json(output / 'generation_metadata.json', dict(model=str(model_path), do_sample=False,
               max_new_tokens=1024, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28,
               attention=args.attention, dtype=args.dtype, timings=timings,
               timing_scope='Image loading and generation, excluding checkpoint loading and CV inference.',
               parsing='Strict JSON; malformed output fails visibly instead of repairing or treating failure as safe.'))
    write_json(output / 'metrics.json', dict(images=len(ids), stage='new_qwen_generation', evaluation_metrics=None))
