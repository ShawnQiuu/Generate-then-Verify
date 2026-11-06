"""Record each experiment in a fresh directory, including unsuccessful runs."""
import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from data_io import sha256, write_json
from project_paths import ROOT


def main():
    arguments = sys.argv[1:]
    if not arguments or any(value in arguments for value in ['--help', '-h']) or arguments[0] == 'check':
        return subprocess.call([sys.executable, '-B', str(ROOT / 'src/cli.py'), *(arguments or ['--help'])])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command', choices=['eval', 'reproduce', 'detect', 'validate', 'generate'])
    parser.add_argument('--model', choices=['qwen', 'gpt'], default='qwen')
    parser.add_argument('--run-name', default='final_method')
    parser.add_argument('--run-note', default='Final frozen configuration.')
    parser.add_argument('--baseline-run')
    args, _ = parser.parse_known_args(arguments)
    if not re.fullmatch(r'[a-z0-9_]+', args.run_name):
        parser.error('--run-name must use lowercase snake_case')
    stage = dict(eval='eval', reproduce='replay', detect='detect', validate='cv', generate='vlm')[args.command]
    model = 'cv' if args.command == 'detect' else args.model
    started = datetime.datetime.now(ZoneInfo('Asia/Shanghai'))
    stem = f'{started:%Y%m%d_%H%M%S}__{stage}__{model}__{args.run_name}'
    runs = ROOT / 'experiments/runs'
    runs.mkdir(parents=True, exist_ok=True)
    counter = 0
    while True:
        run = runs / (stem + (f'_{counter}' if counter else ''))
        try:
            run.mkdir()
            break
        except FileExistsError:
            counter += 1
    for name in ['logs', 'outputs', 'source']:
        (run / name).mkdir()
    snapshot = {}
    for path in [ROOT / 'run.sh', *sorted((ROOT / 'src').glob('*.py'))]:
        target = run / 'source' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        snapshot[str(path.relative_to(ROOT))] = sha256(target)
    shutil.copytree(ROOT / 'prompts', run / 'source/prompts')
    snapshot['prompts/safety_report.txt'] = sha256(ROOT / 'prompts/safety_report.txt')
    write_json(run / 'source/hashes.json', snapshot)
    write_json(run / 'config.json', json.loads((ROOT / 'configs/final.json').read_text()))
    write_json(run / 'inputs.json', {})
    (run / 'image_ids.txt').write_text('')
    (run / 'command.sh').write_text('#!/usr/bin/env bash\n' + shlex.join(['./run.sh', *arguments]) + '\n')
    packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata['Name']}
    write_json(run / 'environment.json', dict(python=sys.version, executable=sys.executable,
               platform=platform.platform(), packages=packages,
               environment={key: os.environ[key] for key in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'] if key in os.environ}))
    manifest = dict(run_id=run.name, command=args.command, model=model, run_note=args.run_note,
                    baseline_run=args.baseline_run, start=started.isoformat(), status='running')
    write_json(run / 'manifest.json', manifest)
    environment = dict(os.environ, GTV_RUN_DIR=str(run))
    started_clock = time.monotonic()
    process = None
    error = None
    try:
        with (run / 'logs/console.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-B', str(ROOT / 'src/cli.py'), *arguments],
                                       cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    except KeyboardInterrupt:
        if process is not None:
            process.terminate()
            process.wait()
        code, error = 130, 'Interrupted by user'
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        code, error = 1, str(exc)
    metrics_path = run / 'outputs/metrics.json'
    report = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    write_json(run / 'metrics.json', report)
    manifest.update(status='completed' if code == 0 else ('interrupted' if code == 130 else 'failed'),
                    end=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    elapsed_seconds=time.monotonic() - started_clock, exit_code=code, error=error)
    write_json(run / 'manifest.json', manifest)
    lines = [f'# {run.name}', '', args.run_note, '', f'Status: {manifest["status"]}.',
             f'Command: `{args.command}`; images: {report.get("images", "see console")}.', '',
             'Evaluation is conditional on a frozen development configuration. No training or threshold selection occurs.',
             'Saved-evidence replay does not repeat model inference. Detection and generation costs are recorded separately.',
             'No end-to-end latency or resource reduction is established by this run.']
    if report.get('reference_check'):
        lines += ['', 'Reference check: ' + json.dumps(report['reference_check'])]
    if code:
        lines += ['', 'Incomplete run; inspect logs/console.log. Partial outputs are not a complete reproduction.']
    (run / 'summary.md').write_text('\n'.join(lines) + '\n')
    with (ROOT / 'experiments/index.jsonl').open('a') as handle:
        handle.write(json.dumps(manifest) + '\n')
    print(f'Run recorded: {run.relative_to(ROOT)}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
