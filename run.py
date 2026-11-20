"""Run video detection and offline road asset tracking."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def verify_package(root):
    record = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for item in record['files']:
        path = (root / item['path']).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('Manifest path escapes package')
        if not path.is_file() or sha(path) != item['sha256']:
            raise ValueError('Missing or changed package file: ' + item['path'])
    return len(record['files'])

def inspect_inputs(source, cv2):
    files = sorted(source.glob('*.avi'))
    if not files:
        raise ValueError('No lowercase .avi files found in input directory')
    records = {}
    for path in files:
        if path.is_symlink():
            raise ValueError('Symlink inputs are not supported')
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError('Cannot open video: ' + path.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if abs(fps - 10) > .01:
                raise ValueError('Require 10 FPS (no silent resampling): ' + path.name)
            frames = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (1080, 1920):
                    raise ValueError('Require 1920x1080: ' + path.name)
                frames += 1
            if not frames or frames != round(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                raise ValueError('Empty/truncated video or inconsistent frame count: ' + path.name)
        finally:
            cap.release()
        records[path.stem] = {'sha256': sha(path), 'frames': frames}
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-package', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile', choices=['h20', '4090'], default='h20')
    parser.add_argument('--yolo-batch', type=int, default=1)
    parser.add_argument('--moco-batch', type=int, default=32)
    args = parser.parse_args()
    print('PACKAGE_FILES_VERIFIED', verify_package(ROOT), flush=True)
    if args.verify_package:
        return
    if args.input is None or args.output is None:
        parser.error('--input and --output required')
    if min(args.yolo_batch, args.moco_batch) < 1:
        parser.error('Batch sizes must be positive')
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_dir() or output.exists():
        parser.error('Input must exist; output must be a NEW directory')
    import torch
    import torchvision
    import ultralytics
    import cv2
    import scipy
    import psutil
    import pynvml
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is prohibited')
    if torch.__version__ != '2.6.0+cu124' or torchvision.__version__ != '0.21.0+cu124':
        raise RuntimeError('Require frozen torch 2.6.0+cu124 / torchvision 0.21.0+cu124')
    if ultralytics.__version__ != '8.4.146':
        raise RuntimeError('Require ultralytics 8.4.146')
    device = torch.cuda.get_device_name(0)
    if (args.profile == 'h20' and 'H20' not in device) or (args.profile == '4090' and '4090' not in device):
        raise RuntimeError('Profile does not match GPU: ' + device)
    inputs = inspect_inputs(source, cv2)
    print('INPUTS_VALIDATED', len(inputs), 'GPU', device, flush=True)
    if args.check_only:
        print('ENVIRONMENT_AND_INPUT_CHECK_ONLY; inference/accuracy not tested')
        return
    output.mkdir(parents=True)
    runtime = output / '_runtime'
    shutil.copytree(ROOT / 'runtime', runtime)
    (runtime / 'source_manifest.json').write_text(json.dumps(inputs, indent=2), encoding='utf-8')
    hashes = {p.name: sha(p) for p in sorted(runtime.iterdir()) if p.is_file() and p.name != 'package_manifest.json'}
    (runtime / 'package_manifest.json').write_text(json.dumps({'files': hashes}, indent=2), encoding='utf-8')
    command = [sys.executable, '-u', str(runtime / 'h20_full_chain_20261010.py'),
               '--sources', str(source), '--weight', str(ROOT / 'models/last.pt'),
               '--moco', str(ROOT / 'models/moco-v3-r50-1000ep.pth.tar'),
               '--out', str(output / 'results'), '--all29', '--gpu-profile', args.profile,
               '--yolo-batch', str(args.yolo_batch), '--moco-batch', str(args.moco_batch)]
    if args.profile == '4090':
        command += ['--release-warm-cache', '--release-phase-cache']
    (output / 'invocation.json').write_text(json.dumps({'command': command, 'input_manifest': inputs,
        'legacy_all29_flag_means_all_current_manifest': True,
        'algorithm_changed': False, 'new_wrapper_requires_GPU_acceptance': True}, indent=2), encoding='utf-8')
    with (output / 'execution.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Frozen chain failed; inspect execution.log')
    expected = {name + '_tracks.json' for name in inputs}
    actual = {p.name for p in (output / 'results').glob('*_tracks.json')}
    if actual != expected:
        raise RuntimeError('Output video coverage mismatch')
    print('TRACKS_READY', output / 'results')

if __name__ == '__main__':
    main()
