"""Render the test views of a released DehazePhys-NeRF scene model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if os.name == 'nt':
    os.environ['PATH'] = os.pathsep.join([str(Path(sys.prefix) / 'Library/bin'),
                                         str(Path(sys.prefix) / 'DLLs'), os.environ.get('PATH', '')])
import torch
import numpy as np
from PIL import Image
from dehazephys.model import load_model
from dehazephys.cameras import generate_rays


def render_scene(root, scene, output, chunk, device, start=0, count=0):
    folder = root / scene
    settings = json.loads((folder / 'scene.json').read_text(encoding='utf8'))
    if hashlib.sha256((folder / 'weights.npz').read_bytes()).hexdigest() != settings['weights_sha256']:
        raise ValueError('Model checksum mismatch: ' + scene)
    model = load_model(folder / 'weights.npz', device)
    with np.load(str(folder / 'cameras.npz'), allow_pickle=False) as archive:
        cameras = {name: archive[name].copy() for name in archive.files}
    total = len(cameras['image_sizes'])
    if start < 0 or start >= total or count < 0:
        raise ValueError('Invalid view range')
    stop = min(total, start + count) if count else total
    destination = output / scene
    (destination / 'raw').mkdir(parents=True, exist_ok=True)
    (destination / 'evaluation').mkdir(exist_ok=True)
    records = []
    for index in range(start, stop):
        begin = time.time()
        h, w = (int(v) for v in cameras['image_sizes'][index])
        rays = generate_rays(cameras['intrinsics'][index], cameras['extrinsics'][index], (h, w))
        pieces = []
        with torch.no_grad():
            for offset in range(0, h * w, chunk):
                batch = {name: torch.as_tensor(value[offset:offset + chunk], device=device)
                         for name, value in rays.items()}
                rgb = model(batch, settings['near'], settings['far'])
                if not torch.isfinite(rgb).all():
                    raise ValueError('Non-finite model output')
                pieces.append(rgb.cpu())
        raw = (255 * np.clip(torch.cat(pieces).reshape(h, w, 3).numpy(), 0, 1)).astype(np.uint8)
        name = 'view_{:03d}.png'.format(index)
        Image.fromarray(raw).save(str(destination / 'raw' / name))
        crop = settings['crop_box']
        if crop:
            left, top, right, bottom = crop
            evaluated = raw[top:bottom, left:right]
        else:
            evaluated = raw
        Image.fromarray(evaluated).save(str(destination / 'evaluation' / name))
        record = {'view': index, 'file': name, 'seconds': time.time() - begin}
        records.append(record)
        print(json.dumps({'scene': scene, **record}), flush=True)
    (destination / 'render_report.json').write_text(json.dumps({'scene': scene, 'views': records}, indent=2), encoding='utf8')
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--scene', default='all', help='Scene name, or all')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--chunk', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=0, help='0 renders every remaining test view')
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error('--chunk must be positive')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    names = json.loads((args.models / 'models.json').read_text(encoding='utf8'))['scenes']
    scenes = names if args.scene == 'all' else [args.scene]
    if any(name not in names for name in scenes):
        parser.error('Unknown scene; available scenes: ' + ', '.join(names))
    for scene in scenes:
        render_scene(args.models, scene, args.output, args.chunk, args.device, args.start, args.count)


if __name__ == '__main__':
    main()
