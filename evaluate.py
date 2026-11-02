"""Evaluate aligned RGB predictions against clear reference images."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

if os.name == 'nt':
    os.environ['PATH'] = os.pathsep.join([str(Path(sys.prefix) / 'Library/bin'),
                                         str(Path(sys.prefix) / 'DLLs'), os.environ.get('PATH', '')])
import torch
import numpy as np
from PIL import Image


def image_tensor(path, device):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('metrics.csv'))
    parser.add_argument('--metrics', nargs='+', choices=['psnr', 'ssim', 'lpips-alex'], default=['psnr', 'ssim', 'lpips-alex'])
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    predictions = sorted(args.predictions.glob('view_*.png'))
    if not predictions:
        parser.error('No view_*.png prediction images found')
    scorers = {}
    if 'ssim' in args.metrics:
        from piqa import SSIM
        scorers['ssim'] = SSIM().eval().to(args.device)
    if 'lpips-alex' in args.metrics:
        from piqa import LPIPS
        scorers['lpips-alex'] = LPIPS(network='alex').eval().to(args.device)
    rows = []
    with torch.no_grad():
        for prediction in predictions:
            reference = args.references / prediction.name
            if not reference.is_file():
                raise FileNotFoundError('Missing matching reference: ' + prediction.name)
            pred, gt = image_tensor(prediction, args.device), image_tensor(reference, args.device)
            if pred.shape != gt.shape:
                raise ValueError('Image size mismatch: ' + prediction.name)
            row = {'image': prediction.name}
            for metric in args.metrics:
                if metric == 'psnr':
                    mse = torch.mean((pred.clamp(0, 1) - gt.clamp(0, 1)) ** 2)
                    row[metric] = float(-10.0 * torch.log(mse) / np.log(10))
                else:
                    row[metric] = float(scorers[metric](pred.clamp(0, 1), gt.clamp(0, 1)))
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image'] + args.metrics)
        writer.writeheader()
        writer.writerows(rows)
        averages = {metric: float(np.mean([row[metric] for row in rows])) for metric in args.metrics}
        writer.writerow({'image': 'AVERAGE', **averages})
    print(json.dumps({'images': len(rows), 'averages': averages}, indent=2))


if __name__ == '__main__':
    main()
