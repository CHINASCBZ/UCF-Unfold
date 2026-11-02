
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from module import UCFUnfoldNet
from data_utils_sup import binary_metrics, load_gt, load_pair, save_probability


MODEL_CONFIG = {
    't1_channels': 3,
    't2_channels': 3,
    'base_channels': 32,
    'latent_channels': 48,
    'prior_channels': 64,
    'stages': 2,        #ltaly,shuguang,Franch :3    SutterCounty:2  YellowRiver:1
    'change_prior': 0.1,
    'mean_field_iterations': 3,
    'adaptive_kappa': True,
    'min_variance': 0.02,
    'max_variance': 5.0,
}


def _starts(length: int, tile: int, overlap: int):
    if tile >= length:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_probability(model: UCFUnfoldNet, x1: torch.Tensor, x2: torch.Tensor,
                        device: torch.device, tile_size: int = 0,
                        overlap: int = 32) -> Dict[str, np.ndarray]:
    model.eval()
    h, w = x1.shape[-2:]
    if tile_size <= 0 or (tile_size >= h and tile_size >= w):
        out = model(x1[None].to(device), x2[None].to(device))
        prob = out['change_probability'][0, 0].detach().float().cpu().numpy()
        gain = out['stage_outputs'][-1]['gain'][0, 0].detach().float().cpu().numpy()
        return {'probability': prob, 'energy_gain': gain}

    if tile_size <= overlap:
        raise ValueError('tile_size must exceed overlap')

    sums = np.zeros((h, w), dtype=np.float64)
    gains = np.zeros_like(sums)
    counts = np.zeros_like(sums)
    for top in _starts(h, tile_size, overlap):
        for left in _starts(w, tile_size, overlap):
            bottom = min(top + tile_size, h)
            right = min(left + tile_size, w)
            out = model(x1[:, top:bottom, left:right][None].to(device),
                        x2[:, top:bottom, left:right][None].to(device))
            p = out['change_probability'][0, 0].float().cpu().numpy()
            g = out['stage_outputs'][-1]['gain'][0, 0].float().cpu().numpy()
            ph, pw = p.shape
            wy = np.hanning(ph) if ph > 2 else np.ones(ph)
            wx = np.hanning(pw) if pw > 2 else np.ones(pw)
            blend = np.maximum(np.outer(wy, wx), 0.05)
            sums[top:bottom, left:right] += blend * p
            gains[top:bottom, left:right] += blend * g
            counts[top:bottom, left:right] += blend
    return {
        'probability': (sums / counts.clip(min=1e-8)).astype(np.float32),
        'energy_gain': (gains / counts.clip(min=1e-8)).astype(np.float32),
    }


def load_checkpoint(path: str, device: torch.device) -> UCFUnfoldNet:
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise TypeError('Expected a raw model.state_dict() dictionary.')
    if 'model_state' in state or 'model_config' in state:
        raise ValueError('This appears to be an old training checkpoint. '
                         'Please load last_weights_only.pth instead.')
    model = UCFUnfoldNet(**MODEL_CONFIG).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='weights/SutterCounty/best.pth',
                   help='Path to the raw state_dict .pth')
    p.add_argument('--t1', default='datasets/SutterCounty_1.png')
    p.add_argument('--t2', default='datasets/SutterCounty_2.png')
    p.add_argument('--gt', default='datasets/SutterCounty_gt.png',
                   help='Optional GT, evaluated only; ')
    p.add_argument('--outdir', default='runs/prediction/SutterCounty')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--tile-size', type=int, default=0)
    p.add_argument('--tile-overlap', type=int, default=32)
    p.add_argument('--log-t1', action='store_true')
    p.add_argument('--log-t2', action='store_true')
    p.add_argument('--device', default='auto')
    a = p.parse_args()

    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if a.device == 'auto' else a.device)
    model = load_checkpoint(a.checkpoint, device)


    x1, x2, _ = load_pair(a.t1, a.t2, None,
                          log_t1=a.log_t1, log_t2=a.log_t2)
    out = predict_probability(model, x1, x2, device, a.tile_size, a.tile_overlap)

    dest = Path(a.outdir)
    dest.mkdir(parents=True, exist_ok=True)
    save_probability(out['probability'], str(dest / 'probability.png'))
    save_probability((out['probability'] >= a.threshold).astype(np.float32),
                     str(dest / 'binary.png'))
    np.save(dest / 'probability.npy', out['probability'])
    np.save(dest / 'energy_gain.npy', out['energy_gain'])

    if a.gt:
        truth = load_gt(a.gt, *out['probability'].shape)[0].numpy()
        metrics = binary_metrics(out['probability'], truth, a.threshold)
        metrics['Kapp'] = metrics['Kappa']
        (dest / 'metrics.json').write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
        metric_fields = ('OA', 'F1', 'Kapp', 'precision', 'recall',
                         'TP', 'TN', 'FP', 'FN', 'IoU', 'Kappa')
        with (dest / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=metric_fields)
            writer.writeheader()
            writer.writerow({key: metrics[key] for key in metric_fields})
        print('Evaluation:', metrics)
    print('Saved to:', dest.resolve())


if __name__ == '__main__':
    main()
