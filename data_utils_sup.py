"""Paired image/GT IO, synchronized patch augmentation and change metrics."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def _read_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'File not found: {p}')
    with Image.open(p) as image:
        if image.mode in ('P', 'RGBA', 'LA'):
            image = image.convert('RGB')
        array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f'Expected a HWC image at {p}; found {array.shape}')
    return array


def load_image(path: str, stats: Optional[Dict[str, Any]] = None, *, log_transform: bool = False):
    arr = _read_image(path)
    if stats is not None:
        log_transform = bool(stats['log_transform'])
    if log_transform:
        if float(arr.min()) < 0:
            raise ValueError('Log transform requires nonnegative values')
        arr = np.log1p(arr)
    if stats is None:
        low = np.percentile(arr, 1.0, axis=(0, 1))
        high = np.percentile(arr, 99.0, axis=(0, 1))
        stats = {'lo': low.tolist(), 'hi': high.tolist(), 'log_transform': log_transform}
    else:
        low = np.asarray(stats['lo'], dtype=np.float32)
        high = np.asarray(stats['hi'], dtype=np.float32)
    if low.size != arr.shape[2]:
        raise ValueError('Incompatible channel count and normalization statistics')
    normalized = np.clip((arr - low) / np.maximum(high - low, 1e-6), 0, 1)
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy()).float(), stats


def load_pair(t1: str, t2: str, stats=None, *, log_t1=False, log_t2=False):
    im1, stat1 = load_image(t1, None if stats is None else stats['t1'], log_transform=log_t1)
    im2, stat2 = load_image(t2, None if stats is None else stats['t2'], log_transform=log_t2)
    if im1.shape[-2:] != im2.shape[-2:]:
        raise ValueError(f'Images must be aligned and share H,W: {im1.shape}, {im2.shape}')
    return im1, im2, {'t1': stat1, 't2': stat2}


def load_gt(path: str, height: int, width: int, *, gt_threshold: float = 127.0) -> torch.Tensor:
    """Load binary GT (0=unchanged, 1=changed), JPG or PNG; no resizing."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Ground truth not found: {path}')
    with Image.open(p) as image:
        gt = np.array(image.convert('L'), dtype=np.float32)
    if gt.shape != (height, width):
        raise ValueError(f'GT size {gt.shape} != image {(height, width)}')
    threshold = 0.5 if gt.max() <= 1.0 else gt_threshold
    return torch.from_numpy((gt > threshold).astype(np.float32))[None]


def load_triplet(t1: str, t2: str, gt: str, stats=None, *, log_t1=False, log_t2=False):
    x1, x2, stats_new = load_pair(t1, t2, stats, log_t1=log_t1, log_t2=log_t2)
    label = load_gt(gt, *x1.shape[-2:])
    return x1, x2, label, stats_new


def random_triplet_batch(x1: torch.Tensor, x2: torch.Tensor, gt: torch.Tensor,
                         batch_size: int, patch_size: int, augment=True,
                         positive_prob: float = 0.5, positives=None):
    """Train on paired crops and the matching GT crop (all transforms synchronous)."""
    h, w = x1.shape[-2:]
    if x2.shape[-2:] != (h, w) or gt.shape != (1, h, w):
        raise ValueError('GT and both images must share spatial shape')
    if min(h, w) < patch_size:
        pad = (0, max(0, patch_size - w), 0, max(0, patch_size - h))
        x1 = F.pad(x1, pad, mode='replicate')
        x2 = F.pad(x2, pad, mode='replicate')
        gt = F.pad(gt, pad, mode='replicate')
        h, w = x1.shape[-2:]
        positives = None
    if positives is None:
        positives = torch.nonzero(gt[0] > 0.5).cpu().numpy()
    batch1, batch2, batchgt = [], [], []
    for _ in range(batch_size):
        if len(positives) and random.random() < positive_prob:
            y, x = positives[random.randrange(len(positives))]
            top = min(max(0, int(y) - random.randrange(patch_size)), h - patch_size)
            left = min(max(0, int(x) - random.randrange(patch_size)), w - patch_size)
        else:
            top = random.randrange(h - patch_size + 1)
            left = random.randrange(w - patch_size + 1)
        a = x1[:, top:top + patch_size, left:left + patch_size]
        b = x2[:, top:top + patch_size, left:left + patch_size]
        c = gt[:, top:top + patch_size, left:left + patch_size]
        if augment:
            if random.random() < 0.5:
                a, b, c = a.flip(-1), b.flip(-1), c.flip(-1)
            if random.random() < 0.5:
                a, b, c = a.flip(-2), b.flip(-2), c.flip(-2)
            k = random.randrange(4)
            a, b, c = (torch.rot90(t, k, (-2, -1)) for t in (a, b, c))
        batch1.append(a.contiguous()); batch2.append(b.contiguous()); batchgt.append(c.contiguous())
    return torch.stack(batch1), torch.stack(batch2), torch.stack(batchgt)


def save_probability(probability: np.ndarray, path: str):
    dst = Path(path); dst.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(probability, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(arr, mode='L').save(dst)


def binary_metrics(probability: np.ndarray, truth: np.ndarray, threshold=0.5) -> Dict[str, float]:
    pred = probability >= threshold
    label = truth.astype(bool)
    tp = int(np.logical_and(pred, label).sum())
    tn = int(np.logical_and(~pred, ~label).sum())
    fp = int(np.logical_and(pred, ~label).sum())
    fn = int(np.logical_and(~pred, label).sum())
    return confusion_metrics(tp, tn, fp, fn)


def confusion_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    n = tp + tn + fp + fn
    oa = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n * n, 1)
    kappa = (oa - pe) / max(1 - pe, 1e-10)
    return {'OA': oa, 'Kappa': kappa, 'IoU': iou, 'F1': f1,
            'precision': precision, 'recall': recall,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
