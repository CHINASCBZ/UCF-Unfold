"""Image IO, robust per-channel normalization, sampling, and evaluation-only metrics."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def _read_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(p) as img:
        if img.mode == "P":
            img = img.convert("RGB")
        elif img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or HWC image, got shape {arr.shape} at {path}")
    return arr


def load_image(
    path: str,
    stats: Optional[Dict[str, Any]] = None,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    log_transform: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    arr = _read_image(path)
    if stats is not None:
        log_transform = bool(stats["log_transform"])
    if log_transform:
        if float(arr.min()) < 0:
            raise ValueError("Log transform requires nonnegative pixel values")
        arr = np.log1p(arr)
    if stats is None:
        lo = np.percentile(arr, lower_percentile, axis=(0, 1))
        hi = np.percentile(arr, upper_percentile, axis=(0, 1))
        stats = {"lo": lo.tolist(), "hi": hi.tolist(), "log_transform": log_transform}
    else:
        lo = np.asarray(stats["lo"], dtype=np.float32)
        hi = np.asarray(stats["hi"], dtype=np.float32)
    if len(lo) != arr.shape[2]:
        raise ValueError(f"Normalization expects {len(lo)} channels, image has {arr.shape[2]}")
    arr = np.clip((arr - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float(), stats


def load_pair(t1: str, t2: str, stats: Optional[Dict[str, Any]] = None, *, log_t1=False, log_t2=False):
    x1, n1 = load_image(t1, None if stats is None else stats["t1"], log_transform=log_t1)
    x2, n2 = load_image(t2, None if stats is None else stats["t2"], log_transform=log_t2)
    if x1.shape[-2:] != x2.shape[-2:]:
        raise ValueError(f"Images must be spatially aligned and share H,W: {x1.shape} vs {x2.shape}")
    return x1, x2, {"t1": n1, "t2": n2}


def random_pair_batch(x1: torch.Tensor, x2: torch.Tensor, batch_size: int, patch_size: int, augment: bool = True):
    """Sample matched image crops without using labels. Tensors live on CPU."""
    h, w = x1.shape[-2:]
    if min(h, w) < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        x1 = F.pad(x1, (0, pad_w, 0, pad_h), mode="replicate")
        x2 = F.pad(x2, (0, pad_w, 0, pad_h), mode="replicate")
        h, w = x1.shape[-2:]
    parts1, parts2 = [], []
    for _ in range(batch_size):
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        crop1 = x1[:, top:top + patch_size, left:left + patch_size]
        crop2 = x2[:, top:top + patch_size, left:left + patch_size]
        if augment:
            if random.random() < 0.5:
                crop1, crop2 = crop1.flip(-1), crop2.flip(-1)
            if random.random() < 0.5:
                crop1, crop2 = crop1.flip(-2), crop2.flip(-2)
            k = random.randint(0, 3)
            crop1, crop2 = torch.rot90(crop1, k, [-2, -1]), torch.rot90(crop2, k, [-2, -1])
        parts1.append(crop1.contiguous())
        parts2.append(crop2.contiguous())
    return torch.stack(parts1), torch.stack(parts2)


def save_probability(prediction: np.ndarray, path: str) -> None:
    output = (np.clip(prediction, 0, 1) * 255).round().astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="L").save(path)


def load_gt_for_evaluation(path: str, height: int, width: int) -> np.ndarray:
    """GT must NEVER enter model forward, training batches or criterion."""
    with Image.open(path) as img:
        gt = np.asarray(img.convert("L"))
    if gt.shape != (height, width):
        raise ValueError(f"Ground truth has shape {gt.shape}, expected {(height, width)}")
    return gt > (127 if gt.max() > 1 else 0)


def binary_metrics(probability: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = probability >= threshold
    truth = gt.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    eps = 1e-10
    total = tp + tn + fp + fn
    oa = (tp + tn) / max(total, 1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(total * total, 1)
    kappa = (oa - pe) / (1.0 - pe + eps)
    return {"OA": oa, "Kappa": kappa, "IoU": iou, "F1": f1,
            "precision": precision, "recall": recall,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn}
