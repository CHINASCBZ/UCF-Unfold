"""Train UCF-Unfold on ONE aligned heterogeneous image pair, without labels.

EXACTLY FOUR losses after warm-up:
  1. observation reconstruction (self + final state)
  2. cross-modal feature consistency (weighted toward predicted unchanged regions)
  3. dual-hypothesis free energy

During warm-up only self-reconstruction is active. No other loss is computed.
GT, if supplied, is loaded only inside no-grad evaluation; it is NEVER used
for losses, gradients, optimization, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from module import UCFUnfoldNet
from losses import FourLossObjective, LossWeights
from data_utils import load_pair, random_pair_batch, save_probability, load_gt_for_evaluation, binary_metrics
from test import predict_probability


def arguments():
    p = argparse.ArgumentParser(description="UCF-Unfold four-loss unlabeled training")
    p.add_argument("--t1", default="datasets/SutterCounty_1.png")
    p.add_argument("--t2", default="datasets/SutterCounty_2.png")
    p.add_argument("--gt", default="datasets/SutterCounty_gt.png", help="Optional GT, for evaluation only; not used in training or checkpoint selection")
    p.add_argument("--outdir", default="weights/SutterCounty")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup-epochs", type=int, default=10)         #warmup-epochs:前10轮预热表示网络，仅使用自重建损失
    p.add_argument("--ramp-epochs", type=int, default=10)           #ramp-epochs:预热后，逐步增加完整展开损失权重的轮数
    p.add_argument("--steps-per-epoch", type=int, default=10)         #每轮执行10次参数更新
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--no-augmentation", action="store_true")       #添加该参数后关闭数据增强
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW optimizer decay, not a loss term")
    p.add_argument("--grad-clip", type=float, default=5.0, help="Numerical gradient clipping, not a loss term")
    p.add_argument("--lambda-rec", type=float, default=0.6)     #损失权重1
    p.add_argument("--lambda-cross", type=float, default=0.0)  #This loss is useless
    p.add_argument("--lambda-con", type=float, default=0.20)    #损失权重3 默认0.1
    p.add_argument("--lambda-fe", type=float, default=0.20)     #损失权重4   默认0.1
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--latent-channels", type=int, default=48)
    p.add_argument("--prior-channels", type=int, default=64)
    p.add_argument("--stages", type=int, default=3)     #控制深度展开次数
    p.add_argument("--mean-field-iterations", type=int, default=3)
    p.add_argument("--change-prior", type=float, default=0.10)  #默认0.1  影响双假设能量增益的自适应判别阈值。  降低该参数会提高所选分位数，使阈值倾向于更严格；但最终响应还取决于可学习的缩放、偏置和温度参数。
    p.add_argument("--fixed-kappa", action="store_true")
    p.add_argument("--eval-every", type=int, default=1, help="Evaluate and save all metrics every N epochs; 1=every epoch")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--eval-tile", type=int, default=0, help="0 uses full-image inference, a positive tile size uses local approximation")
    p.add_argument("--eval-overlap", type=int, default=32)
    p.add_argument("--log-t1", action="store_true",default=False)     #对第一时相影像进行对数预处理,默认False
    p.add_argument("--log-t2", action="store_true",default=False)     #对第二时相影像进行对数预处理，默认False
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_checkpoint(path: Path, model):
    """Save only model parameters and registered buffers (state_dict)."""
    torch.save(model.state_dict(), path)


def main():
    args = arguments()
    if not (args.epochs >= 1 and args.steps_per_epoch >= 1 and args.batch_size >= 1 and args.patch_size >= 2):
        raise ValueError("epochs/steps/batch must be positive, patch-size must be at least 2")
    if args.eval_every < 1:
        raise ValueError("eval-every must be >= 1")
    if args.warmup_epochs < 0 or args.ramp_epochs < 0:
        raise ValueError("warmup-epochs and ramp-epochs must be nonnegative")
    if args.warmup_epochs >= args.epochs:
        raise ValueError("warmup-epochs must be smaller than epochs to train the unfolding stages")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    x1, x2, norm = load_pair(args.t1, args.t2, log_t1=args.log_t1, log_t2=args.log_t2)
    height, width = x1.shape[-2:]
    model_kwargs = {
        "t1_channels": int(x1.shape[0]), "t2_channels": int(x2.shape[0]),
        "base_channels": args.base_channels,
        "latent_channels": args.latent_channels,
        "prior_channels": args.prior_channels,
        "stages": args.stages,
        "mean_field_iterations": args.mean_field_iterations,
        "change_prior": args.change_prior,
        "adaptive_kappa": not args.fixed_kappa,
    }
    model = UCFUnfoldNet(**model_kwargs).to(device)
    weights = LossWeights(rec=args.lambda_rec, cross=args.lambda_cross, con=args.lambda_con, fe=args.lambda_fe)
    loss_fn = FourLossObjective(weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    dest = Path(args.outdir)
    ckp_dir = dest / "checkpoints"
    image_dir = dest / "maps"
    ckp_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}; size: {height}x{width}; channels: {x1.shape[0]}/{x2.shape[0]}")
    print("Exactly FOUR losses: rec, cross, con, fe. No auxiliary losses or GT training.")
    best_loss = float("inf")
    # Evaluation metrics have NO influence on optimizer, LR scheduler or checkpoint choice.
    # The GT is loaded lazily at the first evaluation, and never enters a train step.
    gt_eval = None
    metric_names = ["OA", "F1", "Kapp", "precision", "recall",
                    "TP", "TN", "FP", "FN", "IoU", "Kappa"]
    fields = ["epoch", "phase", "ramp", "total", "rec", "cross", "con", "fe",
              "lr", "seconds", *metric_names]
    metric_fields = ["epoch", *metric_names]
    latest_metrics = None
    with (dest / "history.csv").open("w", newline="", encoding="utf-8-sig") as history_file, \
         (dest / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as metrics_file:
        history_writer = csv.DictWriter(history_file, fieldnames=fields)
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
        history_writer.writeheader()
        metrics_writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            warmup = epoch <= args.warmup_epochs
            ramp = 0.0 if warmup else min(1.0, (epoch - args.warmup_epochs) / max(args.ramp_epochs, 1))
            model.train()
            totals = {"total": 0.0, "rec": 0.0, "cross": 0.0, "con": 0.0, "fe": 0.0}
            for _ in range(args.steps_per_epoch):
                a, b = random_pair_batch(x1, x2, args.batch_size, args.patch_size, not args.no_augmentation)
                a, b = a.to(device), b.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(a, b, n_stages=0 if warmup else None)
                total, items = loss_fn(output, a, b, warmup=warmup, ramp=ramp)
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {total.item()}")
                total.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                totals["total"] += total.detach().item()
                for k, v in items.items():
                    totals[k] += v.detach().item()
            scheduler.step()
            averages = {key: value / args.steps_per_epoch for key, value in totals.items()}
            phase = "warmup" if warmup else "unfolding"

            # Save checkpoints using *only* training losses (never GT-based scores).
            save_checkpoint(ckp_dir / "last.pth", model)
            if not warmup and averages["total"] < best_loss:
                best_loss = averages["total"]
                save_checkpoint(ckp_dir / "best.pth", model)

            record = {"epoch": epoch, "phase": phase, "ramp": ramp, **averages,
                      "lr": optimizer.param_groups[0]["lr"]}
            should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
            if should_evaluate:
                # Even during representation warm-up, report full K-stage inference,
                # making the metric trend comparable across all evaluated epochs.
                prediction = predict_probability(model, x1, x2, device, args.eval_tile, args.eval_overlap)
                prob = prediction["probability"]
                save_probability(prob, str(image_dir / f"epoch_{epoch:04d}_probability.png"))
                save_probability((prob >= args.threshold).astype(np.float32),
                                 str(image_dir / f"epoch_{epoch:04d}_binary.png"))
                if args.gt:
                    if gt_eval is None:
                        gt_eval = load_gt_for_evaluation(args.gt, height, width)
                    m = binary_metrics(prob, gt_eval, args.threshold)
                    # Kapp and Kappa refer to the same Cohen's Kappa statistic.
                    m = {**m, "Kapp": m["Kappa"]}
                    latest_metrics = m
                    for key in metric_names:
                        record[key] = m[key]
                    metrics_writer.writerow({"epoch": epoch, **m})
                    metrics_file.flush()
                    (dest / "latest_metrics.json").write_text(
                        json.dumps({"epoch": epoch, **m}, indent=2, ensure_ascii=False),
                        encoding="utf-8")

            record["seconds"] = time.time() - start
            history_writer.writerow(record)
            history_file.flush()
            text = (f"Epoch {epoch:04d}/{args.epochs} [{phase}] "
                    f"total={averages['total']:.6f} rec={averages['rec']:.5f} "
                    f"cross={averages['cross']:.5f} con={averages['con']:.5f} "
                    f"fe={averages['fe']:.5f} ramp={ramp:.2f}")
            if should_evaluate and latest_metrics is not None:
                text += (f" OA={latest_metrics['OA']:.4f} F1={latest_metrics['F1']:.4f}"
                         f" Kapp={latest_metrics['Kapp']:.4f}"
                         f" Precision={latest_metrics['precision']:.4f}"
                         f" Recall={latest_metrics['recall']:.4f}"
                         f" IoU={latest_metrics['IoU']:.4f}")
            print(text)

    if args.gt and latest_metrics is not None:
        (dest / "final_metrics.json").write_text(
            json.dumps(latest_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Final diagnostic metrics (GT NEVER entered training):", latest_metrics)
    elif not args.gt:
        print("No GT specified: loss history and probability maps saved, metric fields left blank.")
    print(f"Finished. Outputs at: {dest.resolve()}")


if __name__ == "__main__":
    main()

