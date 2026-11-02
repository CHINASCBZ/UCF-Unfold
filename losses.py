"""Exactly four paper-defined unsupervised losses for UCF-Unfold.

L_total = lambda_rec * L_rec + lambda_cross * L_cross
        + lambda_con * L_con + lambda_fe * L_FE.

No regularizers, auxiliary losses or GT-based supervision are present.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    rec: float = 1.0
    cross: float = 0.30
    con: float = 0.10
    fe: float = 0.10


def charbonnier_map(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Pixelwise Charbonnier error, averaged across target modality channels."""
    if prediction.shape != target.shape:
        raise ValueError(f"Reconstruction shape mismatch: {prediction.shape} vs {target.shape}")
    return torch.sqrt((prediction - target).square() + eps * eps).mean(dim=1, keepdim=True)


def weighted_average(error_map: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted mean over samples and pixels. Expects Bx1xHxW for both inputs."""
    if error_map.shape != weight.shape:
        raise ValueError(f"Pixel weight shape mismatch: {error_map.shape} vs {weight.shape}")
    return (error_map * weight).sum() / weight.sum().clamp_min(eps)


class FourLossObjective(nn.Module):
    """Paper Section 3.5 losses only.

    Training warm-up uses self-reconstruction (part of L_rec) exclusively,
    since no unfolded stage exists before warm-up concludes.

    Cross-reconstruction and latent alignment are downweighted at locations
    with high predicted change, using a detached unchanged probability.
    """

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        if any(w < 0 for w in asdict(self.weights).values()):
            raise ValueError("All loss weights must be nonnegative")

    def forward(
        self,
        outputs: Mapping[str, object],
        target1: torch.Tensor,
        target2: torch.Tensor,
        *,
        warmup: bool = False,
        ramp: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ramp = max(0.0, min(1.0, float(ramp)))
        f1, f2 = outputs["feature1"], outputs["feature2"]
        rec_auto1, rec_auto2 = outputs["recon_auto1"], outputs["recon_auto2"]
        self_rec = charbonnier_map(rec_auto1, target1).mean() + charbonnier_map(rec_auto2, target2).mean()
        zero = self_rec.new_zeros(())

        # Pre-training: self-reconstruction is the sole active loss term.
        if warmup:
            components = {"rec": self_rec, "cross": zero, "con": zero, "fe": zero}
            return self.weights.rec * self_rec, components

        if not outputs["stage_outputs"]:
            raise ValueError("Unfolding losses require at least one stage; set warmup=True if n_stages=0")

        # (1) Observation reconstruction: self + final-state reconstruction.
        state_rec = (
            charbonnier_map(outputs["recon_state1"], target1).mean()
            + charbonnier_map(outputs["recon_state2"], target2).mean()
        )
        loss_rec = self_rec + state_rec

        probability = outputs["change_probability"]
        unchanged = (1.0 - probability.detach()).clamp(0.0, 1.0)

        # (2) Cross reconstruction: D1(F2) <-> I1, D2(F1) <-> I2.
        loss_cross = weighted_average(
            charbonnier_map(outputs["cross_recon1"], target1), unchanged
        ) + weighted_average(
            charbonnier_map(outputs["cross_recon2"], target2), unchanged
        )

        # (3) Cross-modal latent feature consistency (channelwise normalized MSE).
        f1_norm = F.normalize(f1, p=2, dim=1, eps=1e-6)
        f2_norm = F.normalize(f2, p=2, dim=1, eps=1e-6)
        loss_con = weighted_average((f1_norm - f2_norm).square().mean(dim=1, keepdim=True), unchanged)

        # (4) Dual-hypothesis free energy from the FINAL unfolding stage.
        final_stage = outputs["stage_outputs"][-1]
        e0 = final_stage["e0"]
        e1 = final_stage["e1"]
        kappa = final_stage["kappa"]
        temperature = final_stage["temperature"]
        safe_prob = probability.clamp(1e-6, 1.0 - 1e-6)
        neg_entropy = safe_prob * safe_prob.log() + (1.0 - safe_prob) * (1.0 - safe_prob).log()
        loss_fe = ((1.0 - probability) * e0 + probability * (e1 + kappa) + temperature * neg_entropy).mean()

        # During progressive activation, ramp affects only the three newly introduced losses.
        total = (
            self.weights.rec * loss_rec
            + ramp * self.weights.cross * loss_cross
            + ramp * self.weights.con * loss_con
            + ramp * self.weights.fe * loss_fe
        )
        components = {"rec": loss_rec, "cross": loss_cross, "con": loss_con, "fe": loss_fe}
        return total, components
