"""Main model for unsupervised heterogeneous change detection.

The model implements a finite-stage unfolding of two nested hypotheses:

H0 (unchanged): one latent state jointly explains both observations.
H1 (changed):   two coupled latent states explain the observations.

At every stage, the H0 and H1 latent states are solved in closed form,
their optimal energies are compared, and an edge-aware mean-field solver
produces a soft binary change posterior. A shared learned prior then refines
the latent scene states before the next stage.

This is a research prototype for target-pair-specific, label-free training.
Ground truth is not consumed by this module or by its training loss.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, 3, dilation=dilation)
        self.conv2 = ConvNormAct(
            channels, channels, 3, dilation=dilation, activation=False
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.conv2(self.conv1(x)))


class ModalityStem(nn.Module):


    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(in_channels, base_channels, 3),
            ResidualBlock(base_channels, dilation=1),
            ResidualBlock(base_channels, dilation=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedContentBackbone(nn.Module):

    def __init__(self, base_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(base_channels, dilation=1),
            ResidualBlock(base_channels, dilation=2),
            ResidualBlock(base_channels, dilation=4),
            ConvNormAct(base_channels, latent_channels, 3, activation=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityDecoder(nn.Module):

    def __init__(
        self, latent_channels: int, base_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(latent_channels, base_channels, 3),
            ResidualBlock(base_channels, dilation=4),
            ResidualBlock(base_channels, dilation=2),
            ResidualBlock(base_channels, dilation=1),
            nn.Conv2d(base_channels, out_channels, 3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.body(x))


class SharedScenePrior(nn.Module):


    def __init__(self, latent_channels: int, hidden_channels: int) -> None:
        super().__init__()
        in_channels = latent_channels + 3
        self.head = ConvNormAct(in_channels, hidden_channels, 3)
        self.body = nn.Sequential(
            ResidualBlock(hidden_channels, dilation=1),
            ResidualBlock(hidden_channels, dilation=2),
            ResidualBlock(hidden_channels, dilation=4),
            ResidualBlock(hidden_channels, dilation=1),
        )
        self.tail = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        self.raw_residual_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(0.10), dtype=torch.float32)
        )

    def forward(
        self,
        state: torch.Tensor,
        change_probability: torch.Tensor,
        reliability: torch.Tensor,
        stage_fraction: float,
    ) -> torch.Tensor:
        stage_map = torch.full_like(change_probability, float(stage_fraction))
        side = torch.cat(
            [state, change_probability, reliability, stage_map], dim=1
        )
        residual = self.tail(self.body(self.head(side)))
        scale = F.softplus(self.raw_residual_scale).clamp(max=0.5)
        return state + scale * residual


# class EdgeAwareMeanField(nn.Module):
#     """Differentiable four-neighbour mean-field solver for a Potts prior."""
#
#     def __init__(self, iterations: int = 3) -> None:
#         super().__init__()
#         self.iterations = int(iterations)
#
#     @staticmethod
#     def _neighbours(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
#         h, w = x.shape[-2:]
#         padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
#         left = padded[..., 1 : 1 + h, 0:w]
#         right = padded[..., 1 : 1 + h, 2 : 2 + w]
#         up = padded[..., 0:h, 1 : 1 + w]
#         down = padded[..., 2 : 2 + h, 1 : 1 + w]
#         return left, right, up, down
#
#     def forward(
#  #The complete code will be organized and published after the paper is published
#         return q


class HypothesisStage(nn.Module):
    """The complete code will be organized and published after the paper is published"""
        # return next_z1, next_z2, outputs


class UCFUnfoldNet(nn.Module):

    def __init__(
        self,
        t1_channels: int,
        t2_channels: int,
        base_channels: int = 32,
        latent_channels: int = 48,
        prior_channels: int = 64,
        stages: int = 5,
        change_prior: float = 0.10,
        mean_field_iterations: int = 3,
        adaptive_kappa: bool = True,
        min_variance: float = 0.02,     #  1e-3
        max_variance: float = 5.0,
    ) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be at least 1.")
        self.t1_channels = int(t1_channels)
        self.t2_channels = int(t2_channels)
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)
        self.prior_channels = int(prior_channels)
        self.num_stages = int(stages)
        self.change_prior = float(change_prior)
        self.mean_field_iterations = int(mean_field_iterations)
        self.adaptive_kappa = bool(adaptive_kappa)
        self.min_variance = float(min_variance)
        self.max_variance = float(max_variance)

        self.stem1 = ModalityStem(self.t1_channels, self.base_channels)
        self.stem2 = ModalityStem(self.t2_channels, self.base_channels)
        self.shared_backbone = SharedContentBackbone(
            self.base_channels, self.latent_channels
        )

        self.uncertainty1 = nn.Conv2d(self.latent_channels, 1, 1)
        self.uncertainty2 = nn.Conv2d(self.latent_channels, 1, 1)
        nn.init.zeros_(self.uncertainty1.weight)
        nn.init.zeros_(self.uncertainty2.weight)
        initial_variance = 0.10
        initial_bias = _inverse_softplus(
            max(initial_variance - self.min_variance, 1e-4)
        )
        nn.init.constant_(self.uncertainty1.bias, initial_bias)
        nn.init.constant_(self.uncertainty2.bias, initial_bias)

        self.decoder1 = ModalityDecoder(
            self.latent_channels, self.base_channels, self.t1_channels
        )
        self.decoder2 = ModalityDecoder(
            self.latent_channels, self.base_channels, self.t2_channels
        )
        self.shared_prior = SharedScenePrior(
            self.latent_channels, self.prior_channels
        )
        self.stages = nn.ModuleList(
            [
                HypothesisStage(
                    change_prior=self.change_prior,
                    mean_field_iterations=self.mean_field_iterations,
                    adaptive_kappa=self.adaptive_kappa,
                    init_mu=1.0 + 0.15 * index,
                    init_rho=max(0.15, 0.8 - 0.1 * index),
                    init_temperature_ratio=0.30,
                    init_smoothness=1.0,
                    init_edge_sigma=0.5,
                )
                for index in range(self.num_stages)
            ]
        )

    def config_dict(self) -> Dict[str, object]:
        return {
            "t1_channels": self.t1_channels,
            "t2_channels": self.t2_channels,
            "base_channels": self.base_channels,
            "latent_channels": self.latent_channels,
            "prior_channels": self.prior_channels,
            "stages": self.num_stages,
            "change_prior": self.change_prior,
            "mean_field_iterations": self.mean_field_iterations,
            "adaptive_kappa": self.adaptive_kappa,
            "min_variance": self.min_variance,
            "max_variance": self.max_variance,
        }

    def _encode(
        self, t1: torch.Tensor, t2: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        f1 = self.shared_backbone(self.stem1(t1))
        f2 = self.shared_backbone(self.stem2(t2))

        variance1 = (
            self.min_variance + F.softplus(self.uncertainty1(f1))
        ).clamp(max=self.max_variance)
        variance2 = (
            self.min_variance + F.softplus(self.uncertainty2(f2))
        ).clamp(max=self.max_variance)
        precision1 = variance1.reciprocal().clamp(max=100.0)
        precision2 = variance2.reciprocal().clamp(max=100.0)
        return f1, f2, variance1, variance2, precision1, precision2

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        n_stages: Optional[int] = None,
    ) -> Dict[str, object]:
        #The complete code will be organized and published after the paper is published
        return {
            "feature1": f1,
            "feature2": f2,
            "variance1": var1,
            "variance2": var2,
            "precision1": w1,
            "precision2": w2,
            "recon_auto1": recon_auto1,
            "recon_auto2": recon_auto2,
            "cross_recon1": cross_recon1,
            "cross_recon2": cross_recon2,
            "state1": z1,
            "state2": z2,
            "recon_state1": recon_state1,
            "recon_state2": recon_state2,
            "common": final_common,
            "delta": final_delta,
            "change_probability": final_mask,
            "stage_outputs": stage_outputs,
        }
