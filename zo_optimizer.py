"""
zo_optimizer.py — GRS with cosine LR decay, bias boost (Optuna-optimized).
"""

from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
import math

class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.0157,
        eps: float = 0.0003,
        perturbation_mode: str = "rademacher",
        n_directions: int = 11,
        total_steps: int = 256,
        bias_lr_mult: float = 4.0,
    ) -> None:
        self.model = model
        self.initial_lr = lr
        self.eps = eps
        self.n_directions = n_directions
        self.total_steps = total_steps
        self.current_step = 0
        self.bias_lr_mult = bias_lr_mult

        if perturbation_mode not in ("gaussian", "uniform", "rademacher"):
            raise ValueError(f"Unknown perturbation_mode: {perturbation_mode}")
        self.perturbation_mode = perturbation_mode

        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        if self.perturbation_mode == "gaussian":
            u = torch.randn_like(param)
        elif self.perturbation_mode == "uniform":
            u = torch.rand_like(param) * 2.0 - 1.0
        else:
            u = torch.where(
                torch.rand_like(param) > 0.5,
                torch.ones_like(param),
                -torch.ones_like(param),
            )
        return u

    def _apply_perturbation(self, params, deltas, sign):
        for name, param in params.items():
            param.data.add_(sign * self.eps * deltas[name])

    def step(self, loss_fn: Callable[[], float]) -> float:
        params = self._active_params()
        self.current_step += 1

        lr_current = self.initial_lr * 0.5 * (1 + math.cos(math.pi * self.current_step / self.total_steps))

        with torch.no_grad():
            loss_before = loss_fn()
            best_loss = float("inf")
            best_deltas = None
            best_sign = 0

            for _ in range(self.n_directions):
                deltas = {}
                for name, param in params.items():
                    deltas[name] = self._sample_direction(param)

                self._apply_perturbation(params, deltas, +1)
                loss_plus = loss_fn()
                self._apply_perturbation(params, deltas, -1)

                if loss_plus < best_loss:
                    best_loss = loss_plus
                    best_deltas = {n: d.clone() for n, d in deltas.items()}
                    best_sign = +1

                self._apply_perturbation(params, deltas, -1)
                loss_minus = loss_fn()
                self._apply_perturbation(params, deltas, +1)

                if loss_minus < best_loss:
                    best_loss = loss_minus
                    best_deltas = {n: d.clone() for n, d in deltas.items()}
                    best_sign = -1

            if best_deltas is not None:
                scale_weight = lr_current / self.eps * best_sign
                scale_bias = lr_current * self.bias_lr_mult / self.eps * best_sign
                params["fc.weight"].data.add_(scale_weight * self.eps * best_deltas["fc.weight"])
                params["fc.bias"].data.add_(scale_bias * self.eps * best_deltas["fc.bias"])

        return float(loss_before)
