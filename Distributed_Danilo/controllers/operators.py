"""
PB operator building blocks (ported from Context_PB ``pb_core/controller.py``).

All tensors are shaped (B, T, N), with T=1 in step-by-step closed-loop usage.
The main operator is the factorization

    u = M_b(w, z) ⊠ M_p(w, z),

where M_p is a (possibly contextual) recurrent l2-stable operator producing s
features and M_b is a bounded, memoryless mixer producing a (nu x s) matrix.
"""

from __future__ import annotations

import inspect
from typing import Callable, Optional

import torch
import torch.nn as nn


def as_bt(x: torch.Tensor) -> torch.Tensor:
    """Ensure input is shaped (B, T, N). Accepts (B, N) or (B, T, N)."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected tensor with 2 or 3 dims, got shape {tuple(x.shape)}")


def accepts_context(module: nn.Module) -> bool:
    """True if ``module.forward`` takes a second positional argument (the context z)."""
    params = [
        p for p in inspect.signature(module.forward).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(params) >= 2


class WIntegralAugmenter(nn.Module):
    """Augments w_t with its causal leaky integral w̄_t = γ * w̄_{t-1} + w_t.

    Output: cat([w_t, w̄_t], dim=-1)  →  shape (B, T, 2*Nw).

    When the true disturbance is near-zero after t=0 (w_0 = x_0, w_{t≥1} ≈ 0),
    the integral reduces to γ^t * x_0 — a smoothly decaying encoding of the
    initial condition that keeps the operator input non-trivial throughout the
    rollout, making training significantly easier.
    """

    def __init__(self, decay: float = 0.97):
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self._bar_w: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._bar_w = None

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w: (B, T, Nw) — expects T=1 in step-by-step usage."""
        if self._bar_w is None:
            self._bar_w = w
        else:
            self._bar_w = self.decay * self._bar_w + w
        return torch.cat([w, self._bar_w], dim=-1)


class OperatorBase(nn.Module):
    """Base class for PB operators. Accepts w and optional z."""

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def reset(self) -> None:
        """Optional: reset internal dynamical state."""
        return None


def boxtimes_timewise(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Timewise matrix–vector product.

      A: (B, T, r, s)
      v: (B, T, s)
      y: (B, T, r)
    """
    if A.ndim != 4:
        raise ValueError(f"A must be 4D (B,T,r,s). Got shape {tuple(A.shape)}")
    if v.ndim != 3:
        raise ValueError(f"v must be 3D (B,T,s). Got shape {tuple(v.shape)}")
    if A.shape[:2] != v.shape[:2]:
        raise ValueError(
            f"Leading dims of A and v must match. Got A{tuple(A.shape)} vs v{tuple(v.shape)}"
        )
    if A.shape[-1] != v.shape[-1]:
        raise ValueError(
            f"Incompatible inner dims: A.shape[-1]={A.shape[-1]} vs v.shape[-1]={v.shape[-1]}"
        )
    return torch.einsum("btrs,bts->btr", A, v)


class FactorizedOperator(OperatorBase):
    """M(w,z) = M_b(w,z) ⊠ M_p(w,z) with configurable product ⊠.

    Context reaches M_p in one of two ways:
      * M_p takes z itself (e.g. ``MpContextualSSM``): called as ``mp(w, z)``,
        and the context ports (input/gate/select/mixer) are chosen there;
      * otherwise, if ``mp_context_lifter`` is set, M_p sees ``[w, lift(z)]``.

    When mp_only=True the M_b branch is skipped entirely and the operator
    reduces to u = M_p(.). In this mode mb may be None and M_p must already
    output (B,T,nu).
    """

    def __init__(
        self,
        mp: nn.Module,
        mb: Optional[nn.Module] = None,
        mp_context_lifter: Optional[nn.Module] = None,
        product_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = boxtimes_timewise,
        mp_only: bool = False,
        w_augmenter: Optional[nn.Module] = None,
    ):
        super().__init__()
        if not mp_only and mb is None:
            raise ValueError("mb must be provided when mp_only=False.")
        self.mp = mp
        self.mb = mb
        self.mp_context_lifter = mp_context_lifter
        self.product_fn = product_fn
        self.mp_only = mp_only
        self.w_augmenter = w_augmenter
        self.mp_takes_context = accepts_context(mp)
        if self.mp_takes_context and mp_context_lifter is not None:
            raise ValueError("mp_context_lifter is redundant when M_p consumes the context itself.")

    def reset(self) -> None:
        for module in (self.mp, self.mb, self.mp_context_lifter, self.w_augmenter):
            if module is not None and hasattr(module, "reset"):
                module.reset()

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = as_bt(w)
        if self.w_augmenter is not None:
            w = self.w_augmenter(w)  # (B, T, 2*Nw) — M_p and M_b both see augmented w

        if self.mp_takes_context:
            v = self.mp(w, z)
        elif self.mp_context_lifter is not None:
            if z is None:
                raise ValueError("z must be provided when mp_context_lifter is enabled.")
            z_lift = as_bt(self.mp_context_lifter(z))
            if w.shape[:2] != z_lift.shape[:2]:
                raise ValueError(
                    "w and lifted z must share leading dims. "
                    f"Got w{tuple(w.shape)} vs z_lift{tuple(z_lift.shape)}"
                )
            v = self.mp(torch.cat([w, z_lift], dim=-1))
        else:
            v = self.mp(w)
        v = as_bt(v)

        if self.mp_only:
            return v
        A = self.mb(w, z) if z is not None else self.mb(w)
        return self.product_fn(A, v)
