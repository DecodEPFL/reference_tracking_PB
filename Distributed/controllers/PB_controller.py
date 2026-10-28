"""
Performance Boosting (PB) controller (ported from Context_PB ``pb_core/controller.py``).

- Causal, step-by-step operation using (B, T, N) tensors (T=1 typical).
- Reconstructs the disturbance w from the nominal model and the last state/control.
- The operator is any ``OperatorBase``, e.g. the factorized M_b(w,z) ⊠ M_p(w,z).
- In the distributed setting w is still reconstructed jointly here (the nominal model is
  block-diagonal, so this is exact per agent); ``DistributedOperator`` then splits w, z
  into per-agent channels and computes each u^i with its own operator M^i.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

import torch
import torch.nn as nn

from Distributed.controllers.operators import as_bt, OperatorBase


class NominalPlant(Protocol):
    """Nominal plant dynamics used to reconstruct the disturbance w.

    Extra keyword arguments carry exogenous signals of the transition that are not
    part of the plant state (for the robots: integrator state and reference).
    """

    def nominal_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, t: Optional[int] = None, **transition_kwargs: Any
    ) -> torch.Tensor:
        """Return x_next_nominal given x and u. Shapes (B, T, Nx) -> (B, T, Nx)."""
        ...


@dataclass
class PBState:
    x_tm1: torch.Tensor
    u_tm1: torch.Tensor


class PBController(nn.Module):
    """
    Performance Boosting controller.

    At time t, the controller receives x_t and the context z_t.
    It reconstructs w_t = x_t - f_nom(x_{t-1}, u_{t-1}; transition_kwargs)
    and returns u_t = output_scale * M(w_t, z_t) (causal).

    ``reset(x_init, u_init)`` stores the state/input *before* the first plant
    step, so the first reconstruction recovers w_0 (e.g. the initial condition
    injected as a disturbance) through the same nominal model.

    Shapes are (B, T, N), with T=1 in step-by-step usage.
    """

    def __init__(
        self,
        plant: NominalPlant,
        operator: OperatorBase,
        u_dim: int,
        output_scale: float = 1.0,
        detach_state: bool = False,
    ):
        super().__init__()
        # Keep the plant out of the module tree: it is not trained and is saved separately.
        object.__setattr__(self, "plant", plant)
        self.operator = operator
        self.u_dim = int(u_dim)
        self.output_scale = float(output_scale)
        self.detach_state = detach_state
        self.state: Optional[PBState] = None

    def reset(self, x_init: torch.Tensor, u_init: Optional[torch.Tensor] = None) -> None:
        x_init = as_bt(x_init)
        if u_init is None:
            u_init = torch.zeros(*x_init.shape[:2], self.u_dim, device=x_init.device, dtype=x_init.dtype)
        self.state = PBState(x_tm1=x_init, u_tm1=as_bt(u_init))
        self.operator.reset()

    def reconstruct_w(self, x_t: torch.Tensor, t: Optional[int] = None, **transition_kwargs: Any) -> torch.Tensor:
        """w_t = x_t - f_nom(x_{t-1}, u_{t-1}) for the transition (t-1 -> t)."""
        assert self.state is not None
        t_prev = None if t is None else (t - 1)
        x_nom = self.plant.nominal_dynamics(self.state.x_tm1, self.state.u_tm1, t_prev, **transition_kwargs)
        return x_t - x_nom

    def forward_step(
        self,
        x_t: torch.Tensor,
        z_t: Optional[torch.Tensor] = None,
        t: Optional[int] = None,
        **transition_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute u_t and return (u_t, w_t)."""
        if self.state is None:
            raise RuntimeError("Call reset(x_init) before forward_step().")

        x_t = as_bt(x_t)
        z_t = as_bt(z_t) if z_t is not None else None

        w_t = self.reconstruct_w(x_t, t, **transition_kwargs)
        if getattr(self.operator, "uses_feedback", False):
            # neighbor signals: state measured now and the input applied at the previous step
            # (u_t is computed by all agents simultaneously, so u_{t-1} is the latest available)
            op_out = self.operator(w_t, z_t, x=x_t, u_prev=self.state.u_tm1)
        else:
            op_out = self.operator(w_t, z_t)
        u_t = self.output_scale * as_bt(op_out)

        if self.detach_state:
            self.state.x_tm1, self.state.u_tm1 = x_t.detach(), u_t.detach()
        else:
            self.state.x_tm1, self.state.u_tm1 = x_t, u_t
        return u_t, w_t

    def forward(self, x_t, z_t=None, t=None, **transition_kwargs):
        return self.forward_step(x_t, z_t, t, **transition_kwargs)
