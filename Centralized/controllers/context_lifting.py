"""Context lifting modules for feeding bounded context into an l_p disturbance processor."""

from __future__ import annotations

import torch
import torch.nn as nn

from Centralized.controllers.operators import as_bt


class LpContextLifter(nn.Module):
    """
    Lift bounded context z into an l_p-compatible sequence.

    The lifted output has the form:
      zeta_t = scale * decay_t * psi(z_t),
    where psi(z_t) is uniformly bounded and decay_t belongs to l_p.
    """

    def __init__(
        self,
        *,
        z_dim: int,
        out_dim: int,
        lift_type: str = "identity",
        hidden_dim: int = 32,
        decay_law: str = "poly",
        decay_rate: float = 0.08,
        decay_power: float = 1.2,
        decay_horizon: int = 80,
        lp_p: float = 2.0,
        scale: float = 1.0,
    ):
        super().__init__()
        if z_dim <= 0 or out_dim <= 0:
            raise ValueError(f"z_dim and out_dim must be > 0, got z_dim={z_dim}, out_dim={out_dim}")
        if lp_p < 1.0:
            raise ValueError(f"lp_p must be >= 1, got {lp_p}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        valid_lift = {"identity", "linear", "mlp"}
        if lift_type not in valid_lift:
            raise ValueError(f"lift_type must be in {valid_lift}, got {lift_type!r}")
        valid_decay = {"exp", "poly", "finite"}
        if decay_law not in valid_decay:
            raise ValueError(f"decay_law must be in {valid_decay}, got {decay_law!r}")
        if decay_law == "exp" and decay_rate <= 0:
            raise ValueError(f"decay_rate must be > 0 for exp law, got {decay_rate}")
        if decay_law == "poly":
            if decay_power <= 1.0 / lp_p:
                raise ValueError(
                    "decay_power must satisfy decay_power > 1/p for l_p compatibility. "
                    f"Got decay_power={decay_power}, p={lp_p}"
                )
        if decay_law == "finite" and decay_horizon <= 0:
            raise ValueError(f"decay_horizon must be > 0 for finite law, got {decay_horizon}")

        self.z_dim = int(z_dim)
        self.out_dim = int(out_dim)
        self.lift_type = lift_type
        self.decay_law = decay_law
        self.decay_rate = float(decay_rate)
        self.decay_power = float(decay_power)
        self.decay_horizon = int(decay_horizon)
        self.lp_p = float(lp_p)
        self.scale = float(scale)

        if lift_type == "identity":
            if self.out_dim != self.z_dim:
                raise ValueError(
                    "identity lift requires out_dim == z_dim. "
                    f"Got out_dim={self.out_dim}, z_dim={self.z_dim}"
                )
            self.mapper = nn.Identity()
        elif lift_type == "linear":
            self.mapper = nn.Linear(self.z_dim, self.out_dim, bias=False)
        else:
            self.mapper = nn.Sequential(
                nn.Linear(self.z_dim, int(hidden_dim), bias=False),
                nn.Tanh(),
                nn.Linear(int(hidden_dim), self.out_dim, bias=False),
            )

        # Streaming step counter kept as tensor ARITHMETIC end to end: reading
        # it via `.item()` would cost a GPU->CPU sync at every closed-loop
        # timestep (160 pipeline stalls per rollout) and break torch.compile
        # graphs, while a Python int would bake the step value into dynamo
        # guards (one recompile per timestep). Non-persistent: never saved.
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

    def reset(self) -> None:
        with torch.no_grad():
            self._step.zero_()

    def _decay(self, t_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        offsets = torch.arange(int(t_steps), device=device, dtype=dtype)
        idx = (self._step.to(device=device, dtype=dtype) + offsets).view(1, t_steps, 1)
        if self.decay_law == "exp":
            d = torch.exp(-self.decay_rate * idx)
        elif self.decay_law == "poly":
            d = (1.0 + idx) ** (-self.decay_power)
        else:
            d = (idx < float(self.decay_horizon)).to(dtype)
        with torch.no_grad():
            self._step += int(t_steps)
        return d

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = as_bt(z)
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"z last dim must be {self.z_dim}, got {z.shape[-1]}")
        feat = self.mapper(z)
        # Keep psi(z_t) uniformly bounded.
        feat = torch.tanh(feat)
        decay = self._decay(z.shape[1], z.device, z.dtype)
        return self.scale * decay * feat
