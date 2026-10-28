"""
Bounded MLP operator for M_b(w, z).

Outputs a matrix-valued sequence A_t with shape (B, T, r, s).
The operator is bounded via spectral normalization plus a configurable output map
(`tanh`, `softsign`, or hard `clamp`).
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from Centralized.controllers.operators import as_bt


class BoundedMLPOperator(nn.Module):
    """Bounded MLP for M_b(w, z)."""

    def __init__(
        self,
        w_dim: int,
        z_dim: int,
        r: int,
        s: int,
        hidden_dim: int = 64,
        num_layers: int = 6,
        gamma: float = 1.0,
        activation: Optional[nn.Module] = None,
        time_first: bool = False,
        use_z_residual: bool = True,
        z_residual_gain: float = 5.0,
        z_hidden_dim: Optional[int] = None,
        bound_mode: str = "tanh",
        clamp_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.w_dim = w_dim
        self.z_dim = z_dim
        self.r = r
        self.s = s
        self.gamma = float(gamma)
        self.time_first = time_first
        self.use_z_residual = use_z_residual
        self.z_residual_gain = float(z_residual_gain)

        valid_bound_modes = {"tanh", "softsign", "clamp"}
        if bound_mode not in valid_bound_modes:
            raise ValueError(f"bound_mode must be one of {valid_bound_modes}, got {bound_mode!r}")
        self.bound_mode = bound_mode
        self.bound_value = float(self.gamma if clamp_value is None else clamp_value)
        if self.bound_value <= 0:
            raise ValueError(f"bound_value must be > 0, got {self.bound_value}")

        in_dim = w_dim + z_dim
        def make_act() -> nn.Module:
            if activation is None:
                return nn.ReLU()
            return copy.deepcopy(activation)

        dims = [in_dim] + [hidden_dim] * (num_layers - 2) + [r * s]
        layers = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            linear = spectral_norm(linear)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(make_act())
        self.net = nn.Sequential(*layers)

        if self.use_z_residual:
            zh = z_hidden_dim if z_hidden_dim is not None else hidden_dim
            self.z_net = nn.Sequential(
                spectral_norm(nn.Linear(z_dim, zh)),
                make_act(),
                spectral_norm(nn.Linear(zh, r * s)),
            )
        else:
            self.z_net = None

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is None:
            raise ValueError("z must be provided for M_b(w, z)")

        if self.time_first:
            # Expect (T, B, N) -> (B, T, N)
            w = w.permute(1, 0, 2)
            z = z.permute(1, 0, 2)

        w = as_bt(w)
        z = as_bt(z)

        if w.shape[:2] != z.shape[:2]:
            raise ValueError(f"w and z must share leading dims, got w{tuple(w.shape)} vs z{tuple(z.shape)}")
        if w.shape[-1] != self.w_dim:
            raise ValueError(f"w last dim must be {self.w_dim}, got {w.shape[-1]}")
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"z last dim must be {self.z_dim}, got {z.shape[-1]}")

        B, T, _ = w.shape
        inp = torch.cat([w, z], dim=-1).reshape(B * T, -1)
        out = self.net(inp)

        if self.z_net is not None:
            z_in = z.reshape(B * T, -1)
            out = out + self.z_residual_gain * self.z_net(z_in)

        if self.bound_mode == "tanh":
            # Uses a larger bound_value to reduce early saturation while keeping hard bounds.
            out = self.bound_value * torch.tanh(out / self.bound_value)
        elif self.bound_mode == "softsign":
            out = self.bound_value * (out / (1.0 + torch.abs(out)))
        else:  # clamp
            out = torch.clamp(out, min=-self.bound_value, max=self.bound_value)

        out = out.view(B, T, self.r, self.s)

        if self.time_first:
            out = out.permute(1, 0, 2, 3)
        return out

    def reset(self) -> None:
        return None


class DiagonalBoundedMLPOperator(nn.Module):
    """Bounded diagonal mixer for the rPB special case.

    The network emits only the diagonal entries g_t in R^dim and returns
    diag(g_t) with shape (B, T, dim, dim), so the standard PB factorized
    product computes u_t = diag(g_t) M_p(w_t).
    """

    def __init__(
        self,
        w_dim: int,
        z_dim: int,
        dim: int,
        hidden_dim: int = 64,
        num_layers: int = 6,
        gamma: float = 1.0,
        activation: Optional[nn.Module] = None,
        time_first: bool = False,
        use_z_residual: bool = True,
        z_residual_gain: float = 5.0,
        z_hidden_dim: Optional[int] = None,
        bound_mode: str = "tanh",
        clamp_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        self.w_dim = int(w_dim)
        self.z_dim = int(z_dim)
        self.dim = int(dim)
        self.gamma = float(gamma)
        self.time_first = time_first
        self.use_z_residual = use_z_residual
        self.z_residual_gain = float(z_residual_gain)

        valid_bound_modes = {"tanh", "softsign", "clamp"}
        if bound_mode not in valid_bound_modes:
            raise ValueError(f"bound_mode must be one of {valid_bound_modes}, got {bound_mode!r}")
        self.bound_mode = bound_mode
        self.bound_value = float(self.gamma if clamp_value is None else clamp_value)
        if self.bound_value <= 0:
            raise ValueError(f"bound_value must be > 0, got {self.bound_value}")

        in_dim = self.w_dim + self.z_dim

        def make_act() -> nn.Module:
            if activation is None:
                return nn.ReLU()
            return copy.deepcopy(activation)

        dims = [in_dim] + [hidden_dim] * (num_layers - 2) + [self.dim]
        layers = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            linear = spectral_norm(linear)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(make_act())
        self.net = nn.Sequential(*layers)

        if self.use_z_residual:
            zh = z_hidden_dim if z_hidden_dim is not None else hidden_dim
            self.z_net = nn.Sequential(
                spectral_norm(nn.Linear(self.z_dim, zh)),
                make_act(),
                spectral_norm(nn.Linear(zh, self.dim)),
            )
        else:
            self.z_net = None

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is None:
            raise ValueError("z must be provided for diagonal M_b(w, z)")

        if self.time_first:
            w = w.permute(1, 0, 2)
            z = z.permute(1, 0, 2)

        w = as_bt(w)
        z = as_bt(z)

        if w.shape[:2] != z.shape[:2]:
            raise ValueError(f"w and z must share leading dims, got w{tuple(w.shape)} vs z{tuple(z.shape)}")
        if w.shape[-1] != self.w_dim:
            raise ValueError(f"w last dim must be {self.w_dim}, got {w.shape[-1]}")
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"z last dim must be {self.z_dim}, got {z.shape[-1]}")

        B, T, _ = w.shape
        inp = torch.cat([w, z], dim=-1).reshape(B * T, -1)
        diag = self.net(inp)

        if self.z_net is not None:
            z_in = z.reshape(B * T, -1)
            diag = diag + self.z_residual_gain * self.z_net(z_in)

        if self.bound_mode == "tanh":
            diag = self.bound_value * torch.tanh(diag / self.bound_value)
        elif self.bound_mode == "softsign":
            diag = self.bound_value * (diag / (1.0 + torch.abs(diag)))
        else:
            diag = torch.clamp(diag, min=-self.bound_value, max=self.bound_value)

        out = torch.diag_embed(diag.view(B, T, self.dim))

        if self.time_first:
            out = out.permute(1, 0, 2, 3)
        return out

    def reset(self) -> None:
        return None
