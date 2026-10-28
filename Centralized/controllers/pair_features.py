"""
Relative positions between the robots, appended to the context of the centralized controller.

For every pair i < j the controller receives, at time t, the offset p^j_t - p^i_t = (dx, dy)
measured from the state x_t (3 robots: pairs (1,2), (1,3), (2,3), 6 features). They are computed
before u_t, so the controller stays causal. Unlike the reconstructed disturbance w_t (zero after
the initial condition in the robots datasets), they tell M where the robots are relative to each
other at every step, as the neighbor features do for each M^i in the distributed controller.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import torch
import torch.nn as nn


class PairOffsets(nn.Module):
    """x_t (B, T, Nx) -> [p^j_t - p^i_t for i < j] (B, T, 2 * n_pairs).

    Args:
        pos_idx: per-agent position channels in x, each [x-index, y-index].
    """

    def __init__(self, pos_idx: Sequence[Sequence[int]]):
        super().__init__()
        self.pairs = list(combinations(range(len(pos_idx)), 2))
        self.out_dim = 2 * len(self.pairs)
        # non-persistent: checkpoints are unchanged, the indices are rebuilt from the plant
        self.register_buffer("pos_idx", torch.as_tensor(pos_idx, dtype=torch.long), persistent=False)
        self.register_buffer("first", torch.as_tensor([i for i, _ in self.pairs], dtype=torch.long), persistent=False)
        self.register_buffer("second", torch.as_tensor([j for _, j in self.pairs], dtype=torch.long), persistent=False)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        p = x_t[..., self.pos_idx]                                  # (B, T, N, 2)
        rel = p[..., self.second, :] - p[..., self.first, :]        # (B, T, n_pairs, 2)
        return rel.flatten(-2)                                      # [dx_12, dy_12, dx_13, dy_13, ...]
