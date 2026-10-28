"""
Distributed PB operator: one operator M^i per agent.

The disturbance w is reconstructed jointly by the PBController (the nominal model is
block-diagonal across agents, so this recovers every w^i exactly). This operator then
splits w and z into per-agent channels and lets each agent compute its own input,

    u^i = M^i(w[w_idx[i]], z[z_idx[i]]),        u = [u^1, ..., u^N]  (scattered back by ``u_idx``).

``w_idx`` and ``z_idx`` set the information pattern: agent i's own channels (local) or
all of w and z, as the centralized operator sees them (full). With ``neighbor_features``
(info pattern "neighbors") the context of M^i is extended with the signals its neighbors
send over the communication graph, computed causally from x_t, u_{t-1} (and z_t for their targets):

    u^i_t = M^i(w^i_t, [z^i_t, n^i(x_t, u_{t-1})]).

With ``RelayFeatures`` (info pattern "relay") the agents also forward what they receive: M^i
gets the disturbances of every agent j reaching it, delayed by the hops d_ij, in its SSM input,
and their targets, relative positions (and inputs) in its context:

    u^i_t = M^i([w^i_t, w^j_{t-d_ij}, ...], [z^i_t, r^i(x, u, z, w)_t]).

With ``share_weights=True`` a single operator M is applied to every agent: the agents
are folded into the batch dimension, so each agent keeps its own recurrent state.
With independent weights and ``batch_agents=True`` the N operators run in one vectorised
call (``AgentBatch``, torch.func.vmap) instead of one after the other; same results up to
float rounding.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import torch
import torch.nn as nn

from Distributed.controllers.batched_agents import AgentBatch
from Distributed.controllers.interconnection import NeighborFeatures, RelayFeatures
from Distributed.controllers.operators import as_bt, OperatorBase


class DistributedOperator(OperatorBase):
    """u^i = M^i(w[..., w_idx[i]], z[..., z_idx[i]]), concatenated into u[..., u_idx[i]].

    Args:
        agent_ops: N operators (independent weights) or a single one (shared weights).
        w_idx, z_idx, u_idx: per-agent channel indices, each a list of N equal-length lists.
        neighbor_features: optional per-agent neighbor context appended to z^i; the operator
            then needs the measured state x_t and the previous input u_{t-1} at every step.
            A ``RelayFeatures`` also appends the relayed disturbances to w^i.
        batch_agents: run independent agent operators in one vmapped call (AgentBatch). Raises
            NotImplementedError if the operators do not support it (see AgentBatch).
        compile_agents: with batch_agents, also compile the vmapped step with torch.compile.
    """

    def __init__(
        self,
        agent_ops: Sequence[OperatorBase],
        w_idx: Sequence[Sequence[int]],
        z_idx: Sequence[Sequence[int]],
        u_idx: Sequence[Sequence[int]],
        neighbor_features: Optional[NeighborFeatures | RelayFeatures] = None,
        batch_agents: bool = False,
        compile_agents: bool = False,
    ):
        super().__init__()
        self.n_agents = len(w_idx)
        if not (len(z_idx) == len(u_idx) == self.n_agents):
            raise ValueError("w_idx, z_idx and u_idx must list the same number of agents.")
        if len(agent_ops) not in (1, self.n_agents):
            raise ValueError(f"Expected 1 (shared) or {self.n_agents} agent operators, got {len(agent_ops)}.")
        self.share_weights = len(agent_ops) == 1 and self.n_agents > 1
        self.agent_ops = nn.ModuleList(agent_ops)
        self.neighbor_features = neighbor_features
        self.relays_w = isinstance(neighbor_features, RelayFeatures)
        if self.share_weights and neighbor_features is not None and (
                len(set(neighbor_features.out_dims)) > 1
                or (self.relays_w and len(set(neighbor_features.w_out_dims)) > 1)):
            raise ValueError("Shared weights need every agent to have the same number of neighbors "
                             f"(got neighbor context sizes {neighbor_features.out_dims}); use independent weights.")

        # (N, d) index tables; non-persistent so checkpoints only hold the operator weights
        self.register_buffer("w_idx", torch.as_tensor(w_idx, dtype=torch.long), persistent=False)
        self.register_buffer("z_idx", torch.as_tensor(z_idx, dtype=torch.long), persistent=False)
        self.register_buffer("u_idx", torch.as_tensor(u_idx, dtype=torch.long), persistent=False)
        self.u_dim = int(self.u_idx.numel())
        # plain attribute (not a submodule): holds a template copy that must stay out of the parameters
        self.agent_batch = AgentBatch(self.agent_ops, compile=compile_agents) if batch_agents and not self.share_weights else None

    @property
    def uses_feedback(self) -> bool:
        """True if forward needs x_t and u_{t-1} (neighbor signals)."""
        return self.neighbor_features is not None

    def reset(self) -> None:
        if hasattr(self.neighbor_features, "reset"):
            self.neighbor_features.reset()         # relay history
        if self.agent_batch is not None:
            self.agent_batch.reset()               # resets every agent, then stacks them
            return
        for op in self.agent_ops:
            op.reset()

    def agent_operator(self, i: int) -> OperatorBase:
        return self.agent_ops[0 if self.share_weights else i]

    def unique_operators(self) -> Iterator[OperatorBase]:
        return iter(self.agent_ops)

    @staticmethod
    def _split(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """(B, T, N_x) -> (B, T, N, d) per-agent channels."""
        return x[..., idx]

    def forward(
        self,
        w: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        u_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """w, z: (B, T, ·); x: state measured at t, u_prev: input applied at t-1 (neighbors only)."""
        w = as_bt(w)
        z = as_bt(z) if z is not None else None
        B, T = w.shape[:2]
        w_ag = self._split(w, self.w_idx)                                   # (B, T, N, dw)
        z_ag = self._split(z, self.z_idx) if z is not None else None        # (B, T, N, dz)
        nb = w_nb = None
        if self.neighbor_features is not None:
            if x is None or u_prev is None:
                raise ValueError("The neighbors info pattern needs x_t and u_{t-1}.")
            if self.relays_w:
                w_nb, nb = self.neighbor_features(as_bt(x), as_bt(u_prev), z, w)   # N x (B, T, ·) each
            else:
                nb = self.neighbor_features(as_bt(x), as_bt(u_prev), z)        # N x (B, T, d_i)

        if self.share_weights:
            if nb is not None:   # equal sizes, checked at construction
                nb_ag = torch.stack(nb, dim=2)
                z_ag = nb_ag if z_ag is None else torch.cat([z_ag, nb_ag], dim=-1)
            if w_nb is not None:
                w_ag = torch.cat([w_ag, torch.stack(w_nb, dim=2)], dim=-1)

            # fold agents into the batch: (B, T, N, d) -> (B*N, T, d)
            def fold(x):
                return x.permute(0, 2, 1, 3).reshape(B * self.n_agents, T, x.shape[-1])
            u_fold = as_bt(self.agent_ops[0](fold(w_ag), fold(z_ag) if z_ag is not None else None))
            u_ag = u_fold.reshape(B, self.n_agents, T, -1).permute(0, 2, 1, 3)   # (B, T, N, du)
        else:
            def w_of(i):
                return w_ag[:, :, i] if w_nb is None else torch.cat([w_ag[:, :, i], w_nb[i]], dim=-1)

            def z_of(i):
                z_i = z_ag[:, :, i] if z_ag is not None else None
                if nb is None:
                    return z_i
                return nb[i] if z_i is None else torch.cat([z_i, nb[i]], dim=-1)
            if self.agent_batch is not None:
                z_all = [z_of(i) for i in range(self.n_agents)]
                z_all = None if z_all[0] is None else torch.stack(z_all)     # (N, B, T, dz)
                w_all = torch.stack([w_of(i) for i in range(self.n_agents)])  # (N, B, T, dw)
                u_ag = self.agent_batch(w_all, z_all)                        # (N, B, T, du)
                u_ag = u_ag.permute(1, 2, 0, 3)                              # (B, T, N, du)
            else:
                u_ag = torch.stack([
                    as_bt(op(w_of(i), z_of(i))) for i, op in enumerate(self.agent_ops)
                ], dim=2)                                                    # (B, T, N, du)

        u = w.new_zeros(B, T, self.u_dim)
        return u.index_copy(-1, self.u_idx.reshape(-1), u_ag.reshape(B, T, -1))
