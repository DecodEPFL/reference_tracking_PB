"""
Communication graph between agents and the neighbor signals it carries.

Convention: ``adjacency[i, j] = 1`` means agent i receives information from agent j
(directed in general, no self-loops). Under ``--info-pattern neighbors`` agent i receives,
for every neighbor j, at time t:

    [p^j_t - p^i_t,  ||p^j_t - p^i_t||,  u^j_{t-1}  (, z^j_t)]

with z^j_t the target of agent j when ``ref_idx`` is given (--neighbor-targets).

Causality: positions come from the state x_t measured before u_t is computed, while the
neighbors compute u^j_t at the same instant as agent i, so only u^j_{t-1} is available.

Under ``--info-pattern relay`` (``RelayFeatures``) every agent forwards what it receives, so
agent i hears from every agent j with a path j -> ... -> i, each hop costing one step: with
h_ij hops, agent i receives at time t the centralized signals of agent j at time t - h_ij,

    w^j_{t-h},  z^j_{t-h},  p^j_{t-h} - p^i_{t-h}  (, u^j_{t-h})

(on the complete graph every agent is one hop away: the centralized inputs delayed by one step).

With ``pad=True`` every agent gets as many neighbor (source) slots as the agent with the most,
the missing ones filled with zeros, so that all agents have the same input sizes and can run
in one vmapped call (``AgentBatch``). ``w_masks`` and ``context_masks`` mark the real (1) and
padded (0) channels of each agent's output; the agent operators keep the weights reading the
padded channels at zero (``padding.mask_padded_inputs``).
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

import torch
import torch.nn as nn

ADJACENCY_GRAPHS = ("complete", "ring", "cycle", "line", "none")


def make_adjacency(spec: str, n_agents: int) -> torch.Tensor:
    """(N, N) 0/1 adjacency from a named graph or an explicit matrix "0,1;1,0" (rows by ';').

    ring: undirected, each agent receives from both cycle neighbours i-1 and i+1 (= complete for 3 agents).
    cycle: directed, agent i receives from agent i+1 only (mod N): 1 <- 2, 2 <- 3, ..., N <- 1.
    """
    n = int(n_agents)
    spec = str(spec).strip()
    if spec == "complete":
        adj = torch.ones(n, n) - torch.eye(n)
    elif spec == "ring":
        adj = torch.zeros(n, n)
        for i in range(n):
            adj[i, (i + 1) % n] = adj[i, (i - 1) % n] = 1.0
        adj.fill_diagonal_(0.0)                      # n = 1
    elif spec == "cycle":
        adj = torch.zeros(n, n)
        for i in range(n):
            adj[i, (i + 1) % n] = 1.0
        adj.fill_diagonal_(0.0)                      # n = 1
    elif spec == "line":
        adj = torch.zeros(n, n)
        for i in range(n - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
    elif spec == "none":
        adj = torch.zeros(n, n)
    else:
        try:
            adj = torch.tensor([[float(v) for v in row.split(",")] for row in spec.split(";")])
        except ValueError as exc:
            raise ValueError(f"--adjacency must be one of {ADJACENCY_GRAPHS} or a matrix like '0,1;1,0', got {spec!r}") from exc
        if adj.shape != (n, n):
            raise ValueError(f"--adjacency matrix must be {n}x{n} for {n} agents, got {tuple(adj.shape)}.")
        if not ((adj == 0) | (adj == 1)).all():
            raise ValueError("--adjacency matrix entries must be 0 or 1.")
        if adj.diagonal().any():
            raise ValueError("--adjacency matrix must have a zero diagonal (no self-loops).")
    return adj


def adjacency_folder(spec: str) -> str:
    """Folder name of an --adjacency spec: the graph name, or "matrix_<rows>" for an explicit
    matrix ("0,1;1,0" -> "matrix_01-10")."""
    spec = str(spec).strip()
    if spec in ADJACENCY_GRAPHS:
        return spec
    rows = ["".join("%g" % float(v) for v in row.split(",")) for row in spec.split(";")]
    return "matrix_" + "-".join(rows)


def neighbor_lists(adjacency: torch.Tensor) -> list[list[int]]:
    """Neighbors of each agent, ordered relative to it ((j - i) mod N), so with shared weights
    a given slot always holds the same relative neighbor (e.g. next / previous on a ring)."""
    n = adjacency.shape[0]
    return [
        [j for j in sorted(range(n), key=lambda j: (j - i) % n) if adjacency[i, j] != 0]
        for i in range(n)
    ]


class NeighborFeatures(nn.Module):
    """Per-agent neighbor context [p^j_t - p^i_t, ||p^j_t - p^i_t||, u^j_{t-1} (, z^j_t)] for j in N_i
    ([p^j_t - p^i_t, s^j_t, u^j_{t-1} (, z^j_t)] with drag_idx: the neighbor's measured drag s^j_t in place of the distance).

    With targets_first (needs ref_idx), the targets of all the neighbors come first, as one group, then the other
    signals of every neighbor: [z^j_t for j in N_i, then (p^j_t - p^i_t, ., u^j_{t-1}) for j in N_i], so that the
    context [z^i, targets of the neighbors, ...] has the exogenous signals as a prefix that can drive the SSM input
    port, as the relayed targets of --info-pattern relay in Distributed/ (--neighbor-targets-input).
    target_dims[i]: width of that targets group (0 without targets_first).

    Args:
        adjacency: (N, N) 0/1, adjacency[i, j] = 1 if i receives from j.
        pos_idx: per-agent position channels in x (each of length 2).
        u_idx: per-agent input channels in u.
        ref_idx: optional per-agent target channels in z; if given, every neighbor also sends its target.
        pad: give every agent the slots of the agent with the most neighbors, the missing ones zero (with
            targets_first, each group is padded at its own end, so the prefix above holds).
        drag_idx: optional per-agent measured drag channels in z (sticky patches); if given, every neighbor sends
            its drag instead of the distance (--neighbor-drag: the signals relayed by --info-pattern relay).
        targets_first: put the neighbors' targets first, as one group (see above).
        delay: steps of delay of the neighbors' positions, drags and targets (--neighbor-delay): agent i forms
            p^j_{t-d} - p^i_{t-d} and reads s^j_{t-d}, z^j_{t-d} (d = 1: a one-hop message of --info-pattern relay
            with --relay-hop-delay 1). Before anything arrives (t < d) they hold their values at t = 0 (the initial
            configuration is known), as in RelayFeatures. u^j_{t-1} is not delayed further (u_tm1 is already one
            step old, as the relayed inputs). The history is kept across calls and cleared by reset().
    """

    def __init__(self, adjacency: torch.Tensor, pos_idx: Sequence[Sequence[int]], u_idx: Sequence[Sequence[int]],
                 ref_idx: Optional[Sequence[Sequence[int]]] = None, pad: bool = False,
                 drag_idx: Optional[Sequence[Sequence[int]]] = None, targets_first: bool = False, delay: int = 0):
        super().__init__()
        if int(delay) < 0:
            raise ValueError("delay must be >= 0.")
        self.delay = int(delay)
        self._hist: Optional[dict[str, torch.Tensor]] = None
        self.neighbors = neighbor_lists(adjacency)
        self.n_agents = len(self.neighbors)
        self.sends_targets = ref_idx is not None
        self.sends_drag = drag_idx is not None
        if targets_first and not self.sends_targets:
            raise ValueError("targets_first needs the neighbors' targets (ref_idx).")
        self.targets_first = bool(targets_first)
        dz = len(ref_idx[0]) if self.sends_targets else 0
        self.target_dim = dz
        self.feat_per_neighbor = (len(pos_idx[0]) + (len(drag_idx[0]) if self.sends_drag else 1) + len(u_idx[0]) + dz)
        slots = max(len(nb) for nb in self.neighbors)
        self.slots = [slots if pad else len(nb) for nb in self.neighbors]
        self.out_dims = [k * self.feat_per_neighbor for k in self.slots]
        self.target_dims = [k * dz if self.targets_first else 0 for k in self.slots]
        group = lambda d, real, k: [1.0] * real * d + [0.0] * (k - real) * d
        if self.targets_first:
            rest = self.feat_per_neighbor - dz
            self.context_masks = [group(dz, len(nb), k) + group(rest, len(nb), k) for nb, k in zip(self.neighbors, self.slots)]
        else:
            self.context_masks = [group(self.feat_per_neighbor, len(nb), k) for nb, k in zip(self.neighbors, self.slots)]
        self.register_buffer("pos_idx", torch.as_tensor(pos_idx, dtype=torch.long), persistent=False)
        self.register_buffer("u_idx", torch.as_tensor(u_idx, dtype=torch.long), persistent=False)
        if self.sends_targets:
            self.register_buffer("ref_idx", torch.as_tensor(ref_idx, dtype=torch.long), persistent=False)
        if self.sends_drag:
            self.register_buffer("drag_idx", torch.as_tensor(drag_idx, dtype=torch.long), persistent=False)

    def reset(self) -> None:
        self._hist = None

    def forward(self, x_t: torch.Tensor, u_tm1: torch.Tensor, z_t: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        """x_t: (B, T, Nx) measured now, u_tm1: (B, T, Nu) applied at the previous step,
        z_t: (B, T, Nz) current targets (only needed if the neighbors send them).
        Returns one (B, T, out_dims[i]) tensor per agent."""
        u = u_tm1[..., self.u_idx]                   # (B, T, N, du)
        if (self.sends_drag or self.sends_targets) and z_t is None:
            raise ValueError("Neighbor drags and targets need the context z_t.")
        cur = {"p": x_t[..., self.pos_idx]}          # (B, T, N, 2)
        if self.sends_drag:
            cur["s"] = z_t[..., self.drag_idx]
        if self.sends_targets:
            cur["z"] = z_t[..., self.ref_idx]        # (B, T, N, dz)
        if self.delay:                               # signals of t - d, held at their t = 0 values before
            T = x_t.shape[1]
            if self._hist is None:
                self._hist = {k: v[:, :1].expand(-1, self.delay, -1, -1) for k, v in cur.items()}
            seq = {k: torch.cat([self._hist[k], v], dim=1) for k, v in cur.items()}
            cur = {k: v[:, :T] for k, v in seq.items()}
            self._hist = {k: v[:, T:] for k, v in seq.items()}
        p, zr = cur["p"], cur.get("z")
        zeros = lambda k, d: x_t.new_zeros(*x_t.shape[:2], k * d)
        feats = []
        for i, nb in enumerate(self.neighbors):
            missing = self.slots[i] - len(nb)
            if not nb:
                feats.append(zeros(missing, self.feat_per_neighbor))
                continue
            rel = p[..., nb, :] - p[..., i:i + 1, :]                      # (B, T, |N_i|, 2)
            if self.sends_drag:                                           # measured drag s^j_t of the neighbors
                second = cur["s"][..., nb, :]
            else:
                second = torch.sqrt(rel.pow(2).sum(-1, keepdim=True) + 1e-12)  # distance, smooth at 0 (collision)
            parts = [rel, second, u[..., nb, :]]
            if self.targets_first:                                        # [targets group, other signals group]
                rest = self.feat_per_neighbor - self.target_dim
                feats.append(torch.cat([zr[..., nb, :].flatten(-2), zeros(missing, self.target_dim),
                                        torch.cat(parts, dim=-1).flatten(-2), zeros(missing, rest)], dim=-1))
            else:
                parts += [zr[..., nb, :]] if self.sends_targets else []
                feats.append(torch.cat([torch.cat(parts, dim=-1).flatten(-2), zeros(missing, self.feat_per_neighbor)],
                                       dim=-1))
        return feats


class PairOffsetFeatures(nn.Module):
    """Relative position of every pair of agents, given to every M^i (--info-pattern centralized).

    For each pair i < j: p^j_t - p^i_t = (dx, dy), measured from x_t, in the same order as the
    centralized controller's PairOffsets (3 agents: pairs (1,2), (1,3), (2,3), 6 features), so that
    with all of w and z each M^i sees exactly what the centralized M sees. Same interface as
    NeighborFeatures: returns one (B, T, out_dims[i]) tensor per agent.
    """

    def __init__(self, pos_idx: Sequence[Sequence[int]]):
        super().__init__()
        n = len(pos_idx)
        self.n_agents = n
        self.pairs = list(combinations(range(n), 2))
        self.out_dims = [2 * len(self.pairs)] * n
        self.register_buffer("pos_idx", torch.as_tensor(pos_idx, dtype=torch.long), persistent=False)
        self.register_buffer("first", torch.as_tensor([i for i, _ in self.pairs], dtype=torch.long), persistent=False)
        self.register_buffer("second", torch.as_tensor([j for _, j in self.pairs], dtype=torch.long), persistent=False)

    def forward(self, x_t: torch.Tensor, u_tm1: Optional[torch.Tensor] = None, z_t: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        p = x_t[..., self.pos_idx]                                   # (B, T, N, 2)
        rel = (p[..., self.second, :] - p[..., self.first, :]).flatten(-2)   # [dx_12, dy_12, dx_13, ...]
        return [rel] * self.n_agents


def hop_counts(adjacency: torch.Tensor) -> list[list[Optional[int]]]:
    """hops[i][j]: fewest edges on a path j -> ... -> i (adjacency[a, b] = 1: a receives from b),
    None if the information of j never reaches i; hops[i][i] = 0."""
    n = adjacency.shape[0]
    hops = []
    for i in range(n):
        dist, frontier = {i: 0}, [i]
        while frontier:                              # breadth-first, backwards along the edges
            nxt = []
            for a in frontier:
                for b in range(n):
                    if adjacency[a, b] != 0 and b not in dist:
                        dist[b] = dist[a] + 1
                        nxt.append(b)
            frontier = nxt
        hops.append([dist.get(j) for j in range(n)])
    return hops


class RelayFeatures(nn.Module):
    """Relayed communication (--info-pattern relay): agent i hears from every agent j that reaches it
    on the graph, with a delay of hop_delay steps per hop, d = hop_delay * h_ij. With max_hops = k,
    only from the agents at most k hops away (k = 1: its in-neighbors only, nothing is forwarded).

    Sources are ordered by hops, then relative index ((j - i) mod N). Agent i receives, per source j,
        disturbance (SSM input):  w^j_{t-d}
        context:                  z^j_{t-d},  p^j_{t-d} - p^i_{t-d}  (, u^j_{t-e} with --relay-inputs)
    where the context lists the targets of all sources first, then the offsets, then the inputs, so
    that [z^i, targets of the sources] is a prefix (only exogenous signals drive the SSM input port).
    With sticky patches (drag_idx: the channels of z holding the measured drag of every agent) every
    source also sends its measured drag, relayed like its position (s^j_{t-d}), listed right after the
    targets: [targets, drags, offsets, inputs].
    The offset is formed at agent i from the relayed position and its own position at the same step.
    A message sent at step s carries u^j_s, so e = d; with hop_delay = 0 (instant sharing) u^j_t is
    not computed yet and e = 1.

    The relayed inputs are multiplied by input_scale: they are an order of magnitude larger than
    the other context signals in the transient, and as a direct controller-to-controller loop they
    otherwise reach a gain above one through the context heads.

    Before anything arrives (t < d) the disturbances and inputs are zero (a delay started at rest,
    so w^j_0 = x^j_0 is not injected twice) and the positions, targets and drags hold their values at t = 0
    (the initial configuration is known). The history is kept across calls and cleared by reset().

    With pad=True every agent gets the slots of the agent with the most sources, each group
    (disturbances; targets, drags, offsets, inputs) padded with zeros at its end, so the prefixes above hold.
    """

    def __init__(self, adjacency: torch.Tensor, pos_idx: Sequence[Sequence[int]], w_idx: Sequence[Sequence[int]],
                 ref_idx: Sequence[Sequence[int]], u_idx: Optional[Sequence[Sequence[int]]] = None, hop_delay: int = 1,
                 input_scale: float = 1.0, max_hops: Optional[int] = None, pad: bool = False,
                 drag_idx: Optional[Sequence[Sequence[int]]] = None):
        super().__init__()
        if hop_delay < 0:
            raise ValueError("hop_delay must be >= 0.")
        if max_hops is not None and max_hops < 1:
            raise ValueError("max_hops must be >= 1 (or None: no limit).")
        hops = hop_counts(adjacency)
        n = len(hops)
        self.n_agents = n
        heard = lambda h: h is not None and (max_hops is None or h <= max_hops)
        self.sources = [sorted((j for j in range(n) if j != i and heard(hops[i][j])),
                               key=lambda j: (hops[i][j], (j - i) % n)) for i in range(n)]
        self.delays = [[hop_delay * hops[i][j] for j in src] for i, src in enumerate(self.sources)]
        self.sends_inputs = u_idx is not None
        self.input_scale = float(input_scale)
        dw, dz, dp = len(w_idx[0]), len(ref_idx[0]), len(pos_idx[0])
        du = len(u_idx[0]) if self.sends_inputs else 0
        self.sends_drag = drag_idx is not None
        ds = len(drag_idx[0]) if self.sends_drag else 0
        self.group_dims = {"w": dw, "z": dz, "s": ds, "p": dp, "u": du}
        slots = max(len(src) for src in self.sources)
        self.slots = [slots if pad else len(src) for src in self.sources]
        self.w_out_dims = [k * dw for k in self.slots]
        self.target_dims = [k * dz for k in self.slots]
        self.drag_dims = [k * ds for k in self.slots]
        self.out_dims = [k * (dz + ds + dp + du) for k in self.slots]
        self.input_dims = [k * du for k in self.slots]                     # the last channels of the context
        group = lambda d, real, k: [1.0] * real * d + [0.0] * (k - real) * d
        self.w_masks = [group(dw, len(src), k) for src, k in zip(self.sources, self.slots)]
        self.context_masks = [group(dz, len(src), k) + group(ds, len(src), k) + group(dp, len(src), k) + group(du, len(src), k)
                              for src, k in zip(self.sources, self.slots)]
        # steps of history needed: d for w, z, p; e - 1 <= d for the inputs (u_prev holds u_{t-1})
        self.memory = max((d for ds in self.delays for d in ds), default=0)
        self.register_buffer("pos_idx", torch.as_tensor(pos_idx, dtype=torch.long), persistent=False)
        self.register_buffer("w_idx", torch.as_tensor(w_idx, dtype=torch.long), persistent=False)
        self.register_buffer("ref_idx", torch.as_tensor(ref_idx, dtype=torch.long), persistent=False)
        if self.sends_inputs:
            self.register_buffer("u_idx", torch.as_tensor(u_idx, dtype=torch.long), persistent=False)
        if self.sends_drag:
            self.register_buffer("drag_idx", torch.as_tensor(drag_idx, dtype=torch.long), persistent=False)
        self._hist: Optional[dict[str, torch.Tensor]] = None

    def reset(self) -> None:
        self._hist = None

    def forward(self, x_t: torch.Tensor, u_tm1: torch.Tensor, z_t: torch.Tensor,
                w_t: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """x_t: (B, T, Nx) measured now, u_tm1: (B, T, Nu) applied at the previous step,
        z_t: (B, T, Nz) targets, w_t: (B, T, Nx) reconstructed disturbances.
        Returns (w_extra, context): one (B, T, w_out_dims[i]) and one (B, T, out_dims[i]) tensor per agent."""
        if z_t is None:
            raise ValueError("Relayed communication needs the context z_t (the targets).")
        cur = {"p": x_t[..., self.pos_idx], "w": w_t[..., self.w_idx], "z": z_t[..., self.ref_idx]}   # (B, T, N, ·)
        if self.sends_inputs:
            cur["u"] = self.input_scale * u_tm1[..., self.u_idx]
        if self.sends_drag:
            cur["s"] = z_t[..., self.drag_idx]
        T = x_t.shape[1]
        if self._hist is None:
            pad = lambda v, hold: (v[:, :1] if hold else torch.zeros_like(v[:, :1])).expand(-1, self.memory, -1, -1)
            self._hist = {k: pad(v, hold=k in ("p", "z", "s")) for k, v in cur.items()}
        seq = {k: torch.cat([self._hist[k], v], dim=1) for k, v in cur.items()}
        L = T + self.memory
        at = lambda k, d: seq[k][:, L - T - d:L - d]                 # signal k delayed by d steps

        w_extra, ctx = [], []
        for i, (src, delays) in enumerate(zip(self.sources, self.delays)):
            # zero slots closing each group (pad=True), none otherwise
            pad = lambda g: [x_t.new_zeros(*x_t.shape[:2], (self.slots[i] - len(src)) * self.group_dims[g])]
            ws, zs, ss, offs, us = [], [], [], [], []
            for j, d in zip(src, delays):
                ws.append(at("w", d)[:, :, j])
                zs.append(at("z", d)[:, :, j])
                if self.sends_drag:
                    ss.append(at("s", d)[:, :, j])
                p = at("p", d)
                offs.append(p[:, :, j] - p[:, :, i])
                if self.sends_inputs:
                    us.append(at("u", max(d, 1) - 1)[:, :, j])    # u^j_{t-e} = u_prev of step t-e+1
            w_extra.append(torch.cat(ws + pad("w"), dim=-1))
            drags = ss + pad("s") if self.sends_drag else []
            ctx.append(torch.cat(zs + pad("z") + drags + offs + pad("p") + us + pad("u"), dim=-1))
        self._hist = {k: v[:, L - self.memory:] for k, v in seq.items()}
        return w_extra, ctx
