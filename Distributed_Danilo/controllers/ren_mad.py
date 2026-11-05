"""
Distributed REN-MAD performance booster, ported from dPB-microgrid/ren_mad (Saccani, Massai and
Ferrari-Trecate, "Distributed Performance Boosting through Neural Control for Plug-and-Play Networks")
to the robots. The certificate, the REN parametrization and the MAD readout are theirs; only the
plant-specific parts (which residual coordinates an agent owns, which context it sees) change.

Agent i runs a gain-certified REN R^i. Its input is its own reconstructed disturbance and the weighted sum
of the outputs that its neighbors published, so the RENs talk to each other (no relaying):

    v^i_t = [ w^i_t ;  sum_j a_ij y^j_t ],       a_ij > 0 iff adjacency[i, j] = 1 (i receives from j),
    y^i_{t+1} = R^i(v^i)_t,                      one-step output buffer: y^i_t is published before v_t is formed,

and a bounded MAD (magnitude and direction) readout gives the input of robot i,

    u^i_t = |y^i_t| * d^i(w^i_t, c^i_t),

where the direction d^i in R^{nu} is the bounded mixer M_b^i of the SSM controller (BoundedMLPOperator with
rank s = 1, flags --mb-*), with the same context c^i_t as there: the target z^i_t, the measured drag
(sticky patches), and with --info-pattern neighbors what NeighborFeatures sends over the --adjacency graph
(relative position, distance, u^j_{t-1} and target of every neighbor).

Certificate (network_certificate). The local gains gamma_i are not free parameters: they are computed from
the free variables (eta_alpha, eta_c) and from the interconnection matrices (general_bounds,
gain_parameters) so that

    [M_vz M_ve]' Pi_v [M_vz M_ve] - diag(Pi_z - M_uz' M_uz, gamma_R^2 I) <= 0,

with Pi_v = diag(alpha_i gamma_i^2), Pi_z = diag(alpha_i). Each R^i is (gamma_i, storage) dissipative by
construction (CertifiedREN.realize), so the network of RENs satisfies ||y||_2 <= gamma_R ||w||_2 for any
parameter value. With |d^i| <= mb_bound entrywise, ||u||_2 <= sqrt(nu) * mb_bound * gamma_R * ||w||_2.

The realization (every matrix derived from the parameters) is computed once per rollout, in reset(), and
reused at every step; gradients flow through that single computation. The agents run in one batched step:
every robot has the same residual size (4) and the same REN size. The RENs run in float64, as in ren_mad;
their output is cast back to the dtype of w.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.nn import functional

from Distributed_Danilo.controllers.operators import as_bt, OperatorBase

DTYPE = torch.float64


# ----------------------------------------------------------------------------------------------------------
# Certificate (ren_mad/src/controller.py, unchanged but for the directed weighted adjacency)


@dataclass(frozen=True)
class Interconnection:
    M_ve: torch.Tensor          # (q, n) unit injection of every owned residual coordinate
    M_vz: torch.Tensor          # (q, N) weighted neighbor outputs into the communication rows
    M_uz: torch.Tensor          # (N, N) = I: u is read from y
    n_sizes: tuple[int, ...]
    q_sizes: tuple[int, ...]
    r_sizes: tuple[int, ...]


@dataclass(frozen=True)
class Bounds:
    hbar: torch.Tensor
    rho_z: torch.Tensor
    rho_v0: torch.Tensor
    rho_v1: torch.Tensor
    cbar: torch.Tensor


@dataclass(frozen=True)
class Gains:
    alpha: torch.Tensor
    c: torch.Tensor
    gamma: torch.Tensor


def _positive(value: float, name: str) -> None:
    if not (math.isfinite(float(value)) and float(value) > 0):
        raise ValueError(f"{name} must be finite and strictly positive")


def build_interconnection(weights: torch.Tensor, n_sizes, out_dim: int = 1,
                         comm_rows: Optional[Sequence[int]] = None) -> Interconnection:
    """Matrices of v = M_ve e + M_vz y for agents with exogenous inputs e^i of sizes n_sizes and outputs y^i in
    R^{out_dim}.

    comm_rows None: agent i's input block is [e^i (n_i rows); sum_j a_ij y^j (out_dim rows)] (ren_mad).
    comm_rows given (out_dim offsets into e^i): sum_j a_ij y^j is added to those rows of e^i, the block keeps
    n_i rows (e.g. the neighbors' commands added to the velocity channels of w^i).

    weights: (N, N) communication weights, weights[i, j] = a_ij >= 0 (i receives y^j), zero diagonal.
    n_sizes: an int (the same for every agent) or one size per agent. The order of the rows inside a block
    does not matter to the certificate (Pi_v is constant on a block), so the operator may feed them in any order.
    """
    n_agents = weights.shape[0]
    if weights.shape != (n_agents, n_agents) or bool(weights.diagonal().ne(0).any()):
        raise ValueError("Communication weights must be a square matrix with a zero diagonal")
    if not bool(torch.isfinite(weights).all() & (weights >= 0).all()):
        raise ValueError("Communication weights must be finite and nonnegative")
    n_sizes = (int(n_sizes),) * n_agents if isinstance(n_sizes, int) else tuple(int(n) for n in n_sizes)
    s = int(out_dim)
    if comm_rows is not None and (len(comm_rows) != s or any(not 0 <= r < min(n_sizes) for r in comm_rows)):
        raise ValueError("comm_rows must give one row of e^i per output channel")
    q_sizes = tuple(n + (0 if comm_rows is not None else s) for n in n_sizes)
    eye = lambda k: torch.eye(k, dtype=weights.dtype, device=weights.device)
    mve = weights.new_zeros(sum(q_sizes), sum(n_sizes))
    rows, row, column = [], 0, 0
    for n_i, q_i in zip(n_sizes, q_sizes):
        mve[row:row + n_i, column:column + n_i] = eye(n_i)
        rows.append(torch.arange(row + n_i, row + q_i) if comm_rows is None
                    else row + torch.as_tensor(list(comm_rows), dtype=torch.long))
        row, column = row + q_i, column + n_i
    # communication rows of agent i: sum_j a_ij y^j, i.e. the rows of kron(W, I_s); keeps the gradient path to W
    mvz = weights.new_zeros(sum(q_sizes), n_agents * s).index_put(
        (torch.cat(rows, 0).to(weights.device),), torch.kron(weights, eye(s)))
    return Interconnection(mve, mvz, eye(n_agents * s), n_sizes, q_sizes, (s,) * n_agents)


def general_bounds(interconnection: Interconnection, gamma_R: float, *, cbar0: float = 1.0) -> Bounds:
    """Bounds from the matrix entries, independently of the graph degrees (ren_mad.general_bounds)."""
    _positive(gamma_R, "network gain")
    me, mz, mu = interconnection.M_ve, interconnection.M_vz, interconnection.M_uz
    nonzero = me != 0
    if (not bool(((me == 0) | (me == 1)).all()) or not bool((nonzero.sum(dim=0) == 1).all())
            or not bool((nonzero.sum(dim=1) <= 1).all())):
        raise ValueError("M_ve must be a unit coordinate injection")
    gram = mu.T @ mu
    h = torch.diagonal(gram)
    if not torch.allclose(gram, torch.diag(h), atol=1e-12, rtol=0):
        raise ValueError("M_uz columns must be orthogonal")
    direct = nonzero.any(dim=1)
    row_sums, column_sums = mz.abs().sum(dim=1), mz.abs().sum(dim=0)
    hbars, rz, rv0, rv1, cbars = [], [], [], [], []
    row = output = 0
    zero = me.new_zeros(())
    target_squared = me.new_tensor(float(gamma_R) ** 2)
    for qi, ri in zip(interconnection.q_sizes, interconnection.r_sizes):
        own_direct = direct[row:row + qi]
        rows = row_sums[row:row + qi]
        bound0 = rows[~own_direct].max() if bool((~own_direct).any()) else zero
        bound1 = rows[own_direct].max() if bool(own_direct.any()) else zero
        choices = []
        if bool(own_direct.any()):
            choices.append(target_squared / (1 + bound1 * target_squared))
        if bool(bound0 > 0):
            choices.append(1 / bound0)
        cbars.append(torch.stack(choices).min() if choices else me.new_tensor(cbar0))
        hbars.append(h[output:output + ri].max())
        rz.append(column_sums[output:output + ri].max())
        rv0.append(bound0)
        rv1.append(bound1)
        row += qi
        output += ri
    return Bounds(*(torch.stack(values) for values in (hbars, rz, rv0, rv1, cbars)))


def gain_parameters(interconnection: Interconnection, eta_alpha: torch.Tensor, eta_c: torch.Tensor,
                    gamma_R: float, epsilon: float) -> Gains:
    """Local gains gamma_i = sqrt(c_i / alpha_i) that satisfy the network certificate by construction."""
    _positive(epsilon, "certificate epsilon")
    bounds = general_bounds(interconnection, gamma_R)
    fraction = torch.sigmoid(eta_c)
    if bool(((fraction <= 0) | (fraction >= 1)).any()):
        raise FloatingPointError("Sigmoid reached a floating-point endpoint; no projection applied")
    alpha = bounds.hbar + bounds.rho_z + eta_alpha.square() + float(epsilon)
    c = bounds.cbar * fraction
    gamma = torch.sqrt(c / alpha)
    if not bool(torch.isfinite(gamma).all() & (gamma > 0).all()):
        raise FloatingPointError("Local gain cannot be represented as a finite positive value")
    return Gains(alpha, c, gamma)


def network_certificate(interconnection: Interconnection, gains: Gains, gamma_R: float) -> torch.Tensor:
    """The full symmetric certificate matrix, negative semidefinite iff the network gain is <= gamma_R."""
    me, mz, mu = interconnection.M_ve, interconnection.M_vz, interconnection.M_uz
    q_repeats = torch.tensor(interconnection.q_sizes, device=me.device)
    r_repeats = torch.tensor(interconnection.r_sizes, device=me.device)
    piv = torch.diag(torch.repeat_interleave(gains.alpha * gains.gamma.square(), q_repeats))
    piz = torch.diag(torch.repeat_interleave(gains.alpha, r_repeats))
    zz = mz.T @ piv @ mz - piz + mu.T @ mu
    ze = mz.T @ piv @ me
    ee = me.T @ piv @ me - float(gamma_R) ** 2 * torch.eye(me.shape[1], dtype=me.dtype, device=me.device)
    matrix = torch.cat((torch.cat((zz, ze), dim=1), torch.cat((ze.T, ee), dim=1)), dim=0)
    return (matrix + matrix.T) / 2


# ----------------------------------------------------------------------------------------------------------
# REN (ren_mad.CertifiedREN, unchanged)


@dataclass(frozen=True)
class RENRealization:
    matrices: dict[str, torch.Tensor]
    P: torch.Tensor
    gamma: torch.Tensor


class CertifiedREN(nn.Module):
    """REN with a prescribed l2 gain gamma from v to its scalar output, for any value of its free parameters."""

    def __init__(self, input_dim: int, state_dim: int = 6, width: int = 6, *,
                 construction_epsilon: float = 0.01, init_std: float = 0.1):
        super().__init__()
        if min(input_dim, state_dim, width) < 1 or not construction_epsilon > 0:
            raise ValueError("REN dimensions and construction epsilon must be positive")
        self.input_dim, self.state_dim, self.width = input_dim, state_dim, width
        self.construction_epsilon = float(construction_epsilon)
        total = 2 * state_dim + width

        def free(*shape: int) -> nn.Parameter:
            return nn.Parameter(init_std * torch.randn(shape, dtype=DTYPE))

        self.metric_factor = free(total, total)
        self.state_skew = free(state_dim, state_dim)
        self.input_to_state = free(state_dim, input_dim)
        self.input_to_nonlinearity = free(width, input_dim)
        self.state_to_output = free(1, state_dim)
        self.nonlinearity_to_output = free(1, width)
        self.feedthrough_vector = free(input_dim)

    def realize(self, gamma: torch.Tensor) -> RENRealization:
        """Differentiable matrices from the current free parameters, for the local gain gamma."""
        q, n, width = self.input_dim, self.state_dim, self.width
        gamma = torch.as_tensor(gamma, dtype=DTYPE, device=self.metric_factor.device).reshape(())
        if not bool(torch.isfinite(gamma) & (gamma > 0)):
            raise ValueError("A REN realization needs one finite positive local gain")
        identity_input = torch.eye(q, dtype=DTYPE, device=gamma.device)
        vector = self.feedthrough_vector
        direct = (vector / torch.sqrt(1 + vector.square().sum())).unsqueeze(0)
        input_slack = identity_input - direct.T @ direct
        torch.linalg.cholesky(input_slack)      # a failure is a numerical-contract failure, never repaired
        c2, d21 = self.state_to_output, self.nonlinearity_to_output
        b2, d12 = self.input_to_state, self.input_to_nonlinearity
        supply_cross = torch.cat((-c2.T @ direct, -d21.T @ direct - d12, b2), dim=0)
        output_stack = torch.cat((c2.T, d21.T, c2.new_zeros(n, 1)), dim=0)
        total = 2 * n + width
        metric = (self.metric_factor.T @ self.metric_factor
                  + self.construction_epsilon * torch.eye(total, dtype=DTYPE, device=gamma.device)
                  + supply_cross @ torch.linalg.solve(input_slack, supply_cross.T)
                  + output_stack @ output_stack.T)
        metric = (metric + metric.T) / 2
        linear, nonlinear, following = slice(0, n), slice(n, n + width), slice(n + width, total)
        p_cal = metric[following, following]
        torch.linalg.cholesky(p_cal)
        e = (metric[linear, linear] + p_cal + self.state_skew - self.state_skew.T) / 2
        storage_matrix = e.T @ torch.linalg.solve(p_cal, e)
        storage_matrix = (storage_matrix + storage_matrix.T) / 2
        if not bool(torch.isfinite(storage_matrix).all()):
            raise FloatingPointError("Nonfinite REN storage matrix")
        torch.linalg.cholesky(storage_matrix)
        nonlinear_metric = metric[nonlinear, nonlinear]
        matrices = {
            "E": e,
            "F": metric[following, linear],
            "B1": metric[following, nonlinear],
            "B2": gamma * b2,
            "C1": -metric[nonlinear, linear],
            "D11": -torch.tril(nonlinear_metric, diagonal=-1),
            "D12": gamma * d12,
            "C2": c2,
            "D21": d21,
            "D22": gamma * direct,
            "Lambda": torch.diagonal(nonlinear_metric) / 2,
        }
        if not all(bool(torch.isfinite(value).all()) for value in matrices.values()):
            raise FloatingPointError("Nonfinite derived REN matrices")
        return RENRealization(matrices, storage_matrix, gamma)


def ren_sector_certificate(realization: RENRealization) -> dict:
    """Numerical check that one realized REN is (gamma, x'Px + b^2) dissipative (ren_mad.ren_sector_certificate).

    For t = [x; w; v], L = E^-1 [F B1 B2], O = [C2 D21 D22], Q = diag(P, 0, gamma^2 I) and the sector
    matrix S of the acyclic nonlinearity, D = Q - L'PL - O'O - S > 0 proves the gain bound.
    """
    m = {k: v.detach().to(device="cpu", dtype=DTYPE) for k, v in realization.matrices.items()}
    P = realization.P.detach().to(device="cpu", dtype=DTYPE)
    gamma = realization.gamma.detach().to(device="cpu", dtype=DTYPE).reshape(())
    n, width, q = P.shape[0], m["B1"].shape[1], m["B2"].shape[1]
    symmetric_P = (P + P.T) / 2
    storage_eigenvalues, storage_vectors = torch.linalg.eigh(symmetric_P)
    dynamics = torch.linalg.solve(m["E"], torch.cat((m["F"], m["B1"], m["B2"]), dim=1))
    output = torch.cat((m["C2"], m["D21"], m["D22"]), dim=1)
    supply = torch.block_diag(P, torch.zeros(width, width, dtype=DTYPE), gamma.square() * torch.eye(q, dtype=DTYPE))
    sector = torch.zeros_like(supply)
    xs, ws, vs = slice(0, n), slice(n, n + width), slice(n + width, n + width + q)
    sector[ws, xs], sector[xs, ws] = m["C1"], m["C1"].T
    sector[ws, ws] = m["D11"] + m["D11"].T - 2 * torch.diag(m["Lambda"])
    sector[ws, vs], sector[vs, ws] = m["D12"], m["D12"].T
    dissipation = supply - dynamics.T @ P @ dynamics - output.T @ output - sector
    dissipation = (dissipation + dissipation.T) / 2
    inv_sqrt = (storage_vectors / storage_eigenvalues.clamp_min(1e-300).sqrt().unsqueeze(0)) @ storage_vectors.T
    congruence = torch.block_diag(inv_sqrt, torch.eye(width, dtype=DTYPE), torch.eye(q, dtype=DTYPE) / gamma)
    scaled = congruence.T @ dissipation @ congruence
    eigenvalues = torch.linalg.eigvalsh(dissipation)
    scaled_eigenvalues = torch.linalg.eigvalsh((scaled + scaled.T) / 2)
    acyclic_error = float(torch.triu(m["D11"]).abs().max())
    sector_min = float(m["Lambda"].min())
    verified = (acyclic_error == 0 and sector_min > 0 and bool((storage_eigenvalues > 0).all())
                and bool((eigenvalues > 0).all()) and bool((scaled_eigenvalues > 0).all()))
    return {"verified": bool(verified), "gamma": float(gamma),
            "min_eigenvalue": float(eigenvalues.min()), "min_scaled_eigenvalue": float(scaled_eigenvalues.min()),
            "storage_min_eigenvalue": float(storage_eigenvalues.min())}


def response_jacobian(realization: RENRealization) -> torch.Tensor:
    """State Jacobian at the origin, including the acyclic nonlinear feedback."""
    m = realization.matrices
    sector = torch.diag(m["Lambda"]) - m["D11"]
    return torch.linalg.solve(m["E"], m["F"] + m["B1"] @ torch.linalg.solve(sector, m["C1"]))


# ----------------------------------------------------------------------------------------------------------
# Distributed operators


@dataclass(frozen=True)
class NetworkRealization:
    weights: torch.Tensor                   # (N, N) communication weights a_ij
    interconnection: Interconnection
    gains: Gains
    locals: list                            # per-agent realizations (RENRealization for the RENs, None for the SSMs)
    certificate: torch.Tensor


class CertifiedNetworkOperator(OperatorBase):
    """What the REN and the SSM networks share: the agents' channels and context, the communication weights
    a_ij on the --adjacency graph and the local gains that certify the network gain gamma_R.

    Same call interface as DistributedOperator (Distributed/controllers/distributed.py): forward(w, z, x=x_t,
    u_prev=u_{t-1}) on the joint (B, T, ·) signals, the per-agent channels picked by w_idx / z_idx / u_idx.

    Args:
        w_idx, z_idx, u_idx: per-agent channels of w, z and u (each a list of N equal-length lists).
        adjacency: (N, N) 0/1, adjacency[i, j] = 1 if agent i receives y^j (and, with neighbor_features,
            the context of agent i holds neighbor j's signals).
        neighbor_features: optional NeighborFeatures appended to the context (--info-pattern neighbors).
        drag_idx, drag_pos: optional measured drag channels of z, inserted at drag_pos in the context of
            each agent (sticky patches), as in DistributedOperator of Distributed/.
        gamma_R: prescribed l2 gain of the network of l2 operators, from the exogenous inputs to y.
        epsilon: margin of the certificate (alpha_i >= hbar_i + rho_z_i + epsilon).
        edge_weight: initial communication weight a of every edge.
        learn_edges: train the edge weights (between rollouts: they are fixed within a rollout).
        edge_bounds: (lower, upper) bounds of learned edge weights (sigmoid); None: exp(log a).
    """

    def __init__(self, w_idx, z_idx, u_idx, adjacency: torch.Tensor,
                 neighbor_features: Optional[nn.Module] = None,
                 drag_idx: Optional[Sequence[Sequence[int]]] = None, drag_pos: Optional[Sequence[int]] = None,
                 gamma_R: float = 25.0, epsilon: float = 0.02, edge_weight: float = 1e-3,
                 learn_edges: bool = False, edge_bounds: Optional[Sequence[float]] = (1e-4, 1.0)):
        super().__init__()
        n = len(w_idx)
        if not (len(z_idx) == len(u_idx) == n):
            raise ValueError("w_idx, z_idx and u_idx must list the same number of agents.")
        if len({len(i) for i in w_idx}) != 1:
            raise ValueError("Every agent must own the same number of residual coordinates.")
        adjacency = torch.as_tensor(adjacency, dtype=DTYPE)
        if adjacency.shape != (n, n) or bool(adjacency.diagonal().ne(0).any()):
            raise ValueError(f"adjacency must be {n}x{n} with a zero diagonal.")
        _positive(gamma_R, "--ren-gamma")
        _positive(epsilon, "--ren-epsilon")
        _positive(edge_weight, "--ren-edge-weight")
        self.n_agents = n
        self.w_dim = len(w_idx[0])
        self.gamma_R, self.epsilon, self.edge_weight = float(gamma_R), float(epsilon), float(edge_weight)
        self.neighbor_features = neighbor_features
        self.register_buffer("w_idx", torch.as_tensor(w_idx, dtype=torch.long), persistent=False)
        self.register_buffer("z_idx", torch.as_tensor(z_idx, dtype=torch.long), persistent=False)
        self.register_buffer("u_idx", torch.as_tensor(u_idx, dtype=torch.long), persistent=False)
        self.u_dim = int(self.u_idx.numel())
        if drag_idx is not None:
            if drag_pos is None or len(drag_idx) != n or len(drag_pos) != n:
                raise ValueError("drag_idx and drag_pos must list every agent.")
            self.register_buffer("drag_idx", torch.as_tensor(drag_idx, dtype=torch.long), persistent=False)
        else:
            self.drag_idx = None
        self.drag_pos = None if drag_pos is None else [int(e) for e in drag_pos]

        # communication edges (i receives from j), in row-major order
        edges = adjacency.nonzero().tolist()
        self.register_buffer("edge_rows", torch.tensor([i for i, _ in edges], dtype=torch.long), persistent=False)
        self.register_buffer("edge_cols", torch.tensor([j for _, j in edges], dtype=torch.long), persistent=False)
        self.learn_edges = bool(learn_edges)
        self.edge_bounds = None if edge_bounds is None else tuple(float(b) for b in edge_bounds)
        if self.learn_edges:
            if self.edge_bounds is None:
                self.log_edge_weights = nn.Parameter(torch.full((len(edges),), math.log(self.edge_weight), dtype=DTYPE))
            else:
                lower, upper = self.edge_bounds
                if not 0 < lower < self.edge_weight < upper:
                    raise ValueError("--ren-edge-bounds must satisfy 0 < lower < --ren-edge-weight < upper.")
                logit = math.log((self.edge_weight - lower) / (upper - self.edge_weight))
                self.edge_logits = nn.Parameter(torch.full((len(edges),), logit, dtype=DTYPE))
        self.eta_alpha = nn.Parameter(torch.full((n,), 0.1, dtype=DTYPE))
        self.eta_c = nn.Parameter(torch.zeros(n, dtype=DTYPE))

    # --- PBController interface
    @property
    def uses_feedback(self) -> bool:
        """True if forward needs x_t and u_{t-1} (neighbor signals in the context)."""
        return self.neighbor_features is not None

    # --- communication and certificate
    def edge_weights(self) -> torch.Tensor:
        """(E,) communication weights, in the order of the adjacency's nonzero entries."""
        if not self.learn_edges:
            return self.eta_alpha.new_full((self.edge_rows.numel(),), self.edge_weight)
        if self.edge_bounds is None:
            return torch.exp(self.log_edge_weights)
        lower, upper = self.edge_bounds
        return lower + (upper - lower) * torch.sigmoid(self.edge_logits)

    def weight_matrix(self) -> torch.Tensor:
        """(N, N) matrix of a_ij."""
        W = self.eta_alpha.new_zeros(self.n_agents, self.n_agents)
        return W.index_put((self.edge_rows, self.edge_cols), self.edge_weights())

    def gain_variables(self) -> list[nn.Parameter]:
        """Gain variables and learned edge weights (they set the local gains)."""
        params = [self.eta_alpha, self.eta_c]
        if self.learn_edges:
            params.append(self.edge_logits if self.edge_bounds is not None else self.log_edge_weights)
        return params

    def network_gains(self, n_sizes, out_dim: int, comm_rows: Optional[Sequence[int]] = None):
        """(weights, interconnection, gains, certificate) for exogenous input sizes n_sizes and outputs in R^out_dim
        (fed back on extra rows, or added to the rows comm_rows of the exogenous inputs)."""
        weights = self.weight_matrix()
        interconnection = build_interconnection(weights, n_sizes, out_dim, comm_rows)
        gains = gain_parameters(interconnection, self.eta_alpha, self.eta_c, self.gamma_R, self.epsilon)
        return weights, interconnection, gains, network_certificate(interconnection, gains, self.gamma_R)

    def edge_summary(self) -> list:
        with torch.no_grad():
            weights = self.edge_weights()
        return [round(float(weights.min()), 6), round(float(weights.max()), 6)] if weights.numel() else []

    def check_certificate(self) -> dict:
        result = self.certify()
        if not result["verified"]:
            raise RuntimeError(f"Network certificate failed: {result}")
        return result

    # --- context
    def contexts(self, z: Optional[torch.Tensor], x: Optional[torch.Tensor], u_prev: Optional[torch.Tensor]):
        """Context c^i of every agent: [z^i, drag^i, neighbor signals], as in DistributedOperator of Distributed/;
        with the neighbors' targets first (--neighbor-targets-input) [z^i, neighbor targets, drag^i, other signals]."""
        nb = None
        if self.neighbor_features is not None:
            if x is None or u_prev is None:
                raise ValueError("The neighbors info pattern needs x_t and u_{t-1}.")
            nb = self.neighbor_features(as_bt(x), as_bt(u_prev), z)
        td = getattr(self.neighbor_features, "target_dims", None)
        out = []
        for i in range(self.n_agents):
            c = z[..., self.z_idx[i]] if z is not None else None
            rest = nb[i] if nb is not None else None
            if rest is not None and td is not None and td[i]:       # neighbors' targets group: in the prefix
                c = torch.cat([c, rest[..., :td[i]]], dim=-1)
                rest = rest[..., td[i]:]
            if self.drag_idx is not None:
                e = self.drag_pos[i]
                c = torch.cat([c[..., :e], z[..., self.drag_idx[i]], c[..., e:]], dim=-1)
            if rest is not None:
                c = rest if c is None else torch.cat([c, rest], dim=-1)
            out.append(c)
        return out

    def scatter_u(self, w: torch.Tensor, u_ag: torch.Tensor) -> torch.Tensor:
        """(B, T, N * du) agent inputs -> (B, T, u_dim) joint input."""
        u = w.new_zeros(*u_ag.shape[:2], self.u_dim)
        return u.index_copy(-1, self.u_idx.reshape(-1), u_ag.to(w.dtype))


class RENMADOperator(CertifiedNetworkOperator):
    """u^i = |y^i| d^i(w^i, c^i), y = network of certified RENs fed with [w^i; sum_j a_ij y^j] (module docstring).

    Args (on top of CertifiedNetworkOperator's):
        directions: one bounded direction network per agent, called as d(w^i, c^i) -> (B, T, nu, 1).
        state_dim, width, init_std: size and initialization of each REN.
    """

    def __init__(self, directions: Sequence[nn.Module], w_idx, z_idx, u_idx, adjacency: torch.Tensor,
                 state_dim: int = 6, width: int = 6, init_std: float = 0.1, **network_kwargs):
        super().__init__(w_idx, z_idx, u_idx, adjacency, **network_kwargs)
        if len(directions) != self.n_agents:
            raise ValueError("directions must list every agent.")
        self.directions = nn.ModuleList(directions)
        self.rens = nn.ModuleList(CertifiedREN(self.w_dim + 1, state_dim, width, init_std=init_std)
                                  for _ in range(self.n_agents))
        self._mats: Optional[dict[str, torch.Tensor]] = None
        self._x: Optional[torch.Tensor] = None     # (B, N, state_dim) REN states
        self._y: Optional[torch.Tensor] = None     # (B, N) published outputs y_t

    def realize(self) -> NetworkRealization:
        """Gains, local REN matrices and certificate for the current parameters."""
        weights, interconnection, gains, certificate = self.network_gains(self.w_dim, 1)
        locals_ = [ren.realize(gains.gamma[i]) for i, ren in enumerate(self.rens)]
        return NetworkRealization(weights, interconnection, gains, locals_, certificate)

    def reset(self) -> None:
        """Start a rollout: realize the network once from the current parameters and clear the states."""
        if hasattr(self.neighbor_features, "reset"):
            self.neighbor_features.reset()
        real = self.realize()
        stack = lambda key: torch.stack([loc.matrices[key] for loc in real.locals])
        E = stack("E")
        self._mats = {
            "F": torch.linalg.solve(E, stack("F")),            # E^-1 F, E^-1 B1, E^-1 B2: E is fixed in the rollout
            "B1": torch.linalg.solve(E, stack("B1")),
            "B2": torch.linalg.solve(E, stack("B2")),
            "C1": stack("C1"), "D11": stack("D11"), "D12": stack("D12"),
            "C2": stack("C2")[:, 0], "D21": stack("D21")[:, 0], "D22": stack("D22")[:, 0],
            "Lambda": stack("Lambda"),
            "W": real.weights,
        }
        self._x = self._y = None

    def _ren_step(self, x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of the N RENs at once. x: (B, N, n), v: (B, N, q) -> next x, next output (B, N)."""
        m = self._mats
        pre = torch.einsum("bnk,njk->bnj", x, m["C1"]) + torch.einsum("bnk,njk->bnj", v, m["D12"])
        acts = []
        for j in range(pre.shape[-1]):          # acyclic equilibrium layer: lower triangular D11
            value = pre[..., j]
            if j:
                value = value + (torch.stack(acts, dim=-1) * m["D11"][:, j, :j]).sum(-1)
            acts.append(torch.tanh(value / m["Lambda"][:, j]))
        a = torch.stack(acts, dim=-1)
        x_next = (torch.einsum("bnk,njk->bnj", x, m["F"]) + torch.einsum("bnk,njk->bnj", a, m["B1"])
                  + torch.einsum("bnk,njk->bnj", v, m["B2"]))
        y_next = (x * m["C2"]).sum(-1) + (a * m["D21"]).sum(-1) + (v * m["D22"]).sum(-1)
        return x_next, y_next

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None,
                u_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """w, z: (B, T, ·); x: state measured at t, u_prev: input applied at t-1 (neighbors only). T = 1 in closed loop."""
        if self._mats is None:
            raise RuntimeError("Call reset() before the first step of a rollout.")
        w = as_bt(w)
        z = as_bt(z) if z is not None else None
        B, T = w.shape[:2]
        if T != 1 and self.neighbor_features is not None:
            raise ValueError("Neighbor signals are causal step by step: call with T = 1.")
        w_ag = w[..., self.w_idx]                                               # (B, T, N, dw)
        if self._x is None:
            n_state = self._mats["F"].shape[-1]
            self._x = torch.zeros(B, self.n_agents, n_state, dtype=DTYPE, device=w.device)
            self._y = torch.zeros(B, self.n_agents, dtype=DTYPE, device=w.device)
        ctx = self.contexts(z, x, u_prev)
        u_steps = []
        for t in range(T):
            y = self._y                                                          # published at t
            comm = y @ self._mats["W"].T                                         # sum_j a_ij y^j
            v = torch.cat([w_ag[:, t].to(DTYPE), comm.unsqueeze(-1)], dim=-1)    # (B, N, dw + 1)
            self._x, self._y = self._ren_step(self._x, v)
            mag = y.abs().to(w.dtype)                                            # (B, N)
            d = torch.stack([
                self.directions[i](w_ag[:, t:t + 1, i], None if ctx[i] is None else ctx[i][:, t:t + 1])[:, 0, :, 0]
                for i in range(self.n_agents)], dim=1)                           # (B, N, du)
            u_steps.append((mag.unsqueeze(-1) * d).reshape(B, 1, -1))
        return self.scatter_u(w, torch.cat(u_steps, dim=1))

    # --- diagnostics
    def core_parameters(self):
        """REN free parameters, gain variables and learned edge weights (the l2 part, full learning rate)."""
        return list(self.rens.parameters()) + self.gain_variables()

    def direction_bound(self) -> float:
        """Upper bound of ||d^i||_2 (sqrt(nu) times the entrywise bound of M_b)."""
        bound = max(float(getattr(d, "bound_value", float("inf"))) for d in self.directions)
        return math.sqrt(self.u_dim // self.n_agents) * bound

    def certify(self) -> dict:
        """Numerical check of the realized network: network LMI, local dissipativity and origin stability."""
        with torch.no_grad():
            real = self.realize()
            network_max = float(torch.linalg.eigvalsh(real.certificate).max())
            checks = [ren_sector_certificate(loc) for loc in real.locals]
            radii = [float(torch.linalg.eigvals(response_jacobian(loc)).abs().max()) for loc in real.locals]
        verified = (network_max <= 1e-12 and all(c["verified"] for c in checks) and max(radii) < 1)
        return {
            "verified": bool(verified),
            "gamma_R": self.gamma_R,
            "network_max_eigenvalue": network_max,
            "local_gains": [round(c["gamma"], 4) for c in checks],
            "local_min_eigenvalue": min(c["min_eigenvalue"] for c in checks),
            "local_min_scaled_eigenvalue": min(c["min_scaled_eigenvalue"] for c in checks),
            "max_spectral_radius": max(radii),
            "edge_weights": self.edge_summary(),
            "u_gain_bound": self.gamma_R * self.direction_bound(),
        }
