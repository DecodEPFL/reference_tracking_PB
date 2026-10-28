"""
Composable construction of the distributed PB operator: one u^i = M_b^i(w^i, z^i) ⊠ M_p^i(w^i, z^i)
per agent (see ``DistributedOperator``), each built from the flags below.

    --mp-type  ssm             M_p = DeepSSM(w)            (+ optional l_p context lift: DeepSSM([w, lift(z)]))
               contextual_ssm  M_p = ContextualDeepSSM(w, z) with ports from --ctx-modes (+ select)
               ren             M_p = ContractiveREN(w)     (legacy baseline)
    --mb-type  bounded_mlp     M_b(w, z) in R^{nu x s}     (s = --feat-dim; s = 1 is the MAD policy)
               diagonal        M_b(w, z) = diag(g(w, z))   (rPB: s = nu)
               none            u = M_p(.) directly         (s = nu; with contextual_ssm this is the
                                                            fused "contextual_ssm" operator)

    --share-weights            one operator M shared by all agents (default: independent M^i)
    --info-pattern local       M^i sees its own channels w^i, z^i
                   full        M^i sees everything the centralized M sees: all of w and z, in the
                               global order with independent weights (M^i = centralized M restricted
                               to the output u^i); with --share-weights ordered ego-first
                               (agent i, i+1, ..., i-1) so the shared operator can tell agents apart
                   neighbors   local, plus for every neighbor j on the --adjacency graph the context
                               [p^j_t - p^i_t, ||p^j_t - p^i_t||, u^j_{t-1}], plus the target z^j_t
                               with --neighbor-targets (see interconnection.py)
                   centralized exactly the centralized M's inputs: all of w, all targets and the
                               relative position of every pair of agents, measured at t (no delay)
                   relay       the centralized inputs over the --adjacency graph, every agent
                               forwarding what it receives, one step per hop (--relay-hop-delay):
                               M^i gets [w^i_t, w^j_{t-d}, ...] in its SSM input and
                               [z^i_t, z^j_{t-d}, ..., p^j_{t-d} - p^i_{t-d}, ... (, u^j_{t-d}, ...)]
                               in its context, d = hops from j to i (u^j with --relay-inputs,
                               only to M_b, see --relay-inputs-to); --relay-max-hops 1 keeps
                               the in-neighbors only (no forwarding)
    --match-centralized-size   with independent weights, shrink the widths of every M^i so the N
                               operators together have at most the parameters of the centralized M
"""

from __future__ import annotations

import argparse
import warnings
from itertools import chain
from typing import Iterable, Iterator, Optional

import torch
import torch.nn as nn

from Distributed.controllers.batched_agents import check_agent_batch
from Distributed.controllers.bounded_mlp import BoundedMLPOperator, DiagonalBoundedMLPOperator
from Distributed.controllers.context_lifting import LpContextLifter
from Distributed.controllers.contractive_ren import ContractiveREN
from Distributed.controllers.distributed import DistributedOperator
from Distributed.controllers.interconnection import NeighborFeatures, PairOffsetFeatures, RelayFeatures, make_adjacency
from Distributed.controllers.operators import FactorizedOperator, WIntegralAugmenter
from Distributed.controllers.PB_controller import PBController
from Distributed.controllers.ssm_operators import ContextRescale, MpContextualSSM, MpDeepSSM

MP_TYPES = ("ssm", "contextual_ssm", "ren")
MB_TYPES = ("bounded_mlp", "diagonal", "none")
CTX_MODES = ("input", "gate", "mixer", "select")
INFO_PATTERNS = ("local", "full", "neighbors", "centralized", "relay")
CTX_FILTERS = ("auto", "finite_horizon", "taper", "exponential", "polynomial", "difference", "none")


def resolve_context_modes(args) -> tuple[str, ...]:
    modes = [m.strip() for m in str(args.ctx_modes).split(",") if m.strip()]
    if args.ctx_select and "select" not in modes:
        modes.append("select")
    unknown = sorted(set(modes) - set(CTX_MODES))
    if unknown:
        raise ValueError(f"--ctx-modes: unknown port(s) {unknown}; valid ports are {CTX_MODES}.")
    if not modes:
        raise ValueError("--ctx-modes must select at least one context port.")
    return tuple(dict.fromkeys(modes))


def validate_operator_args(args) -> None:
    """Fail fast on flag combinations the operator cannot realize."""
    if args.mp_type not in MP_TYPES:
        raise ValueError(f"--mp-type must be one of {MP_TYPES}, got {args.mp_type!r}")
    if args.mb_type not in MB_TYPES:
        raise ValueError(f"--mb-type must be one of {MB_TYPES}, got {args.mb_type!r}")
    if args.mp_type == "contextual_ssm":
        modes = resolve_context_modes(args)
        if "select" in modes and args.ssm_param not in ("tv", "tvc"):
            raise ValueError("The 'select' context port needs a selective core: use --ssm-param tv or tvc.")
        if args.ctx_filter == "exponential" and not 0.0 < args.ctx_filter_decay < 1.0:
            raise ValueError("--ctx-filter exponential needs 0 < --ctx-filter-decay < 1.")
        if args.ctx_filter == "polynomial" and not args.ctx_filter_power > 0.5:
            raise ValueError("--ctx-filter polynomial needs --ctx-filter-power > 0.5.")
        if args.ctx_filter == "taper" and not 1 <= args.ctx_filter_ramp <= args.horizon:
            raise ValueError("--ctx-filter taper needs 1 <= --ctx-filter-ramp <= --horizon.")
        if args.mp_context_lift:
            raise ValueError("--mp-context-lift only applies to --mp-type ssm/ren; "
                             "contextual_ssm consumes z through its own ports.")
    if args.mb_type == "bounded_mlp" and args.feat_dim < 1:
        raise ValueError("--feat-dim must be >= 1.")


def _ssm_core_kwargs(args) -> dict:
    return dict(
        param=args.ssm_param,
        n_layers=int(args.ssm_layers),
        d_model=int(args.ssm_d_model),
        d_state=int(args.ssm_d_state),
        ff=args.ssm_ff,
        bcd_nonlinearity=args.ssm_bcd_nonlinearity,
    )


def build_mp(args, mp_in_dim: int, z_dim: int, out_dim: int, input_z_dim: Optional[int] = None,
             dynamics_z_dim: Optional[int] = None, mixer_z_dim: Optional[int] = None) -> nn.Module:
    if args.mp_type == "ssm":
        return MpDeepSSM(mp_in_dim, out_dim, mode="loop", **_ssm_core_kwargs(args))
    if args.mp_type == "ren":
        return ContractiveREN(
            dim_in=mp_in_dim, dim_out=out_dim,
            dim_internal=int(args.dim_internal), dim_nl=int(args.dim_nl),
            initialization_std=float(args.cont_init_std),
        )
    ramp = int(args.ctx_filter_ramp)
    gamma = float(args.ctx_gamma)
    encoder = ContextRescale(args.ctx_z_scale) if abs(float(args.ctx_z_scale) - 1.0) > 1e-9 else None
    return MpContextualSSM(
        mp_in_dim, z_dim, out_dim,
        context_modes=resolve_context_modes(args),
        input_context_dim=input_z_dim,
        dynamics_context_dim=dynamics_z_dim,
        mixer_context_dim=mixer_z_dim,
        context_encoder=encoder,
        context_filter=args.ctx_filter,
        horizon=int(args.horizon),
        context_filter_ramp=ramp if ramp > 0 else None,
        context_filter_decay=float(args.ctx_filter_decay),
        context_filter_power=float(args.ctx_filter_power),
        mixer_bound=float(args.ctx_mixer_bound),
        d_features=int(args.ctx_d_features),
        mixer_hidden_dim=int(getattr(args, "ctx_head_hidden", 64)),   # absent from older checkpoints
        gate_hidden_dim=int(getattr(args, "ctx_head_hidden", 64)),
        gate_per_channel=bool(args.ctx_gate_per_channel),
        gamma=gamma if gamma > 0.0 else None,
        **_ssm_core_kwargs(args),
    )


def build_mb(args, w_dim: int, z_dim: int, u_dim: int) -> nn.Module | None:
    common = dict(
        w_dim=w_dim, z_dim=z_dim,
        hidden_dim=int(args.mb_hidden), num_layers=int(args.mb_layers),
        use_z_residual=True, z_residual_gain=float(args.z_residual_gain),
        bound_mode="softsign", clamp_value=float(args.mb_bound),
    )
    if args.mb_type == "bounded_mlp":
        return BoundedMLPOperator(r=u_dim, s=int(args.feat_dim), **common)
    if args.mb_type == "diagonal":
        return DiagonalBoundedMLPOperator(dim=u_dim, **common)
    return None


def build_lifter(args, z_dim: int) -> LpContextLifter | None:
    if not args.mp_context_lift:
        return None
    horizon = int(args.mp_context_decay_horizon)
    return LpContextLifter(
        z_dim=z_dim,
        out_dim=int(args.mp_context_lift_dim),
        lift_type=args.mp_context_lift_type,
        hidden_dim=int(args.mp_context_hidden_dim),
        decay_law=args.mp_context_decay_law,
        decay_rate=float(args.mp_context_decay_rate),
        decay_power=float(args.mp_context_decay_power),
        decay_horizon=horizon if horizon > 0 else int(args.horizon),
        lp_p=float(args.mp_context_lp_p),
        scale=float(args.mp_context_scale),
    )


def build_operator(args, w_dim: int, z_dim: int, u_dim: int, input_z_dim: Optional[int] = None,
                   dynamics_z_dim: Optional[int] = None, mixer_z_dim: Optional[int] = None) -> FactorizedOperator:
    """Assemble M_b ⊠ M_p from the CLI flags (see module docstring).

    input_z_dim: if set, only the first input_z_dim channels of z drive the contextual SSM's
    "input" port (the rest are signals computed from the state, see InputContextMask).
    dynamics_z_dim: if set, only the first dynamics_z_dim channels of z reach its gate and select
    ports; the rest only reach the mixer port and M_b (see MpContextualSSM).
    mixer_z_dim: likewise for its mixer port; the rest only reach M_b.
    """
    validate_operator_args(args)
    augmenter = WIntegralAugmenter(float(args.w_augment_decay)) if args.use_w_augment else None
    w_op_dim = 2 * w_dim if augmenter is not None else w_dim

    lifter = build_lifter(args, z_dim) if args.mp_type != "contextual_ssm" else None
    mp_in_dim = w_op_dim + (lifter.out_dim if lifter is not None else 0)
    mp_out_dim = int(args.feat_dim) if args.mb_type == "bounded_mlp" else u_dim

    return FactorizedOperator(
        mp=build_mp(args, mp_in_dim, z_dim, mp_out_dim, input_z_dim, dynamics_z_dim, mixer_z_dim),
        mb=build_mb(args, w_op_dim, z_dim, u_dim),
        mp_context_lifter=lifter,
        mp_only=(args.mb_type == "none"),
        w_augmenter=augmenter,
    )


def agent_channels(agent_idx: list[list[int]], info_pattern: str, ego_first: bool = False) -> list[list[int]]:
    """Channels of a signal (w or z) fed to each M^i under the information pattern.

    agent_idx[j] lists the channels of agent j; "full" gives every agent all of them, in the
    global order or, with ego_first, starting from its own; "centralized" as "full", in the global order.
    """
    n = len(agent_idx)
    if info_pattern in ("local", "neighbors", "relay"):   # neighbor signals are added on top of w^i, z^i
        return [list(idx) for idx in agent_idx]
    if info_pattern == "centralized":
        return [[c for idx in agent_idx for c in idx] for _ in range(n)]
    if info_pattern == "full":
        order = lambda i: [(i + k) % n for k in range(n)] if ego_first else range(n)
        return [[c for j in order(i) for c in agent_idx[j]] for i in range(n)]
    raise ValueError(f"--info-pattern must be one of {INFO_PATTERNS}, got {info_pattern!r}")


# internal widths shrunk by --match-centralized-size; the rank --feat-dim and the l_p lift
# keep their meaning and are left untouched
WIDTH_ARGS = ("ssm_d_model", "ssm_d_state", "ctx_d_features", "ctx_head_hidden", "mb_hidden", "dim_internal", "dim_nl")


def _n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def scale_widths(args, factor: float):
    """Copy of args with every internal width scaled by factor (at least 1)."""
    scaled = argparse.Namespace(**vars(args))
    for name in WIDTH_ARGS:
        setattr(scaled, name, max(1, round(factor * getattr(args, name))))
    return scaled


def match_centralized_size(args, plant, w_dims: list[int], z_dims: list[int], u_dim: int, central_z_dim: Optional[int] = None):
    """Shrink the widths so the operators of sizes (w_dims[i], z_dims[i], u_dim) together have at
    most the parameters of the centralized M (full w, z and u) built from the same args.
    central_z_dim: context width of that centralized M (default: the targets, plant.in_dim).

    Deterministic in args, so a checkpoint rebuilds the same architecture from its saved args.
    """
    with torch.random.fork_rng():   # trial builds must not shift the RNG of the real initialization
        central_z = plant.in_dim if central_z_dim is None else central_z_dim
        budget = _n_params(build_operator(args, w_dim=plant.state_dim, z_dim=central_z, u_dim=plant.in_dim))
        total = lambda f: sum(
            _n_params(build_operator(scale_widths(args, f), w_dim=w_dim, z_dim=z_dim, u_dim=u_dim))
            for w_dim, z_dim in zip(w_dims, z_dims))
        if total(0.0) > budget:
            raise ValueError(
                f"--match-centralized-size: {len(z_dims)} operators exceed the centralized budget of "
                f"{budget} parameters even at width 1; enlarge the widths or use --share-weights.")
        lo, hi = 0.0, 1.0           # largest factor within budget (parameter count grows with it)
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if total(mid) <= budget else (lo, mid)
    return scale_widths(args, lo), budget


def build_distributed_operator(args, plant) -> DistributedOperator:
    """One M^i per agent (or a single shared M), each mapping the (w, z) channels its
    information pattern grants to u^i."""
    # checkpoints saved before --info-pattern / --match-centralized-size existed were local, full size
    info_pattern = getattr(args, "info_pattern", "local")
    if info_pattern == "centralized" and args.share_weights:
        raise ValueError("--info-pattern centralized gives every agent the same inputs: with --share-weights "
                         "all agents would compute the same thing; use independent weights.")
    w_idx = agent_channels(plant.state_idx, info_pattern, ego_first=args.share_weights)
    z_idx = agent_channels(plant.ref_idx, info_pattern, ego_first=args.share_weights)
    u_dim = plant.agent_in_dim
    neighbors = None
    if info_pattern == "neighbors":
        adjacency = make_adjacency(getattr(args, "adjacency", "complete"), plant.n_agents)
        # checkpoints saved before --neighbor-targets existed did not send the targets
        ref_idx = plant.ref_idx if getattr(args, "neighbor_targets", False) else None
        neighbors = NeighborFeatures(adjacency, pos_idx=plant.agent_pos_idx, u_idx=plant.input_idx, ref_idx=ref_idx)
    elif info_pattern == "centralized":
        # exactly the centralized controller's inputs: all of w, all targets, and the pair offsets
        neighbors = PairOffsetFeatures(plant.agent_pos_idx)
    elif info_pattern == "relay":
        # the centralized inputs, relayed over the graph with one step of delay per hop
        neighbors = RelayFeatures(make_adjacency(args.adjacency, plant.n_agents), pos_idx=plant.agent_pos_idx,
                                  w_idx=plant.state_idx, ref_idx=plant.ref_idx,
                                  u_idx=plant.input_idx if args.relay_inputs else None,
                                  hop_delay=args.relay_hop_delay,
                                  # checkpoints saved before --relay-inputs-scale existed were not scaled
                                  input_scale=getattr(args, "relay_inputs_scale", 1.0),
                                  max_hops=getattr(args, "relay_max_hops", 0) or None)   # 0: no limit
    n = plant.n_agents
    relay = info_pattern == "relay"
    # input widths of each M^i: its own w and z channels plus what its neighbors send
    w_dims = [len(w_idx[i]) + (neighbors.w_out_dims[i] if relay else 0) for i in range(n)]
    z_dims = [len(z_idx[i]) + (neighbors.out_dims[i] if neighbors is not None else 0) for i in range(n)]
    # neighbor signals computed from x and u (relative positions, inputs) would close a feedback
    # loop around the plant through the SSM input port: only the targets z drive it (z^i, all of
    # them under "centralized", z^i and the relayed targets under "relay")
    input_z_dims = [None if neighbors is None else len(z_idx[i]) + (neighbors.target_dims[i] if relay else 0)
                    for i in range(n)]
    # relayed inputs kept out of the gate and select ports (mixers, mb) and of the SSM mixer port (mb),
    # so they never reach the SSM state; checkpoints saved before --relay-inputs-to existed fed them to every port
    route = getattr(args, "relay_inputs_to", "all") if relay else "all"
    no_inputs = lambda i: z_dims[i] - neighbors.input_dims[i]
    dynamics_z_dims = [no_inputs(i) if route in ("mixers", "mb") else None for i in range(n)]
    mixer_z_dims = [no_inputs(i) if route == "mb" else None for i in range(n)]
    if args.share_weights:
        if len(set(z_dims)) > 1 or len(set(w_dims)) > 1:
            raise ValueError("--share-weights needs every agent to have the same number of neighbors "
                             f"(context sizes {z_dims}); use --no-share-weights or a regular --adjacency graph.")
        w_dims, z_dims, input_z_dims, dynamics_z_dims, mixer_z_dims = (
            w_dims[:1], z_dims[:1], input_z_dims[:1], dynamics_z_dims[:1], mixer_z_dims[:1])
    budget = None
    if len(z_dims) > 1 and getattr(args, "match_centralized_size", False):
        # under "centralized" and "relay" the reference M is the centralized one with pair features
        central_z_dim = (plant.in_dim + PairOffsetFeatures(plant.agent_pos_idx).out_dims[0]
                         if info_pattern in ("centralized", "relay") else None)
        args, budget = match_centralized_size(args, plant, w_dims, z_dims, u_dim, central_z_dim)
    agent_ops = [build_operator(args, w_dim=w_dim, z_dim=z_dim, u_dim=u_dim, input_z_dim=input_z_dim,
                                dynamics_z_dim=dynamics_z_dim, mixer_z_dim=mixer_z_dim)
                 for w_dim, z_dim, input_z_dim, dynamics_z_dim, mixer_z_dim
                 in zip(w_dims, z_dims, input_z_dims, dynamics_z_dims, mixer_z_dims)]
    # one vmapped call for the N independent operators, if they support it and match the loop
    batch_agents = bool(getattr(args, "batch_agents", False)) and len(agent_ops) > 1
    if batch_agents and (len(set(w_dims)) > 1 or len(set(z_dims)) > 1):
        warnings.warn(f"--batch-agents disabled: the agents have different input sizes (w {w_dims}, z {z_dims}).")
        batch_agents = False
    if batch_agents:
        try:
            check_agent_batch(agent_ops, w_dim=w_dims[0], z_dim=z_dims[0])
        except Exception as exc:          # unsupported cell / architecture, or mismatch: keep the loop
            warnings.warn(f"--batch-agents disabled, running the agents one by one: {exc}")
            batch_agents = False
    operator = DistributedOperator(agent_ops, w_idx=w_idx, z_idx=z_idx, u_idx=plant.input_idx,
                                   neighbor_features=neighbors, batch_agents=batch_agents)
    operator.param_budget = budget                                   # None unless size-matched
    operator.widths = {k: getattr(args, k, None) for k in WIDTH_ARGS}   # widths actually built
    return operator


def build_controller(args, plant) -> PBController:
    operator = build_distributed_operator(args, plant)
    return PBController(
        plant=plant,
        operator=operator,
        u_dim=plant.in_dim,
        output_scale=float(args.output_amplification),
        detach_state=False,
    )


def agent_operators(operator) -> Iterator[FactorizedOperator]:
    """The distinct per-agent operators (a single one with shared weights)."""
    if isinstance(operator, DistributedOperator):
        return operator.unique_operators()
    return iter([operator])


def ssm_core_parameters(operator) -> Iterable[nn.Parameter]:
    """Parameters of the recurrent core of every M_p, trained at the full learning rate.

      MpDeepSSM / ContractiveREN -> the whole M_p
      MpContextualSSM            -> the DeepSSM inside ContextualDeepSSM; its
                                    mixer/gate/filter heads go with the slow group.
    """
    def core(op: FactorizedOperator) -> Iterable[nn.Parameter]:
        mp = op.mp
        if isinstance(mp, MpContextualSSM):
            return getattr(mp.core, "core", mp.core).parameters()
        return mp.parameters()
    return chain.from_iterable(core(op) for op in agent_operators(operator))
