"""
Construction of the distributed certified network (see ren_mad.py, ssm_network.py) from the CLI flags.

Default: REN-MAD (ren_mad.py)
    u^i = |y^i| * d^i(w^i, c^i),     y = network of certified RENs, v^i = [w^i; sum_j a_ij y^j]

    --ren-gamma, --ren-epsilon         prescribed network gain gamma_R from w to y, certificate margin
    --ren-state-dim, --ren-width       size of every REN
    --ren-edge-weight, --ren-learn-edges, --ren-edge-bounds
                                       communication weights a_ij on the --adjacency graph
    --mb-*                             the direction network d^i: the bounded mixer M_b of the SSM controller
                                       (Distributed/) with rank s = 1, i.e. its MAD policy

--use-ssm (ssm_network.py): the same certified network with the M_p and M_b of Distributed/
    u^i = M_b^i(w^i, c^i) ⊠ M_p^i(w^i + E_v sum_j a_ij u^j_{t-1} / K_j, c^i)
    the neighbors' commands (2-D) added to the velocity channels of w^i at the core input, normalized by
    their readout bound K_j (--feedback-norm constant) or raw (none)
    built from the flags of Distributed/ (--mp-type ssm | contextual_ssm, --ssm-*, --ctx-*, --mb-type,
    --feat-dim, --mb-*, --match-centralized-size), with the --ren-gamma / --ren-epsilon / --ren-edge-* flags
    for the certificate and the communication. Every M_p is built with a prescribed gain (--ssm-param must be
    a certified parametrization, e.g. tv / tvc); --ctx-gamma is replaced by the certificate's local gains.

    --info-pattern local               c^i = own target (and drag)
                   neighbors           c^i = own target (and drag) + for every neighbor j on --adjacency
                                       [p^j_t - p^i_t, ||p^j_t - p^i_t||, u^j_{t-1} (, z^j_t)]
                                       (--neighbor-drag: the measured drag s^j_t in place of the distance;
                                       --neighbor-delay d: positions, drags and targets of t - d)

The l2 operators always talk over --adjacency (as in ren_mad); --info-pattern only sets the context.
full / centralized / relay, --share-weights, the l_p context lift and --use-w-augment are not ported.
Flags of the parser that a controller does not use (e.g. the SSM flags without --use-ssm, --batch-agents,
--compile) are accepted so that the command lines of Distributed/ run unchanged, and ignored.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Iterable, Iterator, Optional

import torch
import torch.nn as nn

from Distributed_Danilo.controllers.bounded_mlp import BoundedMLPOperator, DiagonalBoundedMLPOperator
from Distributed_Danilo.controllers.interconnection import NeighborFeatures, make_adjacency
from Distributed_Danilo.controllers.padding import mask_padded_inputs
from Distributed_Danilo.controllers.PB_controller import PBController
from Distributed_Danilo.controllers.ren_mad import CertifiedNetworkOperator, RENMADOperator
from Distributed_Danilo.controllers.ssm_network import SSMNetworkOperator
from Distributed_Danilo.controllers.ssm_operators import ContextRescale, MpContextualSSM, MpDeepSSM

MP_TYPES = ("ssm", "contextual_ssm", "ren")
MB_TYPES = ("bounded_mlp", "diagonal", "none")
CTX_MODES = ("input", "gate", "mixer", "select")
INFO_PATTERNS = ("local", "full", "neighbors", "centralized", "relay")
CTX_FILTERS = ("auto", "finite_horizon", "taper", "exponential", "polynomial", "difference", "none")
SUPPORTED_INFO_PATTERNS = ("local", "neighbors")
CERTIFIED_SSM_PARAMS = ("l2ru", "zak", "l2n", "l2nt", "tv", "tvc", "raven")


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
    info = getattr(args, "info_pattern", "local")
    if info not in SUPPORTED_INFO_PATTERNS:
        raise ValueError(f"The certified network supports --info-pattern {SUPPORTED_INFO_PATTERNS}, got {info!r} "
                         "(full / centralized / relay are SSM-controller patterns, not ported).")
    if getattr(args, "share_weights", False):
        raise ValueError("The certified network has one operator per agent: --share-weights is not supported.")
    if getattr(args, "use_w_augment", False):
        raise ValueError("--use-w-augment is not supported (the l2 operators read w^i directly).")
    if not getattr(args, "use_ssm", False):
        return
    if args.mp_type not in ("ssm", "contextual_ssm"):
        raise ValueError("--use-ssm needs --mp-type ssm or contextual_ssm.")
    if args.mb_type not in MB_TYPES:
        raise ValueError(f"--mb-type must be one of {MB_TYPES}, got {args.mb_type!r}")
    if args.ssm_param not in CERTIFIED_SSM_PARAMS:
        raise ValueError(f"--use-ssm prescribes the gain of every M_p: --ssm-param must be certified "
                         f"(one of {CERTIFIED_SSM_PARAMS}), got {args.ssm_param!r}.")
    if args.mp_context_lift:
        raise ValueError("--mp-context-lift is not supported with --use-ssm.")
    if args.mp_type == "contextual_ssm":
        modes = resolve_context_modes(args)
        if "select" in modes and args.ssm_param not in ("tv", "tvc"):
            raise ValueError("The 'select' context port needs a selective core: use --ssm-param tv or tvc.")
        if "input" in modes and args.ctx_filter in ("difference", "none"):
            raise ValueError("--use-ssm with the 'input' port needs an l2 context filter (not difference / none): "
                             "the filtered targets are exogenous inputs of the certificate.")
    if args.mb_type == "bounded_mlp" and args.feat_dim < 1:
        raise ValueError("--feat-dim must be >= 1.")


# ----------------------------------------------------------------------------------------------------------
# M_p and M_b of Distributed/ (controllers/factory.py there)


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
             gamma: Optional[float] = None) -> nn.Module:
    """M_p of Distributed/ for --mp-type ssm / contextual_ssm. gamma: prescribed gain cap of its core
    (None: --ctx-gamma for the contextual SSM, free for the plain one, as in Distributed/)."""
    if args.mp_type == "ssm":
        return MpDeepSSM(mp_in_dim, out_dim, mode="loop", gamma=gamma, **_ssm_core_kwargs(args))
    if args.mp_type != "contextual_ssm":
        raise ValueError(f"build_mp: --mp-type {args.mp_type!r} is not an SSM.")
    ramp = int(args.ctx_filter_ramp)
    if gamma is None:
        gamma = float(args.ctx_gamma) if float(args.ctx_gamma) > 0.0 else None
    encoder = ContextRescale(args.ctx_z_scale) if abs(float(args.ctx_z_scale) - 1.0) > 1e-9 else None
    return MpContextualSSM(
        mp_in_dim, z_dim, out_dim,
        context_modes=resolve_context_modes(args),
        input_context_dim=input_z_dim,
        context_encoder=encoder,
        context_filter=args.ctx_filter,
        horizon=int(args.horizon),
        context_filter_ramp=ramp if ramp > 0 else None,
        context_filter_decay=float(args.ctx_filter_decay),
        context_filter_power=float(args.ctx_filter_power),
        mixer_bound=float(args.ctx_mixer_bound),
        d_features=int(args.ctx_d_features),
        mixer_hidden_dim=int(getattr(args, "ctx_head_hidden", 64)),
        gate_hidden_dim=int(getattr(args, "ctx_head_hidden", 64)),
        gate_per_channel=bool(args.ctx_gate_per_channel),
        gamma=gamma,
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


def ssm_out_dim(args, u_dim: int) -> int:
    """Size s of y^i = M_p output: the rank --feat-dim under a bounded M_b, else the agent input size."""
    return int(args.feat_dim) if args.mb_type == "bounded_mlp" else int(u_dim)


# internal widths shrunk by --match-centralized-size (as in Distributed/)
WIDTH_ARGS = ("ssm_d_model", "ssm_d_state", "ctx_d_features", "ctx_head_hidden", "mb_hidden", "dim_internal", "dim_nl")


def _n_params(modules) -> int:
    return sum(p.numel() for m in modules if m is not None for p in m.parameters())


def scale_widths(args, factor: float):
    """Copy of args with every internal width scaled by factor (at least 1)."""
    scaled = argparse.Namespace(**vars(args))
    for name in WIDTH_ARGS:
        setattr(scaled, name, max(1, round(factor * getattr(args, name))))
    return scaled


def readout_bound(args, u_dim: int) -> float:
    """Gain from an M_p core's output to u^i: mixer bound of the contextual SSM x spectral bound of M_b."""
    bound = 1.0
    if args.mp_type == "contextual_ssm" and "mixer" in resolve_context_modes(args):
        bound *= float(args.ctx_mixer_bound)
    if args.mb_type == "bounded_mlp":
        bound *= float(args.mb_bound) * (u_dim * int(args.feat_dim)) ** 0.5
    elif args.mb_type == "diagonal":
        bound *= float(args.mb_bound)
    return bound


def certified_cap(args, u_dim: int) -> float:
    """Prescribed gain cap of every M_p core: at least any core gain override the certificate can set,
    gamma_i (constant normalization) or gamma_i / readout_bound (none), with gamma_i < gamma_R."""
    return 2.0 * float(args.ren_gamma) / min(1.0, readout_bound(args, u_dim))


# ----------------------------------------------------------------------------------------------------------
# Distributed operators


def _network_kwargs(args, plant, neighbors, adjacency) -> dict:
    drag = getattr(plant, "drag_idx", None)
    return dict(
        w_idx=plant.state_idx, z_idx=plant.ref_idx, u_idx=plant.input_idx, adjacency=adjacency,
        neighbor_features=neighbors,
        # the drag comes after the exogenous prefix [z^i, targets of the neighbors (--neighbor-targets-input)]
        drag_idx=drag, drag_pos=None if drag is None else [
            len(plant.ref_idx[i]) + (neighbors.target_dims[i] if neighbors is not None else 0) for i in range(plant.n_agents)],
        gamma_R=float(args.ren_gamma), epsilon=float(args.ren_epsilon),
        edge_weight=float(args.ren_edge_weight), learn_edges=bool(args.ren_learn_edges),
        edge_bounds=tuple(args.ren_edge_bounds) if args.ren_edge_bounds is not None else None,
    )


@torch.no_grad()
def match_free_initialization(mp: nn.Module) -> None:
    """Initialize a gain-capped M_p core as DeepSSM initializes a free one (gamma=None), as in Distributed/.

    With a prescribed gamma, neural_ssm initializes the block gains from that cap (e.g. 4.07 instead of 1 for a
    cap of 50 over 4 blocks) and the encoder / decoder orthogonally instead of as nn.Linear: a different
    starting network that has nothing to do with the method. This puts the block gains back at 1 (the free
    initialization) and the encoder / decoder at the nn.Linear initialization (bias-free in both cases); the
    certificate (spectral caps, decoder attenuation) is still applied at every step.
    """
    from neural_ssm.ssm.layers import _set_cell_gamma
    core = mp.core.core if isinstance(mp, MpContextualSSM) else mp.core
    for block in core.blocks:
        _set_cell_gamma(block.lru, 1.0)
    for weight in (core.encoder_w, core.decoder_w):
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)     # nn.Linear's default initialization


def build_ssm_agents(args, plant, z_dims, input_z_dims):
    """M_p^i (input w^i, the size of Distributed/; prescribed gain cap) and M_b^i of every agent."""
    u_dim, s = plant.agent_in_dim, ssm_out_dim(args, plant.agent_in_dim)
    cap = certified_cap(args, u_dim)
    mps, mbs = [], []
    for i in range(plant.n_agents):
        w_dim = len(plant.state_idx[i])
        mps.append(build_mp(args, w_dim, z_dims[i], s, input_z_dim=input_z_dims[i], gamma=cap))
        if getattr(args, "ssm_init", "free") == "free":
            match_free_initialization(mps[-1])
        mbs.append(build_mb(args, w_dim, z_dims[i], u_dim))
    return mps, (None if args.mb_type == "none" else mbs)


def match_centralized_size(args, plant, z_dims, input_z_dims, central_z_dim):
    """--match-centralized-size of Distributed/: shrink the widths so that the N agents have at most the
    parameters of the centralized M_b ⊠ M_p built from the same args (same agent sizes as Distributed/)."""
    with torch.random.fork_rng():   # trial builds must not shift the RNG of the real initialization
        budget = _n_params([build_mp(args, plant.state_dim, central_z_dim, ssm_out_dim(args, plant.in_dim)),
                            build_mb(args, plant.state_dim, central_z_dim, plant.in_dim)])

        def total(f):
            mps, mbs = build_ssm_agents(scale_widths(args, f), plant, z_dims, input_z_dims)
            return _n_params(list(mps) + list(mbs or []))
        if total(0.0) > budget:
            raise ValueError(f"--match-centralized-size: {plant.n_agents} operators exceed the centralized budget of "
                             f"{budget} parameters even at width 1; enlarge the widths.")
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if total(mid) <= budget else (lo, mid)
    return scale_widths(args, lo), budget


def build_distributed_operator(args, plant) -> CertifiedNetworkOperator:
    """One certified l2 operator (REN, or SSM with --use-ssm) and one readout per agent, over --adjacency."""
    validate_operator_args(args)
    info_pattern = getattr(args, "info_pattern", "local")
    n = plant.n_agents
    adjacency = make_adjacency(getattr(args, "adjacency", "complete"), n)
    neighbors = None
    if info_pattern == "neighbors":
        ref_idx = plant.ref_idx if getattr(args, "neighbor_targets", False) else None
        # --neighbor-drag: the neighbors send their measured drag (sticky patches) instead of their distance
        if getattr(args, "neighbor_drag", False) and getattr(plant, "drag_idx", None) is None:
            raise ValueError("--neighbor-drag needs the measured drag of the sticky patches (--patches).")
        nb_drag = plant.drag_idx if getattr(args, "neighbor_drag", False) else None
        # --neighbor-targets-input: the neighbors' targets right after z^i, so they also drive the SSM input port
        targets_first = bool(getattr(args, "neighbor_targets_input", False))
        if targets_first and ref_idx is None:
            raise ValueError("--neighbor-targets-input needs --neighbor-targets.")
        neighbors = NeighborFeatures(adjacency, pos_idx=plant.agent_pos_idx, u_idx=plant.input_idx, ref_idx=ref_idx,
                                     drag_idx=nb_drag, targets_first=targets_first,
                                     delay=int(getattr(args, "neighbor_delay", 0)))
    elif getattr(args, "neighbor_targets_input", False) or getattr(args, "neighbor_delay", 0):
        raise ValueError("--neighbor-targets-input and --neighbor-delay need --info-pattern neighbors.")
    # context of agent i: [z^i, drag^i (sticky patches), neighbor signals], as in Distributed/; with
    # --neighbor-targets-input [z^i, targets of the neighbors, drag^i, other neighbor signals]
    drag = getattr(plant, "drag_idx", None)
    z_dims = [len(plant.ref_idx[i]) + (len(drag[i]) if drag is not None else 0)
              + (neighbors.out_dims[i] if neighbors is not None else 0) for i in range(n)]
    network = _network_kwargs(args, plant, neighbors, adjacency)
    budget = None

    if getattr(args, "use_ssm", False):
        # agents hearing different numbers of neighbors cannot be vmapped: zero-pad their neighbor slots to the
        # largest, the weights reading the padding held at zero, as --pad-agent-inputs in Distributed/
        real_z_dims = z_dims
        padded = (bool(getattr(args, "batch_agents", False)) and getattr(args, "pad_agent_inputs", False)
                  and neighbors is not None and len(set(z_dims)) > 1)
        if padded:
            neighbors = NeighborFeatures(adjacency, pos_idx=plant.agent_pos_idx, u_idx=plant.input_idx,
                                         ref_idx=neighbors.ref_idx.tolist() if neighbors.sends_targets else None, pad=True,
                                         drag_idx=nb_drag, targets_first=targets_first, delay=neighbors.delay)
            network = _network_kwargs(args, plant, neighbors, adjacency)
            z_dims = [len(plant.ref_idx[i]) + (len(drag[i]) if drag is not None else 0) + neighbors.out_dims[i]
                      for i in range(n)]
        # signals computed from the state (neighbors, drag) never drive the "input" port: only the targets (own, and
        # the neighbors' with --neighbor-targets-input, as the relayed targets of Distributed/)
        input_z_dims = [None if (neighbors is None and drag is None)
                        else len(plant.ref_idx[i]) + (neighbors.target_dims[i] if neighbors is not None else 0)
                        for i in range(n)]
        if n > 1 and getattr(args, "match_centralized_size", False):
            central_z_dim = plant.in_dim + (len(drag) if drag is not None else 0)
            # the real sizes: the padded columns are held at zero (as Distributed/)
            args, budget = match_centralized_size(args, plant, real_z_dims, input_z_dims, central_z_dim)
        mps, mbs = build_ssm_agents(args, plant, z_dims, input_z_dims)
        if padded:
            for i in range(n):          # [targets, neighbor targets group, drag, neighbor slots], as the context
                td, mask = neighbors.target_dims[i], neighbors.context_masks[i]
                z_keep = ([1.0] * len(plant.ref_idx[i]) + mask[:td]
                          + [1.0] * (len(drag[i]) if drag is not None else 0) + mask[td:])
                mask_padded_inputs(SimpleNamespace(mp=mps[i], mb=None if mbs is None else mbs[i]),
                                   w_keep=[1.0] * len(plant.state_idx[i]), z_keep=z_keep)
        # the neighbors' commands are added to the velocity channels of w^i (u is a reference offset: through the
        # inner PI loop, it acts on the velocities first)
        velocity = [k for k, c in enumerate(plant.state_idx[0]) if c in plant.vel_idx]
        operator = SSMNetworkOperator(mps, mbs, feedback_channels=velocity,
                                      feedback_norm=getattr(args, "feedback_norm", "constant"),
                                      batch_agents=bool(getattr(args, "batch_agents", False)),
                                      compile=bool(getattr(args, "compile", False)), **network)
        widths = ("ssm_d_model", "ssm_d_state", "ctx_d_features", "ctx_head_hidden", "mb_hidden")
    else:
        directions = [BoundedMLPOperator(
            w_dim=len(plant.state_idx[i]), z_dim=z_dims[i], r=plant.agent_in_dim, s=1,
            hidden_dim=int(args.mb_hidden), num_layers=int(args.mb_layers),
            use_z_residual=True, z_residual_gain=float(args.z_residual_gain),
            bound_mode="softsign", clamp_value=float(args.mb_bound)) for i in range(n)]
        operator = RENMADOperator(directions, state_dim=int(args.ren_state_dim), width=int(args.ren_width),
                                  init_std=float(args.ren_init_std), **network)
        widths = ("ren_state_dim", "ren_width", "mb_hidden", "mb_layers")
    # same attributes as the SSM operator of Distributed/, read by the run scripts
    operator.param_budget = budget                                    # None unless size-matched
    operator.input_padding = None
    if getattr(args, "use_ssm", False) and padded:                    # padded sizes and real sizes, as Distributed/
        operator.input_padding = (len(plant.state_idx[0]), z_dims[0], [len(i) for i in plant.state_idx], real_z_dims)
    operator.widths = {k: getattr(args, k) for k in widths}           # widths actually built
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


def agent_operators(operator) -> Iterator[nn.Module]:
    """The per-agent l2 operators (RENs or M_p)."""
    return iter(operator.mps if isinstance(operator, SSMNetworkOperator) else operator.rens)


def core_parameters(operator) -> Iterable[nn.Parameter]:
    """The l2 part (REN or SSM cores, gain variables, learned edge weights), trained at the full learning rate;
    the readouts (MAD directions, M_b, contextual heads) go with the slow group (--lr-ratio), as in Distributed/."""
    return operator.core_parameters()


def controller_tag(args) -> str:
    """Prefix of the run folders: renmad, or ssmnet_<mp-type>_<mb-type> with --use-ssm."""
    return 'ssmnet_%s_%s' % (args.mp_type, args.mb_type) if getattr(args, 'use_ssm', False) else 'renmad'
