"""
Composable construction of the PB operator u = M_b(w, z) ⊠ M_p(w, z).

    --mp-type  ssm             M_p = DeepSSM(w)            (+ optional l_p context lift: DeepSSM([w, lift(z)]))
               contextual_ssm  M_p = ContextualDeepSSM(w, z) with ports from --ctx-modes (+ select)
               ren             M_p = ContractiveREN(w)     (legacy baseline)
    --mb-type  bounded_mlp     M_b(w, z) in R^{nu x s}     (s = --feat-dim; s = 1 is the MAD policy)
               diagonal        M_b(w, z) = diag(g(w, z))   (rPB: s = nu)
               none            u = M_p(.) directly         (s = nu; with contextual_ssm this is the
                                                            fused "contextual_ssm" operator)
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch.nn as nn

from Centralized.controllers.bounded_mlp import BoundedMLPOperator, DiagonalBoundedMLPOperator
from Centralized.controllers.context_lifting import LpContextLifter
from Centralized.controllers.contractive_ren import ContractiveREN
from Centralized.controllers.operators import FactorizedOperator, WIntegralAugmenter
from Centralized.controllers.pair_features import PairOffsets
from Centralized.controllers.PB_controller import PBController
from Centralized.controllers.ssm_operators import ContextRescale, MpContextualSSM, MpDeepSSM

MP_TYPES = ("ssm", "contextual_ssm", "ren")
MB_TYPES = ("bounded_mlp", "diagonal", "none")
CTX_MODES = ("input", "gate", "mixer", "select")
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


def build_mp(args, mp_in_dim: int, z_dim: int, out_dim: int, input_z_dim: Optional[int] = None) -> nn.Module:
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
        context_encoder=encoder,
        context_filter=args.ctx_filter,
        horizon=int(args.horizon),
        context_filter_ramp=ramp if ramp > 0 else None,
        context_filter_decay=float(args.ctx_filter_decay),
        context_filter_power=float(args.ctx_filter_power),
        mixer_bound=float(args.ctx_mixer_bound),
        d_features=int(args.ctx_d_features),
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


def build_operator(args, w_dim: int, z_dim: int, u_dim: int, input_z_dim: Optional[int] = None) -> FactorizedOperator:
    """Assemble M_b ⊠ M_p from the CLI flags (see module docstring).

    input_z_dim: if set, only the first input_z_dim channels of z drive the contextual SSM's
    "input" port (the rest are signals computed from the state, see InputContextMask).
    """
    validate_operator_args(args)
    augmenter = WIntegralAugmenter(float(args.w_augment_decay)) if args.use_w_augment else None
    w_op_dim = 2 * w_dim if augmenter is not None else w_dim

    lifter = build_lifter(args, z_dim) if args.mp_type != "contextual_ssm" else None
    mp_in_dim = w_op_dim + (lifter.out_dim if lifter is not None else 0)
    mp_out_dim = int(args.feat_dim) if args.mb_type == "bounded_mlp" else u_dim

    return FactorizedOperator(
        mp=build_mp(args, mp_in_dim, z_dim, mp_out_dim, input_z_dim),
        mb=build_mb(args, w_op_dim, z_dim, u_dim),
        mp_context_lifter=lifter,
        mp_only=(args.mb_type == "none"),
        w_augmenter=augmenter,
    )


def build_controller(args, plant, z_dim: int) -> PBController:
    """z_dim: width of the exogenous context (the targets). With --pair-features the controller also
    receives the relative positions of every pair of robots, measured from x_t (PairOffsets); as
    signals of the state they only enter the gate/mixer/select ports, not the SSM input port."""
    features = None
    input_z_dim = None
    if getattr(args, "pair_features", False):          # absent from older checkpoints
        features = PairOffsets([plant.pos_idx[2 * i:2 * i + 2] for i in range(plant.n_agents)])
        input_z_dim = z_dim
        z_dim = z_dim + features.out_dim
    operator = build_operator(args, w_dim=plant.state_dim, z_dim=z_dim, u_dim=plant.in_dim, input_z_dim=input_z_dim)
    return PBController(
        plant=plant,
        operator=operator,
        u_dim=plant.in_dim,
        output_scale=float(args.output_amplification),
        detach_state=False,
        context_features=features,
    )


def ssm_core_parameters(operator: FactorizedOperator) -> Iterable[nn.Parameter]:
    """Parameters of the recurrent core of M_p, trained at the full learning rate.

      MpDeepSSM / ContractiveREN -> the whole M_p
      MpContextualSSM            -> the DeepSSM inside ContextualDeepSSM; its
                                    mixer/gate/filter heads go with the slow group.
    """
    mp = operator.mp
    if isinstance(mp, MpContextualSSM):
        return getattr(mp.core, "core", mp.core).parameters()
    return mp.parameters()
