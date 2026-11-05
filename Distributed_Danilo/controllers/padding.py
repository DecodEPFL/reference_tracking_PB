"""
Zero-padded agent inputs (``--pad-agent-inputs``): agents hearing fewer neighbors than others get
zero channels in place of the missing ones, so that all N operators have the same input sizes and
run in one vmapped call (``AgentBatch``), on any communication graph.

A padded channel is always zero, so it carries no information; ``mask_padded_inputs`` also makes
the operator blind to it, so each padded M^i computes exactly what an operator of its real input
size would. Every weight reading the inputs (w, z) directly is masked, W -> W ⊙ m with m = 0 on
the padded columns, realised by holding those columns at exactly zero: zeroed at construction and
their gradient masked. With zero columns the spectral norms (spectral_norm, exact-SVD and capped
weights) are those of the real columns, and the masked gradient keeps them zero under any optimizer
(even the tiny gradients through a norm, which Adam would rescale to full steps); parameter names
and checkpoints are unchanged. The gradient hooks live on the parameters: a deepcopy of the
operator does not keep them (load_state_dict into a built operator does).

Ported from Distributed/ for the agents of --use-ssm (FeedbackAgent: the same M_p and M_b; the fed-back commands
are added to w, so they need no column of their own). Masked weights (first layer reading w or z of each branch):
    M_b (bounded_mlp, diagonal)   net[0] on [w, z], z_net[0] on z
    M_p contextual_ssm            SSM encoder on [w (, z with the input port)], the gate and mixer
                                  heads on [(w,) z], the selective cells' param_net on [x, z]
    M_p ssm                       SSM encoder on [w (, lift(z))], the l_p lifter on z
with w = [w, w_bar] under --use-w-augment.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import SpectralNorm

from Distributed_Danilo.controllers.bounded_mlp import BoundedMLPOperator, DiagonalBoundedMLPOperator
from Distributed_Danilo.controllers.ssm_operators import MpContextualSSM, MpDeepSSM


def _first_linear(net: nn.Module) -> nn.Linear:
    return net if isinstance(net, nn.Linear) else next(m for m in net.modules() if isinstance(m, nn.Linear))


def _mask_columns(module: nn.Module, name: str, keep: torch.Tensor) -> None:
    """Hold the columns of the weight ``module.<name>`` (d_out, d_in) where keep = 0 at zero."""
    weight = getattr(module, name)
    if weight.shape[-1] != keep.numel():
        raise ValueError(f"{type(module).__name__}.{name}: {weight.shape[-1]} input columns, mask of {keep.numel()}.")
    if bool(keep.all()):
        return
    keep = keep.to(weight)
    with torch.no_grad():
        weight.mul_(keep)
    weight.register_hook(lambda g: g * keep.to(g))
    # spectral_norm: start the power iteration in the row space, as for the unpadded weight
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, SpectralNorm) and name == hook.name + "_orig":
            v = getattr(module, hook.name + "_v")
            with torch.no_grad():
                v.copy_(nn.functional.normalize(v * keep, dim=0, eps=hook.eps))


def _mask_linear(layer: nn.Linear, keep: torch.Tensor) -> None:
    _mask_columns(layer, "weight_orig" if hasattr(layer, "weight_orig") else "weight", keep)


def _mask_deep_ssm(ssm: nn.Module, input_keep: torch.Tensor, context_keep: torch.Tensor | None) -> None:
    """neural_ssm DeepSSM: encoder on its input, the selective cells' param_net on [x, select context]."""
    if hasattr(ssm, "encoder_w"):                  # certified (gamma) encoder, spectrally capped in forward
        _mask_columns(ssm, "encoder_w", input_keep)
    else:
        _mask_linear(ssm.encoder, input_keep)
    for block in ssm.blocks:
        if getattr(block.lru, "context_dim", 0) > 0:
            _mask_linear(_first_linear(block.lru.param_net),
                         torch.cat([torch.ones(block.lru.D), context_keep]))


def mask_padded_inputs(op: nn.Module, w_keep: Sequence[float], z_keep: Sequence[float] | None) -> None:
    """Make the agent operator blind to its padded input channels (w_keep, z_keep: 1 real, 0 padded)."""
    if not (hasattr(op, "mp") and hasattr(op, "mb")):    # an agent: FeedbackAgent here (FactorizedOperator in Distributed/)
        raise NotImplementedError(f"padded inputs are not supported for {type(op).__name__}.")
    w_keep = torch.as_tensor(w_keep, dtype=torch.float32)
    z_keep = None if z_keep is None else torch.as_tensor(z_keep, dtype=torch.float32)
    if getattr(op, "w_augmenter", None) is not None:     # [w, w_bar]
        w_keep = torch.cat([w_keep, w_keep])

    mp = op.mp
    if isinstance(mp, MpContextualSSM):
        c = mp.core
        _mask_deep_ssm(c.core, torch.cat([w_keep, z_keep]) if "input" in c.context_modes else w_keep, z_keep)
        for head in (c.gate, c.mixer):
            if head is not None and getattr(head, "net", None) is not None:
                _mask_linear(_first_linear(head.net), torch.cat([w_keep, z_keep]) if head.include_disturbance else z_keep)
    elif isinstance(mp, MpDeepSSM):
        if getattr(op, "mp_context_lifter", None) is not None:   # the l_p lift is not ported (--use-ssm rejects it)
            raise NotImplementedError("padded inputs are not supported with an l_p context lifter.")
        _mask_deep_ssm(mp.core, w_keep, None)
    else:
        raise NotImplementedError(f"padded inputs are not supported for M_p {type(mp).__name__}.")

    mb = op.mb
    if isinstance(mb, (BoundedMLPOperator, DiagonalBoundedMLPOperator)):
        _mask_linear(_first_linear(mb.net), torch.cat([w_keep, z_keep]))
        if mb.z_net is not None:
            _mask_linear(_first_linear(mb.z_net), z_keep)
    elif mb is not None:
        raise NotImplementedError(f"padded inputs are not supported for M_b {type(mb).__name__}.")
