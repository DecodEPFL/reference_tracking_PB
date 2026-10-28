"""Adapters for using neural_ssm modules as the M_p factor of a PB operator
(ported from Context_PB ``ssm_operators.py``)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from Distributed.controllers.operators import as_bt, OperatorBase


class MpDeepSSM(nn.Module):
    """
    Adapter for M_p(w) implemented with neural_ssm.DeepSSM.

    Input:
      w: (B, T, w_dim) or (B, w_dim)
    Output:
      v: (B, T, s_dim)
    """

    def __init__(
        self,
        w_dim: int,
        s_dim: int,
        *,
        mode: str = "loop",
        reset_state_each_call: bool = False,
        detach_state: bool = False,
        **ssm_kwargs: Any,
    ):
        super().__init__()
        try:
            from neural_ssm import DeepSSM
        except ImportError as exc:
            raise ImportError(
                "Could not import neural_ssm.DeepSSM. Install neural_ssm in this environment."
            ) from exc

        self.core = DeepSSM(w_dim, s_dim, **ssm_kwargs)
        self.mode = mode
        self.reset_state_each_call = reset_state_each_call
        self.detach_state = detach_state

    def reset(self) -> None:
        if hasattr(self.core, "reset"):
            self.core.reset()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w = as_bt(w)
        out = self.core(
            w,
            mode=self.mode,
            reset_state=self.reset_state_each_call,
            detach_state=self.detach_state,
        )
        if isinstance(out, tuple):
            out = out[0]
        return as_bt(out)


class ContextRescale(nn.Module):
    """Stateless scalar rescale of the context (streaming-safe, no parameters).

    Pass as ContextualDeepSSM's ``context_encoder`` so the rescale applies
    before every context port (input/gate/mixer/select) uniformly.
    """

    def __init__(self, factor: float):
        super().__init__()
        self.factor = float(factor)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.factor


class InputContextMask(nn.Module):
    """Context filter of the ``"input"`` port that passes only the first ``keep`` channels.

    Wraps ContextualDeepSSM's own filter: the other channels are zeroed before filtering, so
    they still reach the gate/mixer/select ports but never drive the SSM as an input.
    """

    def __init__(self, inner: nn.Module, keep: int, dim: int):
        super().__init__()
        self.inner = inner
        mask = torch.zeros(int(dim))
        mask[:int(keep)] = 1.0
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, context: torch.Tensor, *, time_offset: int = 0, **kwargs: Any) -> torch.Tensor:
        return self.inner(context * self.mask, time_offset=time_offset, **kwargs)


def _channel_mask(keep: int, dim: int) -> torch.Tensor:
    mask = torch.zeros(int(dim))
    mask[:int(keep)] = 1.0
    return mask


def _mask_head_context(head: nn.Module, args: tuple) -> tuple:
    """Forward pre-hook of the ``"gate"`` or ``"mixer"`` head: zero the context channels past its ``context_mask``."""
    disturbance, context = args
    return disturbance, None if context is None else context * head.context_mask


def _mask_select_context(core: nn.Module, args: tuple, kwargs: dict) -> tuple:
    """Forward pre-hook of the SSM core: zero the ``"select"`` context channels past its ``select_context_mask``."""
    if kwargs.get("select_context") is not None:
        kwargs["select_context"] = kwargs["select_context"] * core.select_context_mask
    return args, kwargs


class MpContextualSSM(OperatorBase):
    """M_p(w, z) built on ``neural_ssm.ContextualDeepSSM``.

    Maps a reconstructed disturbance ``w`` and a context signal ``z`` to an
    ``out_dim`` feature vector, injecting context through any combination of the
    ports exposed by ``ContextualDeepSSM``:

      * ``"mixer"`` — a bounded time-wise matrix on the core features (a router:
        output tracks the disturbance feature and vanishes as ``w -> 0``);
      * ``"input"`` — context, ``L2``-projected by ``context_filter``, concatenated
        to ``w`` before the core (a driver: acts on context even when ``w ~ 0``);
      * ``"gate"`` — per-block sigmoid gates that attenuate the SSM/FF branches;
      * ``"select"`` — context conditions a selective core's per-step dynamics.

    With ``input_context_dim = k`` only the first k context channels drive the ``"input"``
    port (the exogenous part of z, e.g. the agent's own target); the rest, e.g. neighbor
    signals computed from the state, only enter the gate/mixer/select ports.
    With ``dynamics_context_dim = m`` only the first m channels reach the gate and select ports,
    which shape the SSM's dynamics; the rest (e.g. the relayed inputs of the other agents) never
    reach the SSM state. With ``mixer_context_dim = m`` only the first m channels reach the
    memoryless ``"mixer"`` port, so the rest only reach the external mixer M_b. The masks are
    forward pre-hooks on the heads and the core, so the parameter names stay those of the library.

    Used either as the M_p factor under an external mixer M_b (``out_dim = s``)
    or alone as the full operator (``out_dim = nu``).

    Designed for the causal, step-by-step (T=1) PB rollout: it keeps the SSM
    recurrent state between steps (cleared by :meth:`reset`) and threads an
    absolute ``time_offset`` so time-windowed context filters (``taper`` /
    ``finite_horizon`` / ``exponential`` / ``polynomial``) stay correct in the
    closed loop.  (The ``difference`` and ``none`` filters are *not* well defined
    across single-step chunks; prefer a windowed filter for closed-loop use.)

    Input:
      w: (B, T, w_dim) or (B, w_dim)         — reconstructed disturbance
      z: (B, T, z_dim) or (B, z_dim) or None — context
    Output:
      v: (B, T, out_dim)
    """

    def __init__(
        self,
        w_dim: int,
        z_dim: int,
        out_dim: int,
        *,
        context_modes: Any = ("mixer", "input", "gate"),
        input_context_dim: int | None = None,
        dynamics_context_dim: int | None = None,
        mixer_context_dim: int | None = None,
        mode: str = "loop",
        detach_state: bool = False,
        **contextual_kwargs: Any,
    ):
        super().__init__()
        try:
            from neural_ssm import ContextualDeepSSM
        except ImportError as exc:
            raise ImportError(
                "Could not import neural_ssm.ContextualDeepSSM. Install the pinned "
                "neural_ssm from requirements.txt in this environment."
            ) from exc

        self.core = ContextualDeepSSM(
            int(w_dim),
            int(z_dim),
            int(out_dim),
            context_modes=context_modes,
            **contextual_kwargs,
        )
        if input_context_dim is not None and self.core.context_filter is not None and input_context_dim < z_dim:
            self.core.context_filter = InputContextMask(self.core.context_filter, input_context_dim, z_dim)
        if dynamics_context_dim is not None and dynamics_context_dim < z_dim:
            if self.core.gate is not None:
                self.core.gate.register_buffer("context_mask", _channel_mask(dynamics_context_dim, z_dim), persistent=False)
                self.core.gate.register_forward_pre_hook(_mask_head_context)
            if "select" in self.core.context_modes:
                self.core.core.register_buffer("select_context_mask", _channel_mask(dynamics_context_dim, z_dim),
                                               persistent=False)
                self.core.core.register_forward_pre_hook(_mask_select_context, with_kwargs=True)
        if mixer_context_dim is not None and mixer_context_dim < z_dim and self.core.mixer is not None:
            self.core.mixer.register_buffer("context_mask", _channel_mask(mixer_context_dim, z_dim), persistent=False)
            self.core.mixer.register_forward_pre_hook(_mask_head_context)
        self.mode = mode
        self.detach_state = detach_state
        self._t = 0

    def reset(self) -> None:
        self.core.reset()
        self._t = 0

    def forward(self, w: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        w = as_bt(w)
        z = as_bt(z) if z is not None else None
        out, _ = self.core(
            w,
            z,
            mode=self.mode,
            reset_state=False,
            detach_state=self.detach_state,
            time_offset=self._t,
        )
        self._t += int(w.shape[1])
        return as_bt(out)

    @torch.no_grad()
    def gain_diagnostics(self) -> dict:
        return self.core.gain_diagnostics()
