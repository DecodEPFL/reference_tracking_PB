"""
--use-ssm: the certified network of REN-MAD (ren_mad.py) with the operators of the SSM controller of
Distributed/ in place of the RENs and the MAD readout.

Agent i runs the M_p^i and M_b^i that Distributed/ builds from the same flags (--mp-type ssm | contextual_ssm,
--ssm-*, --ctx-*, --mb-type, --feat-dim, --mb-*, same context c^i), the only difference being that the
commands of its neighbors (the whole factorized output, u^j = M_b^j ⊠ M_p^j), normalized by a constant, are
added to the velocity channels of its reconstructed disturbance at the input of its recurrent core:

    v^i_t = w^i_t + E_v sum_j a_ij u^j_{t-1} / K_j  E_v: (u_x, u_y) onto the (v_x, v_y) channels of w^i
    y^i_t = M_p^i(v^i_t, c^i_t)                    in R^s, s = --feat-dim (nu without M_b)
    u^i_t = M_b^i(w^i_t, c^i_t) ⊠ y^i_t            as in Distributed/

So every M_p^i has exactly the input size of Distributed/; the mixer and gate heads of the contextual SSM
and M_b^i still read the pure w^i (the feedback only reaches the core input, FeedbackAgent). The feedback is
delayed by one step, which makes the loop well posed (the cores have a direct feedthrough).

Normalization (--feedback-norm). K_j is the readout bound of agent j, the gain from its core's output to u^j:
    K_j = mixer bound x |M_b|,     |M_b| = sup ||M_b^j||_2 (mb_bound sqrt(nu s) for the bounded MLP),
so the fed-back signal u~^j = u^j / K_j has at most the gain of the core of j (constant), or K_j = 1 (none:
the raw commands). The certificate of ren_mad bounds the local gains roughly as gamma_i^2 x (incoming weight)
<= 1: quadratic in the gain, linear in the coupling, so it is far less conservative when the fed-back signals
have gains of the order of the cores' than of the commands' (about 177 times more coupling for the formation
defaults at the same core gain). The edge weights a_ij are in the units of u~.

Certificate. The local operator of agent i maps v^i (and, with the "input" context port, the filtered targets
z~^i) to u~^i. With e^i = [w^i; z~^i] and the feedback added to the velocity rows of w^i, the network LMI of
ren_mad (network_certificate; general_bounds covers rows holding both w and feedback) fixes local gains
gamma_i. Every M_p^i is built with a prescribed gain and run with the override

    gamma_i K_i / (|M_b| * mixer bound)      (gamma_i with the constant normalization),

so that the gain of v^i -> u~^i is at most gamma_i, whatever the parameters. Then
    ||u~||_2 <= gamma_R ||e||_2,   ||u||_2 <= max_j K_j gamma_R ||e||_2    (u before --output-amplification):
the delayed feedback u~_{t-1} has the same l2 norm as u~ over the whole horizon, so the local bound holds from
v^i to the delayed output and the network argument of ren_mad applies.

Realization (as the REN realization of ren_mad). Within a rollout the parameters and the gain overrides are
fixed, so the certified encoder / decoder of every core (spectral caps, decoder attenuation to the override)
are computed once, in reset(), and the step uses them as plain weights; gradients flow through that single
computation. The steps run on private template copies with the realized weights substituted
(torch.func.functional_call): the agents' own modules are never patched, so certify() reads the library's
bounds. With --batch-agents the N agents run in one vmapped (--compile: compiled) call through AgentBatch,
the machinery of Distributed/, which also rescales its exact-SVD layers once per rollout, as there.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.func import functional_call

from Distributed_Danilo.controllers.batched_agents import AgentBatch
from Distributed_Danilo.controllers.operators import as_bt, boxtimes_timewise
from Distributed_Danilo.controllers.ren_mad import CertifiedNetworkOperator, DTYPE
from Distributed_Danilo.controllers.ssm_operators import MpContextualSSM, MpDeepSSM


def mb_norm_bound(mb: Optional[nn.Module]) -> float:
    """Upper bound of the spectral norm of M_b's matrix for any input (1 without M_b)."""
    if mb is None:
        return 1.0
    bound = float(getattr(mb, "bound_value", float("inf")))
    if mb.__class__.__name__.startswith("Diagonal"):
        return bound
    return bound * math.sqrt(mb.r * mb.s)                  # entrywise bound -> Frobenius -> spectral


def deep_ssm_of(mp: nn.Module) -> nn.Module:
    """The DeepSSM recurrent core of an M_p."""
    return mp.core.core if isinstance(mp, MpContextualSSM) else mp.core


FEEDBACK_MODES = ("u_velocity", "y_append", "core_append")
APPEND_MODES = ("y_append", "core_append")


class FeedbackAgent(nn.Module):
    """One agent, called as agent([w, fb], c):

        u_velocity:  u = M_b(w, c) ⊠ M_p(w + E_v fb, c)                  -> u (B, T, nu)
        y_append:    y = M_p([w, fb], c) (fb appended at the core input), u = M_b(w, c) ⊠ y   -> [u, y]
        core_append: as y_append, but returns the core's output h (before the mixer port of the contextual SSM,
                     y = A(w, c) h; h = y for the plain SSM): the l2 block's own output     -> [u, h]

    The same computation as FactorizedOperator(M_p, M_b) of Distributed/ (ContextualDeepSSM.forward for the
    contextual M_p), except for the fed-back channels, which only reach the recurrent core: the mixer / gate
    heads, the select port and M_b read w. In y_append the core input is [w, filtered targets, fb] (the filtered
    targets with the "input" port), so only the inner DeepSSM is wider than in Distributed/.
    gamma: optional gain override of the core (the realized templates have it in their weights).
    """

    def __init__(self, mp: nn.Module, mb: Optional[nn.Module], w_dim: int, feedback_channels: Sequence[int],
                 feedback_mode: str = "u_velocity", fb_dim: Optional[int] = None):
        super().__init__()
        if feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"feedback_mode must be one of {FEEDBACK_MODES}.")
        self.mp, self.mb = mp, mb
        self.mode = feedback_mode
        self.w_dim = int(w_dim)
        self.fb_dim = len(feedback_channels) if feedback_mode == "u_velocity" else int(fb_dim)
        E = torch.zeros(len(feedback_channels), self.w_dim)
        for k, c in enumerate(feedback_channels):
            E[k, c] = 1.0
        self.register_buffer("feedback_map", E, persistent=False)   # u_velocity: fb (nu) -> velocity channels of w

    def reset(self) -> None:
        self.mp.reset()

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None, gamma=None) -> torch.Tensor:
        x = as_bt(x)
        w, fb = x[..., :self.w_dim], x[..., self.w_dim:]
        append = self.mode in APPEND_MODES
        v = w if append else w + fb @ self.feedback_map                 # u_velocity: w^i + E_v fb on the velocities
        mp = self.mp
        if isinstance(mp, MpContextualSSM):                            # ContextualDeepSSM.forward, core input v
            cds = mp.core
            w3, z3, _, filtered = cds._prepare_inputs(w, z, time_offset=mp._t)
            core_input = v if filtered is None else torch.cat((v, filtered), dim=-1)
            if append:
                core_input = torch.cat((core_input, fb), dim=-1)       # y_append: [w, filtered targets, fb]
            gates = cds.gate(w3, z3) if cds.gate is not None else None
            select = z3 if "select" in cds.context_modes else None
            features, _ = cds.core(core_input, gamma=gamma, mode=mp.mode, reset_state=False,
                                   detach_state=mp.detach_state, context_gates=gates, select_context=select)
            y = boxtimes_timewise(cds.mixer(w3, z3), features) if cds.mixer is not None else features
            mp._t += int(w.shape[1])
        else:
            y = features = mp(torch.cat((w, fb), dim=-1) if append else v, gamma=gamma)
        y = as_bt(y)
        u = y if self.mb is None else boxtimes_timewise(self.mb(w, z), y)
        if not append:
            return u
        return torch.cat((u, y if self.mode == "y_append" else as_bt(features)), dim=-1)


def _use_realized_weights(agent: FeedbackAgent) -> FeedbackAgent:
    """Make the core of a (template) agent take its encoder / decoder as already realized weights: the spectral
    caps and the decoder attenuation are applied once per rollout (reset) instead of at every step. Instance
    attributes shadow the DeepSSM methods on this copy only."""
    core = deep_ssm_of(agent.mp)
    core._capped_encoder_decoder = lambda m=core: (m.encoder_w, m.decoder_w)
    core._smooth_capped_scale_from_logs = lambda *args, **kwargs: torch.ones(())
    core._effective_gamma_cap = lambda *args, **kwargs: torch.ones(())
    core._log_block_gain_product = lambda *args, **kwargs: torch.zeros(())
    return agent


class RealizedAgentBatch(AgentBatch):
    """AgentBatch of Distributed/ whose template takes the realized encoder / decoder of every agent."""

    def __init__(self, agents: Sequence[nn.Module], realize, compile: bool = False):
        super().__init__(agents, compile=compile)
        self.realize = realize                                         # () -> {param name: (N, ...) stacked}

    def _build_template(self):
        super()._build_template()
        _use_realized_weights(self.template)

    def reset(self) -> None:
        super().reset()
        self.params = {**self.params, **self.realize()}


class SSMNetworkOperator(CertifiedNetworkOperator):
    """u^i = M_b^i(w^i, c^i) ⊠ M_p^i(w^i + E_v sum_j a_ij u^j_{t-1} / K_j, c^i), with certified local gains.

    Args (on top of CertifiedNetworkOperator's):
        mps: one M_p per agent (MpDeepSSM or MpContextualSSM built with a prescribed gamma and an input of
            size w_dim), mapping to y^i in R^s.
        mbs: one M_b per agent (called as M_b(w^i, c^i) -> (B, T, nu, s)), or None (u^i = y^i).
        feedback_channels: the nu channels of w^i that receive the neighbors' commands (the velocities).
        feedback_norm: 'constant' (fed-back signal / K_j, K_j its gain bound from the core of j) or 'none' (raw).
        feedback_mode: 'u_velocity' (the commands u^j added to the velocity channels of w^i), 'y_append' (the
            M_p outputs y^j appended to the core input; M_b out of the loop) or 'core_append' (the outputs h^j of the
            recurrent cores appended, as the REN outputs in ren_mad: the mixer port and M_b out of the loop).
        batch_agents: run the N agents in one vmapped call (AgentBatch), as --batch-agents in Distributed/;
            falls back to the loop (with a warning) if the agents differ in size or the cell is unsupported.
        compile: with batch_agents, compile the vmapped step (torch.compile), as --compile in Distributed/.
    """

    def __init__(self, mps: Sequence[nn.Module], mbs: Optional[Sequence[nn.Module]], w_idx, z_idx, u_idx,
                 adjacency: torch.Tensor, feedback_channels: Sequence[int], feedback_norm: str = "constant",
                 feedback_mode: str = "u_velocity", batch_agents: bool = False, compile: bool = False,
                 **network_kwargs):
        super().__init__(w_idx, z_idx, u_idx, adjacency, **network_kwargs)
        if len(mps) != self.n_agents or (mbs is not None and len(mbs) != self.n_agents):
            raise ValueError("mps and mbs must list every agent.")
        if feedback_norm not in ("constant", "none"):
            raise ValueError("feedback_norm must be 'constant' or 'none'.")
        if feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"feedback_mode must be one of {FEEDBACK_MODES}.")
        self.feedback_norm, self.feedback_mode = feedback_norm, feedback_mode
        self.out_dim = int(mps[0].core.d_output)                                   # s, the size of y^i
        self.core_dim = int(deep_ssm_of(mps[0]).d_output)                          # size of the core's output h^i
        append = feedback_mode in APPEND_MODES
        append_dim = self.out_dim if feedback_mode == "y_append" else self.core_dim
        self.agent_u_dim = self.u_dim // self.n_agents
        self.feedback_channels = [int(c) for c in feedback_channels]
        if len(self.feedback_channels) != self.agent_u_dim or not all(0 <= c < self.w_dim for c in self.feedback_channels):
            raise ValueError("feedback_channels must give one channel of w^i per input channel of the agent.")
        exo = []
        for mp in mps:
            if not isinstance(mp, (MpDeepSSM, MpContextualSSM)):
                raise TypeError("--use-ssm needs MpDeepSSM or MpContextualSSM operators.")
            if not getattr(deep_ssm_of(mp), "use_cert_scaling", False):
                raise ValueError("Every M_p must be built with a prescribed gamma (certified cap).")
            ctx_in = mp.core.d_context if isinstance(mp, MpContextualSSM) and "input" in mp.core.context_modes else 0
            exo.append(self.w_dim + ctx_in)                                    # e^i = [w^i, filtered targets]
            if isinstance(mp, MpContextualSSM) and mp.core.d_input != self.w_dim:
                raise ValueError("The contextual M_p must read w^i (w_dim) in its heads, as in Distributed/.")
            if deep_ssm_of(mp).d_input != self.w_dim + ctx_in + (append_dim if append else 0):
                raise ValueError("The M_p core input must be [w^i, filtered targets] (u_velocity: the feedback is added "
                                 "to w^i), or [w^i, filtered targets, feedback] (y_append: s, core_append: core output).")
        self.exo_sizes = tuple(exo)
        self.fb_dim = append_dim if append else self.agent_u_dim                  # size of the fed-back signal
        self.agents = nn.ModuleList(FeedbackAgent(mp, None if mbs is None else mbs[i], self.w_dim, self.feedback_channels,
                                                  feedback_mode, fb_dim=append_dim)
                                    for i, mp in enumerate(mps))
        scale = self._feedback_bounds() if feedback_norm == "constant" else [1.0] * self.n_agents
        self.register_buffer("feedback_scale", torch.tensor(scale), persistent=False)   # K_j (hyperparameters)
        prefix = "mp.core.core." if isinstance(mps[0], MpContextualSSM) else "mp.core."
        self._enc_name, self._dec_name = prefix + "encoder_w", prefix + "decoder_w"
        self.compile = bool(compile)
        self.batch_agents = bool(batch_agents) and self.n_agents > 1
        object.__setattr__(self, "_runner", None)       # AgentBatch or per-agent templates: not submodules
        object.__setattr__(self, "_templates", None)
        self._realized: Optional[list[dict]] = None
        self._W: Optional[torch.Tensor] = None
        self._u: Optional[torch.Tensor] = None          # (B, N, nu) commands of the previous step
        self._fb: Optional[torch.Tensor] = None         # (B, N, fb_dim) fed-back signals of the previous step (u or y)

    # --- the agents' parts
    @property
    def mps(self) -> list:
        return [a.mp for a in self.agents]

    @property
    def mbs(self) -> Optional[list]:
        return None if self.agents[0].mb is None else [a.mb for a in self.agents]

    @staticmethod
    def _matrix_norm_bound(mp: nn.Module) -> float:
        return float(mp.core.matrix_norm_bound) if isinstance(mp, MpContextualSSM) else 1.0

    def _readout_bounds(self) -> list[float]:
        """Per agent: mixer bound of M_p x spectral bound of M_b, the gain from the core output to u^i."""
        return [self._matrix_norm_bound(a.mp) * mb_norm_bound(a.mb) for a in self.agents]

    def _feedback_bounds(self) -> list[float]:
        """Per agent: gain bound from the core output to the fed-back signal before normalization: to u^i (mixer x
        M_b) in u_velocity, to y^i (mixer) in y_append."""
        if self.feedback_mode == "y_append":
            return [self._matrix_norm_bound(a.mp) for a in self.agents]
        if self.feedback_mode == "core_append":                                    # the core's own output
            return [1.0] * self.n_agents
        return self._readout_bounds()

    def _core_to_feedback(self) -> list[float]:
        """Per agent: gain bound from the core output to the fed-back signal / K_i."""
        return [r / float(k) for r, k in zip(self._feedback_bounds(), self.feedback_scale.tolist())]

    def local_gains(self):
        """(weights, gains, certificate) of the current parameters: u_velocity adds the feedback onto the velocity
        rows of w^i, y_append gives it rows of its own (as ren_mad)."""
        comm_rows = self.feedback_channels if self.feedback_mode == "u_velocity" else None   # appended: own rows
        weights, _, gains, certificate = self.network_gains(self.exo_sizes, self.fb_dim, comm_rows)
        return weights, gains, certificate

    # --- realization (once per rollout)
    def _realize_core(self, agent: FeedbackAgent, gamma: torch.Tensor) -> dict:
        """Certified encoder / decoder of the core of agent at the gain override gamma (DeepSSM.forward's math)."""
        core = deep_ssm_of(agent.mp)
        dtype, device = core.encoder_w.dtype, core.encoder_w.device
        encoder = core._spectrally_capped_weight(core.encoder_w)
        decoder = core._spectrally_capped_weight(core.decoder_w)
        cap = core._effective_gamma_cap(gamma=gamma, device=device, dtype=dtype)
        log_prod = core._log_block_gain_product(device=device, dtype=dtype)
        scale = core._smooth_capped_scale_from_logs(gamma_t=cap, log_gamma_prod=log_prod,
                                                    temperature=core.config.cert_scale_temperature)
        return {self._enc_name: encoder, self._dec_name: decoder * scale}

    def _stacked_realization(self) -> dict:
        return {k: torch.stack([r[k] for r in self._realized]) for k in (self._enc_name, self._dec_name)}

    def _build_runner(self) -> None:
        device = self.feedback_scale.device
        if self.batch_agents and self._runner is None:
            try:
                runner = RealizedAgentBatch(list(self.agents), self._stacked_realization, compile=self.compile)
            except NotImplementedError as exc:
                warnings.warn(f"--batch-agents disabled, running the agents one by one: {exc}")
                self.batch_agents = False
            else:
                object.__setattr__(self, "_runner", runner)
        if not self.batch_agents and (self._templates is None or
                                      next(self._templates[0].parameters()).device != device):
            object.__setattr__(self, "_templates", [_use_realized_weights(copy.deepcopy(a)) for a in self.agents])

    def reset(self) -> None:
        """Start a rollout: local gains and realized cores from the current parameters (once, differentiably),
        cleared states."""
        if hasattr(self.neighbor_features, "reset"):
            self.neighbor_features.reset()
        weights, gains, _ = self.local_gains()
        dtype = deep_ssm_of(self.agents[0].mp).encoder_w.dtype
        readout = torch.tensor(self._core_to_feedback(), dtype=DTYPE, device=gains.gamma.device)
        overrides = (gains.gamma / readout).to(dtype)
        self._overrides = overrides.detach()                            # core gain overrides of the rollout
        self._realized = [self._realize_core(a, overrides[i]) for i, a in enumerate(self.agents)]
        self._W = weights.to(dtype)
        self._u = self._fb = None
        self._build_runner()
        if self.batch_agents:
            self._runner.reset()                       # resets the agents, stacks their parameters + realization
        else:
            for agent, template in zip(self.agents, self._templates):
                agent.reset()
                template.reset()
                template.train(agent.training)
            self._params = [{**dict(a.named_parameters()), **dict(a.named_buffers()), **r}
                            for a, r in zip(self.agents, self._realized)]

    def forward(self, w: torch.Tensor, z: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None,
                u_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """w, z: (B, T, ·); x: state measured at t, u_prev: input applied at t-1 (neighbors only). T = 1 in closed loop."""
        if self._realized is None:
            raise RuntimeError("Call reset() before the first step of a rollout.")
        w = as_bt(w)
        z = as_bt(z) if z is not None else None
        B, T = w.shape[:2]
        if T != 1 and self.neighbor_features is not None:
            raise ValueError("Neighbor signals are causal step by step: call with T = 1.")
        w_ag = w[..., self.w_idx]                                                   # (B, T, N, dw)
        if self._fb is None:
            self._fb = w.new_zeros(B, self.n_agents, self.fb_dim)
        ctx = self.contexts(z, x, u_prev)
        du = self.agent_u_dim
        u_steps = []
        for t in range(T):
            fb = torch.einsum("ij,bjk->bik", self._W,                               # sum_j a_ij (u or y)^j_{t-1} / K_j
                              self._fb / self.feedback_scale.to(self._fb.dtype)[:, None])
            inputs = [torch.cat([w_ag[:, t:t + 1, i], fb[:, None, i]], dim=-1) for i in range(self.n_agents)]
            c_t = [None if c is None else c[:, t:t + 1] for c in ctx]
            if self.batch_agents:
                out = self._runner(torch.stack(inputs), None if c_t[0] is None else torch.stack(c_t))[:, :, 0]
                out = out.permute(1, 0, 2)                                          # (B, N, nu [+ s])
            else:
                out = torch.stack([as_bt(functional_call(tpl, p, (x_i, c_i)))[:, 0]
                                   for tpl, p, x_i, c_i in zip(self._templates, self._params, inputs, c_t)], dim=1)
            u_t = out[..., :du]
            self._u = u_t
            self._fb = out[..., du:] if self.feedback_mode in APPEND_MODES else u_t
            u_steps.append(u_t.reshape(B, 1, -1))
        return self.scatter_u(w, torch.cat(u_steps, dim=1))

    # --- diagnostics
    def core_parameters(self):
        """Recurrent cores of the M_p (as ssm_core_parameters of Distributed/), gain variables and learned edges."""
        cores = [p for mp in self.mps
                 for p in (mp.core.core.parameters() if isinstance(mp, MpContextualSSM) else mp.parameters())]
        return cores + self.gain_variables()

    def certify(self) -> dict:
        """Network LMI and every agent's certified bound v^i -> (fed-back signal) / K_i at its gain override (the
        library's own bounds, computed on the agents' modules from their current weights)."""
        with torch.no_grad():
            _, gains, certificate = self.local_gains()
            network_max = float(torch.linalg.eigvalsh(certificate).max())
            local, cores = [], []
            for mp, gamma, readout in zip(self.mps, gains.gamma.tolist(), self._core_to_feedback()):
                core_bound = float(deep_ssm_of(mp).certified_gain_bound(gamma=gamma / readout))
                cores.append(core_bound)
                local.append(core_bound * readout)
        gammas = gains.gamma.tolist()
        verified = network_max <= 1e-12 and all(b <= g * (1 + 1e-5) for b, g in zip(local, gammas))
        return {
            "verified": bool(verified),
            "gamma_R": self.gamma_R,
            "network_max_eigenvalue": network_max,
            "local_gains": [round(g, 4) for g in gammas],
            "local_certified_bounds": [round(b, 4) for b in local],
            "core_gains": [round(c, 5) for c in cores],
            "readout_bounds": [round(r, 3) for r in self._readout_bounds()],
            "feedback_mode": self.feedback_mode,
            "feedback_norm": self.feedback_norm,
            "exogenous_inputs": "w" if all(n == self.w_dim for n in self.exo_sizes) else "w and filtered targets",
            "edge_weights": self.edge_summary(),
            # ||fed back / K|| <= gamma_R ||e||; then u = M_b y (y_append), u = M_b A h (core_append)
            "u_gain_bound": self.gamma_R * float(self.feedback_scale.max()) * max(
                r / f for r, f in zip(self._readout_bounds(), self._feedback_bounds())),
        }
