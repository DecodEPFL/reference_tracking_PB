"""
Run the N independent agent operators M^1..M^N of a DistributedOperator in one vectorised call.

With independent weights the agents have the same architecture but their own parameters and
recurrent state, so the closed loop calls N small operators one after the other at every step.
At the widths used here each call is dominated by per-op overhead (dispatch, autograd nodes),
not by FLOPs, so the step cost grows with N however small the operators are.

``AgentBatch`` instead stacks the agents' parameters, buffers and recurrent states along a
leading agent dimension and runs a single template copy of the operator under ``torch.func.vmap``:

    u^{1..N} = vmap(functional_call(template, (params^i, buffers^i), (w^i, z^i)))

The agents' own modules are not modified. Two pieces of runtime state are threaded explicitly:

* the SSM cell states (``cell.state``), carried by the batch between steps;
* the spectral-norm power-iteration vectors (``weight_u``, ``weight_v``), updated at every call
  in training mode exactly as ``torch.nn.utils.spectral_norm`` does, and copied back into every
  agent's buffers so checkpoints and the loop path stay consistent.

On the template copy only, two library pieces that write in place into freshly allocated tensors
(not batchable by vmap) are replaced by out-of-place versions computing the same operations:
``RobustMambaDiagSSM.forward`` (loop mode) and the ``spectral_norm`` pre-hook.

Parameters are stacked at ``reset()`` (once per rollout), inside the autograd graph, so gradients
reach every agent's own parameters. The exact-SVD rescaling of ``L2BoundedLinearExact`` weights,
which depends on the parameters only, is applied there as well (the template layers then use the
rescaled weight as is): one batched SVD per layer and rollout instead of one per layer and step. Results equal the loop up to float rounding (batched matmuls);
``check_agent_batch`` verifies this and the factory falls back to the loop otherwise.

With ``compile=True`` the vmapped step is also compiled with ``torch.compile``, fusing its many tiny
ops (about 2.4x faster per training step on CPU for 5 robots, same results up to float rounding).
The first steps pay the compilation, and the step counter of the context filter (a Python int on
the module) is traced as a symbolic int so that every time step reuses the same graph.
"""

from __future__ import annotations

import copy
import types
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap
from torch.nn.utils.spectral_norm import SpectralNorm


try:
    from neural_ssm.ssm import selective_cells as _sc
    from neural_ssm.static_layers.lipschitz_mlps import L2BoundedLinearExact
except ImportError:      # pragma: no cover - neural_ssm is a hard dependency of the SSM operators
    _sc = L2BoundedLinearExact = None


def _plain_linear_forward(self, x):
    """``L2BoundedLinearExact.forward`` with ``W_raw`` already rescaled to spectral norm <= bound
    (done once per rollout in ``AgentBatch.reset``)."""
    return F.linear(x, self.W_raw, bias=None)


def _bounded_weight(m, W: torch.Tensor) -> torch.Tensor:
    """``L2BoundedLinearExact._effective_weight`` (exact norm) of stacked weights W: (N, d_out, d_in)."""
    sigma = torch.linalg.svdvals(W)[..., 0].clamp_min(m.eps)
    return W / torch.clamp(sigma / m.bound, min=1.0)[:, None, None]


def _mamba_diag_forward(self, u, state=None, *, mode="scan", return_state=True, return_last=False,
                        reset_state=True, detach_state=True, select_context=None):
    """``RobustMambaDiagSSM.forward`` in loop mode, without in-place writes into new tensors."""
    u = _sc._normalize_to_3d(u)
    B, T, _ = u.shape
    z0 = _sc._resolve_runtime_state(
        explicit_state=state, internal_state=self.state, reset_state=reset_state,
        batch_size=B, n_state=self.N, device=u.device, dtype=u.dtype, x0=self.x0_param,
    )
    a_bt, b_bt, c_bt, u_scaled_bt = self._compute_params(u, select_context)
    bu_bt = b_bt * u_scaled_bt
    z, ys, zs = z0, [], [z0]
    for t in range(T):
        ys.append(c_bt[:, t, :] * z)
        z = a_bt[:, t, :] * z + bu_bt[:, t, :]
        zs.append(z)
    y = self.out_proj(torch.stack(ys, 1))
    self.state = z.detach() if detach_state else z
    z_seq = torch.stack(zs, 1) if return_state else None
    if return_state and return_last:
        return y, z_seq, z
    if return_state:
        return y, z_seq
    if return_last:
        return y, z
    return y


def _spectral_norm_hook(sn: SpectralNorm, prefix: str, updates: dict):
    """``SpectralNorm.__call__`` without ``out=``: the new u, v are stored in ``updates``."""
    def hook(module, inputs):
        weight = getattr(module, sn.name + "_orig")
        u = getattr(module, sn.name + "_u")
        v = getattr(module, sn.name + "_v")
        weight_mat = sn.reshape_weight_to_matrix(weight)
        if module.training:
            with torch.no_grad():
                for _ in range(sn.n_power_iterations):
                    v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=sn.eps)
                    u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=sn.eps)
            updates[prefix + sn.name + "_u"] = u
            updates[prefix + sn.name + "_v"] = v
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        setattr(module, sn.name, weight / sigma)
    return hook


class AgentBatch:
    """Vectorised execution of same-architecture stateful agent operators (see module docstring).

    Not an nn.Module on purpose: the template copy must not add parameters to the controller.
    """

    def __init__(self, agent_ops: Sequence[nn.Module], compile: bool = False):
        self.ops = list(agent_ops)
        self.compile = compile
        if len(self.ops) < 2:
            raise ValueError("AgentBatch needs at least 2 agent operators.")
        ref = self.ops[0]
        self.param_names = [n for n, _ in ref.named_parameters()]
        self.buffer_names = [n for n, _ in ref.named_buffers()]
        for op in self.ops[1:]:
            if [(n, p.shape) for n, p in op.named_parameters()] != [(n, p.shape) for n, p in ref.named_parameters()] \
                    or [(n, b.shape) for n, b in op.named_buffers()] != [(n, b.shape) for n, b in ref.named_buffers()]:
                raise NotImplementedError("agent operators have different architectures (e.g. different context sizes).")
        self.state_paths = [n for n, m in ref.named_modules() if "state" in vars(m)]
        for n in self.state_paths:
            m = ref.get_submodule(n)
            if _sc is None or type(m) is not _sc.RobustMambaDiagSSM:
                raise NotImplementedError(f"recurrent cell {type(m).__name__} at {n!r} is not supported by AgentBatch.")
        self.template = None
        self._step_fns = {}
        self.states = {}
        self.params = self.buffers = None

    def _build_template(self):
        """Private copy of agent 0 with the vmap-safe cell forward and spectral-norm hooks."""
        template = copy.deepcopy(self.ops[0])
        for n in self.state_paths:
            cell = template.get_submodule(n)
            cell.forward = types.MethodType(_mamba_diag_forward, cell)
        self._sn_updates = {}
        # exact-SVD bounded linears: their weight only depends on the parameters, so it is rescaled
        # once per rollout in reset() instead of one SVD per layer and time step
        self.bounded_paths = {}
        for path, m in template.named_modules():
            if L2BoundedLinearExact is not None and isinstance(m, L2BoundedLinearExact) and m.exact_norm:
                m.forward = types.MethodType(_plain_linear_forward, m)
                self.bounded_paths[(path + "." if path else "") + "W_raw"] = m
            for k, h in list(m._forward_pre_hooks.items()):
                if isinstance(h, SpectralNorm):
                    del m._forward_pre_hooks[k]
                    m.register_forward_pre_hook(_spectral_norm_hook(h, path + "." if path else "", self._sn_updates))
        # (module, name) of the spectral-norm weights, which the hook sets as plain attributes
        self._sn_weights = [(m, n[:-len("_orig")]) for m in template.modules()
                            for n, _ in m.named_parameters(recurse=False) if n.endswith("_orig")]
        self.template = template
        self._step_fns = {}
        self._state_modules = {n: template.get_submodule(n) for n in self.state_paths}

    def reset(self) -> None:
        """Reset every agent and stack its parameters (differentiably), buffers and initial states."""
        for op in self.ops:
            op.reset()
        if self.template is None or next(self.template.parameters()).device != next(self.ops[0].parameters()).device:
            self._build_template()
        self.template.reset()
        self.template.train(self.ops[0].training)
        mods = [dict(op.named_modules()) for op in self.ops]
        self.states = {}
        for n in self.state_paths:
            ss = [m[n].state for m in mods]
            if any(s is not None for s in ss):
                self.states[n] = torch.stack(ss)
        params = [dict(op.named_parameters()) for op in self.ops]
        self.params = {k: torch.stack([p[k] for p in params]) for k in self.param_names}
        for k, m in self.bounded_paths.items():
            self.params[k] = _bounded_weight(m, self.params[k])
        buffers = [dict(op.named_buffers()) for op in self.ops]
        self.buffers = {k: torch.stack([b[k] for b in buffers]) for k in self.buffer_names}
        self._agent_buffers = buffers

    def _step_fn(self, has_z: bool):
        """vmapped (and compiled with self.compile) step of the template, built once per template."""
        if has_z in self._step_fns:
            return self._step_fns[has_z]
        template, cells, updates, sn_weights = self.template, self._state_modules, self._sn_updates, self._sn_weights

        def step(params, buffers, states, w, z):
            for n, cell in cells.items():
                cell.state = states.get(n)
            updates.clear()
            out = functional_call(template, (params, buffers), (w, z) if z is not None else (w,))
            new_states = {n: cell.state for n, cell in cells.items()}
            for cell in cells.values():
                cell.state = None                      # do not leak batched tensors
            for m, name in sn_weights:
                setattr(m, name, None)
            new_buffers = {k: updates[k] for k in updates}
            updates.clear()
            return out, new_states, new_buffers

        fn = vmap(step, in_dims=(0, 0, 0, 0, 0 if has_z else None))
        if self.compile:
            import torch._dynamo
            torch._dynamo.config.allow_unspec_int_on_nn_module = True   # the step counter, see module docstring
            fn = torch.compile(fn)
        self._step_fns[has_z] = fn
        return fn

    def __call__(self, w: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        """w: (N, B, T, dw), z: (N, B, T, dz) or None -> u: (N, B, T, du)."""
        if self.params is None:
            raise RuntimeError("Call reset() before running the AgentBatch.")
        out, self.states, new_buffers = self._step_fn(z is not None)(self.params, self.buffers, self.states, w, z)
        if new_buffers:
            self.buffers = {**self.buffers, **new_buffers}
            with torch.no_grad():                      # keep the agents' own buffers in sync
                dst = [b[k] for k in new_buffers for b in self._agent_buffers]
                src = [new_buffers[k][i] for k in new_buffers for i in range(len(self.ops))]
                torch._foreach_copy_(dst, src)
        return out


@torch.no_grad()
def check_agent_batch(agent_ops: Sequence[nn.Module], w_dim: int, z_dim: Optional[int],
                      steps: int = 4, batch: int = 3, rtol: float = 1e-4) -> float:
    """
    Run copies of the agents step by step with the loop and with AgentBatch on random inputs and
    return the relative max output difference; raise if it exceeds rtol. Leaves agent_ops untouched.
    """
    loop_ops = copy.deepcopy(list(agent_ops))
    batch_runner = AgentBatch(copy.deepcopy(list(agent_ops)))
    for op in loop_ops:
        op.reset()
    batch_runner.reset()
    gen = torch.Generator().manual_seed(0)
    device = next(loop_ops[0].parameters()).device
    n = len(loop_ops)
    diff = scale = 0.0
    for _ in range(steps):
        w = 0.1 * torch.randn(n, batch, 1, w_dim, generator=gen).to(device)
        z = torch.randn(n, batch, 1, z_dim, generator=gen).to(device) if z_dim else None
        u_loop = torch.stack([op(w[i], z[i] if z is not None else None) for i, op in enumerate(loop_ops)])
        u_batch = batch_runner(w, z)
        diff = max(diff, (u_loop - u_batch).abs().max().item())
        scale = max(scale, u_loop.abs().max().item())
    rel = diff / max(scale, 1e-12)
    if not rel <= rtol:
        raise RuntimeError(f"AgentBatch differs from the per-agent loop (relative max difference {rel:.2e}).")
    return rel
