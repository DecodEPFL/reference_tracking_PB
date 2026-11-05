"""Distributed REN-MAD controller (controllers/ren_mad.py): certificate, gain, communication and closed loop."""

import argparse, os, sys

import pytest
import torch
from torch.nn import functional

from Distributed_Danilo.controllers import build_controller, core_parameters
from Distributed_Danilo.controllers.interconnection import make_adjacency
from Distributed_Danilo.controllers.ren_mad import DTYPE

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'robots_formation'))
from Distributed_Danilo.experiments.robots_formation.arg_parser import argument_parser, print_args   # noqa: E402
from Distributed_Danilo.experiments.robots_formation.run import get_data, make_dataset, make_plant       # noqa: E402

SMALL = ['--horizon', '20', '--num-rollouts', '6', '--mb-hidden', '16', '--mb-layers', '3',
         '--ren-state-dim', '4', '--ren-width', '3']


def make(argv=(), seed=0):
    torch.manual_seed(seed)
    args = argument_parser(SMALL + list(argv))
    plant = make_plant(args)
    return args, plant, build_controller(args, plant)


def perturb(module, scale=1.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))


def network_outputs(op, w):
    """Outputs y_0..y_T of the network of RENs alone for the disturbances w (B, T, Nx), from rest."""
    op.reset()
    w_ag = w[..., op.w_idx].to(DTYPE)
    B, T, N = w.shape[0], w.shape[1], op.n_agents
    x = torch.zeros(B, N, op._mats['F'].shape[-1], dtype=DTYPE)
    y = torch.zeros(B, N, dtype=DTYPE)
    ys = []
    for t in range(T):
        ys.append(y)
        v = torch.cat([w_ag[:, t], (y @ op._mats['W'].T).unsqueeze(-1)], dim=-1)
        x, y = op._ren_step(x, v)
    ys.append(y)
    return torch.stack(ys, 1)                                                    # (B, T + 1, N)


@pytest.mark.parametrize('graph', ['complete', 'cycle', 'none'])
def test_certificate_holds_for_any_parameters(graph):
    _, _, ctl = make(['--adjacency', graph, '--ren-learn-edges', '--ren-edge-weight', '0.1'])
    op = ctl.operator
    assert op.certify()['verified']
    perturb(op, scale=1.0)                                                       # REN, gain and edge variables
    result = op.check_certificate()
    assert result['network_max_eigenvalue'] <= 1e-12 and result['max_spectral_radius'] < 1


@pytest.mark.parametrize('gamma, weight', [(25.0, 1e-3), (0.5, 0.5)])
def test_network_gain_is_below_gamma_R(gamma, weight):
    _, plant, ctl = make(['--adjacency', 'complete', '--ren-gamma', str(gamma), '--ren-edge-weight', str(weight)])
    op = ctl.operator
    perturb(op, scale=0.5)
    g = torch.Generator().manual_seed(0)
    w = 3.0 * torch.randn(16, 40, plant.state_dim, generator=g)
    w[8:, 1:] = 0.0                                                              # impulses, like w_0 = x_0
    with torch.no_grad():
        y = network_outputs(op, w)
    ratio = y.pow(2).sum((1, 2)).sqrt() / w.to(DTYPE).pow(2).sum((1, 2)).sqrt()
    assert ratio.max() <= gamma * (1 + 1e-9)
    assert ratio.max() > 1e-3 * gamma                                            # not trivially zero


def test_u_gain_is_below_the_mad_bound():
    _, plant, ctl = make(['--info-pattern', 'local', '--ren-gamma', '0.5', '--ren-edge-weight', '0.5'])
    op = ctl.operator
    perturb(op, scale=0.5)
    g = torch.Generator().manual_seed(0)
    w = 3.0 * torch.randn(8, 30, plant.state_dim, generator=g)
    z = 5.0 * torch.randn(8, 30, plant.in_dim, generator=g)
    with torch.no_grad():
        op.reset()
        u = torch.cat([op(w[:, t:t + 1], z[:, t:t + 1]) for t in range(30)], 1)
    bound = op.certify()['u_gain_bound']
    assert bound == pytest.approx(0.5 * 2 ** 0.5 * 8.0)                         # gamma_R sqrt(nu) mb_bound
    assert (u.pow(2).sum((1, 2)).sqrt() / w.pow(2).sum((1, 2)).sqrt()).max() <= bound


def reference_step(matrices, x, b, v):
    """ren_mad CertifiedREN.step (one agent, E solved at every step)."""
    acts = []
    for j in range(matrices['Lambda'].shape[0]):
        value = functional.linear(x, matrices['C1'][j]) + functional.linear(v, matrices['D12'][j])
        if j:
            value = value + functional.linear(torch.stack(acts, dim=-1), matrices['D11'][j, :j])
        acts.append(torch.tanh(value / matrices['Lambda'][j]))
    a = torch.stack(acts, dim=-1)
    rhs = functional.linear(x, matrices['F']) + functional.linear(a, matrices['B1']) + functional.linear(v, matrices['B2'])
    next_x = torch.linalg.solve(matrices['E'], rhs.unsqueeze(-1)).squeeze(-1)
    next_b = (functional.linear(x, matrices['C2']) + functional.linear(a, matrices['D21'])
              + functional.linear(v, matrices['D22'])).squeeze(-1)
    return next_x, next_b


def test_batched_step_matches_the_per_agent_ren_mad_step():
    _, _, ctl = make(['--adjacency', 'cycle'])
    op = ctl.operator
    perturb(op, scale=0.3)
    with torch.no_grad():
        op.reset()
        real = op.realize()
        g = torch.Generator().manual_seed(0)
        x = torch.randn(5, op.n_agents, 4, generator=g, dtype=DTYPE)
        v = torch.randn(5, op.n_agents, op.w_dim + 1, generator=g, dtype=DTYPE)
        x_next, y_next = op._ren_step(x, v)
        for i, loc in enumerate(real.locals):
            rx, rb = reference_step(loc.matrices, x[:, i], None, v[:, i])
            torch.testing.assert_close(x_next[:, i], rx)
            torch.testing.assert_close(y_next[:, i], rb)


def test_cycle_rens_and_context_reach_only_the_successor():
    """Cycle A <- B <- C <- A: u^i at t = 2 depends on its own w, on the w_0 of its successor through the
    REN communication (one step per hop), and on the successor's signals in its context; never on the third robot."""
    args, plant, ctl = make(['--formation-shape', 'isosceles', '--info-pattern', 'neighbors', '--adjacency', 'cycle'])
    assert 'neighbors on cycle graph' in print_args(args)
    op = ctl.operator
    torch.manual_seed(0)
    w = [torch.randn(1, 1, 12, requires_grad=True) for _ in range(3)]
    z = torch.randn(1, 1, 6, requires_grad=True)
    x = torch.randn(1, 1, 12, requires_grad=True)
    u_prev = torch.randn(1, 1, 6, requires_grad=True)
    op.reset()
    u0 = op(w[0], z, x=x, u_prev=u_prev)
    assert u0.abs().max() == 0                                                   # y_0 = 0: nothing published yet
    op(w[1], z, x=x, u_prev=u_prev)
    u = op(w[2], z, x=x, u_prev=u_prev)
    blk = lambda g, j, d: g[..., d * j:d * j + d].abs().sum().item() if g is not None else 0.0
    for i, (own, seen, unseen) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
        grads = torch.autograd.grad(u[..., 2 * i:2 * i + 2].sum(), (*w, z, x, u_prev), retain_graph=True, allow_unused=True)
        gw0, gw1, gw2, gz, gx, gu = grads
        assert blk(gw0, own, 4) > 0 and blk(gw1, own, 4) > 0 and blk(gw2, own, 4) > 0   # own REN input and direction
        assert blk(gw0, seen, 4) > 0 and blk(gw1, seen, 4) == 0                      # successor's y_1 reached i at t = 2
        assert all(blk(g, unseen, 4) == 0 for g in (gw0, gw1, gw2))                  # two hops away: not yet
        assert blk(gz, own, 2) > 0 and blk(gz, seen, 2) > 0 and blk(gz, unseen, 2) == 0
        assert blk(gx, seen, 4) > 0 and blk(gx, unseen, 4) == 0
        assert blk(gu, seen, 2) > 0 and blk(gu, unseen, 2) == 0 and blk(gu, own, 2) == 0


def test_closed_loop_with_patches_trains_and_sees_the_drag():
    args, plant, ctl = make(['--patches', '--info-pattern', 'neighbors'])
    op = ctl.operator
    seen = [[] for _ in range(plant.n_agents)]
    for i, d in enumerate(op.directions):
        d.register_forward_pre_hook(lambda m, inp, i=i: seen[i].append(inp[1]))
    train, _ = get_data(args, make_dataset(args))
    x, e, u = plant.rollout(ctl, train[:4], train=True)
    assert torch.isfinite(x).all() and u[:, 1:].abs().max() > 0
    log_s = torch.log(plant.drag_log)                                            # (4, T, n)
    for i in range(plant.n_agents):
        c = torch.cat(seen[i], 1)
        torch.testing.assert_close(c[..., 2], log_s[..., i])                     # [target, log s^i, neighbors]
    (x.pow(2).mean() + u.pow(2).mean()).backward()
    for p in ctl.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    core = {id(p) for p in core_parameters(op)}
    assert any(p.grad.abs().sum() > 0 for p in ctl.parameters() if id(p) in core)
    assert any(p.grad.abs().sum() > 0 for p in ctl.parameters() if id(p) not in core)


def test_checkpoint_rebuilds_the_same_controller():
    args, plant, ctl = make(['--info-pattern', 'neighbors', '--adjacency', 'ring', '--ren-learn-edges'])
    perturb(ctl, scale=0.2)
    train, _ = get_data(args, make_dataset(args))
    # before the rollout: in train mode the spectral norms of M_b advance their power iteration at every step
    state = {k: v.clone() for k, v in ctl.state_dict().items()}
    with torch.no_grad():
        _, _, u = plant.rollout(ctl, train[:2])
    torch.manual_seed(123)                                                       # another initialization
    again = build_controller(argparse.Namespace(**vars(args)), plant)
    again.load_state_dict(state)
    with torch.no_grad():
        _, _, u2 = plant.rollout(again, train[:2])
    torch.testing.assert_close(u2, u)


@pytest.mark.parametrize('argv', [['--info-pattern', 'relay'], ['--info-pattern', 'centralized'],
                                  ['--info-pattern', 'full'], ['--share-weights', '--adjacency', 'ring']])
def test_unported_patterns_are_rejected(argv):
    with pytest.raises(ValueError):
        make(argv)


def test_isolated_agents_have_no_communication():
    _, _, ctl = make(['--adjacency', 'none'])
    op = ctl.operator
    assert op.edge_rows.numel() == 0 and op.weight_matrix().abs().sum() == 0
    gains = op.realize().gains.gamma
    assert torch.isfinite(gains).all() and (gains > 0).all()
    assert make_adjacency('none', 3).sum() == 0
