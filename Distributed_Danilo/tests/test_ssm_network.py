"""--use-ssm (controllers/ssm_network.py): the certified network with the M_p / M_b of the SSM controller."""

import argparse, os, sys

import pytest
import torch

from Distributed_Danilo.controllers import SSMNetworkOperator, build_controller, core_parameters
from Distributed_Danilo.controllers.operators import as_bt, boxtimes_timewise

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'robots_formation'))
from Distributed_Danilo.experiments.robots_formation.arg_parser import argument_parser, print_args   # noqa: E402
from Distributed_Danilo.experiments.robots_formation.run import get_data, make_dataset, make_plant       # noqa: E402

SMALL = ['--use-ssm', '--horizon', '20', '--num-rollouts', '6', '--ssm-layers', '2', '--ssm-d-model', '8',
         '--ssm-d-state', '8', '--ctx-d-features', '8', '--mb-hidden', '16', '--ctx-head-hidden', '16',
         '--feat-dim', '4', '--no-match-centralized-size']


def make(argv=(), seed=0):
    torch.manual_seed(seed)
    args = argument_parser(SMALL + list(argv))
    plant = make_plant(args)
    return args, plant, build_controller(args, plant)


def perturb(module, scale=0.5, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))


def outputs(op, w, z):
    """Commands u_0..u_{T-1} of the network (before --output-amplification) for w and the contexts z, from rest."""
    op.reset()
    us = []
    for t in range(w.shape[1]):
        op(w[:, t:t + 1], z[:, t:t + 1])
        us.append(op._u)
    return torch.stack(us, 1)                                                    # (B, T, N, nu)


def test_controller_and_summary():
    args, _, ctl = make()
    assert isinstance(ctl.operator, SSMNetworkOperator)
    assert 'certified SSM network' in print_args(args) and 'contextual_ssm' in print_args(args)
    assert ctl.operator.certify()['verified']


@pytest.mark.parametrize('argv', [['--mp-type', 'ssm'], ['--ctx-modes', 'mixer,gate,select'],
                                  ['--mb-type', 'none', '--ctx-modes', 'mixer,gate'],
                                  ['--ctx-modes', 'mixer,gate,select', '--feedback-norm', 'none']])
def test_network_gain_is_below_gamma_R(argv):
    """No 'input' port: the exogenous input is w alone, ||u / K|| <= gamma_R ||w|| for any context."""
    _, plant, ctl = make(['--info-pattern', 'local', '--adjacency', 'complete', '--ren-gamma', '0.5',
                          '--ren-edge-weight', '0.5'] + argv)
    op = ctl.operator
    perturb(op)
    assert op.check_certificate()['exogenous_inputs'] == 'w'
    g = torch.Generator().manual_seed(0)
    w = 3.0 * torch.randn(8, 30, plant.state_dim, generator=g)
    w[4:, 1:] = 0.0                                                              # impulses, like w_0 = x_0
    z = 5.0 * torch.randn(8, 30, plant.in_dim, generator=g)                      # arbitrary context
    with torch.no_grad():
        u = outputs(op, w, z) / op.feedback_scale[:, None]                  # the fed-back u^j / K_j
    ratio = u.pow(2).sum((1, 2, 3)).sqrt() / w.pow(2).sum((1, 2)).sqrt()
    assert ratio.max() <= 0.5 * (1 + 1e-4)
    assert ratio.max() > 1e-6


@pytest.mark.parametrize('batch', [['--batch-agents'], ['--no-batch-agents']])
def test_feedback_reaches_the_velocities_of_the_successor_only(batch):
    """Cycle A <- B <- C <- A, local context: the core of robot i gets w^i + a u^j_{t-1} / K_j on its velocities,
    so u^i at t = 1 depends on the w_0 of its successor, never on the third robot."""
    _, plant, ctl = make(['--formation-shape', 'isosceles', '--info-pattern', 'local', '--adjacency', 'cycle',
                          '--ren-edge-weight', '0.5'] + batch)
    op = ctl.operator
    assert op.feedback_channels == [2, 3] and all(mp.core.d_input == 4 for mp in op.mps)   # w^i only, as Distributed/
    torch.manual_seed(0)
    w0, w1 = (torch.randn(1, 1, 12, requires_grad=True) for _ in range(2))
    z = torch.randn(1, 1, 6)
    op.reset()
    assert op.batch_agents == (batch == ['--batch-agents'])
    u0 = op(w0, z)
    u1 = op(w1, z)
    blk = lambda g, j: g[..., 4 * j:4 * j + 4].abs().sum().item() if g is not None else 0.0
    for i, (own, seen_j, unseen) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
        g00, = torch.autograd.grad(u0[..., 2 * i:2 * i + 2].sum(), (w0,), retain_graph=True)
        assert blk(g00, own) > 0 and blk(g00, seen_j) == 0 and blk(g00, unseen) == 0   # no feedback at t = 0
        g10, g11 = torch.autograd.grad(u1[..., 2 * i:2 * i + 2].sum(), (w0, w1), retain_graph=True)
        assert blk(g10, own) > 0 and blk(g10, seen_j) > 0 and blk(g10, unseen) == 0
        assert blk(g11, own) > 0 and blk(g11, seen_j) == 0 and blk(g11, unseen) == 0


def test_feedback_agent_routes_the_feedback_to_the_core_only():
    """FeedbackAgent([w, fb], c): the core reads w + E_v fb, the mixer head and M_b read w; with fb = 0 it is
    exactly M_b(w, c) ⊠ M_p(w, c), the operator of Distributed/."""
    _, _, ctl = make(['--formation-shape', 'isosceles', '--info-pattern', 'local'])
    ctl.eval()                                     # spectral norms of M_b: no power-iteration step between calls
    agent = ctl.operator.agents[0]
    seen = {}
    agent.mp.core.mixer.register_forward_pre_hook(lambda m, inp: seen.__setitem__('mixer', inp[0]))
    agent.mb.register_forward_pre_hook(lambda m, inp: seen.__setitem__('mb', inp[0]))
    agent.mp.core.core.register_forward_pre_hook(lambda m, inp: seen.__setitem__('core', inp[0][..., :4]))
    torch.manual_seed(0)
    w, fb, c = torch.randn(2, 1, 4), torch.randn(2, 1, 2), torch.randn(2, 1, 2)
    g = torch.tensor(3.0)
    agent.reset()
    agent(torch.cat([w, fb], -1), c, gamma=g)
    torch.testing.assert_close(seen['mixer'], w)
    torch.testing.assert_close(seen['mb'], w)
    torch.testing.assert_close(seen['core'], w + torch.cat([torch.zeros(2, 1, 2), fb], -1))
    agent.reset()
    u = agent(torch.cat([w, torch.zeros_like(fb)], -1), c, gamma=g)
    agent.reset()
    ref = boxtimes_timewise(agent.mb(w, c), agent.mp(w, c, gamma=g))
    torch.testing.assert_close(u, ref)


def test_rollout_realization_reproduces_the_per_step_capping():
    """The realized cores (caps and decoder attenuation once per rollout) give what DeepSSM computes at every step
    with the gain override on the agents' own modules (eval: the spectral norms of M_b do not move)."""
    args, plant, ctl = make(['--patches', '--info-pattern', 'neighbors', '--ren-edge-weight', '0.05', '--no-batch-agents'])
    perturb(ctl, scale=0.05)
    ctl.eval()
    op = ctl.operator
    train, _ = get_data(args, make_dataset(args))
    with torch.no_grad():
        _, _, u_realized = plant.rollout(ctl, train[:3])
        realized_forward, realized_reset = op.forward, op.reset

        def direct_reset():
            realized_reset()                                                  # gains, overrides, contexts
            for a in op.agents:
                a.reset()

        def direct_forward(w, z=None, x=None, u_prev=None):
            w3 = as_bt(w)
            ctx = op.contexts(as_bt(z), x, u_prev)
            if op._u is None:
                op._u = w3.new_zeros(w3.shape[0], op.n_agents, 2)
            fb = torch.einsum("ij,bjk->bik", op._W, op._u / op.feedback_scale[:, None])
            w_ag = w3[..., op.w_idx]
            op._u = torch.stack([a(torch.cat([w_ag[:, :, i], fb[:, None, i]], -1), ctx[i], gamma=op._overrides[i])[:, 0]
                                 for i, a in enumerate(op.agents)], 1)
            return op.scatter_u(w3, op._u.reshape(w3.shape[0], 1, -1))
        op.forward, op.reset = direct_forward, direct_reset
        _, _, u_direct = plant.rollout(ctl, train[:3])
        op.forward, op.reset = realized_forward, realized_reset
    torch.testing.assert_close(u_direct, u_realized, rtol=1e-4, atol=1e-6)


def test_batched_agents_match_the_loop():
    """--batch-agents (vmap through AgentBatch, realized cores stacked) = the per-agent loop: outputs and gradients."""
    results = {}
    for name, extra in (('loop', ['--no-batch-agents']), ('batch', ['--batch-agents'])):
        args, plant, ctl = make(['--patches', '--info-pattern', 'neighbors', '--ren-edge-weight', '0.05'] + extra)
        perturb(ctl, scale=0.05)
        train, _ = get_data(args, make_dataset(args))
        _, _, u = plant.rollout(ctl, train[:3], train=True)
        u.pow(2).mean().backward()
        assert ctl.operator.batch_agents == (name == 'batch')
        results[name] = (u.detach(), [p.grad.clone() for p in ctl.parameters()])
    torch.testing.assert_close(results['batch'][0], results['loop'][0], rtol=1e-4, atol=1e-6)
    for gb, gl in zip(results['batch'][1], results['loop'][1]):
        torch.testing.assert_close(gb, gl, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize('graph', ['complete', 'ring', 'none'])
def test_certificate_holds_for_any_parameters(graph):
    _, _, ctl = make(['--adjacency', graph, '--ren-learn-edges', '--ren-edge-weight', '0.1', '--patches'])
    perturb(ctl.operator, scale=1.0)
    result = ctl.operator.check_certificate()
    assert result['exogenous_inputs'] == 'w and filtered targets'                   # the 'input' port is on
    assert all(b <= g * (1 + 1e-5) for b, g in zip(result['local_certified_bounds'], result['local_gains']))


def test_closed_loop_trains_through_the_gains():
    args, plant, ctl = make(['--patches', '--info-pattern', 'neighbors'])
    train, _ = get_data(args, make_dataset(args))
    x, _, u = plant.rollout(ctl, train[:4], train=True)
    assert torch.isfinite(x).all() and u.abs().max() > 0
    (x.pow(2).mean() + u.pow(2).mean()).backward()
    op = ctl.operator
    assert op.eta_alpha.grad.abs().sum() > 0 and op.eta_c.grad.abs().sum() > 0      # the certificate is in the loop
    core = {id(p) for p in core_parameters(op)}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ctl.parameters() if id(p) in core)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ctl.parameters() if id(p) not in core)


def test_checkpoint_rebuilds_the_same_controller():
    args, plant, ctl = make(['--info-pattern', 'neighbors', '--adjacency', 'ring', '--ren-learn-edges'])
    perturb(ctl, scale=0.1)
    train, _ = get_data(args, make_dataset(args))
    state = {k: v.clone() for k, v in ctl.state_dict().items()}
    with torch.no_grad():
        _, _, u = plant.rollout(ctl, train[:2])
    torch.manual_seed(123)
    again = build_controller(argparse.Namespace(**vars(args)), plant)
    again.load_state_dict(state)
    with torch.no_grad():
        _, _, u2 = plant.rollout(again, train[:2])
    torch.testing.assert_close(u2, u)


def test_agents_have_the_sizes_of_the_free_controller():
    """No input channel is added for the feedback: the M_p^i have the input size of Distributed/ (w^i)."""
    _, _, ctl = make(['--patches'])
    for mp in ctl.operator.mps:
        assert mp.core.d_input == 4 and mp.core.core.encoder_w.shape[1] == 4 + mp.core.d_context   # w^i + filtered context


def test_match_centralized_size_stays_within_the_budget():
    _, _, ctl = make(['--match-centralized-size', '--ssm-d-model', '16', '--ssm-d-state', '16'])
    op = ctl.operator
    assert op.param_budget is not None
    n_agents = sum(p.numel() for m in list(op.mps) + list(op.mbs) for p in m.parameters())
    assert n_agents <= op.param_budget
    assert op.widths['ssm_d_model'] < 16


@pytest.mark.parametrize('argv', [['--ssm-param', 'lru'], ['--mp-type', 'ren'], ['--mp-context-lift', '--mp-type', 'ssm'],
                                  ['--ctx-filter', 'none'], ['--info-pattern', 'relay']])
def test_uncertifiable_configurations_are_rejected(argv):
    with pytest.raises(ValueError):
        make(argv)


def test_capped_cores_start_from_the_free_initialization():
    """--ssm-init free (default): block gains 1 as the uncapped cores of Distributed/; capped: neural_ssm's own."""
    gammas = lambda ctl: {round(float(b.lru.gamma), 4) for mp in ctl.operator.mps for b in mp.core.core.blocks}
    _, _, free = make()
    _, _, capped = make(['--ssm-init', 'capped'])
    assert gammas(free) == {1.0} and gammas(capped) != {1.0}
    assert free.operator.certify()['verified'] and capped.operator.certify()['verified']


def test_constant_normalization_gives_the_cores_the_local_gains():
    """u^j / K_j has the gain of the core of j: the cores run at gamma_i, not gamma_i / K_i as with the raw commands."""
    _, _, norm = make()
    _, _, raw = make(['--feedback-norm', 'none'])
    cn, cr = norm.operator.certify(), raw.operator.certify()
    assert cn['verified'] and cr['verified']
    K = cn['readout_bounds'][0]
    assert K > 1 and norm.operator.feedback_scale.tolist() == pytest.approx(cn['readout_bounds'], rel=1e-4)
    assert raw.operator.feedback_scale.tolist() == [1.0] * 3
    norm.operator.reset(); raw.operator.reset()
    torch.testing.assert_close(norm.operator._overrides, torch.tensor(cn['local_gains'], dtype=torch.float32), rtol=1e-4, atol=0)
    torch.testing.assert_close(raw.operator._overrides, torch.tensor(cr['local_gains'], dtype=torch.float32) / K, rtol=1e-4, atol=0)


@pytest.mark.parametrize('ctx', [[], ['--neighbor-drag', '--neighbor-targets-input']])
def test_padded_batch_on_an_irregular_graph_equals_the_unpadded_loop(ctx):
    """Leader graph (robots with 1 to 4 neighbors): the neighbor slots are zero-padded to batch the agents, the
    weights reading the padding held at zero (padding.py of Distributed/); same closed loop as the unpadded loop."""
    graph = '0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0'
    argv = ['--n-agents', '5', '--formation-shape', 'bowtie', '--info-pattern', 'neighbors', '--adjacency', graph,
            '--patches', '--ren-edge-weight', '0.05'] + ctx

    def run(extra):
        args, plant, ctl = make(argv + extra)
        return args, plant, ctl
    args, plant, loop = run(['--no-batch-agents'])
    _, _, batch = run(['--batch-agents', '--pad-agent-inputs'])
    assert loop.operator.input_padding is None and batch.operator.input_padding is not None
    assert batch.operator.batch_agents
    sd_loop, sd_pad = loop.state_dict(), batch.state_dict()
    for k, v in sd_pad.items():                       # copy the unpadded weights into the real columns
        if v.shape == sd_loop[k].shape:
            sd_pad[k] = sd_loop[k]
            continue
        weight = sd_pad[k.removesuffix('_v') + '_orig'] if k.endswith('_v') else v
        keep = ~(weight == 0).all(0)
        assert int(keep.sum()) == sd_loop[k].shape[-1], k
        sd_pad[k] = torch.zeros_like(v)
        sd_pad[k][..., keep] = sd_loop[k]
    batch.load_state_dict(sd_pad)
    loop.eval(); batch.eval()
    train, _ = get_data(args, make_dataset(args))
    with torch.no_grad():
        xl, _, ul = plant.rollout(loop, train[:3])
        xb, _, ub = plant.rollout(batch, train[:3])
    assert (ub - ul).abs().max() <= 1e-4 * ul.abs().max()
    assert (xb - xl).abs().max() <= 1e-4 * xl.abs().max()


@pytest.mark.parametrize('batch', [['--no-batch-agents'], ['--batch-agents', '--pad-agent-inputs']])
def test_neighbor_drag_and_targets_input_context_layout(batch):
    """--neighbor-drag --neighbor-targets-input: the context of agent i is [z^i, z^j of its neighbors, s^i, then per
    neighbor (p^j - p^i, s^j, u^j_{t-1})], the signals relayed by --info-pattern relay (plus u^j_{t-1}), and the input
    port reads the targets prefix [z^i, z^j] only."""
    graph = '0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0'
    _, plant, ctl = make(['--n-agents', '5', '--formation-shape', 'bowtie', '--info-pattern', 'neighbors',
                          '--adjacency', graph, '--patches', '--neighbor-drag', '--neighbor-targets-input'] + batch)
    op = ctl.operator
    nbs = [[1, 2, 3, 4], [2], [1], [0, 4], [0, 3]]                  # in-neighbors, in (j - i) mod N order
    slots = max(map(len, nbs)) if op.input_padding is not None else None
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 1, plant.state_dim, generator=g)
    u = torch.randn(3, 1, plant.in_dim, generator=g)
    z = torch.randn(3, 1, plant.in_dim + plant.n_agents, generator=g)
    ctx = op.contexts(z, x, u)
    pos = lambda j: x[..., plant.agent_pos_idx[j]]
    for i, nb in enumerate(nbs):
        k = slots or len(nb)
        pad = lambda d: [z.new_zeros(3, 1, (k - len(nb)) * d)]
        expected = torch.cat([z[..., plant.ref_idx[i]]] + [z[..., plant.ref_idx[j]] for j in nb] + pad(2)
                             + [z[..., plant.drag_idx[i]]]
                             + [torch.cat([pos(j) - pos(i), z[..., plant.drag_idx[j]], u[..., plant.input_idx[j]]], -1)
                                for j in nb] + pad(5), dim=-1)
        torch.testing.assert_close(ctx[i], expected)
        keep = op.agents[i].mp.core.context_filter.mask          # the input port: own and neighbors' targets
        assert keep.tolist() == [1.0] * (2 + 2 * k) + [0.0] * (len(keep) - 2 - 2 * k)
    assert op.certify()['verified']


def test_neighbor_delay_sends_the_signals_of_the_previous_step():
    """--neighbor-delay 1: the neighbors' positions, drags and targets are those of t - 1 (held at t = 0 before), the
    offset formed with the own position of t - 1, as a one-hop message of --info-pattern relay; u^j_{t-1} as without
    delay, and the own target and drag of t."""
    graph = '0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0'
    argv = ['--n-agents', '5', '--formation-shape', 'bowtie', '--info-pattern', 'neighbors', '--adjacency', graph,
            '--patches', '--neighbor-drag', '--neighbor-targets-input', '--no-batch-agents']
    _, plant, ctl = make(argv + ['--neighbor-delay', '1'])
    _, _, ref = make(argv)                                       # no delay: what the delayed one must replay
    op, op0 = ctl.operator, ref.operator
    g = torch.Generator().manual_seed(0)
    xs = torch.randn(3, 3, plant.state_dim, generator=g)
    us = torch.randn(3, 3, plant.in_dim, generator=g)
    zs = torch.randn(3, 3, plant.in_dim + plant.n_agents, generator=g)
    op.reset(); op0.reset()
    step = lambda o, t: o.contexts(zs[:, t:t + 1], xs[:, t:t + 1], us[:, t:t + 1])
    now = [step(op0, t) for t in range(3)]
    delayed = [step(op, t) for t in range(3)]
    for i in range(plant.n_agents):
        n_nb = (now[0][i].shape[-1] - 3) // 7                    # [z^i, k targets, s^i, k x (offset, s^j, u^j)]
        own = list(range(2)) + [2 + 2 * n_nb]                    # own target and own drag: never delayed
        nb_u = [3 + 2 * n_nb + 5 * k + c for k in range(n_nb) for c in (3, 4)]
        nb_rest = [c for c in range(now[0][i].shape[-1]) if c not in own + nb_u]
        for t in range(3):
            src = max(t - 1, 0)
            torch.testing.assert_close(delayed[t][i][..., own + nb_u], now[t][i][..., own + nb_u])
            # offsets, drags and targets of the neighbors: of t - 1 (at t = 0, the t = 0 values)
            torch.testing.assert_close(delayed[t][i][..., nb_rest], now[src][i][..., nb_rest])


# --- appended feedback: y_append (the M_p outputs y^j) and core_append (the core outputs h^j, as the REN outputs in
# ren_mad), summed over the neighbors and appended to the core input. With SMALL: s = 4, core output 8.

MODES = {'y_append': dict(fb=4, K=4.0), 'core_append': dict(fb=8, K=1.0)}
mode_param = pytest.mark.parametrize('mode', list(MODES))


@mode_param
def test_append_interconnection_of_three_robots(mode):
    """3 robots, complete graph, no input port: e = [w1; w2; w3] (12), fed-back z = [f1; f2; f3] (3 x r), v (3 x (4 + r)):
    M_ve = blockdiag([I_4; 0_{r x 4}]), block (i, j) of M_vz = [0_{4 x r}; a I_r] (j != i), M_uz = I; the operator
    divides f^j by K_j (the mixer bound for y, 1 for the core output) before M_vz."""
    _, _, ctl = make(['--feedback-mode', mode, '--formation-shape', 'isosceles', '--ctx-modes', 'mixer,gate,select',
                      '--ren-edge-weight', '0.05'])
    op = ctl.operator
    r, q = MODES[mode]['fb'], 4 + MODES[mode]['fb']
    assert op.fb_dim == r and op.exo_sizes == (4, 4, 4)
    from Distributed_Danilo.controllers.ren_mad import build_interconnection
    ic = build_interconnection(op.weight_matrix(), op.exo_sizes, op.fb_dim, None)
    assert ic.M_ve.shape == (3 * q, 12) and ic.M_vz.shape == (3 * q, 3 * r)
    assert torch.equal(ic.M_uz, torch.eye(3 * r, dtype=ic.M_uz.dtype))
    f64 = dict(dtype=torch.float64)
    torch.testing.assert_close(ic.M_ve, torch.block_diag(*[torch.cat([torch.eye(4, **f64), torch.zeros(r, 4, **f64)])] * 3))
    for i in range(3):
        for j in range(3):
            block = ic.M_vz[q * i:q * i + q, r * j:r * j + r]
            torch.testing.assert_close(block[:4], torch.zeros(4, r, **f64))               # no feedback on the w rows
            torch.testing.assert_close(block[4:], (0.05 if i != j else 0.0) * torch.eye(r, **f64))   # a_ij on the comm rows
    assert op.feedback_scale.tolist() == [MODES[mode]['K']] * 3 and op.certify()['verified']


@mode_param
def test_append_agents_read_w_in_the_heads_and_the_feedback_in_the_core(mode):
    _, _, ctl = make(['--feedback-mode', mode, '--formation-shape', 'isosceles', '--info-pattern', 'local', '--patches'])
    ctl.eval()
    r = MODES[mode]['fb']
    agent = ctl.operator.agents[0]
    core = agent.mp.core.core
    assert agent.mp.core.d_input == 4 and core.d_input == 4 + agent.mp.core.d_context + r   # [w, filtered, feedback]
    seen = {}
    agent.mp.core.mixer.register_forward_pre_hook(lambda m, inp: seen.__setitem__('mixer', inp[0]))
    agent.mb.register_forward_pre_hook(lambda m, inp: seen.__setitem__('mb', inp[0]))
    core.register_forward_pre_hook(lambda m, inp: seen.__setitem__('core', inp[0]))
    torch.manual_seed(0)
    w, fb, c = torch.randn(2, 1, 4), torch.randn(2, 1, r), torch.randn(2, 1, agent.mp.core.d_context)
    agent.reset()
    out = agent(torch.cat([w, fb], -1), c, gamma=torch.tensor(3.0))
    assert out.shape[-1] == 2 + r                                                   # [u, fed-back signal]
    torch.testing.assert_close(seen['mixer'], w)
    torch.testing.assert_close(seen['mb'], w)
    torch.testing.assert_close(seen['core'][..., :4], w)
    torch.testing.assert_close(seen['core'][..., -r:], fb)                           # feedback at the end
    y = out[..., 2:] if mode == 'y_append' else boxtimes_timewise(agent.mp.core.mixer(w, c), out[..., 2:])   # y = A h
    torch.testing.assert_close(out[..., :2], boxtimes_timewise(agent.mb(w, c), y))   # u = M_b ⊠ y


@mode_param
def test_append_network_gain_is_below_gamma_R(mode):
    _, plant, ctl = make(['--feedback-mode', mode, '--info-pattern', 'local', '--ctx-modes', 'mixer,gate,select', '--ren-gamma', '0.5',
                                   '--ren-edge-weight', '0.5'])
    op = ctl.operator
    perturb(op)
    g = torch.Generator().manual_seed(0)
    w = 3.0 * torch.randn(8, 30, plant.state_dim, generator=g)
    w[4:, 1:] = 0.0
    z = 5.0 * torch.randn(8, 30, plant.in_dim, generator=g)
    with torch.no_grad():
        op.reset()
        ys = []
        for t in range(30):
            op(w[:, t:t + 1], z[:, t:t + 1])
            ys.append(op._fb / op.feedback_scale[:, None])                           # the fed-back signal / K_j
    y = torch.stack(ys, 1)
    ratio = y.pow(2).sum((1, 2, 3)).sqrt() / w.pow(2).sum((1, 2)).sqrt()
    assert 1e-6 < ratio.max() <= 0.5 * (1 + 1e-4)


@mode_param
@pytest.mark.parametrize('batch', [['--batch-agents'], ['--no-batch-agents']])
def test_append_feedback_reaches_the_successor_only(mode, batch):
    _, _, ctl = make(['--feedback-mode', mode, '--formation-shape', 'isosceles', '--info-pattern', 'local', '--adjacency', 'cycle',
                               '--ren-edge-weight', '0.5'] + batch)
    op = ctl.operator
    torch.manual_seed(0)
    w0, w1 = (torch.randn(1, 1, 12, requires_grad=True) for _ in range(2))
    z = torch.randn(1, 1, 6)
    op.reset()
    u0 = op(w0, z)
    u1 = op(w1, z)
    blk = lambda g, j: g[..., 4 * j:4 * j + 4].abs().sum().item() if g is not None else 0.0
    for i, (own, seen_j, unseen) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
        g00, = torch.autograd.grad(u0[..., 2 * i:2 * i + 2].sum(), (w0,), retain_graph=True)
        assert blk(g00, own) > 0 and blk(g00, seen_j) == 0 and blk(g00, unseen) == 0
        g10, g11 = torch.autograd.grad(u1[..., 2 * i:2 * i + 2].sum(), (w0, w1), retain_graph=True)
        assert blk(g10, own) > 0 and blk(g10, seen_j) > 0 and blk(g10, unseen) == 0
        assert blk(g11, own) > 0 and blk(g11, seen_j) == 0 and blk(g11, unseen) == 0


@mode_param
def test_append_batch_padding_and_realization_match_the_loop(mode):
    """Leader graph: padded + batched = unpadded loop; and the loop's realized cores = per-step capping."""
    graph = '0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0'
    argv = ['--feedback-mode', mode, '--n-agents', '5', '--formation-shape', 'bowtie', '--info-pattern', 'neighbors',
                     '--adjacency', graph, '--patches', '--ren-edge-weight', '0.05']
    args, plant, loop = make(argv + ['--no-batch-agents'])
    _, _, batch = make(argv + ['--batch-agents', '--pad-agent-inputs'])
    assert batch.operator.batch_agents and batch.operator.input_padding is not None
    sd_loop, sd_pad = loop.state_dict(), batch.state_dict()
    for k, v in sd_pad.items():
        if v.shape == sd_loop[k].shape:
            sd_pad[k] = sd_loop[k]
            continue
        weight = sd_pad[k.removesuffix('_v') + '_orig'] if k.endswith('_v') else v
        keep = ~(weight == 0).all(0)
        assert int(keep.sum()) == sd_loop[k].shape[-1], k
        sd_pad[k] = torch.zeros_like(v)
        sd_pad[k][..., keep] = sd_loop[k]
    batch.load_state_dict(sd_pad)
    loop.eval(); batch.eval()
    train, _ = get_data(args, make_dataset(args))
    with torch.no_grad():
        _, _, ul = plant.rollout(loop, train[:3])
        _, _, ub = plant.rollout(batch, train[:3])
    # float rounding (vmapped vs looped matmuls) grows through the feedback over the horizon (~1e-8 at the first steps)
    assert ul.abs().max() > 0 and (ub - ul).abs().max() <= 1e-3 * ul.abs().max()
    # direct: the agents' own modules (DeepSSM caps at every step with the override)
    op = loop.operator
    with torch.no_grad():
        realized_forward, realized_reset = op.forward, op.reset

        def direct_reset():
            realized_reset()
            for a in op.agents:
                a.reset()

        def direct_forward(w, z=None, x=None, u_prev=None):
            w3 = as_bt(w)
            ctx = op.contexts(as_bt(z), x, u_prev)
            if op._fb is None:
                op._fb = w3.new_zeros(w3.shape[0], op.n_agents, op.fb_dim)
            fb = torch.einsum("ij,bjk->bik", op._W, op._fb / op.feedback_scale[:, None])
            w_ag = w3[..., op.w_idx]
            out = torch.stack([a(torch.cat([w_ag[:, :, i], fb[:, None, i]], -1), ctx[i], gamma=op._overrides[i])[:, 0]
                               for i, a in enumerate(op.agents)], 1)
            op._fb = out[..., 2:]
            return op.scatter_u(w3, out[..., :2].reshape(w3.shape[0], 1, -1))
        op.forward, op.reset = direct_forward, direct_reset
        _, _, ud = plant.rollout(loop, train[:3])
        op.forward, op.reset = realized_forward, realized_reset
    torch.testing.assert_close(ud, ul, rtol=1e-4, atol=1e-6)
