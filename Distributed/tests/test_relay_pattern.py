"""--info-pattern relay: every robot forwards what it receives, so M^i gets the centralized inputs of
every robot j reaching it on --adjacency, delayed by one step per hop: w^j_{t-d} in the SSM input,
z^j_{t-d} and p^j_{t-d} - p^i_{t-d} (and u^j_{t-d} with --relay-inputs) in the context."""

import pytest
import torch

from arg_parser import argument_parser
from Distributed.controllers import build_controller, build_distributed_operator, hop_counts, make_adjacency, RelayFeatures
from Distributed.controllers.factory import build_operator
from Distributed.controllers.ssm_operators import InputContextMask
from Distributed.plants import RobotsDataset, RobotsSystem

SMALL = ['--ssm-layers', '2', '--ssm-d-model', '8', '--ssm-d-state', '8', '--ctx-d-features', '8',
         '--mb-hidden', '16', '--ctx-head-hidden', '16', '--feat-dim', '4', '--horizon', '25',
         '--n-agents', '3', '--ssm-param', 'tv', '--ctx-modes', 'mixer,gate,select,input',
         '--info-pattern', 'relay']
CYCLE = '0,0,1;1,0,0;0,1,0'          # 1 -> 2 -> 3 -> 1: robot 1 receives from 3, 2 from 1, 3 from 2
B, T = 2, 7


def make(extra=(), seed=0):
    torch.manual_seed(seed)
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    return plant, build_distributed_operator(argument_parser(SMALL + list(extra)), plant).eval()


def signals(plant, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, d, generator=g) for d in (plant.state_dim, plant.in_dim, plant.state_dim, plant.in_dim)]


def run(op, w, z, x, u_prev):
    """Causal step-by-step evaluation from a fresh state, as in the closed loop."""
    op.reset()
    with torch.no_grad():
        return torch.cat([op(w[:, t:t + 1], z[:, t:t + 1], x=x[:, t:t + 1], u_prev=u_prev[:, t:t + 1])
                          for t in range(T)], dim=1)


def test_hop_counts():
    assert hop_counts(make_adjacency(CYCLE, 3)) == [[0, 2, 1], [1, 0, 2], [2, 1, 0]]
    assert hop_counts(make_adjacency('cycle', 3)) == [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    assert hop_counts(make_adjacency('line', 4))[0] == [0, 1, 2, 3]
    assert hop_counts(make_adjacency('complete', 3)) == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert hop_counts(make_adjacency('0,1,0;0,0,0;0,1,0', 3)) == [[0, 1, None], [None, 0, None], [None, 1, 0]]


def test_relayed_values_are_the_centralized_signals_delayed_by_the_hops():
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    rf = RelayFeatures(make_adjacency(CYCLE, 3), pos_idx=plant.agent_pos_idx, w_idx=plant.state_idx,
                       ref_idx=plant.ref_idx, u_idx=plant.input_idx)
    assert rf.sources == [[2, 1], [0, 2], [1, 0]] and rf.delays == [[1, 2]] * 3
    assert rf.w_out_dims == [8] * 3 and rf.target_dims == [4] * 3 and rf.out_dims == [12] * 3
    w, z, x, u_prev = signals(plant)
    outs = [rf(x[:, t:t + 1], u_prev[:, t:t + 1], z[:, t:t + 1], w[:, t:t + 1]) for t in range(T)]
    w_ex = [torch.cat([o[0][i] for o in outs], 1) for i in range(3)]
    ctx = [torch.cat([o[1][i] for o in outs], 1) for i in range(3)]

    def delayed(sig, d, hold):     # sig_{t-d}: zeros (w, u) or the value at t = 0 (p, z) before the start
        pad = sig[:, :1].expand(-1, d, -1) if hold else torch.zeros_like(sig[:, :1]).expand(-1, d, -1)
        return torch.cat([pad, sig], 1)[:, :T]

    p = lambda k: x[..., plant.agent_pos_idx[k]]
    for i, src in enumerate(rf.sources):
        ws, zs, offs, us = [], [], [], []
        for j, d in zip(src, rf.delays[i]):
            ws.append(delayed(w[..., plant.state_idx[j]], d, hold=False))
            zs.append(delayed(z[..., plant.ref_idx[j]], d, hold=True))
            offs.append(delayed(p(j) - p(i), d, hold=True))
            us.append(delayed(u_prev[..., plant.input_idx[j]], d - 1, hold=False))   # u^j_{t-d} = u_prev of t-d+1
        torch.testing.assert_close(w_ex[i], torch.cat(ws, -1))
        torch.testing.assert_close(ctx[i], torch.cat(zs + offs + us, -1))


def test_reset_clears_the_history():
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    rf = RelayFeatures(make_adjacency(CYCLE, 3), pos_idx=plant.agent_pos_idx, w_idx=plant.state_idx, ref_idx=plant.ref_idx)
    w, z, x, u_prev = signals(plant)
    first = rf(x[:, :1], u_prev[:, :1], z[:, :1], w[:, :1])
    rf(x[:, 1:2], u_prev[:, 1:2], z[:, 1:2], w[:, 1:2])
    rf.reset()
    again = rf(x[:, :1], u_prev[:, :1], z[:, :1], w[:, :1])
    for a, b in zip(first[0] + first[1], again[0] + again[1]):
        torch.testing.assert_close(a, b)


def test_a_sequence_equals_the_step_by_step_calls():
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    rf = RelayFeatures(make_adjacency(CYCLE, 3), pos_idx=plant.agent_pos_idx, w_idx=plant.state_idx,
                       ref_idx=plant.ref_idx, u_idx=plant.input_idx)
    w, z, x, u_prev = signals(plant)
    whole = rf(x, u_prev, z, w)
    rf.reset()
    steps = [rf(x[:, t:t + 1], u_prev[:, t:t + 1], z[:, t:t + 1], w[:, t:t + 1]) for t in range(T)]
    for k in range(2):
        for i in range(3):
            torch.testing.assert_close(whole[k][i], torch.cat([s[k][i] for s in steps], 1))


@pytest.mark.parametrize('extra', [[], ['--relay-inputs']], ids=['', 'inputs'])
def test_input_sizes_and_input_port(extra):
    _, op = make(['--adjacency', CYCLE] + extra)
    z_dim = 2 + 2 * (2 + 2) + (4 if extra else 0)               # own target, then per source: target, offset (, u)
    for o in op.agent_ops:
        assert o.mp.core.context_filter.mask.tolist() == [1.0] * 6 + [0.0] * (z_dim - 6)   # targets only
        assert isinstance(o.mp.core.context_filter, InputContextMask)
    w_seen = []
    for o in op.agent_ops:
        o.register_forward_pre_hook(lambda m, inp: w_seen.append(inp))
    op.agent_batch = None                                       # hooks only fire on the per-agent loop
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    w, z, x, u_prev = signals(plant)
    run(op, w, z, x, u_prev)
    assert [tuple(a.shape[-1] for a in inp) for inp in w_seen[:3]] == [(12, z_dim)] * 3


@pytest.mark.parametrize('extra', [[], ['--share-weights'], ['--batch-agents']], ids=['loop', 'shared', 'batched'])
def test_information_reaches_each_robot_after_its_hop_delay(extra):
    """Perturbing the w, target, position or input of robot j at step s changes u^i from step
    s + d_ij on, never before. For i = j: w^i and z^i at s; the own position only through the
    offsets, formed at the delayed steps (s + 1 here); the own input never."""
    plant, op = make(['--adjacency', CYCLE, '--relay-inputs'] + extra)
    hops = hop_counts(make_adjacency(CYCLE, 3))
    sigs = signals(plant)
    u = run(op, *sigs)
    s = 2
    for k, idx in enumerate([plant.state_idx, plant.ref_idx, plant.agent_pos_idx, plant.input_idx]):
        for j in range(3):
            pert = [v.clone() for v in sigs]
            pert[[0, 1, 2, 3][k]][:, s, idx[j]] += 1.0           # w, z, x (positions), u_prev
            u2 = run(op, *pert)
            for i in range(3):
                # u_prev at step s holds u_{s-1}: it reaches robot i at s + d - 1, and never robot j itself
                if k == 3:
                    first = None if i == j else s + hops[i][j] - 1
                elif k == 2 and i == j:
                    first = s + 1                                   # nearest source: one hop
                else:
                    first = s + hops[i][j]
                diff = (u[..., plant.input_idx[i]] - u2[..., plant.input_idx[i]]).abs().amax(dim=(0, 2))
                assert torch.all(diff[:first if first is not None else T] == 0), (k, j, i)
                if first is not None:
                    assert diff[first] > 0, (k, j, i)


def test_zero_delay_on_the_complete_graph_gives_the_centralized_inputs():
    """With instant sharing every robot receives all of w, all targets and all its offsets at t."""
    plant, op = make(['--adjacency', 'complete', '--relay-hop-delay', '0'])
    seen = []
    for o in op.agent_ops:
        o.register_forward_pre_hook(lambda m, inp: seen.append(inp))
    op.agent_batch = None
    w, z, x, u_prev = [v[:, :1] for v in signals(plant)]
    op.reset()
    with torch.no_grad():
        op(w, z, x=x, u_prev=u_prev)
    p = x.view(B, 1, 3, 4)[..., :2]
    for i, (w_i, z_i) in enumerate(seen):
        src = [(i + 1) % 3, (i + 2) % 3]
        order = [i] + src
        torch.testing.assert_close(w_i, torch.cat([w[..., plant.state_idx[j]] for j in order], -1))
        torch.testing.assert_close(z_i, torch.cat([z[..., plant.ref_idx[j]] for j in order]
                                                  + [p[:, :, j] - p[:, :, i] for j in src], -1))


def test_unreachable_robots_are_not_heard_and_sizes_may_differ():
    _, op = make(['--adjacency', '0,1,0;0,0,0;0,1,0', '--batch-agents', '--no-pad-agent-inputs'])
    assert op.neighbor_features.sources == [[1], [], [1]]
    assert op.agent_batch is None                               # unequal input sizes: per-agent loop
    _, op = make(['--adjacency', '0,1,0;0,0,0;0,1,0', '--batch-agents'])
    assert op.agent_batch is not None and op.input_padding[3] == [6, 2, 6]   # zero-padded (test_padded_agents.py)
    with pytest.raises(ValueError, match='same number of neighbors'):
        make(['--adjacency', '0,1,0;0,0,0;0,1,0', '--share-weights'])


def test_size_matched_to_the_centralized_controller_with_pair_features():
    plant, op = make(['--adjacency', CYCLE])
    central = build_operator(argument_parser(SMALL), w_dim=12, z_dim=6 + 6, u_dim=6)
    assert op.param_budget == sum(p.numel() for p in central.parameters())
    assert sum(p.numel() for p in op.parameters()) <= op.param_budget


def test_closed_loop_is_causal_and_runs():
    args = argument_parser(SMALL + ['--adjacency', CYCLE, '--relay-inputs'])
    torch.manual_seed(0)
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    ctl = build_controller(args, plant).eval()
    data = RobotsDataset(random_seed=0, horizon=args.horizon, x_bar=torch.zeros(12), n_agents=3)._generate_data(2)
    s, j = 10, 1
    data2 = data.clone()
    data2[:, s, plant.state_idx[j]] += 0.5
    with torch.no_grad():
        _, _, u1 = plant.rollout(ctl, data)
        _, _, u2 = plant.rollout(ctl, data2)
    torch.testing.assert_close(u1[:, :s], u2[:, :s])
    assert not torch.allclose(u1[:, s], u2[:, s])


@pytest.mark.parametrize('extra', [[], ['--batch-agents'], ['--share-weights']], ids=['loop', 'batched', 'shared'])
@pytest.mark.parametrize('to', ['mb', 'mixers', 'all'])
def test_relayed_inputs_only_reach_the_mixers(to, extra):
    """With --relay-inputs-to mb (M_b only) or mixers (M_b and the SSM mixer port) a relayed input
    u^j changes u^i at the single step it is received (memoryless, not the SSM state); with all it
    is remembered by the SSM."""
    plant, op = make(['--adjacency', CYCLE, '--relay-inputs', '--relay-inputs-to', to] + extra)
    if '--batch-agents' in extra:
        assert op.agent_batch is not None                       # the masks run under vmap too
    for o in op.agent_ops:
        core = o.mp.core
        masked = to != 'all'
        assert hasattr(core.gate, 'context_mask') == masked and hasattr(core.core, 'select_context_mask') == masked
        assert hasattr(core.mixer, 'context_mask') == (to == 'mb')
        if masked:
            assert core.gate.context_mask.tolist() == [1.0] * 10 + [0.0] * 4    # targets, offsets | inputs
        if to == 'mb':
            assert core.mixer.context_mask.tolist() == [1.0] * 10 + [0.0] * 4
    hops = hop_counts(make_adjacency(CYCLE, 3))
    sigs = signals(plant)
    u = run(op, *sigs)
    s, j = 2, 0
    pert = [v.clone() for v in sigs]
    pert[3][:, s, plant.input_idx[j]] += 1.0                     # u^j_{s-1}
    u2 = run(op, *pert)
    for i in (1, 2):
        diff = (u[..., plant.input_idx[i]] - u2[..., plant.input_idx[i]]).abs().amax(dim=(0, 2))
        arrive = s + hops[i][j] - 1
        assert torch.all(diff[:arrive] == 0) and diff[arrive] > 0
        assert torch.all(diff[arrive + 1:] == 0) == (to != 'all')


def test_relayed_inputs_are_scaled():
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    w, z, x, u_prev = signals(plant)
    feats = lambda scale: RelayFeatures(make_adjacency(CYCLE, 3), pos_idx=plant.agent_pos_idx, w_idx=plant.state_idx,
                                        ref_idx=plant.ref_idx, u_idx=plant.input_idx, input_scale=scale)(x, u_prev, z, w)
    (w1, c1), (w2, c2) = feats(1.0), feats(0.05)
    for i in range(3):
        torch.testing.assert_close(w2[i], w1[i])
        torch.testing.assert_close(c2[i][..., :8], c1[i][..., :8])                # targets, offsets unchanged
        torch.testing.assert_close(c2[i][..., 8:], 0.05 * c1[i][..., 8:])         # relayed inputs
    _, op = make(['--adjacency', CYCLE, '--relay-inputs'])
    assert op.neighbor_features.input_scale == 0.05                               # default 1/20
    args = argument_parser(SMALL + ['--adjacency', CYCLE, '--relay-inputs'])
    del args.relay_inputs_scale                                                   # older checkpoint
    assert build_distributed_operator(args, plant).neighbor_features.input_scale == 1.0


def test_checkpoints_before_relay_inputs_to_feed_every_port():
    args = argument_parser(SMALL + ['--adjacency', CYCLE, '--relay-inputs'])
    del args.relay_inputs_to
    op = build_distributed_operator(args, RobotsSystem(linear_plant=False, n_agents=3))
    assert not hasattr(op.agent_ops[0].mp.core.gate, 'context_mask')


def test_one_hop_hears_the_in_neighbor_only():
    """--relay-max-hops 1 on the cycle: robot i gets w, z, the offset (and u) of its in-neighbor at
    t-1, and nothing of the other robot, ever."""
    plant, op = make(['--adjacency', CYCLE, '--relay-max-hops', '1', '--relay-inputs'])
    nf = op.neighbor_features
    assert nf.sources == [[2], [0], [1]] and nf.delays == [[1]] * 3
    assert nf.w_out_dims == [4] * 3 and nf.out_dims == [6] * 3            # target, offset, input
    hops = hop_counts(make_adjacency(CYCLE, 3))
    sigs = signals(plant)
    u = run(op, *sigs)
    s = 1
    for k, idx in enumerate([plant.state_idx, plant.ref_idx, plant.agent_pos_idx, plant.input_idx]):
        for j in range(3):
            pert = [v.clone() for v in sigs]
            pert[k][:, s, idx[j]] += 1.0
            u2 = run(op, *pert)
            for i in range(3):
                changed = not torch.allclose(u[..., plant.input_idx[i]], u2[..., plant.input_idx[i]])
                hears = hops[i][j] == 1 or (i == j and k != 3)               # own w, z, position; never own u_prev
                assert changed == hears, (k, j, i)
    _, unlimited = make(['--adjacency', CYCLE])
    assert unlimited.neighbor_features.sources == [[2, 1], [0, 2], [1, 0]]  # default: no limit
    with pytest.raises(ValueError, match='max-hops'):
        argument_parser(SMALL + ['--relay-max-hops', '-1'])
