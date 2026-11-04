"""Sticky friction patches (--patches): position-dependent linear drag, one layout per rollout, measured by
the controller (exact w reconstruction) and fed to every M^i as the local drag scale in its context, after
the exogenous prefix that drives the SSM input port."""

import os, sys

import pytest
import torch

from Distributed.controllers import OperatorBase, PBController, build_controller
from Distributed.controllers.ssm_operators import InputContextMask
from Distributed.plants import RobotsSystem
from Distributed.plants.robots import append_patches, sample_friction_patches

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'robots_formation'))
from Distributed.experiments.robots_formation.arg_parser import argument_parser   # noqa: E402
from Distributed.experiments.robots_formation.run import (formation_scenarios, get_data, make_dataset,  # noqa: E402
                                                           make_plant, verification_data, verification_patches)

SMALL = ['--ssm-layers', '2', '--ssm-d-model', '8', '--ssm-d-state', '8', '--ctx-d-features', '8',
         '--mb-hidden', '16', '--ctx-head-hidden', '16', '--feat-dim', '4', '--horizon', '20',
         '--num-rollouts', '6', '--patches', '--no-batch-agents']


def test_friction_scale_matches_the_formula():
    plant = RobotsSystem(linear_plant=False, n_agents=2, n_patches=2)
    patches = torch.tensor([[[0.0, 0.0, 4.0, 1.0], [3.0, 1.0, 2.0, 0.5]]])        # (1, 2, 4)
    x = torch.tensor([[[0.0, 0.0, 0, 0, 3.0, 1.5, 0, 0]]])                       # agent 1 at (0, 0), 2 at (3, 1.5)
    s = plant.friction_scale(x, patches)
    g = lambda d2, w, sg: w * torch.exp(torch.tensor(-d2 / (2 * sg ** 2)))
    expected = torch.stack([1 + g(0.0, 4.0, 1.0) + g(10.0, 2.0, 0.5), 1 + g(9 + 2.25, 4.0, 1.0) + g(0.25, 2.0, 0.5)])
    torch.testing.assert_close(s.view(-1), expected)
    far = torch.tensor([[[50.0, 50.0, 0, 0, -50.0, 50.0, 0, 0]]])
    torch.testing.assert_close(plant.friction_scale(far, patches), torch.ones(1, 1, 2))


def test_patches_scale_only_the_linear_drag():
    plain = RobotsSystem(linear_plant=False, n_agents=1)
    sticky = RobotsSystem(linear_plant=False, n_agents=1, n_patches=1)
    x, v, u, xbar = torch.tensor([[[0.0, 0.0, 1.0, -0.5]]]), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)
    patch = torch.tensor([[[0.0, 0.0, 3.0, 1.0]]])                                  # s = 4 at the robot
    f0, _ = plain.noiseless_forward(0, x, v, u, xbar)
    f1, _ = sticky.noiseless_forward(0, x, v, u, xbar, patches=patch)
    # positions untouched; velocities: the extra drag h b/m (s - 1) = 0.05 * 3 taken implicitly
    torch.testing.assert_close(f1[..., :2], f0[..., :2])
    torch.testing.assert_close(f1[..., 2:], f0[..., 2:] / (1 + 0.05 * 3.0))


class RecordingOperator(OperatorBase):
    def __init__(self, u_dim):
        super().__init__()
        self.u_dim, self.ws, self.zs = u_dim, [], []

    def reset(self):
        self.ws, self.zs = [], []

    def forward(self, w, z=None):
        self.ws.append(w)
        self.zs.append(z)
        return 0.1 * torch.tanh(w[..., : self.u_dim] + z[..., : self.u_dim])


def test_measured_drag_gives_exact_reconstruction_and_is_in_the_context():
    args = argument_parser(SMALL)
    dataset = make_dataset(args)
    train, _ = get_data(args, dataset)
    plant = make_plant(args)
    sd = plant.state_dim
    assert train.shape[-1] == 2 * sd + 4 * args.patch_n
    # one layout per rollout, constant over time, in the ranges of --patch-*
    p = plant.split_patches(train)
    assert not torch.allclose(p[0], p[1])
    torch.testing.assert_close(train[:, -1, 2 * sd:], train[:, 0, 2 * sd:])
    assert p[..., 0].min() >= args.wall_x[0] and p[..., 0].max() <= args.wall_x[1]
    assert p[..., 1].min() >= args.patch_y[0] and p[..., 1].max() <= args.patch_y[1]
    assert p[..., 3].min() >= 0.25 and p[..., 3].max() <= 1.5
    assert p[..., 2].min() >= 2.0 and p[..., 2].max() <= 20.0

    op = RecordingOperator(plant.in_dim)
    ctl = PBController(plant=plant, operator=op, u_dim=plant.in_dim, output_scale=20.0)
    with torch.no_grad():
        x_log, _, _ = plant.rollout(ctl, train)
    torch.testing.assert_close(torch.cat(op.ws, 1), train[:, :, :sd], atol=1e-5, rtol=0)
    z = torch.cat(op.zs, 1)
    assert z.shape[-1] == plant.in_dim + plant.n_agents
    torch.testing.assert_close(z[..., plant.in_dim:], torch.log(plant.friction_scale(x_log, p)))   # log s in z
    torch.testing.assert_close(plant.drag_log, plant.friction_scale(x_log, p))                     # s in the log
    assert z[..., plant.in_dim:].min() >= 0 and z[..., plant.in_dim:].max() < 4

    # the patches change the closed loop
    plain = RobotsSystem(linear_plant=False, n_agents=plant.n_agents)
    with torch.no_grad():
        x_plain, _, _ = plain.rollout(PBController(plant=plain, operator=RecordingOperator(plain.in_dim),
                                                   u_dim=plain.in_dim, output_scale=20.0), train[:, :, :2 * sd])
    assert (x_plain - x_log).abs().max() > 1e-2


@pytest.mark.parametrize('pattern', [['--info-pattern', 'local'], ['--info-pattern', 'neighbors'],
                                     ['--info-pattern', 'relay', '--adjacency', 'cycle'],
                                     ['--info-pattern', 'centralized'], ['--info-pattern', 'full'],
                                     ['--info-pattern', 'neighbors', '--share-weights']])
def test_drag_reaches_the_context_after_the_input_prefix(pattern):
    torch.manual_seed(0)
    args = argument_parser(SMALL + pattern)
    plant = make_plant(args)
    ctl = build_controller(args, plant)
    op = ctl.operator
    n = plant.n_agents
    seen = []
    for o in op.agent_ops:
        o.register_forward_pre_hook(lambda m, inp: seen.append(inp[1]))
    dataset = make_dataset(args)
    train, _ = get_data(args, dataset)
    with torch.no_grad():
        plant.rollout(ctl, train[:2])
    s_first = torch.log(plant.drag_log[:, 0])                                     # (2, n) log drag at t = 0
    full = pattern[1] in ('centralized', 'full')
    for i in range(1 if op.share_weights else n):
        e = op.drag_pos[i]
        z = seen[i]                                                               # first step of agent i (or all, folded)
        got = z[..., e:e + (n if full else 1)].reshape(-1, n if full else 1)
        if op.share_weights:
            torch.testing.assert_close(got.view(2, n), s_first)
        elif full:
            torch.testing.assert_close(got, s_first)
        else:
            torch.testing.assert_close(got.view(-1), s_first[:, i])
        # the input port of the contextual SSM only sees the prefix before the drag
        cf = op.agent_ops[i].mp.core.context_filter
        assert isinstance(cf, InputContextMask) and int(cf.mask.sum()) == e
    # a training step goes through
    x, e_log, u = plant.rollout(ctl, train[:2], train=True)
    (x.pow(2).mean() + u.pow(2).mean()).backward()


def test_verification_data_has_one_layout_per_scenario_flipped_alike():
    args = argument_parser(SMALL + ['--start-flip-frac', '0.5'])
    dataset = make_dataset(args)
    scenarios = formation_scenarios(dataset)
    names, data = verification_data(dataset, scenarios, args.horizon, patches=verification_patches(args, len(scenarios)))
    k, sd = len(scenarios), 4 * args.n_agents
    assert data.shape[-1] == 2 * sd + 4 * args.patch_n and len(names) == 2 * k
    torch.testing.assert_close(data[k:, 0, 2 * sd:], data[:k, 0, 2 * sd:])        # '_fs' on the same patches
    off = argument_parser(SMALL[:-2])
    assert verification_patches(off, 3) is None and make_plant(off).drag_idx is None


@pytest.mark.parametrize('extra', [[], ['--relay-inputs'], ['--relay-max-hops', '1']])
def test_relay_forwards_the_measured_drag_like_the_positions(extra):
    """Under relay, agent i gets [z^i, targets of its sources, log s^i_t, log s^j_{t-d} per source, ...]."""
    torch.manual_seed(0)
    args = argument_parser(SMALL + ['--info-pattern', 'relay', '--adjacency', 'cycle', '--no-share-weights'] + extra)
    plant = make_plant(args)
    op = build_controller(args, plant).operator
    rf, n = op.neighbor_features, plant.n_agents
    assert rf.sends_drag and rf.drag_dims == [len(src) for src in rf.sources]
    seen = [[] for _ in range(n)]
    for i, o in enumerate(op.agent_ops):
        o.register_forward_pre_hook(lambda m, inp, i=i: seen[i].append(inp[1]))
    train, _ = get_data(args, make_dataset(args))
    with torch.no_grad():
        plant.rollout(build_controller.__globals__['PBController'](plant, op, plant.in_dim, float(args.output_amplification)),
                      train[:2])
    log_s = torch.log(plant.drag_log)                                                 # (2, T, n)
    for i in range(n):
        z = torch.cat(seen[i], 1)                                                     # (2, T, dz_i)
        e = op.drag_pos[i]
        assert e == 2 * (1 + len(rf.sources[i]))                                      # own + relayed targets
        torch.testing.assert_close(z[..., e], log_s[..., i])                          # own drag, now
        for k, (j, d) in enumerate(zip(rf.sources[i], rf.delays[i])):
            delayed = torch.cat([log_s[:, :1, j].expand(-1, d), log_s[:, :log_s.shape[1] - d, j]], 1)
            torch.testing.assert_close(z[..., e + 1 + k], delayed)                    # s^j_{t-d}, held before
        cf = op.agent_ops[i].mp.core.context_filter
        assert isinstance(cf, InputContextMask) and int(cf.mask.sum()) == e           # no drag in the input port


def test_padded_relay_with_drag_equals_the_unpadded_operators():
    """Irregular graph (unequal agents, inputs padded to batch them): the drag channels sit where the masks
    expect them, so the padded operators compute what the unpadded ones do."""
    graph = '0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0'
    argv = SMALL[:-1] + ['--n-agents', '5', '--formation-shape', 'bowtie', '--info-pattern', 'relay',
                         '--adjacency', graph, '--relay-inputs', '--relay-inputs-to', 'mixers']

    def make(batch):
        torch.manual_seed(0)
        args = argument_parser(argv + (['--batch-agents', '--pad-agent-inputs'] if batch else ['--no-batch-agents']))
        plant = make_plant(args)
        return args, plant, build_controller(args, plant)
    args, plant, small = make(False)
    _, _, padded = make(True)
    assert padded.operator.input_padding is not None
    sd_small, sd_pad = small.state_dict(), padded.state_dict()
    for k, v in sd_pad.items():
        if v.shape == sd_small[k].shape:
            sd_pad[k] = sd_small[k]
            continue
        weight = sd_pad[k.removesuffix('_v') + '_orig'] if k.endswith('_v') else v
        keep = ~(weight == 0).all(0)
        assert int(keep.sum()) == sd_small[k].shape[-1], k
        sd_pad[k] = torch.zeros_like(v)
        sd_pad[k][..., keep] = sd_small[k]
    padded.load_state_dict(sd_pad)
    padded.operator.agent_batch = None
    train, _ = get_data(args, make_dataset(args))
    with torch.no_grad():
        xs, _, us = plant.rollout(small, train[:3])
        xp, _, up = plant.rollout(padded, train[:3])
    assert (xp - xs).abs().max() <= 1e-4 * xs.abs().max()
    assert (up - us).abs().max() <= 1e-4 * us.abs().max()
