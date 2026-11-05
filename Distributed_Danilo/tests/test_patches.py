"""Sticky friction patches (--patches): position-dependent linear drag, one layout per rollout, measured by
the controller (exact w reconstruction) and fed to every agent as the local drag scale in its context
(REN-MAD: see test_ren_mad.py)."""

import os, sys

import pytest
import torch

from Distributed_Danilo.controllers import OperatorBase, PBController, build_controller
from Distributed_Danilo.plants import RobotsSystem
from Distributed_Danilo.plants.robots import append_patches, sample_friction_patches

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'robots_formation'))
from Distributed_Danilo.experiments.robots_formation.arg_parser import argument_parser   # noqa: E402
from Distributed_Danilo.experiments.robots_formation.run import (formation_scenarios, get_data, make_dataset,  # noqa: E402
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
