"""Formation experiment: n-gon formation, formation dataset, formation loss and argument parsers."""

import math

import pytest
import torch

from Centralized.experiments.robots.arg_parser import argument_parser as robots_argument_parser
from Centralized.experiments.robots_formation.arg_parser import FORMATION_DEFAULTS, argument_parser
from Centralized.loss_functions import RobotsFormationLoss
from Centralized.plants import RobotsFormationDataset
from Centralized.plants.robots.formation_dataset import (desired_distances, formation_offsets,
                                                          formation_state)


def positions(state):
    """(..., 4n) state -> (..., n, 2) positions."""
    return state.reshape(*state.shape[:-1], -1, 4)[..., :2]


def make_loss(n=3, side=1.5, alpha_form=1.0, deadband=0.0, **kwargs):
    return RobotsFormationLoss(
        Q=torch.eye(2 * n), Qs=torch.eye(2 * n), alpha_u=0.0, xbar=None, n_agents=n,
        alpha_col=None, min_dist=None, alpha_obst=None,
        formation_dist=desired_distances(formation_offsets(n, side)),
        alpha_form=alpha_form, formation_deadband=deadband, **kwargs,
    )


def rollout_of(states):
    """(T, 4n) states -> xs (1, T, 4n) and zero us, es."""
    xs = states.unsqueeze(0)
    n = xs.shape[-1] // 4
    return xs, torch.zeros(1, xs.shape[1], 2 * n), torch.zeros(1, xs.shape[1], 2 * n)


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_ngon_has_equal_sides_and_zero_centroid(n):
    off = formation_offsets(n, side=1.5)
    sides = torch.norm(off - off.roll(-1, dims=0), dim=-1)
    assert torch.allclose(sides, torch.full((n,), 1.5), atol=1e-5)
    assert torch.allclose(off.mean(0), torch.zeros(2), atol=1e-6)


def test_formation_state_is_a_rigid_motion():
    off = formation_offsets(3, 1.5)
    state = formation_state((1.0, -2.0), 0.7, off)
    assert state.shape == (12,)
    assert torch.all(state.view(3, 4)[:, 2:] == 0)                           # at rest
    p = positions(state)
    assert torch.allclose(torch.cdist(p, p), desired_distances(off), atol=1e-5)
    assert torch.allclose(p.mean(0), torch.tensor([1.0, -2.0]), atol=1e-5)


def test_dataset_targets_are_rotated_copies_of_the_formation():
    ds = RobotsFormationDataset(random_seed=0, horizon=20, n_agents=3, side=1.5, std_ini=0.0)
    data = ds._generate_data(16)
    assert data.shape == (16, 20, 24)
    # start in formation
    p0 = positions(data[:, 0, :12])
    assert torch.allclose(torch.cdist(p0, p0), ds.distances.expand(16, 3, 3), atol=1e-5)
    # constant targets, in formation, with various orientations
    xbar = data[:, 1:, 12:]
    assert torch.all(xbar == xbar[:, :1])
    pt = positions(xbar[:, 0])
    assert torch.allclose(torch.cdist(pt, pt), ds.distances.expand(16, 3, 3), atol=1e-5)
    heading = pt[:, 0] - pt.mean(1)                                           # agent 0 relative to the centroid
    angles = torch.atan2(heading[:, 1], heading[:, 0])
    assert angles.std() > 0.5
    assert torch.all(data[:, 0, 12:] == 0) and torch.all(data[:, 1:, :12] == 0)


def test_dataset_without_rotation_keeps_the_start_orientation():
    ds = RobotsFormationDataset(random_seed=0, horizon=5, n_agents=3, target_rotation=False)
    pt = positions(ds._generate_data(8)[:, 1, 12:])
    assert torch.allclose(pt - pt.mean(1, keepdim=True), ds.offsets.expand(8, 3, 2), atol=1e-5)


def test_formation_loss_is_zero_in_formation_and_rotation_invariant():
    off = formation_offsets(3, 1.5)
    states = torch.stack([formation_state((0.3 * t, 1.0), 0.2 * t, off) for t in range(10)])
    loss = make_loss()
    assert loss.loss_terms(*rollout_of(states))['formation'].item() == pytest.approx(0.0, abs=1e-8)


def test_formation_loss_deadband_and_weight():
    off = formation_offsets(3, 1.5)
    stretched = formation_state((0.0, 0.0), 0.0, off * 1.05)                # every side 1.575: deviation 0.075
    xs, us, es = rollout_of(stretched.repeat(4, 1))
    assert make_loss(deadband=0.1).loss_terms(xs, us, es)['formation'].item() == 0.0
    term = make_loss(alpha_form=2.0, deadband=0.05).loss_terms(xs, us, es)['formation'].item()
    assert term == pytest.approx(2.0 * 3 * 0.025 ** 2, rel=1e-3)             # 3 edges, 4 steps averaged over T=4
    assert make_loss(alpha_form=0.0).loss_terms(xs, us, es)['formation'].item() == 0.0
    assert make_loss(alpha_form=None).loss_terms(xs, us, es)['formation'].item() == 0.0


def test_formation_graph_and_start_restrict_the_penalty():
    off = formation_offsets(3, 1.5)
    bad = formation_state((0.0, 0.0), 0.0, off * 2)
    xs, us, es = rollout_of(bad.repeat(4, 1))
    full = make_loss().loss_terms(xs, us, es)['formation'].item()
    line = make_loss(formation_adjacency=torch.tensor([[0., 1, 0], [0, 0, 1], [0, 0, 0]])).loss_terms(xs, us, es)['formation'].item()
    assert line == pytest.approx(full * 2 / 3, rel=1e-5)                     # 2 of the 3 edges (symmetrised)
    late = make_loss(formation_start=2).loss_terms(xs, us, es)['formation'].item()
    assert late == pytest.approx(full / 2, rel=1e-5)


def test_formation_loss_gradient_is_finite_when_agents_coincide():
    xs = torch.zeros(1, 3, 12, requires_grad=True)
    loss = make_loss()
    loss.forward(xs, torch.zeros(1, 3, 6), torch.zeros(1, 3, 6)).sum().backward()
    assert torch.isfinite(xs.grad).all()


def test_formation_error_metric():
    off = formation_offsets(3, 1.5)
    states = torch.stack([formation_state((0, 0), 0, off * s) for s in (1.0, 2.0)])
    err_mean, err_max = make_loss().formation_error(states.unsqueeze(0))
    assert err_mean.shape == err_max.shape == (1, 2)
    assert torch.allclose(err_mean[0], torch.tensor([0.0, 1.5]), atol=1e-5)
    assert torch.allclose(err_max[0], torch.tensor([0.0, 1.5]), atol=1e-5)


def test_parsers():
    robots = robots_argument_parser([])
    assert robots.n_agents == 2 and not hasattr(robots, 'alpha_form')
    args = argument_parser([])
    for k, v in FORMATION_DEFAULTS.items():
        assert getattr(args, k) == v, k
    assert args.alpha_form == 500 and args.formation_side == 2.0
    assert argument_parser(['--n-agents', '4', '--alpha-form', '0']).n_agents == 4


def test_centralized_operator_sees_every_robot():
    """One M for the three robots: w of all agents in, u of all agents out."""
    from Centralized.controllers import build_controller
    from Centralized.plants import RobotsSystem
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    ctl = build_controller(argument_parser(['--ssm-layers', '1', '--ssm-d-model', '8', '--ssm-d-state', '8']),
                           plant=plant, z_dim=plant.in_dim)
    data = RobotsFormationDataset(random_seed=0, horizon=6, n_agents=3)._generate_data(2)
    with torch.no_grad():
        x, e, u = plant.rollout(ctl, data)
    assert x.shape == (2, 6, 12) and u.shape == (2, 6, 6) and e.shape == (2, 6, 6)
    assert torch.isfinite(x).all()
