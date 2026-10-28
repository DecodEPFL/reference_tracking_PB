"""Obstacle penalties of RobotsLoss: radius cut-off of the Gaussian and the barrier."""

import torch

from Centralized.loss_functions.robots_loss import RobotsLoss, normpdf, obstacle_barrier

MU = torch.tensor([[4.5, 2.0]])
COV = torch.tensor([[0.44, 0.44]])
R = 1.3


def at_distance(d):
    """Two agents, both at distance d straight above MU: q shape (1, len(d), 4)."""
    d = torch.as_tensor(d, dtype=torch.float32)
    agent = torch.stack([torch.full_like(d, 4.5), 2.0 + d], -1)
    return torch.cat([agent, agent], -1).unsqueeze(0)


def test_barrier_is_zero_outside_and_grows_inside():
    vals = obstacle_barrier(at_distance([0.1, 0.3, 0.65, 1.0, R, 1.5, 2.05]), MU, R)[0]
    assert torch.all(vals[4:] == 0)
    assert torch.all(vals[:4] > 0)
    assert torch.all(vals[:-1] >= vals[1:])            # decreasing with distance
    assert vals[0] > 50 * vals[2]                       # much steeper than the Gaussian bump


def test_barrier_is_c1_and_finite_at_the_centre():
    q = at_distance([R - 1e-3]).requires_grad_()
    grad, = torch.autograd.grad(obstacle_barrier(q, MU, R).sum(), q)
    assert grad.abs().max() < 1e-4                      # zero slope at the edge, no kink
    centre = obstacle_barrier(at_distance([0.0]), MU, R)
    assert torch.isfinite(centre).all() and centre.item() > 1000


def test_gaussian_cutoff_is_continuous_and_keeps_inner_gradient():
    assert normpdf(at_distance([R, 1.5, 2.05]), MU, COV, radius=R).abs().max() < 1e-7
    q1 = at_distance([0.7]).requires_grad_()
    q2 = at_distance([0.7]).requires_grad_()
    g_full, = torch.autograd.grad(normpdf(q1, MU, COV).sum(), q1)
    g_cut, = torch.autograd.grad(normpdf(q2, MU, COV, radius=R).sum(), q2)
    torch.testing.assert_close(g_full, g_cut)


def test_robots_loss_dispatches_to_the_barrier():
    loss = RobotsLoss(xbar=None, Q=torch.eye(4), Qs=torch.eye(4), alpha_obst=1.0,
                      obstacle_centers=[MU], obstacle_covs=[COV], obstacle_radius=R,
                      obstacle_loss='barrier', min_dist=None)
    x = torch.zeros(1, 2, 8, 1)
    x[0, :, [0, 4], 0], x[0, :, [1, 5], 0] = 4.5, 2.0 + 0.65   # both agents at d = 0.65, 2 steps
    expected = 2 * (R / (0.65**2 + 0.05**2) ** 0.5 - 1) ** 2   # two agents, time-averaged
    torch.testing.assert_close(loss.f_loss_obst(x).squeeze(), torch.tensor(expected), atol=1e-5, rtol=1e-5)


def test_alpha_final_replaces_q_after_final_start():
    T, final_start = 8, 5
    es = torch.ones(1, T, 4)
    xs, us = torch.zeros(1, T, 8), torch.zeros(1, T, 4)
    base = dict(xbar=None, Q=2 * torch.eye(4), Qs=torch.eye(4), alpha_u=0.0, min_dist=None)
    plain = RobotsLoss(**base).loss_terms(xs, us, es)
    assert 'final' not in plain
    assert plain['tracking'].item() == 4 * 2 * T / T                       # e^T Q e = 4 * 2 per step
    terms = RobotsLoss(**base, Q_final=10 * torch.eye(4), final_start=final_start).loss_terms(xs, us, es)
    assert terms['tracking'].item() == 4 * 2 * (final_start + 1) / T
    assert terms['final'].item() == 4 * 10 * (T - final_start - 1) / T
    short = RobotsLoss(**base, Q_final=10 * torch.eye(4), final_start=T).loss_terms(xs, us, es)
    assert short['tracking'].item() == plain['tracking'].item() and short['final'].item() == 0   # horizon ends first
