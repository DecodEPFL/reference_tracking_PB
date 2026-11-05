"""Formation experiment: n-gon formation, formation dataset, formation loss and argument parsers."""

import math

import pytest
import torch

from Distributed_Danilo.experiments.robots.arg_parser import argument_parser as robots_argument_parser
from Distributed_Danilo.experiments.robots_formation.arg_parser import FORMATION_DEFAULTS, argument_parser, print_args
from Distributed_Danilo.loss_functions import RobotsFormationLoss
from Distributed_Danilo.plants import RobotsFormationDataset
from Distributed_Danilo.plants.robots.formation_dataset import (desired_distances, formation_offsets, isosceles_offsets, bowtie_offsets,
                                                          double_bowtie_offsets,
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


def test_isosceles_formation_and_bounded_target_angle():
    off = isosceles_offsets(4.0, 1.2)
    assert torch.allclose(off.mean(0), torch.zeros(2), atol=1e-6)           # centred on the centroid
    assert torch.isclose(off[0, 1] - off[1, 1], torch.tensor(1.2))         # apex height
    assert torch.allclose(off[2] - off[1], torch.tensor([4.0, 0.0]))      # horizontal base
    ds = RobotsFormationDataset(random_seed=0, horizon=5, n_agents=3, std_ini=0.0, shape='isosceles',
                                base=4.0, height=1.2, target_max_angle=0.2)
    pt = positions(ds._generate_data(64)[:, 1, 12:])
    base = pt[:, 2] - pt[:, 1]
    angle = torch.atan2(base[:, 1], base[:, 0])
    assert angle.abs().max() <= 0.2 + 1e-5 and angle.abs().max() > 0.1
    assert torch.allclose(torch.cdist(pt, pt), ds.distances.expand(64, 3, 3), atol=1e-4)


def test_target_flip_turns_half_of_the_targets():
    ds = RobotsFormationDataset(random_seed=0, horizon=5, n_agents=3, std_ini=0.0, shape='isosceles',
                                base=4.0, height=1.2, target_rotation=False, target_flip=True)
    pt = positions(ds._generate_data(200)[:, 1, 12:])
    flipped = pt[:, 0, 1] < pt[:, 1, 1]                                    # apex below the base
    assert 0.3 < flipped.float().mean() < 0.7
    assert torch.allclose((pt[:, 2] - pt[:, 1])[:, 1], torch.zeros(200), atol=1e-5)   # base along x


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
    assert robots.n_agents == 2 and robots.info_pattern == 'local' and not hasattr(robots, 'alpha_form')
    args = argument_parser([])
    for k, v in FORMATION_DEFAULTS.items():
        assert getattr(args, k) == v, k
    assert args.alpha_form == 500 and args.formation_side == 2.0
    assert argument_parser(['--n-agents', '4', '--alpha-form', '0']).n_agents == 4
    iso = argument_parser(['--formation-shape', 'isosceles', '--no-target-rotation'])
    assert iso.formation_base == 4.0 and iso.formation_height == 1.2 and 'isosceles' in print_args(iso)


def make_orient_loss(alpha_orient=1.0, margin=0.5, **kwargs):
    off = isosceles_offsets(4.0, 1.2)
    return RobotsFormationLoss(
        Q=torch.eye(6), Qs=torch.eye(6), alpha_u=0.0, xbar=None, n_agents=3,
        alpha_col=None, min_dist=None, alpha_obst=None,
        formation_dist=desired_distances(off), alpha_form=1.0,
        formation_offsets=off, alpha_orient=alpha_orient, orient_margin=margin, **kwargs,
    ), off


def test_orientation_is_zero_for_rigid_motions_and_penalises_mirroring():
    loss, off = make_orient_loss()
    A = abs(0.5 * 4.0 * 1.2)
    rigid = torch.stack([formation_state((c, 2 * c), th, off) for c, th in zip([0.0, 1.0, 3.0], [0.0, 2.0, math.pi])])
    xs, us, es = rollout_of(rigid)
    terms = loss.loss_terms(xs, us, es)
    assert terms['orientation'].abs().item() < 1e-6 and terms['formation'].abs().item() < 1e-6
    assert not loss.mirrored(xs).item()
    # mirror image (apex reflected through the base): same distances, opposite orientation
    mirror = off * torch.tensor([1.0, -1.0])
    xs, us, es = rollout_of(torch.stack([formation_state((0.0, 0.0), 0.0, off), formation_state((0.0, 0.0), 0.0, mirror)]))
    terms = loss.loss_terms(xs, us, es)
    assert terms['formation'].abs().item() < 1e-6                         # distances are blind to it
    torch.testing.assert_close(terms['orientation'].squeeze(), torch.tensor((0.5 * A + A) ** 2 / 2))
    assert loss.mirrored(xs).item()


def test_orientation_margin_and_gradient_on_a_collapsed_triangle():
    loss, off = make_orient_loss(margin=0.5)
    flat = off.clone(); flat[0, 1] = flat[1, 1]                              # apex on the base line: area 0
    xs = formation_state((0.0, 0.0), 0.3, flat).view(1, 1, 12).requires_grad_()
    val = loss.f_loss_orientation(xs).squeeze()
    torch.testing.assert_close(val, torch.tensor((0.5 * 2.4) ** 2))
    grad, = torch.autograd.grad(val, xs)
    assert grad.abs().sum() > 0
    assert 'orientation' not in make_loss().loss_terms(*rollout_of(formation_state((0.0, 0.0), 0.0, formation_offsets(3, 1.5)).view(1, 12)))


def test_orientation_needs_a_non_collinear_triple():
    with pytest.raises(ValueError):
        make_loss(n=2, formation_offsets=formation_offsets(2, 1.5), alpha_orient=1.0)


def test_orientation_max_charges_a_brief_inversion_like_a_long_one():
    loss, off = make_orient_loss(alpha_orient=1.0, alpha_orient_max=3.0)
    A = abs(0.5 * 4.0 * 1.2)
    mirror = off * torch.tensor([1.0, -1.0])
    ok, bad = formation_state((0.0, 0.0), 0.0, off), formation_state((0.0, 0.0), 0.0, mirror)
    worst = torch.tensor(3.0 * (0.5 * A + A) ** 2)
    brief = loss.loss_terms(*rollout_of(torch.stack([ok] * 9 + [bad])))
    long = loss.loss_terms(*rollout_of(torch.stack([ok] + [bad] * 9)))
    torch.testing.assert_close(brief['orientation_max'].squeeze(), worst)
    torch.testing.assert_close(long['orientation_max'].squeeze(), worst)
    assert brief['orientation'].item() < long['orientation'].item() / 5    # the averaged term is not
    rigid = loss.loss_terms(*rollout_of(torch.stack([ok, formation_state((1.0, 2.0), math.pi, off)])))
    assert rigid['orientation_max'].abs().item() < 1e-6
    assert 'orientation_max' not in make_orient_loss()[0].loss_terms(*rollout_of(torch.stack([ok, bad])))
    assert argument_parser(['--alpha-orient-max', '50']).alpha_orient_max == 50


def test_large_rotation_oversampling_applies_to_train_and_test_alike():
    def iso(frac, seed=3):
        torch.manual_seed(seed)
        ds = RobotsFormationDataset(random_seed=seed, horizon=5, n_agents=3, std_ini=0.2, shape='isosceles',
                                    base=4.0, height=1.2, large_rot_frac=frac)
        tr, te = ds.get_data(num_train_samples=400, num_test_samples=400)
        return tr, te, torch.randn(2)

    def large(data):
        pt = positions(data[:, 1, 12:])
        b = pt[:, 2] - pt[:, 1]
        return (torch.atan2(b[:, 1], b[:, 0]).abs() >= 2 * math.pi / 3).float().mean().item()

    tr0, te0, r0 = iso(0.0)
    tr1, te1, r1 = iso(1.0)
    tr5, te5, r5 = iso(0.5)
    assert 0.2 < large(tr0) < 0.45 and 0.2 < large(te0) < 0.45                # uniform: about 1/3
    assert large(tr1) == 1.0 and large(te1) == 1.0
    assert 0.55 < large(tr5) < 0.8 and 0.55 < large(te5) < 0.8
    assert not torch.equal(tr5, te5)
    assert torch.equal(r0, r1) and torch.equal(r0, r5)                         # model init unaffected
    with pytest.raises(ValueError):
        RobotsFormationDataset(random_seed=0, horizon=5, large_rot_frac=0.5, target_rotation=False)


def test_wall_loss_on_x():
    from Distributed_Danilo.loss_functions.robots_loss import RobotsLoss
    loss = RobotsLoss(xbar=None, Q=torch.eye(4), Qs=torch.eye(4), alpha_u=0.0, n_agents=2, alpha_col=None,
                      min_dist=None, alpha_obst=None, alpha_wall=2.0, wall_x=(-3.0, 7.0), wall_margin=0.5)
    # agent 0 at x in {0, -2.75 (half the margin), -3 (on the wall), -4 (beyond)}, agent 1 at x = 7.5 (beyond the right wall)
    xs = torch.zeros(1, 4, 8)
    xs[0, :, 0] = torch.tensor([0.0, -2.75, -3.0, -4.0])
    xs[0, :, 4] = 7.5
    pen = loss.wall_penalty(xs.unsqueeze(-1))[0]
    torch.testing.assert_close(pen[:, 0], torch.tensor([0.0, 0.25, 1.0, 9.0]))
    torch.testing.assert_close(pen[:, 1], torch.full((4,), 4.0))
    terms = loss.loss_terms(xs, torch.zeros(1, 4, 4), torch.zeros(1, 4, 4))
    torch.testing.assert_close(terms['wall'].squeeze(), 2.0 * pen.sum() / 4)
    xs.requires_grad_()
    g, = torch.autograd.grad(loss.f_loss_wall(xs.unsqueeze(-1)).sum(), xs)
    assert g[0, 3, 0] < 0 and g[0, 0, 0] == 0                                  # pushes back in from beyond the wall
    assert 'wall' not in RobotsLoss(xbar=None, Q=torch.eye(4), Qs=torch.eye(4), n_agents=2, alpha_col=None,
                                    min_dist=None, alpha_obst=None).loss_terms(xs.detach(), torch.zeros(1, 4, 4), torch.zeros(1, 4, 4))
    args = argument_parser(['--alpha-wall', '100', '--wall-x', '-4', '8'])
    assert args.wall_x == [-4.0, 8.0] and 'Arena walls' in print_args(args)
    with pytest.raises(ValueError):
        argument_parser(['--alpha-wall', '1', '--wall-x', '5', '1'])


def test_collision_summary_counts_rollouts_and_pair_steps():
    from Distributed_Danilo.loss_functions.robots_loss import RobotsLoss
    loss = RobotsLoss(xbar=None, Q=torch.eye(4), Qs=torch.eye(4), n_agents=2, alpha_col=1.0, min_dist=0.5, alpha_obst=None)
    xs = torch.zeros(3, 5, 8)
    xs[:, :, 4] = 2.0                                       # agents 2 m apart: no collision
    xs[0, 1:4, 4] = 0.3                                     # rollout 0: 3 steps in collision
    xs[2, 4, 4] = 0.1                                       # rollout 2: 1 step
    assert loss.count_collisions(xs) == 4
    assert loss.collision_summary(xs) == ' -- Collisions: 2/3 rollouts (4 pair-steps)'


def test_bowtie_formation():
    off = bowtie_offsets(4.0, 1.2)
    assert off.shape == (5, 2) and torch.allclose(off.mean(0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(off[4] - off[1], torch.tensor([4.0, 0.0]))       # width along x
    assert torch.allclose(off[1] - off[2], torch.tensor([0.0, 1.2]))       # height along y
    ds = RobotsFormationDataset(random_seed=0, horizon=5, n_agents=5, std_ini=0.0, shape='bowtie', base=4.0, height=1.2)
    pt = positions(ds._generate_data(16)[:, 1, 20:])
    assert torch.allclose(torch.cdist(pt, pt), ds.distances.expand(16, 5, 5), atol=1e-4)
    # the diagonals through the knot are collinear: only the 8 non-degenerate triples are kept
    loss = make_loss(n=5, formation_offsets=ds.offsets, alpha_orient=1.0)
    assert len(loss.orient_triples) == 8
    with pytest.raises(ValueError):
        RobotsFormationDataset(random_seed=0, horizon=5, n_agents=3, shape='bowtie', base=4.0, height=1.2)
    with pytest.raises(ValueError):
        argument_parser(['--formation-shape', 'bowtie', '--n-agents', '3'])
    assert 'bowtie' in print_args(argument_parser(['--formation-shape', 'bowtie', '--n-agents', '5']))


def test_double_bowtie_formation():
    off = double_bowtie_offsets(6.0, 1.2)
    assert off.shape == (8, 2) and torch.allclose(off.mean(0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(off[6] - off[0], torch.tensor([6.0, 0.0]))       # total width along x
    assert torch.allclose(off[3] - off[4], torch.tensor([0.0, 1.2]))       # shared middle column
    # each half is a bowtie of width 3 around its knot, the middle column a corner of both
    assert torch.allclose(off[[0, 1, 4, 3]] - off[2], bowtie_offsets(3.0, 1.2)[1:])
    assert torch.allclose(off[[3, 4, 7, 6]] - off[5], bowtie_offsets(3.0, 1.2)[1:])
    assert torch.pdist(off).min() == pytest.approx(1.2)                  # no closer than the 5-robot bowtie
    ds = RobotsFormationDataset(random_seed=0, horizon=5, n_agents=8, std_ini=0.0, shape='double_bowtie', base=6.0,
                                height=1.2, target_x=(1.0, 3.0), target_y=(7.0, 7.1))
    # orientation: the 8 triangles of each bowtie, not the triples spanning both
    loss = make_loss(n=8, formation_offsets=ds.offsets, orient_groups=ds.orient_groups, alpha_orient=1.0)
    assert len(loss.orient_triples) == 16
    # drawn segments: outer rectangle (6, top and bottom split at the middle) + middle side + 4 per knot
    assert len(ds.plot_edges) == 15 and all(ds.distances[i, j] > 0 for i, j in ds.plot_edges)
    assert len(make_loss(n=8, formation_offsets=ds.offsets, alpha_orient=1.0).orient_triples) == 50
    xt = ds._generate_data(16)[:, 1, 32:]
    pt = positions(xt)
    assert torch.allclose(torch.cdist(pt, pt), ds.distances.expand(16, 8, 8), atol=1e-4)
    centre = pt.mean(1)
    assert ((1.0 <= centre[:, 0] + 1e-5) & (centre[:, 0] <= 3.0 + 1e-5) & (7.0 <= centre[:, 1] + 1e-5)).all()
    with pytest.raises(ValueError):
        RobotsFormationDataset(random_seed=0, horizon=5, n_agents=5, shape='double_bowtie', base=6.0, height=1.2)
    with pytest.raises(ValueError):
        argument_parser(['--formation-shape', 'double_bowtie', '--n-agents', '5'])
    args = argument_parser(['--formation-shape', 'double_bowtie', '--n-agents', '8', '--formation-base', '6',
                            '--target-y', '7', '7.1'])
    assert args.target_y == [7.0, 7.1] and 'double bowtie' in print_args(args)


def _min_start_dist(data, n_agents):
    pos = data[:, 0, :4 * n_agents].view(-1, n_agents, 4)[..., :2]
    return (torch.cdist(pos, pos) + 1e9 * torch.eye(n_agents)).flatten(1).amin(1)


def test_init_min_dist_gives_collision_free_starts():
    kw = dict(random_seed=0, horizon=5, n_agents=5, shape='bowtie', base=4.0, height=1.2, std_ini=0.4, large_rot_frac=0.6)
    loose = RobotsFormationDataset(**kw).get_data(500, 500)
    tight = RobotsFormationDataset(**kw, init_min_dist=0.5).get_data(500, 500)
    assert (_min_start_dist(loose[0], 5) < 0.5).any()
    assert all((_min_start_dist(d, 5) >= 0.5).all() for d in tight)
    with pytest.raises(RuntimeError):
        RobotsFormationDataset(**{**kw, 'std_ini': 0.0}, init_min_dist=5.0).sample_initial_state()


def test_init_min_dist_zero_draws_the_initial_state_as_before():
    ds = RobotsFormationDataset(random_seed=0, horizon=5, n_agents=3, std_ini=0.2)
    x = ds.sample_initial_state(generator=torch.Generator().manual_seed(3))
    assert torch.equal(x, ds.x0 + 0.2 * torch.randn(ds.x0.shape, generator=torch.Generator().manual_seed(3)))


def test_resampled_train_data_follows_its_generator():
    ds = RobotsFormationDataset(random_seed=0, horizon=5, n_agents=3, large_rot_frac=0.5, init_min_dist=1.0)
    draw = lambda seed: ds.sample_train(20, torch.Generator().manual_seed(seed))
    assert torch.equal(draw(1), draw(1)) and not torch.equal(draw(1), draw(2))
    assert draw(1).shape == ds.get_data(20, 20)[0].shape
    assert (_min_start_dist(draw(1), 3) >= 1.0).all()


def test_init_min_dist_and_resample_args():
    from Distributed_Danilo.experiments.robots_formation.arg_parser import saved_run_args
    assert argument_parser(['--min-dist', '0.5']).init_min_dist == 0.5        # default: collision free start
    assert argument_parser(['--no-col-av']).init_min_dist == 0.0
    assert argument_parser(['--init-min-dist', '0.8']).init_min_dist == 0.8
    assert not argument_parser([]).resample_data and argument_parser(['--resample-data']).resample_data
    assert saved_run_args({})['init_min_dist'] == 0.0 and saved_run_args({})['resample_data'] is False   # runs saved before


BOWTIE = dict(random_seed=5, horizon=6, n_agents=5, shape='bowtie', base=4.0, height=1.2, large_rot_frac=0.6, init_min_dist=0.5)


def test_flipped_start_puts_the_left_corners_on_the_right():
    ds = RobotsFormationDataset(**BOWTIE)
    p0, p1 = positions(ds.x0), positions(ds.x0_flipped)
    assert torch.allclose(p1[0], p0[0])                                   # knot: the centre
    assert torch.allclose(p1[1], p0[3]) and torch.allclose(p1[2], p0[4])  # 1: bottom right, 2: top right
    assert torch.allclose(p1[3], p0[1]) and torch.allclose(p1[4], p0[2])


def test_start_flip_turns_some_rollouts_by_pi_and_leaves_the_others():
    (a, at), (b, bt) = (RobotsFormationDataset(**BOWTIE, start_flip_frac=f).get_data(600, 500) for f in (0.0, 0.5))
    ds = RobotsFormationDataset(**BOWTIE)
    m, mt = ds.flipped_starts(b), ds.flipped_starts(bt)
    assert not ds.flipped_starts(a).any() and 0.4 < m.float().mean() < 0.6 and 0.4 < mt.float().mean() < 0.6
    assert torch.equal(a[~m], b[~m]) and torch.equal(at[~mt], bt[~mt])
    assert torch.allclose(ds.flip_starts(a, m), b)
    # the noisy start turned by pi: same pairwise distances (init_min_dist holds), velocities negated
    pa, pb = positions(a[m, 0, :20]), positions(b[m, 0, :20])
    assert torch.allclose(torch.cdist(pa, pa), torch.cdist(pb, pb), atol=1e-5)
    assert torch.allclose(b[m, 0, 2:20:4], -a[m, 0, 2:20:4])
    # the target turned by pi about its own centre: same centre, the same turn from the start
    ta, tb = positions(a[m, 1:, 20:]), positions(b[m, 1:, 20:])
    assert torch.allclose(ta.mean(-2), tb.mean(-2), atol=1e-5)
    assert torch.allclose(tb - tb.mean(-2, keepdim=True), -(ta - ta.mean(-2, keepdim=True)), atol=1e-5)


def test_test_start_flip_frac_flips_only_the_test_data():
    a, at = RobotsFormationDataset(**BOWTIE).get_data(100, 100)
    b, bt = RobotsFormationDataset(**BOWTIE, test_start_flip_frac=1.0).get_data(100, 100)
    ds = RobotsFormationDataset(**BOWTIE)
    assert torch.equal(a, b) and ds.flipped_starts(bt).all()
    with pytest.raises(ValueError):
        RobotsFormationDataset(**BOWTIE, start_flip_frac=1.5)


def test_resampled_train_data_flips_its_starts():
    draw = lambda frac, seed: RobotsFormationDataset(**BOWTIE, start_flip_frac=frac).sample_train(200, torch.Generator().manual_seed(seed))
    ds = RobotsFormationDataset(**BOWTIE)
    m = ds.flipped_starts(draw(0.5, 1))
    assert 0.35 < m.float().mean() < 0.65 and torch.equal(draw(0.5, 1), draw(0.5, 1))
    assert torch.equal(draw(0.0, 1)[~m], draw(0.5, 1)[~m])


def test_start_flip_args():
    from Distributed_Danilo.experiments.robots_formation.arg_parser import saved_run_args
    args = argument_parser([])
    assert args.start_flip_frac == 0.0 and args.test_start_flip_frac == 0.0
    args = argument_parser(['--start-flip-frac', '0.5'])
    assert args.test_start_flip_frac == 0.5                                   # default: as the train data
    assert argument_parser(['--test-start-flip-frac', '0.5']).start_flip_frac == 0.0
    with pytest.raises(ValueError):
        argument_parser(['--test-start-flip-frac', '2'])
    assert 'starts turned by pi' in print_args(args)


def test_verification_data_adds_the_flipped_scenarios():
    from Distributed_Danilo.experiments.robots_formation.run import formation_scenarios, verification_data
    ds = RobotsFormationDataset(**BOWTIE)
    names, data = verification_data(ds, formation_scenarios(ds), horizon=6)
    assert names == ['direct', 'flip', 'offset'] and data.shape == (3, 206, 40)
    ds = RobotsFormationDataset(**BOWTIE, test_start_flip_frac=0.5)
    names, data = verification_data(ds, formation_scenarios(ds), horizon=6)
    assert names == ['direct', 'flip', 'offset', 'direct_fs', 'flip_fs', 'offset_fs']
    assert torch.equal(ds.flipped_starts(data), torch.tensor([False] * 3 + [True] * 3))
    # direct with the flipped start: the target formation turned by pi too, i.e. the flip target
    assert torch.allclose(data[3, 1:], data[1, 1:], atol=1e-5)
