import argparse, math
import pytest
import torch

from Distributed.utils.training import GradClipper, SpikeDiagnoser, WeightEMA, lr_factor, term_grad_norms


def _pair(d):
    """Two robots at distance d along x: states of shape (1, 1, 8)."""
    xs = torch.zeros(1, 1, 8)
    xs[..., 4] = d
    return xs


def test_hinge_collision_matches_inverse_at_min_dist_and_is_bounded():
    from Distributed.loss_functions.robots_loss import RobotsLoss
    kw = dict(xbar=None, Q=torch.eye(4), Qs=torch.eye(4), n_agents=2, alpha_col=1.0, min_dist=0.5, alpha_obst=None)
    inverse, hinge = RobotsLoss(**kw), RobotsLoss(**kw, collision_loss='hinge')
    ca = lambda loss, d: loss.f_loss_ca(_pair(d).unsqueeze(-1)).item()
    assert ca(hinge, 0.5) == pytest.approx(ca(inverse, 0.5), rel=1e-2)        # 1 / min_dist^2 (inverse: + 1e-3 in d^2)
    assert ca(hinge, 0.7) == pytest.approx(0.0, abs=1e-4) and ca(hinge, 1.0) == 0.0
    assert ca(hinge, 0.0) == pytest.approx((0.7 / 0.2) ** 2 / 0.25, rel=1e-2)
    # gradient bounded near contact, where the inverse one explodes
    x = _pair(0.01).unsqueeze(-1).requires_grad_()
    hinge.f_loss_ca(x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().max() < 100
    with pytest.raises(ValueError):
        RobotsLoss(**kw, collision_loss='nope')


def test_cosine_lr_factor_warmup_and_decay():
    args = argparse.Namespace(lr_schedule='cosine', lr_warmup=10, lr_final_ratio=0.01, epochs=110)
    assert lr_factor(0, args) == pytest.approx(0.1)
    assert lr_factor(9, args) == pytest.approx(1.0)
    assert lr_factor(10, args) == pytest.approx(1.0)
    assert lr_factor(60, args) == pytest.approx(0.505)
    assert lr_factor(110, args) == pytest.approx(0.01)
    assert lr_factor(500, args) == pytest.approx(0.01)
    assert lr_factor(50, argparse.Namespace(lr_schedule='constant')) == 1.0


def test_grad_clipper_relative_threshold_and_nonfinite_skip():
    p = torch.nn.Parameter(torch.zeros(1))
    clip = GradClipper(rel=5.0, window=100, min_history=3)
    for _ in range(3):
        p.grad = torch.ones(1)
        assert clip([p])
    p.grad = torch.full((1,), 100.0)                  # spike: clipped to 5 x median
    assert clip([p]) and p.grad.item() == pytest.approx(5.0, rel=1e-4)
    assert clip.n_clipped == 1 and max(clip.history) == 1.0     # the clipped norm stays out of the median
    p.grad = torch.full((1,), math.nan)
    assert not clip([p]) and clip.n_skipped == 1
    assert 'clipped 1/4 steps' in clip.summary() and '1 non-finite' in clip.summary()


def test_grad_clipper_threshold_does_not_ratchet_in_a_burst():
    p = torch.nn.Parameter(torch.zeros(1))
    clip = GradClipper(rel=5.0, window=100, min_history=3)
    for _ in range(10):
        p.grad = torch.ones(1)
        clip([p])
    for _ in range(300):                              # long burst: counting the clipped steps would raise the bound
        p.grad = torch.full((1,), 1e6)
        clip([p])
    assert clip.threshold(clip.median()) == pytest.approx(5.0)
    assert p.grad.item() == pytest.approx(5.0, rel=1e-4)


def test_spike_diagnoser_triggers_on_growing_spikes_only():
    p = torch.nn.Parameter(torch.zeros(1))
    clip, diag = GradClipper(min_history=3), SpikeDiagnoser(20.0)
    fired = []
    for g in [1, 1, 1, 1, 30, 40, 100, 150, 1000]:
        p.grad = torch.full((1,), float(g))
        clip([p])
        fired.append(diag.triggered(clip))
    # 1 (no median yet / below 20x), 30 > 20x, 40 < 2 x 30, 100, 150 < 2 x 100, 1000
    assert fired == [False, False, False, False, True, False, True, False, True]
    assert 'norm 1e+03' in diag.report(0, 8, clip, 1.0, {'a': 1.0})
    diag.reset_stats()                                # new log period: a spike below the last diagnosed one fires again
    p.grad = torch.full((1,), 900.0)
    clip([p])
    assert diag.triggered(clip)


def test_term_grad_norms():
    w = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    unused = torch.nn.Parameter(torch.ones(1))
    terms = lambda: {'lin': (w * torch.tensor([1.0, 0.0])).sum().reshape(1, 1, 1).expand(2, 1, 1),
                     'quad': (w ** 2).sum().reshape(1, 1, 1), 'const': torch.ones(2, 1, 1)}
    norms = term_grad_norms(terms, [w, unused])
    assert norms['lin'] == pytest.approx(1.0) and norms['quad'] == pytest.approx(10.0) and norms['const'] == 0.0


def test_weight_ema_tracks_and_swaps():
    m = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.0)
    ema = WeightEMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)                                     # first decay min(0.5, 1/10) = 0.1
    assert ema.state['weight'].item() == pytest.approx(0.9)
    current = ema.swap(m)
    assert m.weight.item() == pytest.approx(0.9)
    m.load_state_dict(current)
    assert m.weight.item() == 1.0


def test_formation_defaults_enable_stabilisers():
    from Distributed.experiments.robots_formation.arg_parser import argument_parser
    args = argument_parser([])
    assert (args.col_loss, args.lr_schedule, args.grad_clip_rel, args.ema_decay) == ('hinge', 'cosine', 5.0, 0.995)
