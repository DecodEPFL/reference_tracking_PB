"""--pair-features: the centralized controller also receives the relative positions of every pair of
robots, measured from x_t; off by default (unchanged controller), state signals kept out of the
SSM input port."""

import torch

from arg_parser import argument_parser
from Centralized.controllers import build_controller, PairOffsets
from Centralized.controllers.ssm_operators import InputContextMask
from Centralized.plants import RobotsFormationDataset, RobotsSystem

SMALL = ['--ssm-layers', '1', '--ssm-d-model', '8', '--ssm-d-state', '8', '--ctx-d-features', '8',
         '--mb-hidden', '16', '--feat-dim', '4', '--n-agents', '3', '--ssm-param', 'tv',
         '--ctx-modes', 'mixer,gate,select,input']


def make(extra, seed=0):
    torch.manual_seed(seed)
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    return plant, build_controller(argument_parser(SMALL + extra), plant=plant, z_dim=plant.in_dim)


def test_offsets_of_every_pair():
    f = PairOffsets([[0, 1], [4, 5], [8, 9]])
    x = torch.zeros(2, 1, 12)
    x[..., [0, 1, 4, 5, 8, 9]] = torch.tensor([0., 0., 2., 1., -1., 3.])
    assert f.pairs == [(0, 1), (0, 2), (1, 2)] and f.out_dim == 6
    assert torch.equal(f(x)[0, 0], torch.tensor([2., 1., -1., 3., -3., 2.]))


def test_off_by_default_and_same_checkpoint_layout():
    assert argument_parser([]).pair_features is False
    _, off = make([])
    _, on = make(['--pair-features'])
    assert off.context_features is None and on.context_features is not None
    assert not any('context_features' in k for k in on.state_dict())


def test_features_reach_the_operator_but_not_the_input_port():
    plant, ctl = make(['--pair-features'])
    ctl.eval()
    filt = ctl.operator.mp.core.context_filter
    assert isinstance(filt, InputContextMask)
    assert filt.mask.tolist() == [1.0] * 6 + [0.0] * 6          # targets drive the input port, offsets do not
    w, targets, offsets = 0.1 * torch.randn(3, 1, 12), torch.randn(3, 1, 6), torch.randn(3, 1, 6)
    with torch.no_grad():
        ctl.operator.reset()
        u1 = ctl.operator(w, torch.cat([targets, offsets], -1))
        ctl.operator.reset()
        u2 = ctl.operator(w, torch.cat([targets, offsets + 1.0], -1))
    assert (u1 - u2).abs().max() > 1e-4


def test_closed_loop_rollout_with_pair_features():
    plant, ctl = make(['--pair-features'])
    data = RobotsFormationDataset(random_seed=0, horizon=8, n_agents=3)._generate_data(2)
    x, e, u = plant.rollout(ctl, data, train=True)
    assert u.shape == (2, 8, 6) and torch.isfinite(x).all()
    u.pow(2).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ctl.parameters() if p.requires_grad)
