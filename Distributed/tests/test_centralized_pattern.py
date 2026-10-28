"""--info-pattern centralized: every M^i sees exactly the inputs of the centralized controller with
--pair-features (all of w, all targets z, and the relative position of every pair of robots measured
from x_t); the pair offsets only reach the gate/mixer/select ports; size-matched to that centralized M."""

import warnings
from itertools import combinations

import pytest
import torch

from arg_parser import argument_parser
from Distributed.controllers import build_controller, build_distributed_operator
from Distributed.controllers.factory import build_operator
from Distributed.controllers.interconnection import PairOffsetFeatures
from Distributed.controllers.ssm_operators import InputContextMask
from Distributed.plants import RobotsDataset, RobotsSystem

SMALL = ['--ssm-layers', '2', '--ssm-d-model', '8', '--ssm-d-state', '8', '--ctx-d-features', '8',
         '--mb-hidden', '16', '--ctx-head-hidden', '16', '--feat-dim', '4', '--horizon', '25',
         '--n-agents', '3', '--ssm-param', 'tv', '--ctx-modes', 'mixer,gate,select,input',
         '--info-pattern', 'centralized']


def make(extra=(), seed=0):
    torch.manual_seed(seed)
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    return plant, build_distributed_operator(argument_parser(SMALL + list(extra)), plant)


def expected_offsets(x):
    p = x.view(*x.shape[:-1], 3, 4)[..., :2]
    return torch.cat([p[..., j, :] - p[..., i, :] for i, j in combinations(range(3), 2)], -1)


def test_every_agent_sees_all_of_w_and_z_in_the_global_order():
    plant, op = make()
    assert op.w_idx.tolist() == [list(range(12))] * 3
    assert op.z_idx.tolist() == [list(range(6))] * 3


def test_pair_offsets_match_the_centralized_features():
    f = PairOffsetFeatures([[0, 1], [4, 5], [8, 9]])
    x = torch.randn(2, 1, 12)
    feats = f(x)
    assert len(feats) == 3 and f.out_dims == [6, 6, 6]
    for feat in feats:
        assert torch.equal(feat, expected_offsets(x))


def test_each_agent_receives_the_centralized_inputs():
    plant, op = make(['--no-batch-agents'])
    seen = []
    for o in op.agent_ops:
        o.register_forward_pre_hook(lambda m, inp: seen.append(inp))
    w, z, x, u_prev = torch.randn(2, 1, 12), torch.randn(2, 1, 6), torch.randn(2, 1, 12), torch.randn(2, 1, 6)
    op.reset()
    with torch.no_grad():
        op(w, z, x=x, u_prev=u_prev)
    assert len(seen) == 3
    for w_i, z_i in seen:
        assert torch.equal(w_i, w)
        assert torch.equal(z_i, torch.cat([z, expected_offsets(x)], -1))


def test_offsets_stay_out_of_the_input_port():
    _, op = make()
    for o in op.agent_ops:
        filt = o.mp.core.context_filter
        assert isinstance(filt, InputContextMask)
        assert filt.mask.tolist() == [1.0] * 6 + [0.0] * 6


def test_size_matched_to_the_centralized_controller_with_pair_features():
    args = argument_parser(SMALL)
    plant, op = make()
    central = build_operator(args, w_dim=12, z_dim=6 + 6, u_dim=6)
    assert op.param_budget == sum(p.numel() for p in central.parameters())
    assert sum(p.numel() for p in op.parameters()) <= op.param_budget


def test_shared_weights_are_rejected():
    with pytest.raises(ValueError, match='centralized'):
        make(['--share-weights'])


def test_closed_loop_with_batched_agents():
    torch.manual_seed(0)
    plant = RobotsSystem(linear_plant=False, n_agents=3)
    with warnings.catch_warnings():
        warnings.simplefilter('error')                     # no fallback of --batch-agents
        ctl = build_controller(argument_parser(SMALL + ['--batch-agents']), plant=plant)
    assert ctl.operator.agent_batch is not None
    data = RobotsDataset(random_seed=0, horizon=8, x_bar=None, n_agents=3)._generate_data(2)
    x, e, u = plant.rollout(ctl, data, train=True)
    assert u.shape == (2, 8, 6) and torch.isfinite(x).all()
    u.pow(2).sum().backward()
