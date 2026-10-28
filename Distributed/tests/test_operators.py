"""Every M_p x M_b combination (and every contextual port subset), with independent or
shared per-agent operators, builds, rolls out in closed loop with the right shapes,
backpropagates, resets cleanly and reloads."""

import argparse

import pytest
import torch

from arg_parser import argument_parser
from Distributed.controllers import agent_operators, build_controller, DistributedOperator, FactorizedOperator, MpContextualSSM
from Distributed.plants import RobotsSystem, RobotsDataset

SMALL = ['--ssm-layers', '2', '--ssm-d-model', '8', '--ssm-d-state', '8',
         '--ctx-d-features', '8', '--mb-hidden', '16', '--feat-dim', '4', '--horizon', '25']


def make(argv):
    args = argument_parser(SMALL + argv)
    torch.manual_seed(0)
    plant = RobotsSystem(linear_plant=False)
    ctl = build_controller(args, plant=plant)
    dataset = RobotsDataset(random_seed=0, horizon=args.horizon, x_bar=torch.zeros(8), n_agents=2)
    data = dataset._generate_data(3)
    return args, plant, ctl, data


COMBOS = [
    ['--mp-type', 'ssm', '--mb-type', 'bounded_mlp'],
    ['--mp-type', 'ssm', '--mb-type', 'bounded_mlp', '--feat-dim', '1'],          # MAD
    ['--mp-type', 'ssm', '--mb-type', 'diagonal'],                                 # rPB
    ['--mp-type', 'ssm', '--mb-type', 'none', '--mp-context-lift'],                # M_p only, lifted z
    ['--mp-type', 'ssm', '--mb-type', 'bounded_mlp', '--mp-context-lift', '--use-w-augment'],
    ['--mp-type', 'ren', '--mb-type', 'bounded_mlp'],                              # legacy REN core
    ['--mp-type', 'contextual_ssm', '--mb-type', 'none'],                          # fused contextual SSM
    ['--mp-type', 'contextual_ssm', '--mb-type', 'bounded_mlp'],
    ['--mp-type', 'contextual_ssm', '--mb-type', 'diagonal', '--use-w-augment'],
    ['--mp-type', 'contextual_ssm', '--ctx-modes', 'input'],
    ['--mp-type', 'contextual_ssm', '--ctx-modes', 'gate'],
    ['--mp-type', 'contextual_ssm', '--ctx-modes', 'mixer'],
    ['--mp-type', 'contextual_ssm', '--ctx-modes', 'input,gate', '--ctx-gate-per-channel'],
    ['--mp-type', 'contextual_ssm', '--ctx-modes', 'input', '--ctx-select', '--ssm-param', 'tv'],
    ['--mp-type', 'contextual_ssm', '--ctx-filter', 'exponential', '--ctx-filter-ramp', '0'],
    ['--mp-type', 'contextual_ssm', '--mb-type', 'bounded_mlp', '--info-pattern', 'full'],
    ['--mp-type', 'ssm', '--mb-type', 'diagonal', '--use-w-augment', '--mp-context-lift', '--info-pattern', 'full'],
    ['--mp-type', 'contextual_ssm', '--mb-type', 'bounded_mlp', '--info-pattern', 'neighbors'],
    ['--mp-type', 'ssm', '--mb-type', 'none', '--mp-context-lift', '--info-pattern', 'neighbors'],
]


SHARE = [['--no-share-weights'], ['--share-weights']]


@pytest.mark.parametrize('share', SHARE, ids=[s[0] for s in SHARE])
@pytest.mark.parametrize('argv', COMBOS, ids=[' '.join(a) for a in COMBOS])
def test_closed_loop_shapes_grads_and_reset(argv, share):
    args, plant, ctl, data = make(argv + share)
    assert isinstance(ctl.operator, DistributedOperator)
    ops = list(agent_operators(ctl.operator))
    assert len(ops) == (1 if args.share_weights else plant.n_agents)
    assert all(isinstance(op, FactorizedOperator) for op in ops)

    x, e, u = plant.rollout(ctl, data, train=True)
    T = data.shape[1]
    assert x.shape == (3, T, plant.state_dim)
    assert e.shape == u.shape == (3, T, plant.in_dim)
    assert torch.isfinite(x).all() and torch.isfinite(u).all()

    u.pow(2).sum().backward()
    grads = [p.grad for p in ctl.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)

    # reset() clears every recurrent state: a second rollout is identical
    # (eval: spectral_norm stops its power iteration, so M_b weights are frozen)
    ctl.eval()
    with torch.no_grad():
        _, _, u1 = plant.rollout(ctl, data)
        _, _, u2 = plant.rollout(ctl, data)
    torch.testing.assert_close(u1, u2)


def test_context_reaches_the_operator():
    args, plant, ctl, data = make(['--mp-type', 'contextual_ssm', '--mb-type', 'none'])
    ctl.eval()
    with torch.no_grad():
        _, _, u_ref = plant.rollout(ctl, data)
        _, _, u_zero = plant.rollout(ctl, data, zero_context=True)
    assert not torch.allclose(u_ref, u_zero)


def test_contextual_mp_receives_z():
    args, plant, ctl, data = make(['--mp-type', 'contextual_ssm', '--mb-type', 'bounded_mlp'])
    for op in agent_operators(ctl.operator):
        assert isinstance(op.mp, MpContextualSSM)
        assert op.mp_takes_context and op.mp_context_lifter is None


@pytest.mark.parametrize('share', SHARE, ids=[s[0] for s in SHARE])
@pytest.mark.parametrize('info', ['local', 'neighbors'])
def test_checkpoint_roundtrip(share, info):
    args, plant, ctl, data = make(['--mp-type', 'contextual_ssm', '--mb-type', 'bounded_mlp', '--info-pattern', info] + share)
    ckpt = {'controller': ctl.state_dict(), 'args': vars(args)}

    torch.manual_seed(123)   # different init, overwritten by the checkpoint
    ctl2 = build_controller(argparse.Namespace(**ckpt['args']), plant=plant)
    ctl2.load_state_dict(ckpt['controller'])
    ctl.eval(), ctl2.eval()
    with torch.no_grad():
        _, _, u1 = plant.rollout(ctl, data)
        _, _, u2 = plant.rollout(ctl2, data)
    torch.testing.assert_close(u1, u2)


@pytest.mark.parametrize('argv, match', [
    (['--mp-type', 'contextual_ssm', '--ctx-select', '--ssm-param', 'lru'], 'selective core'),
    (['--mp-type', 'contextual_ssm', '--ctx-modes', 'input,foo'], 'unknown port'),
    (['--mp-type', 'contextual_ssm', '--ctx-filter', 'taper', '--ctx-filter-ramp', '0'], 'ramp'),
    (['--mp-type', 'contextual_ssm', '--ctx-filter', 'taper', '--ctx-filter-ramp', '40'], 'ramp'),
    (['--mp-type', 'contextual_ssm', '--mp-context-lift'], 'mp-context-lift'),
])
def test_invalid_combinations_fail_fast(argv, match):
    with pytest.raises(ValueError, match=match):
        make(argv)
