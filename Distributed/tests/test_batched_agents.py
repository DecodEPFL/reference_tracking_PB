"""--batch-agents: the N independent agent operators run in one vmapped call (AgentBatch) with the
same closed-loop results as the per-agent loop (up to float rounding), the same parameters and
checkpoints, and a fallback to the loop when the operators do not support it."""

import warnings

import pytest
import torch

from arg_parser import argument_parser
from Distributed.controllers import build_controller
from Distributed.plants import RobotsDataset, RobotsSystem

SMALL = ['--ssm-layers', '2', '--ssm-d-model', '8', '--ssm-d-state', '8',
         '--ctx-d-features', '8', '--mb-hidden', '16', '--feat-dim', '4', '--horizon', '25']
# selective (tv) SSM with every context port: the configuration of the distributed runs
TV = ['--ssm-param', 'tv', '--ctx-modes', 'mixer,gate,select,input', '--ctx-filter', 'exponential',
      '--info-pattern', 'neighbors']
N_AGENTS = 3


def make(argv, batch, seed=0):
    torch.manual_seed(seed)
    plant = RobotsSystem(linear_plant=False, n_agents=N_AGENTS)
    flag = '--batch-agents' if batch else '--no-batch-agents'
    return plant, build_controller(argument_parser(SMALL + argv + [flag]), plant=plant)


def pair(argv=TV):
    """Loop and batched controllers with identical weights and buffers."""
    plant, loop = make(argv, batch=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')       # no silent fallback to the loop
        _, batched = make(argv, batch=True)
    assert batched.operator.agent_batch is not None and loop.operator.agent_batch is None
    batched.load_state_dict(loop.state_dict())
    return plant, loop, batched


def data(T=15, B=4):
    return RobotsDataset(random_seed=0, horizon=T, x_bar=None, n_agents=N_AGENTS)._generate_data(B)


def flat(grads):
    return torch.cat([g.flatten() for g in grads])


def test_same_parameters_and_checkpoints():
    _, loop, batched = pair()
    assert [n for n, _ in loop.named_parameters()] == [n for n, _ in batched.named_parameters()]
    assert loop.state_dict().keys() == batched.state_dict().keys()


def test_closed_loop_matches_the_loop():
    plant, loop, batched = pair()
    d = data()
    out = {}
    for name, ctl in (('loop', loop), ('batch', batched)):
        x, e, u = plant.rollout(ctl, d, train=True)
        grads = torch.autograd.grad(x.pow(2).mean() + u.pow(2).mean(), list(ctl.parameters()))
        out[name] = x, u, flat(grads), ctl.state_dict()
    (xl, ul, gl, sl), (xb, ub, gb, sb) = out['loop'], out['batch']
    assert (xb - xl).abs().max() <= 1e-4 * xl.abs().max()
    assert (ub - ul).abs().max() <= 1e-4 * ul.abs().max()
    assert (gb - gl).norm() <= 1e-3 * gl.norm()
    for k in sl:                                   # spectral-norm power iterations, written back to every agent
        assert torch.allclose(sl[k], sb[k], rtol=1e-5, atol=1e-6), k


def test_eval_mode_and_repeated_rollouts():
    plant, loop, batched = pair()
    loop.eval(), batched.eval()
    d = data()
    with torch.no_grad():
        for _ in range(2):                         # reset() restarts the recurrent states
            xl, _, _ = plant.rollout(loop, d)
            xb, _, _ = plant.rollout(batched, d)
            assert (xb - xl).abs().max() <= 1e-4 * xl.abs().max()


def test_unsupported_cell_falls_back_to_the_loop():
    with pytest.warns(UserWarning, match='--batch-agents disabled'):
        _, ctl = make(['--ssm-param', 'lru', '--info-pattern', 'neighbors'], batch=True)
    assert ctl.operator.agent_batch is None


def test_shared_weights_ignore_the_flag():
    _, ctl = make(TV + ['--share-weights'], batch=True)
    assert ctl.operator.share_weights and ctl.operator.agent_batch is None


def test_default_is_off_for_the_robots_experiment():
    assert argument_parser([]).batch_agents is False
