"""--pad-agent-inputs: on a graph where the robots hear different numbers of others (unequal input
sizes) the inputs are zero-padded to the largest so the agents still run in one vmapped call, with
the weights reading the padding held at zero: every padded M^i computes what it would unpadded."""

import argparse
import warnings

import pytest
import torch

from arg_parser import argument_parser
from Distributed.controllers import build_controller
from Distributed.plants import RobotsSystem
from Distributed.plants.robots.formation_dataset import RobotsFormationDataset

# 5 robots: 0 hears everyone, 1 and 2 only each other, 3 and 4 hear 0 and each other (so 1 and 2 relayed)
GRAPH = '0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0'
SMALL = ['--ssm-layers', '2', '--ssm-d-model', '8', '--ssm-d-state', '8', '--ctx-d-features', '8',
         '--mb-hidden', '16', '--ctx-head-hidden', '16', '--feat-dim', '4', '--horizon', '25', '--n-agents', '5',
         '--ssm-param', 'tv', '--ctx-modes', 'mixer,gate,select,input', '--ctx-filter', 'exponential',
         '--adjacency', GRAPH]
RELAY = ['--info-pattern', 'relay', '--relay-inputs', '--relay-inputs-to', 'mixers']
N, B, T = 5, 3, 12


def make(argv, seed=0):
    torch.manual_seed(seed)
    plant = RobotsSystem(linear_plant=False, n_agents=N)
    with warnings.catch_warnings():
        warnings.simplefilter('error')       # no silent fallback to the loop
        ctl = build_controller(argument_parser(SMALL + argv), plant=plant)
    return plant, ctl


def data():
    return RobotsFormationDataset(random_seed=0, horizon=T, n_agents=N, shape='bowtie', base=4.0, height=1.2)._generate_data(B)


def masked_columns(ctl):
    """(name, parameter, padded columns) of every weight reading padded inputs."""
    out = []
    for name, p in ctl.named_parameters():
        if p.dim() == 2:
            dead = (p == 0).all(0)
            if dead.any():
                out.append((name, p, dead))
    return out


def run_closed_loop(plant, ctl):
    x, _, u = plant.rollout(ctl, data(), train=True)
    loss = x.pow(2).mean() + u.pow(2).mean()
    return x, u, loss


@pytest.mark.parametrize('pattern', [RELAY, ['--info-pattern', 'neighbors']], ids=['relay', 'neighbors'])
def test_unequal_agents_are_padded_and_batched(pattern):
    _, ctl = make(pattern + ['--batch-agents'])
    op = ctl.operator
    assert op.agent_batch is not None and op.input_padding is not None
    w_pad, z_pad, real_w, real_z = op.input_padding
    assert len(set(real_z)) > 1 and max(real_z) == z_pad and max(real_w) == w_pad
    # only the agents with fewer sources have masked weights, agent 0 (hears everyone) none
    names = {n.split('.')[2] for n, _, _ in masked_columns(ctl)}
    assert names == {str(i) for i in range(N) if real_z[i] < z_pad}


def test_off_or_without_batching_the_agents_stay_unpadded():
    plant = RobotsSystem(linear_plant=False, n_agents=N)
    with pytest.warns(UserWarning, match='--batch-agents disabled'):
        ctl = build_controller(argument_parser(SMALL + RELAY + ['--batch-agents', '--no-pad-agent-inputs']), plant=plant)
    assert ctl.operator.agent_batch is None and ctl.operator.input_padding is None
    _, ctl = make(RELAY + ['--no-batch-agents'])
    assert ctl.operator.input_padding is None


def test_checkpoints_saved_before_the_flag_are_unpadded():
    args = vars(argument_parser(SMALL + RELAY + ['--batch-agents']))
    del args['pad_agent_inputs']
    with pytest.warns(UserWarning, match='--batch-agents disabled'):
        ctl = build_controller(argparse.Namespace(**args), plant=RobotsSystem(linear_plant=False, n_agents=N))
    assert ctl.operator.input_padding is None


VARIANTS = {'relay': RELAY, 'neighbors': ['--info-pattern', 'neighbors'],
            'certified_gamma': RELAY + ['--ctx-gamma', '1.0'], 'w_augment': RELAY + ['--use-w-augment'],
            'diagonal_mb': RELAY + ['--mb-type', 'diagonal'],
            'ssm_lifted': RELAY + ['--mp-type', 'ssm', '--mp-context-lift', '--mp-context-lift-type', 'mlp']}


@pytest.mark.parametrize('pattern', VARIANTS.values(), ids=VARIANTS.keys())
def test_padded_equals_the_unpadded_operators(pattern):
    """The unpadded weights copied into the real columns of the padded ones: same closed loop,
    same gradients of the real weights, zero gradient on the padded columns."""
    plant, small = make(pattern + ['--no-batch-agents'])
    _, padded = make(pattern + ['--batch-agents'])
    sd_small, sd_pad = small.state_dict(), padded.state_dict()
    real = {}
    for k, v in sd_pad.items():
        if v.shape == sd_small[k].shape:
            sd_pad[k] = sd_small[k]
            continue
        # padded columns of a weight (all zero), or entries of its spectral-norm v
        weight = sd_pad[k.removesuffix('_v') + '_orig'] if k.endswith('_v') else v
        keep = ~(weight == 0).all(0)
        assert int(keep.sum()) == sd_small[k].shape[-1], k
        sd_pad[k] = torch.zeros_like(v)
        sd_pad[k][..., keep] = sd_small[k]
        real[k] = keep
    padded.load_state_dict(sd_pad)
    padded.operator.agent_batch = None           # the padding alone (batching: test_batched_matches_the_padded_loop)

    xs, us, ls = run_closed_loop(plant, small)
    xp, up, lp = run_closed_loop(plant, padded)
    assert (xp - xs).abs().max() <= 1e-4 * xs.abs().max()
    assert (up - us).abs().max() <= 1e-4 * us.abs().max()
    gs = dict(zip([n for n, _ in small.named_parameters()], torch.autograd.grad(ls, list(small.parameters()))))
    names = [n for n, _ in padded.named_parameters()]
    gp = dict(zip(names, torch.autograd.grad(lp, list(padded.parameters()))))
    for n in names:
        g = gp[n]
        if n in real:
            assert torch.all(g[..., ~real[n]] == 0), n
            g = g[..., real[n]]
        assert (g - gs[n]).abs().max() <= 1e-3 * gs[n].abs().max() + 1e-7, n


def test_batched_matches_the_padded_loop():
    plant, batched = make(RELAY + ['--batch-agents'])
    _, loop = make(RELAY + ['--batch-agents'], seed=1)
    loop.operator.agent_batch = None                 # the same padded agents, one by one
    loop.load_state_dict(batched.state_dict())
    xb, ub, lb = run_closed_loop(plant, batched)
    xl, ul, ll = run_closed_loop(plant, loop)
    assert (xb - xl).abs().max() <= 1e-4 * xl.abs().max()
    assert (ub - ul).abs().max() <= 1e-4 * ul.abs().max()
    for gb, gl in zip(torch.autograd.grad(lb, list(batched.parameters())), torch.autograd.grad(ll, list(loop.parameters()))):
        assert (gb - gl).abs().max() <= 1e-3 * gl.abs().max() + 1e-7


def test_blind_to_whatever_is_in_the_padding():
    """Masked weights: values in the padded channels change nothing."""
    _, ctl = make(RELAY + ['--batch-agents'])
    op = ctl.operator
    _, _, real_w, real_z = op.input_padding
    i = real_z.index(min(real_z))
    agent = op.agent_ops[i]
    w_keep = torch.tensor([1.0] * 4 + op.neighbor_features.w_masks[i])
    z_keep = torch.tensor([1.0] * 2 + op.neighbor_features.context_masks[i])
    g = torch.Generator().manual_seed(0)
    w = torch.randn(B, T, len(w_keep), generator=g) * w_keep
    z = torch.randn(B, T, len(z_keep), generator=g) * z_keep
    junk_w = w + 3 * torch.randn(w.shape, generator=g) * (1 - w_keep)
    junk_z = z + 3 * torch.randn(z.shape, generator=g) * (1 - z_keep)

    def run(w, z):
        agent.reset()
        with torch.no_grad():
            return torch.cat([agent(w[:, t:t + 1], z[:, t:t + 1]) for t in range(T)], dim=1)
    agent.eval()
    torch.testing.assert_close(run(junk_w, junk_z), run(w, z), rtol=0, atol=0)


def test_training_keeps_the_padded_columns_at_zero():
    plant, ctl = make(RELAY + ['--batch-agents'])
    masked = masked_columns(ctl)
    assert masked
    opt = torch.optim.AdamW(ctl.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(3):
        opt.zero_grad()
        run_closed_loop(plant, ctl)[2].backward()
        opt.step()
    for name, p, dead in masked:
        assert torch.all(p[:, dead] == 0), name
        assert not torch.all(p[:, ~dead] == 0), name
