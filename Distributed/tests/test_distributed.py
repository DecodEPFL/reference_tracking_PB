"""Structure of the distributed operator: u^i depends exactly on the w and z channels of its
information pattern (full = the centralized input), shared weights equal N identical independent copies, independent
weights cost N times the parameters, or fit the centralized budget when size-matched."""

import pytest
import torch

from arg_parser import argument_parser
from Distributed.controllers import (agent_operators, build_controller, build_distributed_operator,
                                    make_adjacency, neighbor_lists, NeighborFeatures, PBController)
from Distributed.controllers.factory import agent_channels, build_operator
from Distributed.plants import RobotsDataset, RobotsSystem

SMALL = ['--ssm-layers', '2', '--ssm-d-model', '8', '--ssm-d-state', '8',
         '--ctx-d-features', '8', '--mb-hidden', '16', '--feat-dim', '4', '--horizon', '25']
N_AGENTS, B, T = 3, 2, 6
OPERATORS = [
    ['--mp-type', 'contextual_ssm', '--mb-type', 'bounded_mlp'],
    ['--mp-type', 'ssm', '--mb-type', 'diagonal', '--use-w-augment'],
]


def make(argv, seed=0):
    torch.manual_seed(seed)
    plant = RobotsSystem(linear_plant=False, n_agents=N_AGENTS)
    op = build_distributed_operator(argument_parser(SMALL + argv), plant).eval()
    return plant, op


def run(op, w, z):
    """Causal step-by-step evaluation from a fresh state, as in the closed loop."""
    op.reset()
    with torch.no_grad():
        return torch.cat([op(w[:, t:t + 1], z[:, t:t + 1]) for t in range(w.shape[1])], dim=1)


def inputs(plant):
    g = torch.Generator().manual_seed(1)
    return torch.randn(B, T, plant.state_dim, generator=g), torch.randn(B, T, plant.in_dim, generator=g)


def test_agent_channels():
    idx = [[0, 1], [2, 3], [4, 5]]
    assert agent_channels(idx, 'local') == agent_channels(idx, 'local', ego_first=True) == idx
    assert agent_channels(idx, 'full') == [[0, 1, 2, 3, 4, 5]] * 3
    assert agent_channels(idx, 'full', ego_first=True) == [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 0, 1], [4, 5, 0, 1, 2, 3]]
    with pytest.raises(ValueError, match='info-pattern'):
        agent_channels(idx, 'foo')


def test_full_independent_sees_the_centralized_input():
    plant, op = make(OPERATORS[0] + ['--info-pattern', 'full'])
    for i in range(N_AGENTS):
        assert op.w_idx[i].tolist() == list(range(plant.state_dim))
        assert op.z_idx[i].tolist() == list(range(plant.in_dim))


def assert_dependence(plant, op, perturb, depends):
    """u^i changes when agent j's channels are perturbed iff depends(i, j)."""
    w, z = inputs(plant)
    u = run(op, w, z)
    assert u.shape == (B, T, plant.in_dim)
    for j in range(N_AGENTS):
        u2 = run(op, *perturb(w.clone(), z.clone(), j))
        for i in range(N_AGENTS):
            changed = not torch.allclose(u[..., plant.input_idx[i]], u2[..., plant.input_idx[i]])
            assert changed == depends(i, j), f'perturbing agent {j}: u^{i} changed={changed}'


def perturb_w(plant):
    def f(w, z, j):
        w[..., plant.state_idx[j]] += 1.0
        return w, z
    return f


def perturb_z(plant):
    def f(w, z, j):
        z[..., plant.ref_idx[j]] += 1.0
        return w, z
    return f


@pytest.mark.parametrize('pattern', ['local', 'full'])
@pytest.mark.parametrize('share', [False, True], ids=['independent', 'shared'])
@pytest.mark.parametrize('argv', OPERATORS, ids=[' '.join(a) for a in OPERATORS])
def test_information_pattern(argv, share, pattern):
    plant, op = make(argv + ['--info-pattern', pattern] + (['--share-weights'] if share else []))
    dep = (lambda i, j: i == j) if pattern == 'local' else (lambda i, j: True)
    assert_dependence(plant, op, perturb_w(plant), dep)
    assert_dependence(plant, op, perturb_z(plant), dep)


# local only: under full, shared weights order the input ego-first while independent ones
# use the global order, so identical weights do not give identical outputs
@pytest.mark.parametrize('argv', OPERATORS, ids=[' '.join(a) for a in OPERATORS])
def test_shared_equals_identical_independent_copies(argv):
    plant, shared = make(argv + ['--share-weights'])
    _, indep = make(argv + ['--no-match-centralized-size'], seed=1)
    for op in agent_operators(indep):
        op.load_state_dict(shared.agent_ops[0].state_dict())
    w, z = inputs(plant)
    torch.testing.assert_close(run(shared, w, z), run(indep, w, z))


def test_independent_weights_cost_n_times_the_parameters():
    n_params = lambda m: sum(p.numel() for p in m.parameters())
    _, shared = make(OPERATORS[0] + ['--share-weights'])
    _, indep = make(OPERATORS[0] + ['--no-match-centralized-size'])
    assert n_params(indep) == N_AGENTS * n_params(shared)


@pytest.mark.parametrize('pattern', ['local', 'full'])
@pytest.mark.parametrize('argv', OPERATORS, ids=[' '.join(a) for a in OPERATORS])
def test_matched_size_fits_the_centralized_budget(argv, pattern):
    n_params = lambda m: sum(p.numel() for p in m.parameters())
    plant, op = make(argv + ['--info-pattern', pattern])
    central = build_operator(argument_parser(SMALL + argv), plant.state_dim, plant.in_dim, plant.in_dim)
    assert op.param_budget == n_params(central)
    assert 0.8 * n_params(central) <= n_params(op) <= n_params(central)
    # deterministic in args: a rebuild (as from a checkpoint) has the same architecture
    _, op2 = make(argv + ['--info-pattern', pattern], seed=7)
    assert op2.widths == op.widths
    op2.load_state_dict(op.state_dict())


# ---------------- neighbors info pattern ----------------

def test_make_adjacency():
    torch.testing.assert_close(make_adjacency('complete', 3), torch.ones(3, 3) - torch.eye(3))
    torch.testing.assert_close(make_adjacency('line', 3), torch.tensor([[0., 1, 0], [1, 0, 1], [0, 1, 0]]))
    torch.testing.assert_close(make_adjacency('ring', 4).sum(1), torch.full((4,), 2.))
    torch.testing.assert_close(make_adjacency('none', 2), torch.zeros(2, 2))
    torch.testing.assert_close(make_adjacency('0,1;0,0', 2), torch.tensor([[0., 1], [0, 0]]))   # directed
    for bad, match in [('0,1;1,0', '3x3'), ('1,1;1,0', 'diagonal'), ('0,2;1,0', '0 or 1'), ('star', 'adjacency')]:
        with pytest.raises(ValueError, match=match):
            make_adjacency(bad, 3 if match == '3x3' else 2)


def test_neighbor_lists_are_ordered_relative_to_the_agent():
    # ring of 4: slot 0 is always the next agent, slot 1 the previous one
    assert neighbor_lists(make_adjacency('ring', 4)) == [[1, 3], [2, 0], [3, 1], [0, 2]]


def test_neighbor_feature_values():
    # 2 agents, agent 0 at (0, 0), agent 1 at (3, 4); u_{t-1} = [u^0, u^1]
    nf = NeighborFeatures(make_adjacency('complete', 2), pos_idx=[[0, 1], [4, 5]], u_idx=[[0, 1], [2, 3]])
    x = torch.tensor([0., 0, 9, 9, 3, 4, 9, 9]).view(1, 1, 8)
    u = torch.tensor([1., 2, 3, 4]).view(1, 1, 4)
    f0, f1 = nf(x, u)
    torch.testing.assert_close(f0, torch.tensor([3., 4, 5, 3, 4]).view(1, 1, 5))
    torch.testing.assert_close(f1, torch.tensor([-3., -4, 5, 1, 2]).view(1, 1, 5))
    # with targets z = [z^0, z^1], each neighbor appends its own
    nf = NeighborFeatures(make_adjacency('complete', 2), pos_idx=[[0, 1], [4, 5]], u_idx=[[0, 1], [2, 3]],
                          ref_idx=[[0, 1], [2, 3]])
    z = torch.tensor([5., 6, 7, 8]).view(1, 1, 4)
    f0, f1 = nf(x, u, z)
    torch.testing.assert_close(f0, torch.tensor([3., 4, 5, 3, 4, 7, 8]).view(1, 1, 7))
    torch.testing.assert_close(f1, torch.tensor([-3., -4, 5, 1, 2, 5, 6]).view(1, 1, 7))


def make_nb(argv, adjacency, seed=0):
    return make(argv + ['--info-pattern', 'neighbors', '--adjacency', adjacency], seed=seed)


def run_nb(op, w, z, x, u_prev):
    op.reset()
    with torch.no_grad():
        return torch.cat([op(w[:, t:t + 1], z[:, t:t + 1], x=x[:, t:t + 1], u_prev=u_prev[:, t:t + 1])
                          for t in range(w.shape[1])], dim=1)


@pytest.mark.parametrize('share, adjacency', [(False, 'line'), (False, 'complete'), (True, 'complete')],
                         ids=['independent-line', 'independent-complete', 'shared-complete'])
@pytest.mark.parametrize('argv', OPERATORS, ids=[' '.join(a) for a in OPERATORS])
def test_neighbors_information_pattern(argv, share, adjacency):
    plant, op = make_nb(argv + (['--share-weights'] if share else []), adjacency)
    adj = make_adjacency(adjacency, N_AGENTS)
    w, z = inputs(plant)
    g = torch.Generator().manual_seed(2)
    x, u_prev = torch.randn(B, T, plant.state_dim, generator=g), torch.randn(B, T, plant.in_dim, generator=g)
    u = run_nb(op, w, z, x, u_prev)

    def changed(i, x2=x, u2=u_prev, w2=w, z2=z):
        out = run_nb(op, w2, z2, x2, u2)
        return not torch.allclose(u[..., plant.input_idx[i]], out[..., plant.input_idx[i]])

    for j in range(N_AGENTS):
        pos, vel = x.clone(), x.clone()
        pos[..., plant.agent_pos_idx[j]] += 1.0
        vel[..., [c for c in plant.state_idx[j] if c not in plant.agent_pos_idx[j]]] += 1.0
        up = u_prev.clone()
        up[..., plant.input_idx[j]] += 1.0
        w2, z2 = w.clone(), z.clone()
        w2[..., plant.state_idx[j]] += 1.0
        z2[..., plant.ref_idx[j]] += 1.0
        for i in range(N_AGENTS):
            # relative positions: p^j enters u^i for i receiving from j, and u^j if j has any neighbor
            assert changed(i, x2=pos) == (adj[i, j] == 1 or (i == j and adj[j].any()))
            assert not changed(i, x2=vel)                                   # velocities are not sent
            assert changed(i, u2=up) == (i != j and adj[i, j] == 1)        # neighbors' u_{t-1} only
            assert changed(i, w2=w2) == (i == j)                            # w stays local
            assert changed(i, z2=z2) == (i == j or adj[i, j] == 1)          # neighbors send their targets


def test_neighbors_closed_loop_is_causal():
    """Perturbing the disturbance of agent j at step s leaves every input before s unchanged,
    and reaches agent i != j already at step s (through the measured relative position)."""
    args = argument_parser(SMALL + ['--info-pattern', 'neighbors'])
    torch.manual_seed(0)
    plant = RobotsSystem(linear_plant=False)
    ctl = build_controller(args, plant).eval()
    data = RobotsDataset(random_seed=0, horizon=args.horizon, x_bar=torch.zeros(8), n_agents=2)._generate_data(2)
    s, j, i = 10, 1, 0
    data2 = data.clone()
    data2[:, s, plant.state_idx[j]] += 0.5
    with torch.no_grad():
        _, _, u1 = plant.rollout(ctl, data)
        _, _, u2 = plant.rollout(ctl, data2)
    torch.testing.assert_close(u1[:, :s], u2[:, :s])
    assert not torch.allclose(u1[:, s, plant.input_idx[i]], u2[:, s, plant.input_idx[i]])


def test_neighbors_controller_passes_previous_input():
    """The PBController hands the operator x_t and the u it returned at the previous step."""
    plant, op = make_nb(OPERATORS[0], 'complete')
    seen = []
    forward = op.forward
    op.forward = lambda w, z=None, x=None, u_prev=None: (seen.append((x, u_prev)), forward(w, z, x, u_prev))[1]
    ctl = PBController(plant=plant, operator=op, u_dim=plant.in_dim, output_scale=2.0)
    x0 = torch.randn(B, 1, plant.state_dim)
    ctl.reset(x0)
    xs, us = [torch.randn(B, 1, plant.state_dim) for _ in range(3)], []
    v, xbar = torch.zeros(B, 1, plant.in_dim), torch.zeros(B, 1, plant.in_dim)
    with torch.no_grad():
        for t, x in enumerate(xs):
            us.append(ctl(x, torch.zeros(B, 1, plant.in_dim), t, v_tm1=v, xbar_t=xbar)[0])
    for t, (x, u_prev) in enumerate(seen):
        assert x is xs[t]
        torch.testing.assert_close(u_prev, torch.zeros_like(us[0]) if t == 0 else us[t - 1])


def test_neighbors_context_sizes_and_shared_weights_need_equal_degree():
    plant, op = make_nb(OPERATORS[0], 'line')
    assert op.neighbor_features.out_dims == [7, 14, 7]           # [rel (2), dist (1), u (2), z (2)] per neighbor
    _, op = make_nb(OPERATORS[0] + ['--no-neighbor-targets'], 'line')
    assert op.neighbor_features.out_dims == [5, 10, 5]
    with pytest.raises(ValueError, match='same number of neighbors'):
        make_nb(OPERATORS[0] + ['--share-weights'], 'line')
    _, none = make_nb(OPERATORS[0], 'none')                      # no edges: same operators as local
    _, local = make(OPERATORS[0])
    assert none.widths == local.widths


def test_neighbors_checkpoint_without_targets_rebuilds():
    """Checkpoints saved before --neighbor-targets existed rebuild without the targets."""
    args = argument_parser(SMALL + ['--info-pattern', 'neighbors'])
    del args.neighbor_targets
    op = build_controller(args, RobotsSystem(linear_plant=False)).operator
    assert op.neighbor_features.out_dims == [5, 5]
