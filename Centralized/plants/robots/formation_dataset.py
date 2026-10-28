import math
import torch
from Centralized.plants.costum_dataset import CostumDataset


def formation_offsets(n_agents, side):
    """
    Offsets of the agents in a regular n-gon of side "side", centred on 0: shape (n_agents, 2).
    Agent 0 is on top (+y), the others follow counter-clockwise (3 agents: equilateral triangle).
    With 2 agents: a horizontal segment of length "side"; with 1 agent: the origin.
    """
    if n_agents == 1:
        return torch.zeros(1, 2)
    radius = side / (2 * math.sin(math.pi / n_agents))   # circumradius of the n-gon
    angles = math.pi / 2 + 2 * math.pi * torch.arange(n_agents, dtype=torch.float32) / n_agents
    return radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def rotate(offsets, theta):
    """Rotate (n, 2) offsets by theta (radians) about the origin."""
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s], [s, c]], dtype=offsets.dtype)
    return offsets @ rot.T


def formation_state(center, theta, offsets):
    """Full state [px, py, vx, vy] of every agent, at rest at center + R(theta) offsets[i]: shape (4n,)."""
    pos = rotate(offsets, theta) + torch.as_tensor(center, dtype=offsets.dtype).view(1, 2)
    return torch.cat([pos, torch.zeros_like(pos)], dim=-1).flatten()


def desired_distances(offsets):
    """(n, n) matrix of the desired inter-agent distances d_ij = ||offsets[i] - offsets[j]||."""
    return torch.cdist(offsets, offsets)


class RobotsFormationDataset(CostumDataset):
    """
    Robots that start in formation and must reach a target formation: the same shape,
    translated to a random centroid and (optionally) rotated by a random angle.

    Same layout as RobotsDataset: data[:, :, :4n] is the process noise (the initial state at t=0),
    data[:, 1:, 4n:] the reference state, whose positions form the target formation.
    """

    def __init__(self, random_seed, horizon, n_agents=3, side=2.0, std_ini=0.2,
                 start_center=(2.0, -1.0), start_theta=0.0,
                 target_x=(0.0, 4.0), target_y=(6.0, 6.1), target_rotation=True):
        exp_name = 'robots_formation'
        file_name = 'data_T%i_stdini%s_agents%i_side%s_rot%i_RS%i.pkl' % (
            horizon, std_ini, n_agents, side, int(target_rotation), random_seed)
        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini
        self.n_agents = n_agents
        self.side = side
        self.start_center, self.start_theta = start_center, start_theta
        self.target_x, self.target_y = target_x, target_y
        self.target_rotation = target_rotation

        self.offsets = formation_offsets(n_agents, side)       # (n, 2)
        self.distances = desired_distances(self.offsets)       # (n, n)
        self.x0 = formation_state(start_center, start_theta, self.offsets)

    def sample_target(self):
        """Random target formation: centroid uniform in the target box, angle uniform in [-pi, pi)."""
        center = torch.tensor([
            torch.empty(1).uniform_(*self.target_x).item(),
            torch.empty(1).uniform_(*self.target_y).item(),
        ])
        theta = torch.empty(1).uniform_(-math.pi, math.pi).item() if self.target_rotation else self.start_theta
        return formation_state(center, theta, self.offsets)

    # ---- data generation ----
    def _generate_data(self, num_samples):
        state_dim = 4 * self.n_agents
        data = torch.zeros(num_samples, self.horizon, 2 * state_dim)
        for rollout_num in range(num_samples):
            xbar = self.sample_target()
            data[rollout_num, 0, :state_dim] = self.x0 + self.std_ini * torch.randn(self.x0.shape)
            data[rollout_num, 1:, state_dim:] = xbar
        assert data.shape[0] == num_samples
        return data
