import math
import torch
from Distributed.plants.costum_dataset import CostumDataset


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


def isosceles_offsets(base, height):
    """
    Offsets of an isosceles triangle centred on its centroid: shape (3, 2). Same agent order as the
    regular 3-gon: agent 0 is the apex (+y), agents 1 and 2 the left and right ends of a horizontal
    base of length "base", at distance "height" below the apex.
    A wide flat triangle (base > gap > height) must turn its base along y to pass a narrow gap.
    """
    return torch.tensor([[0.0, 2 * height / 3], [-base / 2, -height / 3], [base / 2, -height / 3]])


def bowtie_offsets(width, height):
    """
    Offsets of a bowtie centred on 0: shape (5, 2). Agent 0 is the knot at the centre, agents 1-4
    the corners counter-clockwise from the top left: (-w/2, h/2), (-w/2, -h/2), (w/2, -h/2), (w/2, h/2).
    Two isosceles triangles of base "height" (vertical, at x = -+w/2) meeting at their apex (agent 0).
    A wide flat bowtie (width > gap > height) must turn to pass a narrow gap.
    """
    w, h = width / 2, height / 2
    return torch.tensor([[0.0, 0.0], [-w, h], [-w, -h], [w, -h], [w, h]])


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
                 target_x=(0.0, 4.0), target_y=(6.0, 6.1), target_rotation=True,
                 target_max_angle=math.pi, target_flip=False, shape='regular', base=None, height=None,
                 large_rot_frac=0.0, large_rot_min_angle=2 * math.pi / 3):
        """
        shape: 'regular' (n-gon of side "side"), 'isosceles' (3 agents, see isosceles_offsets)
            or 'bowtie' (5 agents, see bowtie_offsets: width "base", height "height").
        target_max_angle: with target_rotation, target angles are uniform in
            start_theta + [-target_max_angle, target_max_angle) (pi: any orientation).
        target_flip: also turn half of the targets by pi (a wide triangle keeps its base along x, apex down).
        large_rot_frac: fraction of the targets (train and test alike) whose angle is drawn from the large
            rotations only, |angle - start_theta| in [large_rot_min_angle, target_max_angle] with a random
            sign; the others stay uniform over the whole range. 0 (default) leaves the data unchanged.
        """
        exp_name = 'robots_formation'
        shape_tag = {'regular': 'side%s' % side, 'isosceles': 'iso%sx%s' % (base, height),
                     'bowtie': 'bowtie%sx%s' % (base, height)}.get(shape, shape)
        file_name = 'data_T%i_stdini%s_agents%i_%s_rot%i%s_RS%i.pkl' % (
            horizon, std_ini, n_agents, shape_tag, int(target_rotation), '_flip' if target_flip else '', random_seed)
        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini
        self.n_agents = n_agents
        self.side = side
        self.start_center, self.start_theta = start_center, start_theta
        self.target_x, self.target_y = target_x, target_y
        self.target_rotation = target_rotation
        self.target_max_angle = target_max_angle
        self.target_flip = target_flip
        self.shape, self.base, self.height = shape, base, height
        self.large_rot_frac, self.large_rot_min_angle = float(large_rot_frac), float(large_rot_min_angle)
        if not 0.0 <= self.large_rot_frac <= 1.0:
            raise ValueError(f"large_rot_frac must be in [0, 1], got {large_rot_frac}")
        if self.large_rot_frac > 0 and not (target_rotation and 0 <= self.large_rot_min_angle < min(target_max_angle, math.pi)):
            raise ValueError("large_rot_frac > 0 needs target_rotation and 0 <= large_rot_min_angle < target_max_angle")

        if shape == 'regular':
            self.offsets = formation_offsets(n_agents, side)       # (n, 2)
        elif shape == 'isosceles':
            if n_agents != 3:
                raise ValueError(f"the isosceles formation needs 3 agents, got {n_agents}")
            self.offsets = isosceles_offsets(base, height)
        elif shape == 'bowtie':
            if n_agents != 5:
                raise ValueError(f"the bowtie formation needs 5 agents, got {n_agents}")
            self.offsets = bowtie_offsets(base, height)
        else:
            raise ValueError(f"shape must be 'regular', 'isosceles' or 'bowtie', got {shape!r}")
        self.distances = desired_distances(self.offsets)       # (n, n)
        self.x0 = formation_state(start_center, start_theta, self.offsets)

    def sample_target(self, generator=None, oversample=False):
        """Random target formation: centroid uniform in the target box, angle uniform in
        start_theta + [-target_max_angle, target_max_angle) ([-pi, pi) by default), plus pi for
        half of the targets with target_flip.
        oversample: with probability large_rot_frac, the angle is drawn from the large rotations
        |angle - start_theta| in [large_rot_min_angle, target_max_angle] instead (before the flip)."""
        g = generator
        center = torch.tensor([
            torch.empty(1).uniform_(*self.target_x, generator=g).item(),
            torch.empty(1).uniform_(*self.target_y, generator=g).item(),
        ])
        max_angle = min(self.target_max_angle, math.pi)
        if not self.target_rotation:
            theta = self.start_theta
        elif oversample and self.large_rot_frac > 0 and torch.rand(1, generator=g).item() < self.large_rot_frac:
            sign = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
            theta = self.start_theta + sign * torch.empty(1).uniform_(self.large_rot_min_angle, max_angle, generator=g).item()
        elif self.target_max_angle >= math.pi:
            theta = torch.empty(1).uniform_(-math.pi, math.pi, generator=g).item()
        else:
            theta = self.start_theta + torch.empty(1).uniform_(-self.target_max_angle, self.target_max_angle, generator=g).item()
        if self.target_flip and torch.rand(1, generator=g).item() < 0.5:
            theta += math.pi
        return formation_state(center, theta, self.offsets)

    # ---- data generation ----
    def _generate_data(self, num_samples, generator=None, oversample=False):
        state_dim = 4 * self.n_agents
        data = torch.zeros(num_samples, self.horizon, 2 * state_dim)
        for rollout_num in range(num_samples):
            xbar = self.sample_target(generator=generator, oversample=oversample)
            data[rollout_num, 0, :state_dim] = self.x0 + self.std_ini * torch.randn(self.x0.shape, generator=generator)
            data[rollout_num, 1:, state_dim:] = xbar
        assert data.shape[0] == num_samples
        return data

    def get_data(self, num_train_samples=1024, num_test_samples=1024):
        """Train and test data of CostumDataset. With large_rot_frac > 0, both are redrawn with the large
        rotations oversampled (same distribution), each from its own generator: the global RNG state
        afterwards (hence the model initialisation) is the same as without oversampling."""
        train_data, test_data = super().get_data(num_train_samples, num_test_samples)
        if self.large_rot_frac > 0:
            g_train = torch.Generator().manual_seed(self.random_seed + 1)
            g_test = torch.Generator().manual_seed(self.random_seed + 2)
            train_data = self._generate_data(num_train_samples, generator=g_train, oversample=True)
            test_data = self._generate_data(num_test_samples, generator=g_test, oversample=True)
        return train_data, test_data
