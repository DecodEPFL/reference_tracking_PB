import torch
from . import LQLossFH
from Distributed.config import device


class RobotsLoss(LQLossFH):
    def __init__(
        self, xbar, Q,Qs, alpha_u=1,
        alpha_col=None, alpha_obst=None, alpha_obst_max=None,
        loss_bound=None, sat_bound=None,
        n_agents=2, min_dist=0.5,
        obstacle_centers=None, obstacle_covs=None, obstacle_radius=None,
        obstacle_loss='gaussian',
        Q_final=None, final_start=150,
        legacy_speed_weight=False,
        alpha_wall=None, wall_x=None, wall_margin=0.5,
    ):
        """
        Args:
            - obstacle_radius: if set, each obstacle's penalty is cut off at this distance from
                its center (shifted down to reach 0 there, so it stays continuous and keeps its
                gradient inside). None keeps the untruncated Gaussian. Required by the barrier.
            - obstacle_loss: 'gaussian' (pdf bump with obstacle_covs) or 'barrier'
                ((R/d - 1)^2 inside radius R, see obstacle_barrier).
            - alpha_obst_max: weight of the worst-step obstacle term, sum over agents of the max over
                time of the obstacle penalty. Unlike the time-averaged term, it cannot be made cheap by
                crossing an obstacle in a few fast steps. None or 0 disables it.
            - Q_final: tracking weight replacing Q for the steps t > final_start (terminal
                cost). None uses Q at every step.
            - legacy_speed_weight: weight the speed penalty with Q instead of Qs, reproducing
                the numbers of the original implementation (where Qs was silently unused).
            - alpha_wall: weight of the arena-wall term on the x coordinate (see f_loss_wall), keeping
                the robots from going around the obstacles. None or 0 disables it.
            - wall_x: (x_min, x_max) of the walls.
            - wall_margin: the penalty starts this far inside each wall.
        """
        super().__init__(Q=Q, R=alpha_u, loss_bound=loss_bound, sat_bound=sat_bound, xbar=None)
        self.Qs = Q if legacy_speed_weight else Qs
        self.xbar = xbar
        self.n_agents = n_agents
        self.alpha_col, self.alpha_obst, self.min_dist = alpha_col, alpha_obst, min_dist
        self.alpha_obst_max = alpha_obst_max
        self.obstacle_radius = obstacle_radius
        self.Q_final, self.final_start = Q_final, final_start
        if obstacle_loss not in ('gaussian', 'barrier'):
            raise ValueError(f"obstacle_loss must be 'gaussian' or 'barrier', got {obstacle_loss!r}")
        if obstacle_loss == 'barrier' and obstacle_radius is None:
            raise ValueError("the barrier obstacle loss needs an obstacle_radius")
        self.obstacle_loss = obstacle_loss
        self.alpha_wall, self.wall_margin = alpha_wall, float(wall_margin)
        self.wall_x = None if wall_x is None else (float(wall_x[0]), float(wall_x[1]))
        if self.alpha_wall:
            if self.wall_x is None or not self.wall_x[0] < self.wall_x[1]:
                raise ValueError(f"the wall loss needs wall_x = (x_min, x_max) with x_min < x_max, got {wall_x}")
            if self.wall_margin <= 0:
                raise ValueError(f"wall_margin must be > 0, got {wall_margin}")
        assert (self.alpha_col is None and self.min_dist is None) or not (self.alpha_col is None or self.min_dist is None)
        if self.alpha_col is not None:
            assert self.n_agents is not None
        # define obstacles
        if obstacle_centers is None:
            self.obstacle_centers = [
                torch.tensor([[-2.5, 0]], device=device),
                torch.tensor([[2.5, 0.0]], device=device),
                torch.tensor([[-1.5, 0.0]], device=device),
                torch.tensor([[1.5, 0.0]], device=device),
            ]
        else:
            self.obstacle_centers = obstacle_centers
        if obstacle_covs is None:
            self.obstacle_covs = [
                torch.tensor([[0.2, 0.2]], device=device)
            ] * len(self.obstacle_centers)
        else:
            self.obstacle_covs = obstacle_covs

        # mask
        self.mask = torch.logical_not(torch.eye(self.n_agents, device=device))   # shape = (n_agents, n_agents)

    def loss_terms(self, xs, us, es):
        """
        Individual loss terms, per sample and before any saturation/bound.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)
            - es: tensor of shape (S, T, in_dim), reference-tracking error

        Return:
            - dict name -> tensor of shape (S, 1, 1) with keys
              tracking, speed, input, collision, obstacle (time-averaged, already weighted),
              plus final (the Q_final part of the tracking, t > final_start) when Q_final is set
              and obstacle_max (worst step, already weighted) when alpha_obst_max is set,
              wall (time-averaged, already weighted) when alpha_wall is set.
        """
        x_batch = xs.reshape(*xs.shape, 1)
        u_batch = us.reshape(*us.shape, 1)
        e_batch = es.reshape(*es.shape, 1)
        T = x_batch.shape[1]
        zero = torch.zeros(xs.shape[0], 1, 1, device=xs.device, dtype=xs.dtype)

        # tracking = 1/T sum_{t <= final_start} e_t^T Q e_t, final = 1/T sum_{t > final_start} e_t^T Q_final e_t
        # (without Q_final, tracking uses Q at every step)
        n_q = T if self.Q_final is None else min(T, self.final_start + 1)
        e_q, e_fin = e_batch[:, :n_q], e_batch[:, n_q:]
        eTQe = torch.matmul(torch.matmul(e_q.transpose(-1, -2), self.Q), e_q)   # (S, n_q, 1, 1)
        loss_x = torch.sum(eTQe, 1) / T
        if self.Q_final is not None:
            eTQe_fin = torch.matmul(torch.matmul(e_fin.transpose(-1, -2), self.Q_final), e_fin)
            loss_final = torch.sum(eTQe_fin, 1) / T   # zero when the horizon ends before final_start

        # speed = 1/T sum_t s_t^T Qs s_t, s = [vx, vy] of every agent
        speed = x_batch[:, :, [i for i in range(xs.shape[-1]) if i % 4 in (2, 3)]]
        sTQs = torch.matmul(torch.matmul(speed.transpose(-1, -2), self.Qs), speed)
        loss_speed = torch.sum(sTQs, 1) / T

        # control actions = 1/T sum_t u_t^T R u_t
        uTRu = self.R * torch.matmul(u_batch.transpose(-1, -2), u_batch)
        loss_u = torch.sum(uTRu, 1) / T

        loss_ca = zero if self.alpha_col is None else self.alpha_col * self.f_loss_ca(x_batch)
        loss_obst = zero if self.alpha_obst is None else self.alpha_obst * self.f_loss_obst(x_batch)

        terms = {'tracking': loss_x}
        if self.Q_final is not None:
            terms['final'] = loss_final
        terms.update(speed=loss_speed, input=loss_u, collision=loss_ca, obstacle=loss_obst)
        if self.alpha_obst_max:
            terms['obstacle_max'] = self.alpha_obst_max * self.f_loss_obst_max(x_batch)
        if self.alpha_wall:
            terms['wall'] = self.alpha_wall * self.f_loss_wall(x_batch)
        return terms

    def forward(self, xs, us, es):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)
            - es: tensor of shape (S, T, in_dim)

        Return:
            - loss of shape (1, 1).
        """
        loss_val = sum(self.loss_terms(xs, us, es).values())   # shape = (S, 1, 1)
        # bound
        if self.sat_bound is not None:
            loss_val = torch.tanh(loss_val/self.sat_bound)  # shape = (S, 1, 1)
        if self.loss_bound is not None:
            loss_val = self.loss_bound * loss_val           # shape = (S, 1, 1)
        # average over the samples
        loss_val = torch.sum(loss_val, 0)/xs.shape[0]       # shape = (1, 1)
        return loss_val

    @torch.no_grad()
    def breakdown(self, xs, us, es):
        """Sample-averaged value of each loss term (floats), before saturation/bound."""
        return {k: v.mean().item() for k, v in self.loss_terms(xs, us, es).items()}

    @staticmethod
    def format_breakdown(terms):
        return ' | '.join('%s: %.2f' % (k, v) for k, v in terms.items())

    def obst_penalty(self, x_batched):
        """
        Obstacle penalty of each agent at each step, summed over the obstacles.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - tensor of shape (S, T, n_agents).
        """
        per_agent = []
        for i in range(x_batched.shape[2] // 4):
            q = x_batched[:, :, 4*i:4*i+2, 0]    # [x, y] of agent i. shape = (S, T, 2)
            pen = 0
            for center, cov in zip(self.obstacle_centers, self.obstacle_covs):
                if self.obstacle_loss == 'barrier':
                    pen = pen + obstacle_barrier(q, mu=center, radius=self.obstacle_radius)   # shape = (S, T)
                else:
                    pen = pen + normpdf(q, mu=center, cov=cov, radius=self.obstacle_radius)   # shape = (S, T)
            per_agent.append(pen)
        return torch.stack(per_agent, dim=-1)

    def f_loss_obst(self, x_batched):
        """
        Obstacle avoidance loss, averaged over time and summed over agents.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - obstacle avoidance loss of shape (S, 1, 1).
        """
        pen = self.obst_penalty(x_batched)
        loss_obst = pen.sum((1, 2)) / pen.shape[1]    # shape = (S)
        return loss_obst.reshape(-1, 1, 1)

    def f_loss_obst_max(self, x_batched):
        """
        Worst-step obstacle loss: max over time of each agent's penalty, summed over agents.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)

        Return:
            - shape (S, 1, 1).
        """
        pen = self.obst_penalty(x_batched)
        return pen.amax(1).sum(-1).reshape(-1, 1, 1)

    def f_loss_ca(self, x_batch):
        """
        Collision avoidance loss.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.


        Return:
            - collision avoidance loss of shape (1, 1).
        """
        min_sec_dist = self.min_dist + 0.2
        # compute pairwise distances
        distance_sq = self.get_pairwise_distance_sq(x_batch)              # shape = (S, T, n_agents, n_agents)
        # compute and sum up loss when two agents are too close
        loss_ca = (1/(distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * self.mask).sum((-1, -2))/2        # shape = (S, T)
        # average over time steps
        loss_ca = loss_ca.sum(1)/loss_ca.shape[1]
        # reshape to S,1,1
        loss_ca = loss_ca.reshape(-1,1,1)
        return loss_ca

    def count_collisions(self, x_batch):
        """
        Count the number of collisions between agents.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - number of collisions between agents.
        """
        if len(x_batch.shape) == 3:
            x_batch = x_batch.reshape(*x_batch.shape, 1)
        distance_sq = self.get_pairwise_distance_sq(x_batch)  # shape = (S, T, n_agents, n_agents)
        col_matrix = (0.0001 < distance_sq) * (distance_sq < self.min_dist ** 2)  # Boolean collision matrix of shape (S, T, n_agents, n_agents)
        n_coll = col_matrix.sum().item()    # all collisions at all times and across all rollouts
        return n_coll/2                     # each collision is counted twice

    def collision_summary(self, x_batch):
        """
        ' -- Collisions: r/S rollouts (n pair-steps)': r rollouts with at least one collision out of S,
        n = count_collisions (every pair closer than min_dist, counted at every step).
        """
        if len(x_batch.shape) == 3:
            x_batch = x_batch.reshape(*x_batch.shape, 1)
        distance_sq = self.get_pairwise_distance_sq(x_batch)
        col_matrix = (0.0001 < distance_sq) * (distance_sq < self.min_dist ** 2)
        n_rollouts = col_matrix.flatten(1).any(1).sum().item()
        return ' -- Collisions: %i/%i rollouts (%i pair-steps)' % (
            n_rollouts, x_batch.shape[0], self.count_collisions(x_batch))

    def get_pairwise_distance_sq(self, x_batch):
        """
        Squared distance between pairwise agents.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - matrix of shape (S, T, n_agents, n_agents) of squared pairwise distances.
        """
        state_dim_per_agent = int(x_batch.shape[2]/self.n_agents)
        # collision avoidance:
        x_agents = x_batch[:, :, 0::state_dim_per_agent, :]  # start from 0, pick every state_dim_per_agent. shape = (S, T, n_agents, 1)
        y_agents = x_batch[:, :, 1::state_dim_per_agent, :]  # start from 1, pick every state_dim_per_agent. shape = (S, T, n_agents, 1)
        deltaqx = x_agents.repeat(1, 1, 1, self.n_agents) - x_agents.repeat(1, 1, 1, self.n_agents).transpose(-2, -1)   # shape = (S, T, n_agents, n_agents)
        deltaqy = y_agents.repeat(1, 1, 1, self.n_agents) - y_agents.repeat(1, 1, 1, self.n_agents).transpose(-2, -1)   # shape = (S, T, n_agents, n_agents)
        distance_sq = deltaqx ** 2 + deltaqy ** 2             # shape = (S, T, n_agents, n_agents)
        return distance_sq


    def wall_penalty(self, x_batched):
        """
        Arena walls at x = x_min and x = x_max: with s the distance of a robot inside the closest wall
        (negative beyond it) and m = wall_margin,

            phi(s) = max(m - s, 0)^2 / m^2

        0 farther than m from the walls, 1 on a wall, still growing (with a non-vanishing gradient)
        beyond it, unlike a 1/s barrier that a robot already outside could not come back through.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)

        Return:
            - tensor of shape (S, T, n_agents).
        """
        px = x_batched[:, :, 0::4, 0]                                      # x of every agent: (S, T, n_agents)
        x_min, x_max = self.wall_x
        m = self.wall_margin
        return (torch.clamp(m - (px - x_min), min=0) ** 2 + torch.clamp(m - (x_max - px), min=0) ** 2) / m ** 2

    def f_loss_wall(self, x_batched):
        """Wall loss, averaged over time and summed over agents: shape (S, 1, 1)."""
        pen = self.wall_penalty(x_batched)
        return (pen.sum((1, 2)) / pen.shape[1]).reshape(-1, 1, 1)


def normpdf(q, mu, cov, radius=None):  #TODO
    """
    PDF of normal distribution with mean "mu" and covariance "cov".

    Args:
        - q: shape(S, T, state_dim_per_agent)
        - mu:
        - cov:
        - radius: if set, return max(pdf(q) - pdf(radius), 0) with pdf(radius) the value at
            distance "radius" from mu (exact for isotropic cov, which all obstacles use),
            i.e. a continuous penalty that is exactly 0 beyond "radius".

    Return:
            -
    """
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)  # the diagonal of the covariance matrix
    qs = torch.split(q, 2, dim=-1)
    for ind, qi in enumerate(qs):
        # if qi[1]<1.5 and qi[1]>-1.5:
        den = (2*torch.pi)**(0.5*d) * torch.sqrt(torch.prod(cov))
        nom = torch.exp((-0.5 * (qi - mu)**2 / cov).sum(-1))
        val = nom/den
        if radius is not None:
            edge = torch.exp(-0.5 * radius**2 / cov.mean()) / den
            val = torch.clamp(val - edge, min=0)
        if ind == 0:
            out = val
        else:
            out += val
    return out


def obstacle_barrier(q, mu, radius, eps=0.05):
    """
    Barrier penalty around an obstacle centre "mu", summed over agents.

        phi(d) = (radius / d_eps - 1)^2  for d < radius, 0 beyond,   d_eps = sqrt(d^2 + eps^2)

    Zero with zero slope at d = radius (C^1, no kink), growing like 1/d^2 towards the
    centre; eps keeps it finite there ((radius/eps - 1)^2).

    Args:
        - q: shape(S, T, 2*n_agents), [x1, y1, x2, y2, ...]
        - mu: obstacle centre, 2 entries
        - radius: distance beyond which the penalty is 0

    Return:
        - shape (S, T)
    """
    mu = mu.view(1, 2)
    out = 0
    for qi in torch.split(q, 2, dim=-1):
        d_eps = torch.sqrt(((qi - mu)**2).sum(-1) + eps**2)
        out = out + torch.clamp(radius / d_eps - 1, min=0)**2
    return out
