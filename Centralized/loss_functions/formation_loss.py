import torch
from .robots_loss import RobotsLoss
from Centralized.config import device


class RobotsFormationLoss(RobotsLoss):
    def __init__(
        self, *args, formation_dist, alpha_form=None, formation_adjacency=None,
        formation_deadband=0.0, formation_start=0, **kwargs,
    ):
        """
        RobotsLoss plus a formation term on the inter-agent distances,

            formation = alpha_form / T * sum_{t >= formation_start} sum_{(i, j) in E} max(|d_ij(t) - D_ij| - eps, 0)^2

        with d_ij(t) = ||p^i_t - p^j_t||, D = formation_dist and E the edges of the formation graph.
        Invariant to translations and rotations of the formation; the deadband eps leaves it
        elastic (no cost while every distance is within eps of its desired value).

        Args:
            - formation_dist: (n_agents, n_agents) desired distances D.
            - alpha_form: weight of the formation term. None or 0 disables it (baseline).
            - formation_adjacency: (n_agents, n_agents) 0/1 graph of the constrained pairs, symmetrised
                (an edge in either direction constrains the pair). None: every pair.
            - formation_deadband: eps >= 0.
            - formation_start: first time step where the formation is penalised.
            - args, kwargs: passed to RobotsLoss.
        """
        super().__init__(*args, **kwargs)
        n = self.n_agents
        self.formation_dist = torch.as_tensor(formation_dist, dtype=torch.float32, device=device)
        assert self.formation_dist.shape == (n, n)
        adj = torch.ones(n, n) if formation_adjacency is None else torch.as_tensor(formation_adjacency, dtype=torch.float32)
        assert adj.shape == (n, n)
        adj = ((adj + adj.T) > 0).float()
        self.formation_edges = torch.triu(adj, diagonal=1).to(device)    # each pair once
        self.alpha_form = alpha_form
        if formation_deadband < 0:
            raise ValueError(f"formation_deadband must be >= 0, got {formation_deadband}")
        self.formation_deadband = formation_deadband
        self.formation_start = formation_start

    def loss_terms(self, xs, us, es):
        """RobotsLoss terms plus 'formation' (shape (S, 1, 1))."""
        terms = super().loss_terms(xs, us, es)
        if self.alpha_form:
            terms['formation'] = self.alpha_form * self.f_loss_formation(xs)
        else:
            terms['formation'] = torch.zeros_like(terms['tracking'])
        return terms

    def formation_deviation(self, xs):
        """|d_ij(t) - D_ij| for every pair: shape (S, T, n_agents, n_agents)."""
        if xs.dim() == 3:
            xs = xs.reshape(*xs.shape, 1)
        dist = torch.sqrt(self.get_pairwise_distance_sq(xs) + 1e-12)     # smooth when agents coincide
        return (dist - self.formation_dist.to(dist.device)).abs()

    def f_loss_formation(self, xs):
        """Unweighted formation term: shape (S, 1, 1)."""
        dev = self.formation_deviation(xs)
        pen = torch.clamp(dev - self.formation_deadband, min=0) ** 2
        pen = (pen * self.formation_edges.to(pen.device)).sum((-1, -2))   # (S, T)
        pen = pen[:, self.formation_start:]
        return (pen.sum(1) / xs.shape[1]).reshape(-1, 1, 1)

    @torch.no_grad()
    def formation_error(self, xs):
        """Mean and max over the formation edges of |d_ij(t) - D_ij|: two tensors of shape (S, T)."""
        dev = self.formation_deviation(xs)
        edges = self.formation_edges.to(dev.device).bool()
        dev = dev[..., edges]                                            # (S, T, |E|)
        if dev.shape[-1] == 0:
            zero = xs.new_zeros(xs.shape[:2])
            return zero, zero
        return dev.mean(-1), dev.amax(-1)
