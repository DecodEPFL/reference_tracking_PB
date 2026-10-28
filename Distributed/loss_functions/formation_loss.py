import itertools

import torch
from .robots_loss import RobotsLoss
from Distributed.config import device


class RobotsFormationLoss(RobotsLoss):
    def __init__(
        self, *args, formation_dist, alpha_form=None, formation_adjacency=None,
        formation_deadband=0.0, formation_start=0,
        formation_offsets=None, alpha_orient=None, orient_margin=0.5, alpha_orient_max=None, **kwargs,
    ):
        """
        RobotsLoss plus a formation term on the inter-agent distances,

            formation = alpha_form / T * sum_{t >= formation_start} sum_{(i, j) in E} max(|d_ij(t) - D_ij| - eps, 0)^2

        with d_ij(t) = ||p^i_t - p^j_t||, D = formation_dist and E the edges of the formation graph.
        Invariant to translations and rotations of the formation; the deadband eps leaves it
        elastic (no cost while every distance is within eps of its desired value).

        Distances cannot tell a formation from its mirror image, so robots may cross the line
        between two others (collapsing and mirroring the formation) instead of rotating it.
        The orientation term penalises this,

            orientation = alpha_orient / T * sum_{t >= formation_start} sum_{(i, j, k)} max(m |A*_ijk| - s_ijk A_ijk(t), 0)^2

        with A_ijk(t) the signed area of the triangle (p^i_t, p^j_t, p^k_t), A*_ijk and
        s_ijk = sign(A*_ijk) those of the formation offsets, over every non-collinear triple, and
        m = orient_margin. Zero for any rigid motion, positive once a triangle shrinks below m of
        its area or turns inside out. Being time-averaged, it stays cheap when a robot cuts through
        the line between two others in a few fast steps; the worst-step term

            orientation_max = alpha_orient_max * sum_{(i, j, k)} max_{t >= formation_start} max(m |A*_ijk| - s_ijk A_ijk(t), 0)^2

        charges such a brief inversion as much as a long one.

        Args:
            - formation_dist: (n_agents, n_agents) desired distances D.
            - alpha_form: weight of the formation term. None or 0 disables it (baseline).
            - formation_adjacency: (n_agents, n_agents) 0/1 graph of the constrained pairs, symmetrised
                (an edge in either direction constrains the pair). None: every pair.
            - formation_deadband: eps >= 0.
            - formation_start: first time step where the formation is penalised.
            - formation_offsets: (n_agents, 2) positions of the formation, giving the reference
                orientation (targets are rotations of it). Needed by the orientation term.
            - alpha_orient: weight of the orientation term. None or 0 disables it.
            - orient_margin: m in [0, 1], fraction of each triangle's area below which it is penalised.
            - alpha_orient_max: weight of the worst-step orientation term. None or 0 disables it.
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

        # non-collinear triples of the formation and their signed areas
        self.alpha_orient = alpha_orient
        if not 0 <= orient_margin <= 1:
            raise ValueError(f"orient_margin must be in [0, 1], got {orient_margin}")
        self.orient_margin = orient_margin
        self.alpha_orient_max = alpha_orient_max
        self.orient_triples, self.orient_areas = None, None
        if formation_offsets is not None:
            off = torch.as_tensor(formation_offsets, dtype=torch.float32)
            assert off.shape == (n, 2)
            triples = torch.tensor(list(itertools.combinations(range(n), 3)), dtype=torch.long).view(-1, 3)
            areas = signed_area(off[triples[:, 0]], off[triples[:, 1]], off[triples[:, 2]])
            keep = areas.abs() > 1e-6 * off.norm(dim=-1).max().clamp(min=1) ** 2
            self.orient_triples, self.orient_areas = triples[keep].to(device), areas[keep].to(device)
        if (alpha_orient or alpha_orient_max) and (self.orient_triples is None or len(self.orient_triples) == 0):
            raise ValueError("the orientation term needs formation_offsets with at least one non-collinear triple")

    def loss_terms(self, xs, us, es):
        """RobotsLoss terms plus 'formation', 'orientation' and 'orientation_max' when enabled (shape (S, 1, 1))."""
        terms = super().loss_terms(xs, us, es)
        if self.alpha_form:
            terms['formation'] = self.alpha_form * self.f_loss_formation(xs)
        else:
            terms['formation'] = torch.zeros_like(terms['tracking'])
        if self.alpha_orient:
            terms['orientation'] = self.alpha_orient * self.f_loss_orientation(xs)
        if self.alpha_orient_max:
            terms['orientation_max'] = self.alpha_orient_max * self.f_loss_orientation_max(xs)
        return terms

    def triangle_areas(self, xs):
        """Signed areas of the orientation triples, oriented so the formation's are positive: shape (S, T, n_triples)."""
        if xs.dim() == 4:
            xs = xs[..., 0]
        p = xs.reshape(*xs.shape[:-1], -1, 4)[..., :2]                  # (S, T, n_agents, 2)
        tri = self.orient_triples.to(p.device)
        areas = signed_area(p[..., tri[:, 0], :], p[..., tri[:, 1], :], p[..., tri[:, 2], :])
        return areas * torch.sign(self.orient_areas.to(p.device))

    def orientation_penalty(self, xs):
        """Squared area shortfall of every orientation triple: shape (S, T', n_triples), t >= formation_start."""
        areas = self.triangle_areas(xs)
        floor = self.orient_margin * self.orient_areas.abs().to(areas.device)
        return (torch.clamp(floor - areas, min=0) ** 2)[:, self.formation_start:]

    def f_loss_orientation(self, xs):
        """Unweighted orientation term: shape (S, 1, 1)."""
        pen = self.orientation_penalty(xs).sum(-1)                         # (S, T')
        return (pen.sum(1) / xs.shape[1]).reshape(-1, 1, 1)

    def f_loss_orientation_max(self, xs):
        """Unweighted worst-step orientation term: max over time of each triple's penalty, summed over triples: shape (S, 1, 1)."""
        return self.orientation_penalty(xs).amax(1).sum(-1).reshape(-1, 1, 1)

    @torch.no_grad()
    def mirrored(self, xs):
        """True for the rollouts where some triple is turned inside out (mirrored) at some step: shape (S,)."""
        if self.orient_triples is None or len(self.orient_triples) == 0:
            return xs.new_zeros(xs.shape[0], dtype=torch.bool)
        return (self.triangle_areas(xs) < 0).any(-1).any(-1)

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


def signed_area(a, b, c):
    """Signed area of the triangle (a, b, c), positive when counter-clockwise. Shapes (..., 2) -> (...)."""
    u, v = b - a, c - a
    return 0.5 * (u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
