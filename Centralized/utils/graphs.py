"""Agent graphs given as named graphs or explicit 0/1 matrices (same convention as the
distributed code: adjacency[i, j] = 1 links agent i to agent j, no self-loops)."""

import torch

ADJACENCY_GRAPHS = ("complete", "ring", "line", "none")


def make_adjacency(spec: str, n_agents: int) -> torch.Tensor:
    """(N, N) 0/1 adjacency from a named graph or an explicit matrix "0,1;1,0" (rows by ';')."""
    n = int(n_agents)
    spec = str(spec).strip()
    if spec == "complete":
        adj = torch.ones(n, n) - torch.eye(n)
    elif spec == "ring":
        adj = torch.zeros(n, n)
        for i in range(n):
            adj[i, (i + 1) % n] = adj[i, (i - 1) % n] = 1.0
        adj.fill_diagonal_(0.0)                      # n = 1
    elif spec == "line":
        adj = torch.zeros(n, n)
        for i in range(n - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
    elif spec == "none":
        adj = torch.zeros(n, n)
    else:
        try:
            adj = torch.tensor([[float(v) for v in row.split(",")] for row in spec.split(";")])
        except ValueError as exc:
            raise ValueError(f"--adjacency must be one of {ADJACENCY_GRAPHS} or a matrix like '0,1;1,0', got {spec!r}") from exc
        if adj.shape != (n, n):
            raise ValueError(f"--adjacency matrix must be {n}x{n} for {n} agents, got {tuple(adj.shape)}.")
        if not ((adj == 0) | (adj == 1)).all():
            raise ValueError("--adjacency matrix entries must be 0 or 1.")
        if adj.diagonal().any():
            raise ValueError("--adjacency matrix must have a zero diagonal (no self-loops).")
    return adj
