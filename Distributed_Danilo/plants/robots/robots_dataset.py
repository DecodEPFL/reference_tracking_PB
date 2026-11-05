import torch
from Distributed_Danilo.plants import CostumDataset


def agents_on_line(xs, y=0.0):
    """Full state [px, py, vx, vy] of every agent, at rest at (xs[i], y): shape (4 * len(xs),)."""
    return torch.cat([torch.tensor([float(x), float(y), 0., 0.]) for x in xs])


def start_xs(n_agents):
    """Default start abscissas, evenly spread from x=4 down to x=0 on y=0 (2 agents: 4, 0)."""
    return torch.linspace(4, 0, n_agents).tolist()


def scenario_xbar(name, n_agents, y=4.0):
    """
    Reference state of the verification scenarios, generalising the 2-agent ones:
        diag:   targets mirror the start positions, the agents cross (2 agents: 0, 4)
        direct: targets straight ahead, spread outwards (2 agents: 5, -1)
        center: targets packed around x=1 with spacing 1 (2 agents: 0.5, 1.5)
    """
    if name == 'diag':
        xs = torch.linspace(0, 4, n_agents).tolist()
    elif name == 'direct':
        xs = torch.linspace(5, -1, n_agents).tolist()
    elif name == 'center':
        xs = [1 + (i - (n_agents - 1) / 2) for i in range(n_agents)]
    else:
        raise ValueError(f"unknown scenario {name!r}")
    return agents_on_line(xs, y)


class RobotsDataset(CostumDataset):
    def __init__(self, random_seed, horizon,x_bar, std_ini=0.2, n_agents=2):
        # experiment and file names
        exp_name = 'robots'
        file_name = 'data_T'+str(horizon)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini
        self.n_agents = n_agents

        self.x0 = agents_on_line(start_xs(n_agents))   # 2 agents: [4, 0, 0, 0, 0, 0, 0, 0]
        self.xbar = x_bar

    ### Get target position which are not colliding ###
    def generate_vector_with_min_distance(self,interval_x1=(-3, 3), interval_x2=(2, 3), min_distance=1.0, max_tries=100000):
        """Rejection-sample one position per agent, pairwise at least min_distance apart: shape (2 * n_agents,)."""
        for _ in range(max_tries):
            # one (x1, x2) per agent, x1 in interval_x1, x2 in interval_x2
            vecs = [
                torch.tensor([
                    torch.empty(1).uniform_(interval_x1[0], interval_x1[1]),
                    torch.empty(1).uniform_(interval_x2[0], interval_x2[1])
                ]).flatten()
                for _ in range(self.n_agents)
            ]
            # Check the pairwise Euclidean distances
            if all(torch.norm(vecs[i] - vecs[j]) >= min_distance
                   for i in range(self.n_agents) for j in range(i + 1, self.n_agents)):
                # Concatenate to form a (2 * n_agents,) vector
                return torch.cat(vecs)
        raise RuntimeError(f"no {self.n_agents} targets at least {min_distance} apart found in {max_tries} tries; "
                           f"widen the target intervals or lower min_distance.")

    # ---- data generation ----
    def _generate_data(self, num_samples):

        #Initial conditions
        state_dim_x0 = 4*self.n_agents

        #Reference tracking data
        state_dim_ref = 4*self.n_agents

        #Total dimensions
        state_dim = state_dim_x0+state_dim_ref

        data = torch.zeros(num_samples, self.horizon, state_dim)


        for rollout_num in range(num_samples):
            vecs = self.generate_vector_with_min_distance(interval_x1=(-1, 5), interval_x2=(4, 4.1), min_distance=2.0)
            data[rollout_num, 0, :state_dim_x0] = \
                self.x0 + self.std_ini * torch.randn(self.x0.shape)
            for i in range(self.n_agents):   # target position of agent i
                data[rollout_num, 1:, state_dim_x0+4*i:state_dim_x0+4*i+2] = \
                    vecs[2*i:2*i+2]

        assert data.shape[0]==num_samples
        return data
