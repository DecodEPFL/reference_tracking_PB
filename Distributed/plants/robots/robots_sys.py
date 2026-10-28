import numpy as np
from scipy.linalg import block_diag
from scipy.signal import place_poles

import torch
import torch.nn.functional as F


class RobotsSystem(torch.nn.Module):
    def __init__(self, linear_plant: bool, x_init=None, u_init=None, k: float=1.0, n_agents: int = 2):
        """

        Args:
            xbar: concatenated nominal initial point of all agents
            linear_plant: if True, a linearized model of the system is used.
                             O.w., the model is non-lineardue to the dependence of friction on the speed.
            x_init: concatenated initial point of all agents. Defaults to xbar when None.
            u_init: initial input to the plant. Defaults to zero when None.
            k (float): gain of the pre-stabilizing controller (acts as a spring constant).
        """
        super().__init__()

        self.linear_plant = linear_plant



        # check dimensions
        self.n_agents = n_agents
        self.state_dim = 4*self.n_agents
        self.in_dim = 2*self.n_agents

        # initial state
        x_init = torch.zeros((1,self.state_dim)) if x_init is None else x_init.reshape(1, -1) # shape = (1, state_dim)
        self.register_buffer('x_init', x_init)
        u_init = torch.zeros(1, int(self.x_init.shape[1]/2)) if u_init is None else u_init.reshape(1, -1)   # shape = (1, in_dim)
        self.register_buffer('u_init', u_init)

        assert self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.in_dim

        self.h = 0.05
        self.mass = 1.0
        self.k = k
        self.b = 1.0
        self.b2 = 0.0 if self.linear_plant else 0.1
        m = self.mass

        # Define A and A2 matrices
        A_ = np.eye(4)
        A2 = np.block([[np.zeros((2, 2)), np.eye(2)],
                    [np.diag([-k/m, -k/m]), np.diag([-self.b/m, -self.b/m])]])

        A = A_ + self.h * A2  # Compute A matrix

        # Define B matrix
        B = np.array([[0, 0], [0, 0], [self.h, 0], [0, self.h]])


        # Define output matrix C to track x_0, x_1, x_4, x_5
        C = np.array([[1, 0, 0, 0],   # Track x_0 for agent 1
                    [0, 1, 0, 0]])  # Track x_5 for agent 2

        # Augmented system (including integral states)
        A_aug = np.block([[A, np.zeros((4, 2))], [-C, np.eye(2)]])
        B_aug = np.vstack([B, np.zeros((2, 2))])


        B = torch.kron(torch.eye(self.n_agents),
                       torch.tensor([[0, 0],
                                     [0., 0],
                                     [1/m, 0],
                                     [0, 1/m]])
                       ) * self.h
        """k_i = 0
        self.K_i = torch.kron(torch.eye(self.n_agents),
                       torch.tensor([[0, 0],
                                     [0., 0],
                                     [k_i, 0],
                                     [0, k_i]])
                       ) * self.h"""
        self.register_buffer('B', B)

        _A1 = torch.eye(4*self.n_agents)
        _A2 = torch.cat((torch.cat((torch.zeros(2,2),
                                torch.eye(2)
                                ), dim=1),
                        torch.cat((torch.diag(torch.tensor([-self.k/self.mass, -self.k/self.mass])),
                                   torch.diag(torch.tensor([-self.b/self.mass, -self.b/self.mass]))
                                   ), dim=1),
                        ), dim=0)
        _A2 = torch.kron(torch.eye(self.n_agents), _A2)
        A_lin = _A1 + self.h * _A2



        self.register_buffer('A_lin', A_lin)



        # Choose desired poles for the augmented system
        desired_poles = np.exp(np.array([-3, -3.05, -4, -4.05, -2, -2.05]) * self.h)
        #desired_poles = np.exp(np.array([-3, -3.05, -5, -5.05, -7, -7.05]) * self.h)

        # Compute the state feedback gains (K) and integral gains (K_I) using pole placement
        # Using scipy's place_poles function
        place_obj = place_poles(A_aug, B_aug, desired_poles)
        K_aug = place_obj.gain_matrix

        # Extract state feedback gains and integral gains
        K_p = K_aug[:, :4].astype(np.float32)  # State feedback gains
        K_i = K_aug[:, 4:].astype(np.float32)  # Integral gains
        K_p = torch.kron(torch.eye(self.n_agents), torch.tensor(K_p))
        K_i = torch.kron(torch.eye(self.n_agents), torch.tensor(K_i))

        self.register_buffer('K_p', K_p)
        self.register_buffer('K_i', K_i)



        # Output the results
        """print("State feedback gains (K_p):")
        print(self.K_p)
        print("\nIntegral gains (K_i):")
        print(self.K_i)"""

        mask = torch.tensor([[0, 0], [1, 1]]).repeat(self.n_agents, 1)

        mask_tanh = torch.kron(torch.eye(self.n_agents),
                       torch.tensor([[0, 0],
                                     [0., 0],
                                     [-self.b2/m, 0],
                                     [0, -self.b2/m]])
                       ) * self.h
        self.register_buffer('mask', mask)
        self.register_buffer('mask_tanh', mask_tanh)

        # state layout per agent: [px, py, vx, vy]
        self.pos_idx = [4 * i + j for i in range(self.n_agents) for j in (0, 1)]
        self.vel_idx = [4 * i + j for i in range(self.n_agents) for j in (2, 3)]

        # per-agent channels, used by the distributed operator (one M^i per agent).
        # The dynamics are block-diagonal, so the joint nominal model reconstructs every w^i exactly.
        self.agent_state_dim = 4
        self.agent_in_dim = 2
        self.state_idx = [[4 * i + j for j in range(4)] for i in range(self.n_agents)]    # x^i, w^i in x
        self.agent_pos_idx = [[4 * i + j for j in range(2)] for i in range(self.n_agents)]  # p^i in x
        self.input_idx = [[2 * i + j for j in range(2)] for i in range(self.n_agents)]    # u^i in u
        self.ref_idx = [[2 * i + j for j in range(2)] for i in range(self.n_agents)]      # z^i in z = xbar positions

    def A_nonlin(self, x):
        assert not self.linear_plant
        A3 = torch.norm(
            x.view(-1, 2 * self.n_agents, 2) * self.mask, dim=-1, keepdim=True
        )           # shape = (batch_size, 2 * n_agents, 1)
        A3 = torch.kron(
            A3, torch.ones(2, 1, device=A3.device)
        )           # shape = (batch_size, 4 * n_agents, 1)
        A3 = -self.b2 / self.mass * torch.diag_embed(
            A3.squeeze(dim=-1), offset=0, dim1=-2, dim2=-1
        )           # shape = (batch_size, 4 * n_agents, 4 * n_agents)
        A = self.A_lin + self.h * A3
        return A    # shape = (batch_size, 4 * n_agents, 4 * n_agents)

    def noiseless_forward(self, t, x: torch.Tensor,v:torch.Tensor, u: torch.Tensor, xbar: torch.Tensor):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state of the noise-free dynamics.
        """

        x = x.view(-1, 1, self.state_dim)
        dxref = u.view(-1, 1, self.in_dim)
        
        e = (xbar+dxref) - x[:,:,self.pos_idx]

        v = v + e 

        u = -F.linear(x,self.K_p) -F.linear(v,self.K_i)
        
        tanh_q = torch.tanh(x[:,:,self.vel_idx])

        if self.linear_plant:
            # x is batched but A is not => can use F.linear to compute xA^T
            #f = F.linear(x - xbar, self.A_lin) + F.linear(u, self.B) + xbar

            f = F.linear(x, self.A_lin) + F.linear(u,self.B) 
        else:
            # A depends on x, hence is batched. perform batched matrix multiplication
            f =  F.linear(x, self.A_lin) + F.linear(tanh_q, self.mask_tanh) + F.linear(u, self.B) 
        #


        return (f,v)    # shape = (batch_size, 1, state_dim)

    def forward(self, t, x,v, u, w,xbar):
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next state.
        """

        f,v = self.noiseless_forward(t, x,v, u,xbar)

        f = f + w.view(-1, 1, self.state_dim) 
        

        return (f,v)
    def nominal_dynamics(self, x, u, t=None, *, v_tm1, xbar_t):
        """
        Nominal transition used by the PB controller to reconstruct the disturbance.

        Args:
            - x, u: plant state and PB input (reference offset) at t-1.
            - v_tm1: integrator state of the inner PI loop at t-1.
            - xbar_t: reference applied during the transition (t-1 -> t).

        Returns:
            nominal next state, shape = (batch_size, 1, state_dim).
        """
        f, _ = self.noiseless_forward(t, x, v_tm1, u, xbar_t)
        return f

    def split_data(self, data):
        """data: (B, T, 2*state_dim) = [disturbance | reference state] -> (w, xbar positions)."""
        w = data[:, :, :self.state_dim]
        xbar = data[:, :, self.state_dim:][:, :, self.pos_idx]
        return w, xbar

    # simulation
    def rollout(self, controller, data, train=False, zero_context=False):
        """
        Closed-loop rollout of the PB controller for a batch of disturbance/reference sequences.

        Args:
            - controller: PBController.
            - data: (batch_size, T, 2*state_dim); [:state_dim] is the process noise
                (the initial condition enters as w_0), [state_dim:] the reference state.
            - zero_context: feed z = 0 instead of the reference (disturbance-only baseline).

        Return:
            - x_log of shape = (batch_size, T, state_dim)
            - e_log of shape = (batch_size, T, in_dim), reference-tracking error
            - u_log of shape = (batch_size, T, in_dim)
        """
        w, xbar = self.split_data(data)
        batch = data.shape[0]

        x = self.x_init.detach().clone().repeat(batch, 1, 1)
        u = self.u_init.detach().clone().repeat(batch, 1, 1)
        v = torch.zeros_like(u)
        controller.reset(x, u)

        x_log, e_log, u_log, v_log = [], [], [], []
        for t in range(data.shape[1]):
            xbar_t = xbar[:, t:t+1, :]
            v_tm1 = v
            # x_{k+1} = f(x_k, u_k, xbar_k) + w_k
            x, v = self.forward(t=t, x=x, u=u, v=v, w=w[:, t:t+1, :], xbar=xbar_t)    # shape = (batch_size, 1, state_dim)
            # u_{k+1} = PB(x_{k+1}; z = xbar), reconstructing w with the reference of this transition
            z_t = torch.zeros_like(xbar_t) if zero_context else xbar_t
            u, _ = controller(x, z_t, t, v_tm1=v_tm1, xbar_t=xbar_t)              # shape = (batch_size, 1, in_dim)

            x_log.append(x)
            u_log.append(u)
            v_log.append(v)
            e_log.append(xbar_t - x[:, :, self.pos_idx])

        x_log, e_log, u_log = torch.cat(x_log, 1), torch.cat(e_log, 1), torch.cat(u_log, 1)
        if not train:
            x_log, e_log, u_log = x_log.detach(), e_log.detach(), u_log.detach()

        self.v_log = torch.cat(v_log, 1).detach()
        return x_log, e_log, u_log
