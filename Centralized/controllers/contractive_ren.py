import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ContractiveREN(nn.Module):
    """
    Acyclic contractive recurrent equilibrium network, following the paper:
    "Recurrent equilibrium networks: Flexible dynamic models with guaranteed
    stability and robustness, Revay M et al. ."

    The mathematical model of RENs relies on an implicit layer embedded in a recurrent layer.
    The model is described as,

                    [  E . x_t+1 ]  =  [ F    B_1  B_2   ]   [  x_t ]   +   [  b_x ]
                    [  Λ . v_t   ]  =  [ C_1  D_11  D_12 ]   [  w_t ]   +   [  b_w ]
                    [  y_t       ]  =  [ C_2  D_21  D_22 ]   [  u_t ]   +   [  b_u ]

    where E is an invertible matrix and Λ is a positive-definite diagonal matrix. The model parameters
    are then {E, Λ , F, B_i, C_i, D_ij, b} which form a convex set according to the paper.

    NOTE: REN has input "u", output "y", and internal state "x". When used in closed-loop,
          the REN input "u" would be the noise reconstruction ("w") and the REN output ("y")
          would be the input to the plant. The internal state of the REN ("x") should not be mistaken
          with the internal state of the plant.
    """

    def __init__(
        self, dim_in: int, dim_out: int, dim_internal: int,
        dim_nl: int, internal_state_init = None, initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0
    ):
        """
        Args:
            dim_in (int): Input (u) dimension.
            dim_out (int): Output (y) dimension.
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            internal_state_init (torch.Tensor or None, optional): Initial condition for the internal state. Defaults to 0 when set to None.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
        """
        super().__init__()

        # set dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_internal = dim_internal
        self.dim_nl = dim_nl

        # set functionalities
        self.contraction_rate_lb = contraction_rate_lb

        # auxiliary elements
        self.epsilon = posdef_tol

        # initialize internal state
        if internal_state_init is None:
            self.x = torch.zeros(1, 1, self.dim_internal)
        else:
            assert isinstance(internal_state_init, torch.Tensor)
            self.x = internal_state_init.reshape(1, 1, self.dim_internal)
        self.register_buffer('init_x', self.x.detach().clone())

        # define matrices shapes
        # auxiliary matrices
        self.X_shape = (2 * self.dim_internal + self.dim_nl, 2 * self.dim_internal + self.dim_nl)
        self.Y_shape = (self.dim_internal, self.dim_internal)
        # nn state dynamics
        self.B2_shape = (self.dim_internal, self.dim_in)
        # nn output
        self.C2_shape = (self.dim_out, self.dim_internal)
        self.D21_shape = (self.dim_out, self.dim_nl)
        self.D22_shape = (self.dim_out, self.dim_in)
        # v signal
        self.D12_shape = (self.dim_nl, self.dim_in)

        # define trainble params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        self._init_trainable_params(initialization_std)

        # mask
        self.register_buffer('eye_mask_H', torch.eye(2 * self.dim_internal + self.dim_nl))
        self.register_buffer('eye_mask_w', torch.eye(self.dim_nl))

    def _update_model_param(self):
        """
        Update non-trainable matrices according to the REN formulation to preserve contraction.
        """
        # dependent params
        H = torch.matmul(self.X.T, self.X) + self.epsilon * self.eye_mask_H
        h1, h2, h3 = torch.split(H, [self.dim_internal, self.dim_nl, self.dim_internal], dim=0)
        H11, H12, H13 = torch.split(h1, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H21, H22, _ = torch.split(h2, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H31, H32, H33 = torch.split(h3, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        P = H33

        # nn state dynamics
        self.F = H31
        self.B1 = H32

        # nn output
        self.E = 0.5 * (H11 + self.contraction_rate_lb * P + self.Y - self.Y.T)

        # v signal for strictly acyclic REN
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, u_in):
        """
        Forward pass of REN.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        # update non-trainable model params
        self._update_model_param()

        batch_size = u_in.shape[0]

        w = torch.zeros(batch_size, 1, self.dim_nl, device=u_in.device)

        # update each row of w using Eq. (8) with a lower triangular D11
        for i in range(self.dim_nl):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i,:])
            w = w + (self.eye_mask_w[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.dim_nl)

        # compute next state using Eq. 18
        self.x = F.linear(
            F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2),
            self.E.inverse())

        # compute output
        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)
        return y_out

    def reset(self):
        self.x = self.init_x  # reset the REN state to the initial value

    # init trainable params
    def _init_trainable_params(self, initialization_std):
        for training_param_name in self.training_param_names:  # name of one of the training params, e.g., X
            # read the defined shapes of the selected training param, e.g., X_shape
            shape = getattr(self, training_param_name + '_shape')
            # define the selected param (e.g., self.X) as nn.Parameter
            setattr(self, training_param_name, nn.Parameter((torch.randn(*shape) * initialization_std)))

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict
