import torch
import torch.nn as nn
import numpy as np
import math
import time

from argparse import Namespace
from config import device
from .contractive_ren import ContractiveREN
from .MLP import MLP
from utils.assistive_functions import to_tensor


class PerfBoostController(nn.Module):
    """
    Performance boosting controller, following the paper:
        "Learning to Boost the Performance of Stable Nonlinear Systems".
    Implements a state-feedback controller with stability guarantees.
    NOTE: When used in closed-loop, the controller input is the measured state
        of the plant and the controller output is the input to the plant.
        This controller has a memory for the last input ("self.last_input") and
        the last output ("self.last_output").
    """
    def __init__(
        self, noiseless_forward, input_init: torch.Tensor, output_init: torch.Tensor,
        # acyclic REN properties
        dim_internal: int, dim_nl: int,
        initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0,
        ren_internal_state_init=None,
        # misc
        output_amplification: float=20,
    ):
        """
         Args:
            noiseless_forward: system dynamics without process noise. can be TV.
            input_init (torch.Tensor): initial input to the controller.
            output_init (torch.Tensor): initial output from the controller before anything is calculated.
            output_amplification (float): TODO
            * the following are the same as AcyclicREN args:
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block of REN.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
            ren_internal_state_init (torch.Tensor, optional): initial state of the REN. Defaults to 0 when None.
        """
        super().__init__()

        self.output_amplification = output_amplification

        # set initial conditions
        self.input_init = input_init.reshape(1, -1)
        self.output_init = output_init.reshape(1, -1)
        self.vg_init = torch.zeros(1,4).reshape(1,-1)
        self.xbar_init = torch.zeros(1,4).reshape(1,-1)


        # set dimensions
        self.dim_in = self.input_init.shape[-1]
        self.dim_out = self.output_init.shape[-1]

        # define the REN
        self.c_ren = ContractiveREN(
            dim_in=self.dim_in, dim_out=self.dim_out, dim_internal=dim_internal,
            dim_nl=dim_nl, initialization_std=initialization_std,
            internal_state_init=ren_internal_state_init,
            posdef_tol=posdef_tol, contraction_rate_lb=contraction_rate_lb
        ).to(device)

        self.MLP = MLP(dim_out = self.dim_out)

        # define the system dynamics without process noise
        self.noiseless_forward = noiseless_forward

        self.reset()

    def reset(self):
        """
        set time to 0 and reset to initial state.
        """
        self.t = 0  # time
        self.last_input = self.input_init.detach().clone()
        self.last_output = self.output_init.detach().clone()
        self.last_vg = self.vg_init.detach().clone()
        self.last_xbar = self.xbar_init.detach().clone()

        self.c_ren.x = self.c_ren.init_x    # reset the REN state to the initial value

    def forward(self, input_t: torch.Tensor,vg:torch.Tensor,xbar:torch.Tensor,):
        """
        Forward pass of the controller.

        Args:
            input_t (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).
            NOTE: when used in closed-loop, "input_t" is the measured states.

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """


        # apply noiseless forward to get noise less input (noise less state of the plant)
        u_noiseless, _ = self.noiseless_forward(
            t=self.t,
            x=self.last_input,  # last input to the controller is the last state of the plant
            u=self.last_output, 
            v=self.last_vg,
            xbar = self.last_xbar  # last output of the controller is the last input to the plant
        )  # shape = (self.batch_size, 1, self.dim_in)

        # reconstruct the noise
        w_ = input_t - u_noiseless # shape = (self.batch_size, 1, self.dim_in)

        # apply REN
        output_REN = self.c_ren.forward(w_)
        mlp_input = torch.cat((w_, xbar), dim=2)
        mlp_input = mlp_input.view(input_t.shape[0],1, -1)
   

        # apply MLP on reference plus disturbance 
        output_MLP = self.MLP.forward(mlp_input)

        output = output_REN*output_MLP*self.output_amplification   # shape = (self.batch_size, 1, self.dim_out)
        #output = torch.clamp(output,min = -10,max = 10)
        # update internal states
        self.last_input, self.last_output, self.last_xbar = input_t, output, xbar
        self.last_vg = vg
        self.t += 1
        return output

    # setters and getters
    def get_parameter_shapes(self):
        return self.c_ren.get_parameter_shapes()

    def get_named_parameters(self):
        return self.c_ren.get_named_parameters()

    def get_parameters_as_vector(self):
        # TODO: implement without numpy
        return np.concatenate([p.detach().clone().cpu().numpy().flatten() for p in self.c_ren.parameters()])
    
    def get_mlp_parameters(self):
        # TODO: implement without numpy
        return self.MLP.state_dict()
    
    def set_mlp_parameters(self,params):
        # TODO: implement without numpy
        self.MLP.load_state_dict(params)

    def set_parameter(self, name, value):
        current_val = getattr(self.c_ren, name)
        value = torch.nn.Parameter(to_tensor(value.reshape(current_val.shape)))
        setattr(self.c_ren, name, value)
        self.c_ren._update_model_param()    # update dependent params

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def set_parameters_as_vector(self, value):
        idx = 0
        for name, shape in self.get_parameter_shapes().items():
            if len(shape) == 1:
                dim = shape
            elif len(shape) == 2:
                dim = shape[0]*shape[1]
            else:
                raise NotImplementedError
            idx_next = idx + dim
            # select indx
            if len(value.shape) == 1:
                value_tmp = value[idx:idx_next]
            elif len(value.shape) == 2:
                value_tmp = value[:, idx:idx_next]
            else:
                raise AssertionError
            # set
            with torch.no_grad():
                self.set_parameter(name, value_tmp.reshape(shape))
            idx = idx_next
        assert idx_next == value.shape[-1]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
