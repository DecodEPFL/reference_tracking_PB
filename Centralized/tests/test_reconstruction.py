"""Disturbance reconstruction of the PB controller on the robots plant.

Regression test for the reference lag of the original PerfBoostController, which
reconstructed w_t with the reference of the *previous* step: at the reference jump
(0 -> target at t=1) the plant used x̄_1 while the controller assumed x̄_0, so the
reference step leaked into ŵ.
"""

import torch

from Centralized.controllers import OperatorBase, PBController
from Centralized.plants import RobotsSystem, RobotsDataset


class RecordingOperator(OperatorBase):
    """u = small deterministic function of (w, z), recording every reconstructed w."""

    def __init__(self, u_dim):
        super().__init__()
        self.u_dim = u_dim
        self.ws = []

    def reset(self):
        self.ws = []

    def forward(self, w, z=None):
        self.ws.append(w)
        # nonzero, state-dependent input so the PI loop and the u_{t-1} memory matter
        return 0.1 * torch.tanh(w[..., : self.u_dim] + z)


def test_w_reconstruction_is_exact_across_the_reference_step():
    torch.manual_seed(0)
    plant = RobotsSystem(linear_plant=False)
    op = RecordingOperator(plant.in_dim)
    ctl = PBController(plant=plant, operator=op, u_dim=plant.in_dim, output_scale=20.0)
    data = RobotsDataset(random_seed=0, horizon=30, x_bar=torch.zeros(8), n_agents=2)._generate_data(4)

    with torch.no_grad():
        plant.rollout(ctl, data)
    w_hat = torch.cat(op.ws, dim=1)
    w_true = data[:, :, :plant.state_dim]

    # w_0 is the random initial condition, w_{t>=1} = 0 although x̄ jumps at t=1
    torch.testing.assert_close(w_hat, w_true, atol=1e-5, rtol=0)
    assert (data[:, 1, plant.state_dim:] != data[:, 0, plant.state_dim:]).any()
