import numpy as np
import torch

from value_iteration.pendulum import BaseSystem
from value_iteration.cost_functions import ArcTangent, SineQuadraticCost, BarrierCost
CUDA_AVAILABLE = torch.cuda.is_available()


class Cartpole(BaseSystem):
    name = "Cartpole"
    labels = ('x', 'theta', 'x_dot', 'theta_dot')

    def __init__(self, cuda=CUDA_AVAILABLE, **kwargs):
        super(Cartpole, self).__init__()

        # Define Duration:
        self.T = kwargs.get("T", 7.5)
        self.dt = kwargs.get("dt", 1./500.)

        # Define the System:
        self.n_state = 4
        self.n_dof = 2
        self.n_act = 1
        self.n_parameter = 5

        # Continuous Joints:
        # Right now only one continuous joint is supported
        self.wrap, self.wrap_i = True, 1

        # State Constraints:
        # theta = 0, means the pendulum is pointing upward
        self.x_target = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.x_start = torch.tensor([0.0, np.pi, 0.0, 0.0])
        self.x_start_var = torch.tensor([1.e-3, 5.e-2, 1.e-6, 1.e-6])
        self.x_lim = torch.tensor([0.5, np.pi, 5.0, 20.0])
        self.x_penalty = torch.tensor([0.4, 1.1 * np.pi, 1.1 * 5.0, 1.1 * 20.0])
        self.x_init = torch.tensor([0.15, np.pi, 0.01, 0.01])
        self.u_lim = torch.tensor([20., ])

        # Define dynamics:
        self.g = 9.81          # Gravitational acceleration [m/s^2]
        mc = 0.57              # Mass of the cart [kg]
        mp = 0.127             # Mass of the pole [kg]
        pl = 0.3365 / 2.       # Half of the pole length [m]
        Beq = 0.1              # Equivalent Viscous damping Coefficient 5.4
        Bp = 1.e-3             # Viscous coefficient at the pole 0.0024

        # Dynamics parameter:
        self.theta = torch.tensor([mc, mp, pl, Beq, Bp]).view(1, self.n_parameter, 1)
        self.theta_min = 0.5 * torch.tensor([mc, mp, pl, Beq, Bp]).view(1, self.n_parameter, 1)
        self.theta_max = 1.5 * torch.tensor([mc, mp, pl, Beq, Bp]).view(1, self.n_parameter, 1)

        # Compute Linearized System:
        out = self.dyn(self.x_target, gradient=True)
        self.A = out[2].view(1, self.n_state, self.n_state).transpose(dim0=1, dim1=2).numpy()
        self.B = out[1].view(1, self.n_state, self.n_act).numpy()

        # Test Dynamics:
        self.check_dynamics()

        self.device = None
        Cartpole.cuda(self) if cuda else Cartpole.cpu(self)

    def dyn(self, x, dtheta=None, gradient=False):
        cat = torch.cat

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        n_samples = x.shape[0]

        q, q_dot = x[:, :self.n_dof], x[:, self.n_dof:]
        xc, th = x[:, 0].view(-1, 1, 1), x[:, 1].view(-1, 1, 1)
        x_dot, th_dot = x[:, 2].view(-1, 1, 1), x[:, 3].view(-1, 1, 1)

        sin_th, cos_th = torch.sin(th), torch.cos(th)
        ones_1, zeros_1, zeros_n_dof = torch.ones_like(th), torch.zeros_like(th), torch.zeros((n_samples, 2, 1)).to(x.device)

        # Update the dynamics parameters with disturbance:
        if dtheta is not None:
            dtheta = torch.from_numpy(dtheta).float() if isinstance(dtheta, np.ndarray) else dtheta
            dtheta = dtheta.view(n_samples, self.n_parameter, 1)
            theta = self.theta + dtheta
            theta = torch.min(torch.max(theta, self.theta_min), self.theta_max)

        else:
            theta = self.theta

        # Define mass matrix M = [[a, b], [b, c]]
        H_00 = (theta[:, 1:2] + theta[:, 0:1]) * ones_1
        H_01 = theta[:, 1:2] * theta[:, 2:3] * cos_th
        H_11 = theta[:, 1:2] * theta[:, 2:3] ** 2 * ones_1

        # H = cat([cat([H_00, H_01], dim=2), cat([H_01, H_11], dim=2)], dim=1)
        invH = cat([cat([H_11, -H_01], dim=2), cat([-H_01, H_00], dim=2)], dim=1) / (H_00 * H_11 - H_01 * H_01)

        # Calculate vector n = C(q, qd) + g(q):
        n = cat([-theta[:, 1:2] * theta[:, 2:3] * sin_th * th_dot**2,
                 -theta[:, 1:2] * theta[:, 2:3] * self.g * sin_th], dim=1)

        f = cat([-theta[:, 3:4] * x_dot, -theta[:, 4:5] * th_dot], dim=1)

        # Construct Dynamics:
        a = cat([q_dot, torch.matmul(invH, f - n)], dim=1)
        B = cat([torch.zeros((n_samples, self.n_dof, 1)).to(x.device), invH[:, :, :1]], dim=1)

        assert a.shape == (n_samples, self.n_state, 1)
        assert B.shape == (n_samples, self.n_state, self.n_act)
        out = (a, B)

        if gradient:
            zeros_nxn = torch.zeros((n_samples, self.n_dof, self.n_dof)).to(x.device)
            ones_nxn = torch.ones((n_samples, self.n_dof, self.n_dof)).to(x.device)

            dH_00_dq = zeros_n_dof.view(n_samples, self.n_dof, 1, 1)
            dH_01_dq = cat([zeros_1.view((n_samples, 1, 1, 1)), (-theta[:, 1] * theta[:, 2] * sin_th).view((-1, 1, 1, 1))], dim=1)
            dH_11_dq = zeros_n_dof.view(n_samples, self.n_dof, 1, 1)

            dHdq = cat([cat([dH_00_dq, dH_01_dq], dim=3), cat([dH_01_dq, dH_11_dq], dim=3)], dim=2)
            dinvH_dq = -torch.matmul(invH.view(-1, 1, self.n_dof, self.n_dof), torch.matmul(dHdq, invH.view(-1, 1, self.n_dof, self.n_dof)))

            dn_dx = zeros_n_dof.view(n_samples, 2, 1)
            dn_dth = cat([-theta[:, 1] * theta[:, 2] * cos_th * th_dot ** 2, -theta[:, 1] * theta[:, 2] * self.g * cos_th], dim=1)
            dn_dxd = zeros_n_dof
            dn_dthd = cat([-2. * theta[:, 1] * theta[:, 2] * sin_th * th_dot, zeros_1], dim=1)

            dn_dq = cat([dn_dx, dn_dth], dim=2)
            dn_dqd = cat([dn_dxd, dn_dthd], dim=2)

            df_dqd = cat([cat([-theta[:, 3] * ones_1, zeros_1], dim=1), cat([zeros_1, -theta[:, 4] * ones_1], dim=1)], dim=2)

            # Construct da/dx:
            A_00 = zeros_nxn
            A_01 = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof).to(x.device) * ones_nxn
            A_10 = torch.matmul(dinvH_dq, (f - n).view(-1, 1, self.n_dof, 1)).squeeze(-1).transpose(dim0=1, dim1=2) - torch.matmul(invH, dn_dq)
            A_11 = torch.matmul(invH, df_dqd - dn_dqd)

            dadx = cat([cat([A_00, A_01], dim=2), cat([A_10, A_11], dim=2)], dim=1).transpose(dim0=1, dim1=2)
            dBdx = cat([cat([zeros_nxn.view(n_samples, self.n_dof, self.n_dof, 1), dinvH_dq[:, :, :, :self.n_act]], dim=2),
                        torch.zeros(n_samples, self.n_dof, self.n_state, 1).to(x.device)], dim=1)

            assert dadx.shape == (n_samples, self.n_state, self.n_state,)
            assert dBdx.shape == (n_samples, self.n_state, self.n_state, self.n_act)
            out = (a, B, dadx, dBdx)

        if is_numpy:
            out = [array.numpy() for array in out]

        return out

    def grad_dyn_theta(self, x):
        cat = torch.cat

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        n_samples = x.shape[0]

        xc, th = x[:, 0].view(-1, 1, 1), x[:, 1].view(-1, 1, 1)
        x_dot, th_dot = x[:, 2].view(-1, 1, 1), x[:, 3].view(-1, 1, 1)

        sin_th, cos_th = torch.sin(th), torch.cos(th)
        ones_1, zeros_1, zeros_n_dof = torch.ones_like(th), torch.zeros_like(th), torch.zeros((n_samples, 2, 1)).to(x.device)

        # Define mass matrix M = [[a, b], [b, c]]
        H_00 = (self.theta[:, 1] + self.theta[:, 0]) * ones_1
        H_01 = self.theta[:, 1] * self.theta[:, 2] * cos_th
        H_11 = self.theta[:, 1] * self.theta[:, 2] ** 2 * ones_1

        # H = cat([cat([H_00, H_01], dim=2), cat([H_01, H_11], dim=2)], dim=1)
        invH = cat([cat([H_11, -H_01], dim=2), cat([-H_01, H_00], dim=2)], dim=1) / (H_00 * H_11 - H_01 * H_01)

        # Calculate vector n = C(q, qd) + g(q):
        n = cat([-self.theta[:, 1] * self.theta[:, 2] * sin_th * th_dot**2,
                 -self.theta[:, 1] * self.theta[:, 2] * self.g * sin_th], dim=1).view(-1, 1, self.n_dof, 1)

        f = cat([-self.theta[:, 3] * x_dot, -self.theta[:, 4] * th_dot], dim=1).view(-1, 1, self.n_dof, 1)

        dHdp = torch.zeros(n_samples, self.n_parameter, self.n_dof, self.n_dof).to(x.device)
        dndp = torch.zeros(n_samples, self.n_parameter, self.n_dof, 1).to(x.device)
        dfdp = torch.zeros(n_samples, self.n_parameter, self.n_dof, 1).to(x.device)

        # dM/dm_c
        dHdp[:, 0, 0:1, 0:1] = ones_1

        # dM/dm_p
        dHdp[:, 1, 0:1, 0:1] = ones_1
        dHdp[:, 1, 0:1, 1:2] = self.theta[:, 2] * cos_th
        dHdp[:, 1, 1:2, 0:1] = self.theta[:, 2] * cos_th
        dHdp[:, 1, 1:2, 1:2] = self.theta[:, 2]**2

        # dM/dl_p
        dHdp[:, 2, 0:1, 0:1] = zeros_1
        dHdp[:, 2, 0:1, 1:2] = self.theta[:, 1] * cos_th
        dHdp[:, 2, 1:2, 0:1] = self.theta[:, 1] * cos_th
        dHdp[:, 2, 1:2, 1:2] = self.theta[:, 1] * self.theta[:, 2] * 2

        # dn/dm_p
        dndp[:, 1, 0:1] = -self.theta[:, 2] * sin_th * th_dot**2
        dndp[:, 1, 1:2] = -self.theta[:, 2] * self.g * sin_th

        # dn/dl_p
        dndp[:, 2, 0:1] = -self.theta[:, 1] * sin_th * th_dot**2
        dndp[:, 2, 1:2] = -self.theta[:, 1] * self.g * sin_th

        # df/dB_c
        dfdp[:, 3, 0:1] = -x_dot
        dfdp[:, 4, 1:2] = -th_dot

        invH_4d = invH.view(-1, 1, self.n_dof, self.n_dof)
        dinvHdp = -torch.matmul(invH_4d, torch.matmul(dHdp, invH_4d))

        dadp = torch.zeros(n_samples, self.n_parameter, self.n_state).to(x.device)
        dadp[:, :, self.n_dof:, ] = (torch.matmul(dinvHdp, f - n) + torch.matmul(invH_4d, dfdp - dndp)).view(-1, self.n_parameter, self.n_dof)

        dBdp = torch.zeros(n_samples, self.n_parameter, self.n_state, self.n_act).to(x.device)
        dBdp[:, :, self.n_dof:, ] = dinvHdp[:, :, :, :self.n_act]

        out = (dadp, dBdp)
        if is_numpy:
            out = [array.cpu().detach().numpy() for array in out]

        return out

    def cuda(self, device=None):
        self.u_lim = self.u_lim.cuda(device=device)
        self.theta_min = self.theta_min.cuda(device=device)
        self.theta = self.theta.cuda(device=device)
        self.theta_max = self.theta_max.cuda(device=device)
        self.device = self.theta.device
        return self

    def cpu(self):
        self.u_lim = self.u_lim.cpu()
        self.theta_min = self.theta_min.cpu()
        self.theta = self.theta.cpu()
        self.theta_max = self.theta_max.cpu()
        self.device = self.theta.device
        return self


class CartpoleLogCos(Cartpole):
    name = "Cartpole_LogCosCost"

    def __init__(self, Q, R, cuda=False, **kwargs):

        # Create the dynamics:
        super(CartpoleLogCos, self).__init__(cuda=cuda, **kwargs)
        self.u_lim = torch.tensor([12., ])

        # Create the Reward Function:
        assert Q.size == self.n_state and np.all(Q > 0.0)
        self.Q = np.diag(Q).reshape((self.n_state, self.n_state))

        assert R.size == self.n_act and np.all(R > 0.0)
        self.R = np.diag(R).reshape((self.n_act, self.n_act))

        self._q = SineQuadraticCost(self.Q, np.array([0.0, 1.0, 0.0, 0.0]), cuda=cuda)
        self.q = BarrierCost(self._q,  self.x_penalty, cuda)

        # Determine beta s.t. the curvature at u = 0 is identical to 2R
        beta = 4. * self.u_lim[0] ** 2 / np.pi * self.R
        self.r = ArcTangent(alpha=self.u_lim.numpy()[0], beta=beta.numpy()[0, 0])

    def rwd(self, x, u):
        return self.q(x) + self.r(u)

    def cuda(self, device=None):
        super(CartpoleLogCos, self).cuda(device=device)
        self.q.cuda(device=device)
        return self

    def cpu(self):
        super(CartpoleLogCos, self).cpu()
        self.q.cpu()
        return self


if __name__ == "__main__":
    from deep_differential_network.utils import jacobian

    # GPU vs. CPU:
    cuda = True

    # Seed the test:
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create system:
    sys = Cartpole()

    n_samples = 10000
    x_lim = torch.from_numpy(sys.x_lim).float() if isinstance(sys.x_lim, np.ndarray) else sys.x_lim
    x_test = torch.distributions.uniform.Uniform(-x_lim, x_lim).sample((n_samples,))
    # x_test = torch.tensor([np.pi / 2., 0.5]).view(1, sys.n_state, 1)

    dtheta = torch.zeros(1, sys.n_parameter, 1)

    if cuda:
        sys, x_test, dtheta = sys.cuda(), x_test.cuda(), dtheta.cuda()

    ###################################################################################################################
    # Test dynamics gradient w.r.t. state:
    dadx_shape = (n_samples, sys.n_state, sys.n_state)
    dBdx_shape = (n_samples, sys.n_state, sys.n_state, sys.n_act)

    a, B, dadx, dBdx = sys.dyn(x_test, gradient=True)

    dadx_auto = torch.cat([jacobian(lambda x: sys.dyn(x)[0], x_test[i:i+1]) for i in range(n_samples)], dim=0)
    dBdx_auto = torch.cat([jacobian(lambda x: sys.dyn(x)[1], x_test[i:i+1]) for i in range(n_samples)], dim=0)

    err_a = (dadx_auto.view(dadx_shape) - dadx).abs().sum() / n_samples
    err_B = (dBdx_auto.view(dBdx_shape) - dBdx).abs().sum() / n_samples
    assert err_a <= 1.e-5 and err_B <= 1.e-6

    ###################################################################################################################
    # Test dynamics gradient w.r.t. model parameter:
    dadp_shape = (n_samples, sys.n_parameter, sys.n_state)
    dBdp_shape = (n_samples, sys.n_parameter, sys.n_state, sys.n_act)

    dadp, dBdp = sys.grad_dyn_theta(x_test)

    dadp_auto = torch.cat([jacobian(lambda x: sys.dyn(x_test[i], dtheta=x)[0], dtheta) for i in range(n_samples)], dim=0)
    dBdp_auto = torch.cat([jacobian(lambda x: sys.dyn(x_test[i], dtheta=x)[1], dtheta) for i in range(n_samples)], dim=0)

    err_a = (dadp_auto.view(dadp_shape) - dadp).abs().sum() / n_samples
    err_B = (dBdp_auto.view(dBdp_shape) - dBdp).abs().sum() / n_samples
    assert err_a <= 2.e-4 and err_B <= 2.e-4