import numpy as np
import torch

from deep_differential_network.utils import jacobian
from value_iteration.cost_functions import ArcTangent, SineQuadraticCost
CUDA_AVAILABLE = torch.cuda.is_available()


class BaseSystem:
    def __init__(self):
        self.n_state = 0
        self.n_act = 0
        self.x_lim = []

    def check_dynamics(self, n_samples=10):
        # Checking Gradients:
        to_x_test = torch.distributions.uniform.Uniform(-self.x_lim, self.x_lim).sample((n_samples,))
        to_x_test = to_x_test.view(-1, self.n_state, 1).float().to(self.theta.device)
        np_x_test = to_x_test.cpu().numpy()

        np_a, np_B, np_dadx, np_dBdx = self.dyn(np_x_test, gradient=True)
        to_a, to_B, to_dadx, to_dBdx = [x.cpu().numpy() for x in self.dyn(to_x_test, gradient=True)]

        assert np.allclose(to_a, np_a, atol=1.e-5)
        assert np.allclose(to_B, np_B, atol=1.e-5)
        assert np.allclose(to_dadx, np_dadx, atol=1.e-5)
        assert np.allclose(to_dBdx, np_dBdx, atol=1.e-5)

        grad_auto_dadx = jacobian(lambda x: self.dyn(x)[-2], to_x_test).view(-1, self.n_state, self.n_state).cpu().numpy()
        grad_auto_dBdx = jacobian(lambda x: self.dyn(x)[-1], to_x_test).view(-1, self.n_state, self.n_state, self.n_act).cpu().numpy()

        assert np.allclose(to_dBdx, grad_auto_dBdx, atol=1.e-3)
        assert np.allclose(to_dadx, grad_auto_dadx, atol=1.e-3)

    def dyn(self, x):
        raise AttributeError

    def grad_dyn(self, x):
        raise AttributeError


class Pendulum(BaseSystem):
    name = "Pendulum"
    labels = ('theta', 'theta_dot')

    def __init__(self, cuda=False, **kwargs):
        super(Pendulum, self).__init__()
        device = torch.device('cuda') if cuda else torch.device('cpu')

        # Define Duration:
        self.T = kwargs.get("T", 10.0)
        self.dt = kwargs.get("dt", 1./500.)

        # Define the System:
        self.n_state = 2
        self.n_act = 1
        self.n_joint = 1
        self.n_parameter = 2

        # Continuous Joints:
        # Right now only one continuous joint is supported
        self.wrap, self.wrap_i = True, 0

        # State Constraints:
        # theta = 0, means the pendulum is pointing upward
        self.x_target = torch.tensor([0.0, 0.0])
        self.x_start = torch.tensor([np.pi, 0.01])
        self.x_start_var = torch.tensor([1.e-3, 1.e-6])
        self.x_lim = torch.tensor([np.pi, 8.])
        self.x_init = torch.tensor([np.pi, 0.01])
        self.u_lim = torch.tensor([200., ])

        # Define Dynamics:
        self.gravity = -9.81

        # theta = [mass, length]
        self.theta_min = torch.tensor([0.5, 0.5]).to(device).view(1, self.n_parameter, 1)
        self.theta = torch.tensor([1., 1.]).to(device).view(1, self.n_parameter, 1)
        self.theta_max = torch.tensor([2., 2.]).to(device).view(1, self.n_parameter, 1)

        # LQR Baseline:
        out = self.dyn(self.x_target.numpy(), gradient=True)
        self.A = out[2].reshape((1, self.n_state, self.n_state)).transpose((0, 2, 1))
        self.B = out[1].reshape((1, self.n_state, self.n_act))

        # Test dynamics:
        self.check_dynamics()

        self.device = None
        Pendulum.cuda(self) if cuda else Pendulum.cpu(self)

    def dyn(self, x, dtheta=None, gradient=False):
        cat = torch.cat

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x).to(self.theta.device) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        n_samples = x.shape[0]

        # Update the dynamics parameters with disturbance:
        if dtheta is not None:
            dtheta = torch.from_numpy(dtheta).float() if isinstance(dtheta, np.ndarray) else dtheta
            dtheta = dtheta.view(-1, self.n_parameter, 1)
            assert dtheta.shape[0] in (1, n_samples)

            theta = self.theta + dtheta
            theta = torch.min(torch.max(theta, self.theta_min), self.theta_max)

        else:
            theta = self.theta
            theta = theta

        mgl = theta[:, 0] * theta[:, 1]/2. * self.gravity
        mL2 = theta[:, 0] * theta[:, 1]**2

        a = torch.cat([x[:, 1], -3. / mL2 * mgl * torch.sin(x[:, 0])], dim=1).view(-1, self.n_state, 1)
        B = torch.zeros(x.shape[0], self.n_state, self.n_act).to(self.theta.device)
        B[:, 1] = 3. / mL2

        assert a.shape == (n_samples, self.n_state, 1)
        assert B.shape == (n_samples, self.n_state, self.n_act)
        out = (a, B)

        if gradient:
            zeros, ones = torch.zeros_like(x[:, 1]), torch.ones_like(x[:, 1])

            dadx = cat([cat((zeros, -3. / mL2 * mgl * torch.cos(x[:, 0])), dim=1).unsqueeze(-1),
                        cat((ones, zeros), dim=1).unsqueeze(-1)], dim=1).view(-1, self.n_state, self.n_state)

            dBdx = torch.zeros((x.shape[0], self.n_state, self.n_state, self.n_act), dtype=x.dtype, device=x.device)

            assert dadx.shape == (n_samples, self.n_state, self.n_state)
            assert dBdx.shape == (n_samples, self.n_state, self.n_state, self.n_act)
            out = (a, B, dadx, dBdx)

        if is_numpy:
            out = [array.cpu().detach().numpy() for array in out]

        return out

    def grad_dyn_theta(self, x):
        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        n_samples = x.shape[0]

        dadth = torch.zeros(n_samples, self.n_parameter, self.n_state).to(x.device)
        dadth[:, 1, 1:2] = 1.5 * self.gravity / self.theta[:, 1]**2 * torch.sin(x[:, 0])

        dBdth = torch.zeros(n_samples, self.n_parameter, self.n_state, self.n_act).to(x.device)
        dBdth[:, 0, 1] = -3. / (self.theta[:, 0]**2 * self.theta[:, 1]**2)
        dBdth[:, 1, 1] = -6. / (self.theta[:, 0] * self.theta[:, 1]**3)
        out = dadth, dBdth

        if is_numpy:
            out = [array.numpy() for array in out]

        return out

    def cuda(self, device=None):
        self.theta_min = self.theta_min.cuda(device=device)
        self.theta = self.theta.cuda(device=device)
        self.theta_max = self.theta_max.cuda(device=device)

        self.u_lim = self.u_lim.cuda(device=device)
        self.x_lim = self.x_lim.cuda(device=device)
        self.device = self.theta.device
        return self

    def cpu(self):
        self.theta_min = self.theta_min.cpu()
        self.theta = self.theta.cpu()
        self.theta_max = self.theta_max.cpu()

        self.u_lim = self.u_lim.cpu()
        self.x_lim = self.x_lim.cpu()
        self.device = self.theta.device
        return self


class PendulumLogCos(Pendulum):
    name = "Pendulum_LogCosCost"

    def __init__(self, Q, R, cuda=False, **kwargs):

        # Create the dynamics:
        super(PendulumLogCos, self).__init__(cuda=cuda, **kwargs)
        self.u_lim = torch.tensor([2.5, ])

        # Change the Parameters:
        # self.Q = np.diag(np.array([1.e+0, 1.0e-1]))
        # self.R = np.array([[5.e-1]])

        assert Q.size == self.n_state and np.all(Q > 0.0)
        self.Q = np.diag(Q).reshape((self.n_state, self.n_state))

        assert R.size == self.n_act and np.all(R > 0.0)
        self.R = np.diag(R).reshape((self.n_act, self.n_act))

        # Create the Reward Function:
        self.q = SineQuadraticCost(self.Q, np.array([1.0, 0.0]), cuda=cuda)

        # Determine beta s.t. the curvature at u = 0 is identical to 2R
        beta = (4. * self.u_lim[0] ** 2 / np.pi * self.R)[0, 0].item()
        self.r = ArcTangent(alpha=self.u_lim.numpy()[0], beta=beta)

    def rwd(self, x, u):
        return self.q(x) + self.r(u)

    def cuda(self, device=None):
        super(PendulumLogCos, self).cuda(device=device)
        self.q.cuda(device=device)
        return self

    def cpu(self):
        super(PendulumLogCos, self).cpu()
        self.q.cpu()
        return self


if __name__ == "__main__":
    from deep_differential_network.utils import jacobian

    # GPU vs. CPU:
    cuda = True

    # Seed the test:
    seed = 20
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create system:
    sys = Pendulum()

    n_samples = 10
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
    assert err_a <= 1.e-5 and err_B <= 1.e-6


