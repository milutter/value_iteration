import torch
import numpy as np

from value_iteration.utils import trange


class ValueFunPolicy:
    def __init__(self, sys, val_fun):
        self.v = val_fun
        self.sys = sys

    def __call__(self, x, B):
        if B is None:
            _, B = self.sys.dyn(x)

        Vi, dVidx = self.v(x)  # negative_definite(*val_fun(x[-1]))
        dVidx = dVidx.transpose(dim0=1, dim1=2)

        BT_dVdx = torch.matmul(B.transpose(dim0=1, dim1=2), dVidx)
        ui = self.sys.r.grad_convex_conjugate(BT_dVdx)
        return Vi, dVidx, ui


def sample_data(T, n_seeds, val_fun, hyper, sys, config):
    n_seeds = int(n_seeds)

    with torch.no_grad():
        dt_ctrl, f_ctrl = hyper.get('dt_ctrl', hyper['dt']), 1. / hyper.get('dt_ctrl', hyper['dt'])
        fs_trials = [250., 500., ] # [250., 300., 500.]

        for fs in fs_trials:
            dt = 1. / fs
            n_steps = int(T * fs)
            n_sim_step_per_ctrl = int(fs / f_ctrl)

            if float(n_sim_step_per_ctrl) * f_ctrl == fs:
                break

        assert float(n_sim_step_per_ctrl) * f_ctrl == fs

        verbose = config.get("verbose", False)
        mode = config.get("mode", "init")
        downsample = int(fs / config.get("fs_return", 50.))

        # Actions are determined using the value-function policy:
        pi = ValueFunPolicy(sys, val_fun)
        device = val_fun.device

        x_lim = sys.x_lim.float().to(device).view(1, sys.n_state, 1)
        u_lim = sys.u_lim.float().to(device).view(1, sys.n_act, 1)

        mu_x, mu_u = torch.zeros(sys.n_state), torch.zeros(sys.n_act)
        eye_x, eye_u = torch.eye(sys.n_state), torch.eye(sys.n_act)

        # State Exploration Noise via a Wiener Process:
        # The state-noise is implicitly integrated in execution loop
        xi_x_alpha = config.get('x_noise', 0.0) * sys.x_lim.view(1, 1, sys.n_state, 1)
        dist_x_noise = torch.distributions.multivariate_normal.MultivariateNormal(mu_x, covariance_matrix=eye_x)
        n_x = dist_x_noise.sample((n_steps, n_seeds)).float().to(device).view(n_steps, n_seeds, sys.n_state, 1)
        n_x = xi_x_alpha.to(n_x.device) / 1.96 * np.sqrt(dt) * n_x

        # Action Exploration Noise via an Ornstein–Uhlenbeck process:
        # Explicit integration to represent a Wiener process
        theta = 0.5
        t = torch.arange(0.0, T, dt)
        exp_minus = torch.exp(-theta * t).view(-1, 1, 1, 1).to(n_x.device)
        exp_plus = torch.exp(theta * t).view(-1, 1, 1, 1).to(n_x.device)

        xi_u_alpha = config.get('u_noise', 0.0) * sys.u_lim
        dist_u_noise = torch.distributions.multivariate_normal.MultivariateNormal(mu_u, covariance_matrix=eye_u)
        n_u = dist_u_noise.sample((n_steps, n_seeds)).float().to(device).view(n_steps, n_seeds, sys.n_act, 1)
        n_u = xi_u_alpha.to(n_u.device) / 1.96 * exp_minus * torch.cumsum(exp_plus * np.sqrt(dt) * n_u, dim=0)

        x, a, V, dVdx, dVdt, B, u, r = [], [], [], [], [], [], [], []

        # Sample initial seeds:
        if mode == "test":
            x_start = sys.x_start.float().view(sys.n_state)
            sigma = torch.diag(sys.x_start_var.float()).view(sys.n_state, sys.n_state)
            dist_x = torch.distributions.multivariate_normal.MultivariateNormal(x_start, sigma)
            x0 = dist_x.sample((n_seeds,)).view(-1, sys.n_state, 1).float().to(device)

        elif mode == "init":
            x_init = sys.x_init.float().view(1, sys.n_state, 1)
            dist_x = torch.distributions.uniform.Uniform(-x_init, x_init)
            x0 = dist_x.sample((n_seeds,)).view(-1, sys.n_state, 1).to(device)

        else:
            raise ValueError

        if sys.wrap:
            x0[:, sys.wrap_i] = torch.remainder(x0[:, sys.wrap_i] + np.pi, 2 * np.pi) - np.pi

        x.append(torch.min(torch.max(x0, -x_lim), x_lim))

        t = 0.0
        for i in trange(int(n_steps), prefix=f"Sample Datapoints", ncols=100, verbose=verbose):

            # Compute dynamics:
            a_t, B_t = sys.dyn(x[-1])
            a.append(a_t)
            B.append(B_t)

            # Compute optimal action:
            if np.mod(i, n_sim_step_per_ctrl) == 0:
                Vi, dVidx, ui = pi(x[-1], B[-1])

            else:
                Vi, dVidx, ui = V[-1].clone(), dVdx[-1].clone(), u[-1].clone()

            V.append(Vi)
            dVdx.append(dVidx)
            u.append(torch.min(torch.max(ui, -u_lim), u_lim))

            # Compute reward:
            r.append(-dt * sys.rwd(x[-1], u[-1]))

            # Compute next step:
            xd = (a[-1] + torch.matmul(B[-1], u[-1] + n_u[i])).view(-1, sys.n_state, 1)
            xn = x[-1] + dt * xd + n_x[i]

            # Compute dVdt
            dVdt.append(torch.matmul(dVidx.transpose(dim0=1, dim1=2), xd))

            if sys.wrap:
                xn[:, sys.wrap_i] = torch.remainder(xn[:, sys.wrap_i] + np.pi, 2 * np.pi) - np.pi

            x.append(torch.min(torch.max(xn, -x_lim), x_lim))
            t += dt

        # Compute Value function for the last step:
        x_clip = torch.min(torch.max(x[-1], -x_lim), x_lim)
        Vi, dVidx, _ = pi(x_clip, B[-1])
        V.append(Vi)
        dVdx.append(dVidx)

        x = torch.stack(x, dim=0)[:-1:downsample]
        V = torch.stack(V, dim=0)[:-1:downsample]
        dVdx = torch.stack(dVdx, dim=0)[:-1:downsample]
        dVdt = torch.stack(dVdt, dim=0)[::downsample]
        u = torch.stack(u, dim=0)[::downsample]

        x_flat = torch.flatten(x, 0, 1)
        a_flat, B_flat, dadx_flat, dBdx_flat = sys.dyn(x_flat, gradient=True)

        # Rewards:
        r = torch.stack(r, dim=0)
        R = torch.sum(r, dim=0).squeeze(0)
        r = r[::downsample] * downsample

        out_tra = [xi for xi in [x, u, r, R, V, dVdx, dVdt]]
        out_flat = (x_flat, a_flat, dadx_flat, B_flat, dBdx_flat)
        return out_flat, out_tra
