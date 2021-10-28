import time
import torch
import numpy as np

from torch.optim import Adam

from deep_differential_network.replay_memory import PyTorchReplayMemory
from value_iteration.utils import analyze_error, error_statistics_string, negative_definite, evaluate
from value_iteration.value_function import ValueFunctionMixture


def print_loss(epoch, stats_loss, stats_V, t_comp):
    print(f"{epoch + 1:03d}) ", end="")
    print(f"T = {t_comp:04.1f}s", end=" ")
    print(f"J_cost = {error_statistics_string(stats_loss)}", end=" ")
    print(f"Error |V| = {error_statistics_string(stats_V)}", end=" ")
    print("")


def policy(x, B, r, val_fun):
    Vi, dVidx = negative_definite(*evaluate(val_fun, x))
    dVidx = dVidx.transpose(dim0=1, dim1=2)

    BT_dVidx = torch.matmul(B.transpose(dim0=1, dim1=2), dVidx)
    u = r.grad_convex_conjugate(BT_dVidx)
    return Vi, dVidx, u, None


def norm(z):
    return z / (torch.sum(z**2, dim=1, keepdim=True).sqrt() + 1.e-6)


def bounds(z, mu, range):
    return range * torch.sign(z) + mu


def update_value_function(step_i, value_fun_tar, system, mem_train, hyper, writer, verbose=True):

    # Compute target value function:
    t0_target = time.perf_counter()
    with torch.no_grad():
        x_tar, V_tar, dVdx_tar, V_diff = [], [], [], []

        # Compute trace weights:
        trace_l = hyper["trace_lambda"]
        trace_n = np.ceil(np.log(hyper["trace_weight_n"] / (1. - trace_l)) / np.log(trace_l)).astype(int)
        w_lambda = ((1. - trace_l) * trace_l ** torch.arange(0., trace_n, 1.)).view(1, -1, 1).to(value_fun_tar.device)
        w_lambda[0, -1, 0] = trace_l ** (trace_n - 1)

        x_lim = torch.from_numpy(system.x_lim).float() if isinstance(system.x_lim, np.ndarray) else system.x_lim
        x_lim = x_lim.to(value_fun_tar.device).view(1, system.n_state, 1)

        for n_batch, batch_i in enumerate(mem_train):
            V0_tar, V0_diff, dV0dx_tar = [], [], []

            # Unpack the batch:
            x0, a0, da0dx, B0, dB0dx = batch_i

            # Compute the value function
            V0, dV0dx, u0_star, du0dx_star = policy(x0, B0, system.r, value_fun_tar)

            xj = x0
            aj, dajdx, Bj, dBjdx = a0, da0dx, B0, dB0dx
            dVjdx, uj_star, dujdx_star = dV0dx, u0_star, du0dx_star

            # Compute uniform scaling of adversarial noise:
            noise_shape = (trace_n, xj.shape[0])
            mu_x, mu_u = torch.zeros(system.n_state), torch.zeros(system.n_act)
            eye_x, eye_u = torch.eye(system.n_state), torch.eye(system.n_act)

            # the integration (i.e., cumsum) is implicit in the main loop!
            dist_x_noise = torch.distributions.multivariate_normal.MultivariateNormal(mu_x, covariance_matrix=eye_x)
            x_noise = dist_x_noise.sample(noise_shape).to(xj.device)
            x_noise = np.sqrt(hyper["dt"]) * hyper["xi_x_alpha"] / 1.96 * x_noise
            xi_x_scale = torch.sqrt(torch.sum(x_noise**2, dim=2)).view(trace_n, xj.shape[0], 1, 1)

            dist_u_noise = torch.distributions.multivariate_normal.MultivariateNormal(mu_u, covariance_matrix=eye_u)
            u_noise = dist_u_noise.sample(noise_shape).to(xj.device)
            u_noise = torch.cumsum(np.sqrt(hyper["dt"]) * hyper["xi_u_alpha"] / 1.96 * u_noise, dim=0)
            xi_u_scale = torch.sqrt(torch.sum(u_noise**2, dim=2)).view(trace_n, xj.shape[0], 1, 1)

            dist_o_noise = torch.distributions.multivariate_normal.MultivariateNormal(mu_x, covariance_matrix=eye_x)
            o_noise = dist_o_noise.sample(noise_shape).to(xj.device)
            o_noise = torch.cumsum(np.sqrt(hyper["dt"]) * hyper["xi_o_alpha"] / 1.96 * o_noise, dim=0)
            xi_o_scale = torch.sqrt(torch.sum(o_noise**2, dim=2)).view(trace_n, xj.shape[0], 1, 1)

            min_theta, max_theta = - hyper["xi_m_alpha"] * system.theta, hyper["xi_m_alpha"] * system.theta
            xi_M_range, xi_M_mu = (max_theta - min_theta)/2., (max_theta + min_theta)/2.

            r,  drdx = 0.0, 0.0
            for n in range(trace_n):

                # Compute the reward:
                r_j = -hyper['dt'] * (system.q(xj) + system.r(uj_star))
                r = r + hyper["gamma"] ** n * r_j

                # Compute adversarial state-noise:
                z_x = -dVjdx
                xi_x = float(hyper["robust"]) * norm(z_x) * xi_x_scale[n]

                # Compute adversarial action-noise:
                z_u = -torch.matmul(Bj.transpose(dim0=1, dim1=2), dVjdx)
                xi_u = float(hyper["robust"]) * norm(z_u) * xi_u_scale[n]

                # Compute adversarial observation-noise:
                z_o = -torch.matmul(torch.matmul(dBjdx, uj_star.unsqueeze(-1)).squeeze(-1) + dajdx, dVjdx)
                xi_o = float(hyper["robust"]) * norm(z_o) * xi_o_scale[n]

                # Compute adversarial parameter-noise:
                dajdp, dBjdp = system.grad_dyn_theta(xj)
                z_m = -torch.matmul((torch.matmul(dBjdp, uj_star.unsqueeze(-1)).squeeze(-1) + dajdp).squeeze(-1), dVjdx)
                xi_m = float(hyper["robust"]) * bounds(z_m, xi_M_mu, xi_M_range)

                # Compute next state:
                aj_xi, Bj_xi = system.dyn(xj + xi_o, dtheta=xi_m)
                xdj = aj_xi + torch.matmul(Bj_xi, uj_star + xi_u)
                xn = xj + hyper["dt"] * xdj + xi_x

                # Compute wrap-around for continuous joints
                if system.wrap:
                    xn[:, system.wrap_i] = torch.remainder(xn[:, system.wrap_i] + np.pi, 2 * np.pi) - np.pi

                # Clip to state space:
                xn = torch.min(torch.max(xn, -x_lim), x_lim)

                # Compute dynamics at the next step:
                an, Bn, dandx, dBndx = system.dyn(xn, gradient=True)

                # Compute the value function of the next state:
                Vn, dVndx, un_star, dundx_star = policy(xn, Bn, system.r, value_fun_tar)

                # Compute the target value function:
                V0_tar.append(torch.clamp(r + hyper['gamma'] ** (n+1) * Vn, max=0.0))

                xj, Vj, dVjdx, uj_star, dujdx_star, dajdx, Bj, dBjdx = xn, Vn, dVndx, un_star, dundx_star, dandx, Bn, dBndx

            # Compute Exponential Average of the n-steps:
            Vn = torch.sum(w_lambda * torch.cat(V0_tar, dim=1), dim=1, keepdim=True)

            # Compute the Value function target:
            delta_V = Vn - V0
            Vi_tar = Vn

            # Update Buffers:
            x_tar.append(x0)
            V_tar.append(Vi_tar)
            V_diff.append(delta_V)

        # Stack results:
        x_tar, V_tar, V_diff = torch.cat(x_tar), torch.cat(V_tar), torch.cat(V_diff)
        assert torch.all(V_tar <= 0.0)

        # Compute current performance:
        stats_V_diff = analyze_error(torch.abs(V_diff))

        t_target = time.perf_counter() - t0_target
        if verbose:
            print(f"Epoch {step_i:02d}) "
                  f"|\u0394V| = {error_statistics_string(stats_V_diff)}, "
                  f"Comp Time = {t_target:.2f}s, "
                  f"\u03BB = {trace_l:.3f}, N = {trace_n}")

    # Generate Training Memory:
    mem_dim = ((x_tar.shape[1], x_tar.shape[2]),  (V_tar.shape[1], V_tar.shape[2]),)
    mem = PyTorchReplayMemory(x_tar.shape[0], hyper["n_minibatch"], mem_dim, x_tar.is_cuda)
    mem.add_samples([x_tar, V_tar])

    # Construct Value Function:
    feature = torch.zeros(system.n_state)
    if system.wrap:
        feature[system.wrap_i] = 1.0

    val_fun_kwargs = {'feature': feature}
    value_fun = ValueFunctionMixture(system.n_state, **val_fun_kwargs, **hyper)
    value_fun = value_fun.cuda() if x_tar.is_cuda else value_fun
    value_fun.load_state_dict(value_fun_tar.state_dict())

    optimizer = Adam(value_fun.net.parameters(),
                     lr=hyper["lr_SGD"],
                     weight_decay=hyper["weight_decay"],
                     amsgrad=True)

    # Update Value function to minimize the error between value function and value target:
    t0_start, epoch_i, t_opt = time.perf_counter(), 0, 0.0
    while epoch_i < hyper["max_epoch"]:
        loss, loss_V = [], []

        for n_batch, batch_i in enumerate(mem):
            xi, Vi_tar = batch_i
            optimizer.zero_grad()

            V_hat, dVdx_hat = value_fun(xi, fit=True)
            err_V = torch.mean(torch.abs(V_hat - Vi_tar.unsqueeze(0)) ** hyper['exp'], dim=0)

            J_cost = torch.mean(err_V)
            J_cost.backward()
            optimizer.step()

            loss.append(J_cost.detach())
            loss_V.append(torch.mean(err_V.detach()))

        # Compute average loss statistics
        if verbose and (epoch_i == 0 or np.mod(epoch_i+1, 5) == 0):
            loss, loss_V = torch.stack(loss), torch.stack(loss_V)
            stats_loss, stats_loss_V = analyze_error(loss), analyze_error(loss_V)
            print_loss(epoch_i, stats_loss, stats_loss_V, time.perf_counter() - t0_start)

        epoch_i += 1

    if verbose:
        print("")

    return value_fun, stats_loss, stats_loss_V


def eval_memory(val_fun, hyper, mem, system):
    x_lim = system.x_lim.float().to(val_fun.device).view(1, system.n_state, 1)

    # Compute target value function:
    with torch.no_grad():
        x, u, V, dVdx, d2Vd2x, Vn_tar, V0_tar, V_diff = [], [], [], [], [], [], [], []

        # Compute trace weights:
        trace_l = hyper["trace_lambda"]
        trace_n = np.ceil(np.log(hyper["trace_weight_n"] / (1. - trace_l)) / np.log(trace_l)).astype(int)
        w_lambda = ((1. - trace_l) * trace_l ** torch.arange(0., trace_n, 1.)).view(1, -1, 1).to(val_fun.device)
        w_lambda[0, -1, 0] = trace_l ** (trace_n - 1)

        for n_batch, batch_i in enumerate(mem):
            Vn, Vn_diff = [], []
            xi, ai, daidx, Bi, dBidx = batch_i

            # Compute the value function
            Vi, dVidx = negative_definite(*evaluate(val_fun, xi))
            dVidx = dVidx.transpose(dim0=1, dim1=2)

            # Compute the optimal action:
            BT_dVidx = torch.matmul(Bi.transpose(dim0=1, dim1=2), dVidx)
            ui_star = system.r.grad_convex_conjugate(BT_dVidx)

            aj, Bj = ai, Bi
            dVjdx = dVidx
            xj = xi
            r, u_star = 0.0, 0.0

            for n in range(trace_n):
                # Compute the optimal action:
                BT_dVjdx = torch.matmul(Bj.transpose(dim0=1, dim1=2), dVjdx)
                u_star = system.r.grad_convex_conjugate(BT_dVjdx)

                # Compute the reward:
                r_j = -hyper['dt'] * (system.q(xj) + system.r(u_star))
                r = r + hyper["gamma"] ** n * r_j

                # Compute next state:
                xdj = aj + torch.matmul(Bj, u_star)
                xj = xj + hyper["dt"] * xdj

                # Compute wrap-around for continuous joints
                if system.wrap:
                    xj[:, system.wrap_i] = torch.remainder(xj[:, system.wrap_i] + np.pi, 2 * np.pi) - np.pi

                # Clip to state-space:
                xj = torch.min(torch.max(xj, -x_lim), x_lim)

                # Compute the value function of the next state:
                Vj, dVjdx = negative_definite(*evaluate(val_fun, xj))

                dVjdx = dVjdx.transpose(dim0=1, dim1=2)

                # Compute new dynamics:
                aj, Bj = system.dyn(xj)

                # Compute the value function difference:
                Vn.append(r + hyper['gamma'] * Vj)
                Vn_diff.append(Vn[-1] - Vi)

            # Compute Exponential Average of the n-steps:
            Vn = torch.cat(Vn, dim=1)
            Vn_lambda = torch.sum(w_lambda * Vn, dim=1, keepdim=True)
            Vn_diff = torch.sum(w_lambda * torch.cat(Vn_diff, dim=1), dim=1, keepdim=True)

            # Compute the Value function target:
            delta_V = Vn_lambda - Vi
            V0i_tar = Vn[:, 0].view(-1, 1, 1)
            Vni_tar = Vn_lambda

            # Update Buffers:
            x.append(xi)
            u.append(ui_star)
            V.append(Vi)
            dVdx.append(dVidx)
            V0_tar.append(V0i_tar)
            Vn_tar.append(Vni_tar)
            V_diff.append(delta_V)

        # Stack results:
        x = torch.cat(x).cpu()
        u = torch.cat(u).cpu()
        V = torch.cat(V).cpu()
        dVdx = torch.cat(dVdx).cpu()
        V0_tar = torch.cat(V0_tar).cpu()
        Vn_tar = torch.cat(Vn_tar).cpu()
        V_diff = torch.cat(V_diff).cpu()

    return x, u, V, dVdx, V0_tar, Vn_tar, V_diff
