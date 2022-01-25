import time
import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

except ImportError:
    pass

from deep_differential_network.replay_memory import PyTorchReplayMemory, PyTorchTestMemory
from value_iteration.update_value_function import update_value_function, eval_memory
from value_iteration.value_function import ValueFunctionMixture
from value_iteration.sample_rollouts import sample_data
from value_iteration.utils import linspace, add_nan


def _sample_rollout(value_fun, hyper, system, run_config):
    t0 = time.perf_counter()

    # Perform the roll-out
    n_samples = int(hyper["test_minibatch"])
    n_steps = int(np.ceil(hyper['T'] * run_config['fs_return']))
    n_trajectories = int(np.ceil(n_samples / n_steps))
    mem_data, trajectory_data = sample_data(hyper['T'], n_trajectories, value_fun, hyper, system, run_config)
    R = trajectory_data[3].squeeze()
    R_mean = torch.mean(R).item()
    R_std = torch.std(R).item()

    # Compute distribution around last time-step
    x_n_avg = torch.mean(torch.abs(trajectory_data[0][-1, :, :, 0]), dim=0)
    x_n_std = torch.std(torch.abs(trajectory_data[0][-1, :, :, 0]), dim=0)
    x_n = zip(x_n_avg, x_n_std)

    t_comp = time.perf_counter() - t0
    return (mem_data, trajectory_data, (R_mean, R_std), t_comp, x_n)


def run_experiment(hyper):
    cuda = torch.cuda.is_available()
    alg_name = "rFVI" if hyper['robust'] else "cFVI"

    # Configuration for sampling trajectories from the system
    run_config = {"verbose": False, 'mode': 'init', 'fs_return': 10.,
                  'x_noise': hyper['x_noise'], 'u_noise': hyper['u_noise']}

    # Build the dynamical system:
    Q = np.array([float(x) for x in hyper['state_cost'].split(',')])
    R = np.array([float(x) for x in hyper['action_cost'].split(',')])
    system = hyper['system_class'](Q, R, cuda=cuda, **hyper)

    if system.name.split("_")[0] == "Pendulum":
        mat_shape = (150, 150)
    elif system.name.split("_")[0] == "Cartpole":
        mat_shape = (1, 150, 1, 150)
    else:
        raise ValueError


    # Compute Gamma s.t., the weight of the reward at time T is \eps, i.e., exp(-rho T) = gamma^(T/\Delta t) = eps:
    rho = -np.log(hyper['eps']) / hyper["T"]
    hyper["gamma"] = np.exp(-rho * hyper["dt"])

    # Construct Value Function:
    feature = torch.zeros(system.n_state)
    if system.wrap:
        feature[system.wrap_i] = 1.0

    val_fun_kwargs = {'feature': feature}
    value_fun = ValueFunctionMixture(system.n_state, **val_fun_kwargs, **hyper)

    if hyper['checkpoint'] is not None:
        data = torch.load(hyper['checkpoint'], map_location=torch.device('cpu'))

        hyper = data['hyper']
        hyper['n_iter'] = 0

        value_fun = ValueFunctionMixture(system.n_state, **val_fun_kwargs, **data['hyper'])
        value_fun.load_state_dict(data["state_dict"])

    value_fun = value_fun.cuda() if cuda else value_fun.cpu()

    print("\n\n################################################")
    print(f"{'Sample Data:':>25}", end="\t")
    t0_data = time.perf_counter()

    # Sample uniformly from the n-d hypercube
    n_samples = hyper["eval_minibatch"]
    x_lim = torch.from_numpy(system.x_lim).float() if isinstance(system.x_lim, np.ndarray) else system.x_lim

    x = torch.distributions.uniform.Uniform(-x_lim, x_lim).sample((n_samples, ))
    x = x.view(-1, system.n_state, 1).float().cuda() if cuda else x.view(-1, system.n_state, 1).float()

    ax, Bx, dadx, dBdx = system.dyn(x, gradient=True)
    mem_data = [x, ax, dadx, Bx, dBdx]

    # Sample the testing data:
    grid = [linspace(-x_lim[i].item(), x_lim[i].item(), mat_shape[i]) for i in range(system.n_state)]
    x_grid = torch.meshgrid(grid, indexing='ij')
    x_grid = torch.cat([x.reshape(-1, 1) for x in x_grid], dim=1).view(-1, system.n_state, 1)
    x_grid = x_grid.cuda() if cuda else x_grid

    ax_grid, Bx_grid, dadx_grid, dBdx_grid = system.dyn(x_grid, gradient=True)
    mem_grid_data = [x_grid.cpu(), ax_grid.cpu(), dadx_grid.cpu(), Bx_grid.cpu(), dBdx_grid.cpu()]

    # Memory Dimensions:
    mem_dim = ((system.n_state, 1),                                 # x
               (system.n_state, 1),                                 # a(x)
               (system.n_state, system.n_state),                    # da(x)/dx
               (system.n_state, system.n_act),                      # B(x)
               (system.n_state, system.n_state, system.n_act))      # dB(x)dx

    # Generate Replay Memory:
    mem = PyTorchReplayMemory(mem_data[0].shape[0], min(mem_data[0].shape[0], int(hyper["eval_minibatch"]/2)), mem_dim, cuda)
    mem_test = PyTorchTestMemory(x_grid.shape[0], min(mem_data[0].shape[0], hyper["n_minibatch"]), mem_dim, cuda)

    mem.add_samples(mem_data)
    mem_test.add_samples(mem_grid_data)

    print(f"{time.perf_counter() - t0_data:05.2e}s")

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    print("\n\n################################################")
    print("Learn the Value Function:")

    t0_training = time.perf_counter()
    step_i = -1

    try:
        for step_i in range(hyper["n_iter"]):
            t0_iter = time.perf_counter()

            # Compute the roll-out:
            args = (value_fun, hyper, system, run_config)
            mem_data, uniform_trajectory_data, R_uniform, t_rollout, x_last = _sample_rollout(*args)
            t_wait = 0.0

            # Update the Value Function:
            out = update_value_function(step_i, value_fun.cuda(), system, mem, hyper, None)
            value_fun, _, _ = out

            print("Rollout Computation:")
            str_x_n = "[" + ", ".join([f"{x[0]:.2f} \u00B1 {x[1]:.2f}" for x in x_last]) + "]"
            print(f"x_0 reward = {R_uniform[0]:+.2f} \u00B1 {1.96*R_uniform[1]:.2f}, x_N = {str_x_n} "
                  f"Comp Time = {t_rollout:.2f}s, Wait Time = {t_wait:.2f}s\n"
                  f"Iteration Time = {time.perf_counter() - t0_iter:.2f}s")
            print("")

            # Sample new data:
            if hyper['mode'] == 'RTDP':
                mem.add_samples(mem_data)

            # Save the model:
            if np.mod(step_i+1, 25) == 0:
                model_file = f"data/{alg_name}_{system.name}_step_{step_i+1:03d}.torch"
                torch.save({"epoch": step_i, "hyper": hyper, "state_dict": value_fun.state_dict()}, model_file)

    except KeyboardInterrupt as err:
        t_train = time.perf_counter() - t0_training
        print(f"Training stopped due to Keyboard Interrupt. Comp Time = {t_train:.2f}s\n")

    finally:
        # Training Time:
        t_train = time.perf_counter() - t0_training

        # Save the Model:
        if step_i > 10:
            model_file = f"data/{alg_name}_{system.name}.torch"
            torch.save({"epoch": step_i, "hyper": hyper, "state_dict": value_fun.state_dict()}, model_file)

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    print("\n################################################")
    print("Evaluate the Value Function:")
    t0 = time.perf_counter()
    n_test = 100

    # Compute the value-function error:
    value_fun = value_fun.cuda() if cuda else value_fun.cpu()
    x, u, V, dVdx, V0_tar, Vn_tar, V_diff = eval_memory(value_fun, hyper, mem_test, system)
    test_err_hjb = 1. / float(x_grid.shape[0]) * torch.sum(V_diff**2).numpy()

    # Evaluate expected reward with uniform initial state distribution:
    test_config = {"verbose": False, 'mode': 'init', 'fs_return': 100., 'x_noise': 0.0, 'u_noise': 0.0}
    _, uniform_trajectory_data = sample_data(hyper["T"], n_test, value_fun, hyper, system, test_config)
    R_uniform = uniform_trajectory_data[3].squeeze()
    R_uniform_mean = torch.mean(R_uniform).item()
    R_uniform_std = torch.std(R_uniform).item()

    # Evaluate expected reward with downward initial state distribution:
    test_config = {"verbose": False, 'mode': 'test', 'fs_return': 100., 'x_noise': 0.0, 'u_noise': 0.0}
    _, downward_trajectory_data = sample_data(hyper["T"], n_test, value_fun, hyper, system, test_config)
    R_downward = downward_trajectory_data[3].squeeze()
    R_downward_mean = torch.mean(R_downward).item()
    R_downward_std = torch.std(R_downward).item()

    print("\nPerformance:")
    print(f"        Value Function MSE = {test_err_hjb:.3e}")
    print(f" Expected Reward - Uniform = {R_uniform_mean:.2f} \u00B1 {1.96*R_uniform_std:.2f}")
    print(f"Expected Reward - Downward = {R_downward_mean:.2f} \u00B1 {1.96*R_downward_std:.2f}")
    print(f"             Training Time = {t_train/60.:.2f}min")
    print(f"                 Test Time = {time.perf_counter() - t0:.2f}s")

    print("\n################################################")

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    print("\n################################################")
    print("Plot the Value Function:")

    if hyper["plot"]:
        n_plot = 20
        scale = 1.0

        x_tra = downward_trajectory_data[0].cpu().numpy()
        u_tra = downward_trajectory_data[1].cpu().numpy()
        x_mat = x.reshape(*mat_shape, system.n_state)
        xx, xy = x_mat[:, :, 0], x_mat[:, :, 1]
        u_mat = u.reshape(*mat_shape, system.n_act)[:, :, 0]
        V_mat = V.reshape(mat_shape)

        x_lim = scale * torch.tensor([system.x_lim[0], system.x_lim[1]]).float()
        norm_V = cm.colors.Normalize(vmax=0.0, vmin=-torch.abs(V).max())

        u_max = torch.abs(u).max()
        norm_u = cm.colors.Normalize(vmax=u_max, vmin=-u_max)

        fig = plt.figure(figsize=(12, 4))
        fig.subplots_adjust(left=0.05, bottom=0.12, right=1.0, top=0.93, wspace=0.1, hspace=0.3)

        def format_space_ax(ax):
            y_ticks = [-7.5, 0.0, +7.5]
            x_ticks = [-np.pi / 1., -np.pi / 2., 0.0, np.pi / 2., np.pi]
            x_tick_label = [r"$\pm\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$\pm\pi$"]

            ax.set_xlabel(r"Angle [Rad]")
            ax.set_ylabel(r"Velocity [Rad/s]")
            ax.set_xlim(-x_lim[0], x_lim[0])
            ax.set_ylim(-x_lim[1], x_lim[1])
            ax.yaxis.set_label_coords(-0.09, 0.5)

            ax.set_yticks(y_ticks)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_label)
            return ax

        ax_val = format_space_ax(fig.add_subplot(1, 2, 1))
        ax_val.set_title(r"$V(x)$")

        plot_hyper = {'levels': 50, 'norm': norm_V, 'cmap': cm.get_cmap(cm.Spectral, 50)}
        cset = ax_val.contourf(xx, xy, V_mat, **plot_hyper)
        plt.colorbar(cset, ax=ax_val)

        ax_pi = format_space_ax(fig.add_subplot(1, 2, 2))
        ax_pi.set_title( r"$\pi(x)$")

        plot_hyper = {'levels': 50, 'norm': norm_u, 'cmap': cm.get_cmap(cm.Spectral, 50)}
        cset = ax_pi.contourf(xx, xy, u_mat, **plot_hyper)
        plt.colorbar(cset, ax=ax_pi)

        for i in range(n_plot):
            xi_tra = add_nan(x_tra[:, i, :, 0], system.wrap_i)
            ax_val.plot(xi_tra[:, 0], xi_tra[:, 1], c="k", alpha=0.25)
            ax_pi.plot(xi_tra[:, 0], xi_tra[:, 1], c="k", alpha=0.25)

        fig = plt.figure(figsize=(12, 5))
        fig.subplots_adjust(left=0.065, bottom=0.1, right=0.98, top=0.97, wspace=0.1, hspace=0.3)

        def format_time_ax(ax, i):
            v_ticks = [-7.5, 0.0, +7.5]
            x_ticks = [-np.pi / 1., -np.pi / 2., 0.0, np.pi / 2., np.pi]
            x_tick_label = [r"$\pm\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$\pm\pi$"]

            if i == 1:
                ax.set_ylabel(r"Angle [Rad]")
                ax.set_ylim(-x_lim[0], x_lim[0])
                ax.set_yticks(x_ticks)
                ax.set_yticklabels(x_tick_label)

            elif i == 2:
                ax.set_ylabel(r"Velocity [Rad/s]")
                ax.set_ylim(-x_lim[1], x_lim[1])
                ax.set_yticks(v_ticks)

            elif i == 3:
                ax.set_ylabel(r"Torque [Nm]")
                ax.set_xlabel("Time [s]")
                ax.set_ylim(-system.u_lim[0], system.u_lim[0])

            else:
                raise ValueError

            ax.yaxis.set_label_coords(-0.045, 0.5)
            ax.set_xlim(0, hyper["T"])
            return ax

        ax_xp = format_time_ax(fig.add_subplot(3, 1, 1), 1)
        ax_xv = format_time_ax(fig.add_subplot(3, 1, 2), 2)
        ax_u = format_time_ax(fig.add_subplot(3, 1, 3), 3)

        t = np.linspace(0, hyper["T"], x_tra.shape[0])
        for i in range(n_plot):
            xi_tra = add_nan(np.concatenate((x_tra[:, i, :, 0], t[:, np.newaxis]), axis=-1), system.wrap_i)

            ax_xp.plot(xi_tra[:, -1], xi_tra[:, 0], c="k", alpha=0.25)
            ax_xv.plot(xi_tra[:, -1], xi_tra[:, 1], c="k", alpha=0.25)
            ax_u.plot(t, u_tra[:, i, 0, 0], c="k", alpha=0.25)

        plt.show()