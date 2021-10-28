import time
import torch
import numpy as np

try:
    import matplotlib as mpl
    mpl.use("Qt5Agg")

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

except ImportError:
    pass

from deep_differential_network.replay_memory import PyTorchTestMemory
from value_iteration.value_function import ValueFunctionMixture
from value_iteration.update_value_function import eval_memory
from value_iteration.sample_rollouts import sample_data
from value_iteration.utils import linspace, add_nan


if __name__ == "__main__":
    n_test = 50
    scale = 1.0
    duration = 5.0
    mat_shape = (150, 150)
    n_plot = min(50, n_test)
    cuda = torch.cuda.is_available()

    cfvi_data = torch.load('data/cFVI.torch', map_location=torch.device('cpu'))
    cfvi_hyper = cfvi_data['hyper']
    cfvi_weights = cfvi_data["state_dict"]

    rfvi_data = torch.load('data/rFVI.torch', map_location=torch.device('cpu'))
    rfvi_hyper = rfvi_data['hyper']
    rfvi_weights = rfvi_data["state_dict"]

    # Build the dynamical system:
    Q = np.array([float(x) for x in cfvi_hyper['state_cost'].split(',')])
    R = np.array([float(x) for x in cfvi_hyper['action_cost'].split(',')])
    system = cfvi_hyper['system_class'](Q, R, cuda=cuda, **cfvi_hyper)

    # Construct Value Function:
    feature = torch.zeros(system.n_state)
    if system.wrap:
        feature[system.wrap_i] = 1.0

    val_fun_kwargs = {'feature': feature}
    cfvi_value_fun = ValueFunctionMixture(system.n_state, **val_fun_kwargs, **cfvi_hyper)
    cfvi_value_fun.load_state_dict(cfvi_weights)
    cfvi_value_fun = cfvi_value_fun.cuda() if cuda else cfvi_value_fun.cpu()

    rfvi_value_fun = ValueFunctionMixture(system.n_state, **val_fun_kwargs, **rfvi_hyper)
    rfvi_value_fun.load_state_dict(rfvi_weights)
    rfvi_value_fun = rfvi_value_fun.cuda() if cuda else rfvi_value_fun.cpu()

    # Sample the testing data:
    x_lim = torch.from_numpy(system.x_lim).float() if isinstance(system.x_lim, np.ndarray) else system.x_lim
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

    mem_test = PyTorchTestMemory(x_grid.shape[0], min(mem_grid_data[0].shape[0], cfvi_hyper["n_minibatch"]), mem_dim, cuda)
    mem_test.add_samples(mem_grid_data)

    print("\n################################################")
    print("Evaluate the Value Functions:")
    t0 = time.perf_counter()

    # Compute the value-function error:
    cfvi_x, cfvi_u, cfvi_V, _, _, _, _ = eval_memory(cfvi_value_fun, cfvi_hyper, mem_test, system)
    rfvi_x, rfvi_u, rfvi_V, _, _, _, _ = eval_memory(rfvi_value_fun, rfvi_hyper, mem_test, system)

    # Evaluate expected reward with uniform initial state distribution:
    uniform_test_config = {"verbose": False, 'mode': 'init', 'fs_return': 100., 'x_noise': 0.0, 'u_noise': 0.0}
    _, cfvi_uniform_trajectories = sample_data(duration, n_test, cfvi_value_fun, cfvi_hyper, system, uniform_test_config)
    cfvi_R_uniform = cfvi_uniform_trajectories[3].squeeze()
    cfvi_R_uniform_mean = torch.mean(cfvi_R_uniform).item()
    cfvi_R_uniform_std = torch.std(cfvi_R_uniform).item()

    _, rfvi_uniform_trajectories = sample_data(duration, n_test, rfvi_value_fun, rfvi_hyper, system, uniform_test_config)
    rfvi_R_uniform = rfvi_uniform_trajectories[3].squeeze()
    rfvi_R_uniform_mean = torch.mean(rfvi_R_uniform).item()
    rfvi_R_uniform_std = torch.std(rfvi_R_uniform).item()

    # Evaluate expected reward with downward initial state distribution:
    downward_test_config = {"verbose": False, 'mode': 'test', 'fs_return': 100., 'x_noise': 0.0, 'u_noise': 0.0}
    _, cfvi_downward_trajectories = sample_data(duration, n_test, cfvi_value_fun, cfvi_hyper, system, downward_test_config)
    cfvi_R_downward = cfvi_downward_trajectories[3].squeeze()
    cfvi_R_downward_mean = torch.mean(cfvi_R_downward).item()
    cfvi_R_downward_std = torch.std(cfvi_R_downward).item()

    _, rfvi_downward_trajectories = sample_data(duration, n_test, rfvi_value_fun, rfvi_hyper, system, downward_test_config)
    rfvi_R_downward = rfvi_downward_trajectories[3].squeeze()
    rfvi_R_downward_mean = torch.mean(rfvi_R_downward).item()
    rfvi_R_downward_std = torch.std(rfvi_R_downward).item()

    print("\nPerformance:")
    print(f" Expected Reward - Uniform: "
          f"cFVI = {cfvi_R_uniform_mean:.2f} \u00B1 {1.96 * cfvi_R_uniform_std:.2f}  /"
          f"rFVI = {rfvi_R_uniform_mean:.2f} \u00B1 {1.96 * rfvi_R_uniform_std:.2f}")

    print(f"Expected Reward - Downward: "
          f"cFVI = {cfvi_R_downward_mean:.2f} \u00B1 {1.96 * cfvi_R_downward_std:.2f} /"
          f"rFVI = {rfvi_R_downward_mean:.2f} \u00B1 {1.96 * rfvi_R_downward_std:.2f}")
    print(f"                 Test Time: {time.perf_counter() - t0:.2f}s")

    print("\n################################################")
    print("Plot the Value Function:")
    x_lim = scale * torch.tensor([system.x_lim[0], system.x_lim[1]]).float()
    norm_V = cm.colors.Normalize(vmax=0.0, vmin=-max(torch.abs(cfvi_V).max(), torch.abs(rfvi_V).max()))
    norm_u = cm.colors.Normalize(vmax=system.u_lim[0], vmin=-system.u_lim[0])

    v_plot_hyper = {'levels': 50, 'norm': norm_V, 'cmap': cm.get_cmap(cm.Spectral, 50)}
    pi_plot_hyper = {'levels': 50, 'norm': norm_u, 'cmap': cm.get_cmap(cm.Spectral, 50)}

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

        ax.yaxis.set_label_coords(-0.065, 0.5)
        ax.set_xlim(0, duration)
        return ax

    cfvi_x_tra = cfvi_downward_trajectories[0].cpu().numpy()
    cfvi_u_tra = cfvi_downward_trajectories[1].cpu().numpy()
    cfvi_x_mat = cfvi_x.reshape(*mat_shape, system.n_state)
    cfvi_xx, cfvi_xy = cfvi_x_mat[:, :, 0], cfvi_x_mat[:, :, 1]
    cfvi_u_mat = cfvi_u.reshape(*mat_shape, system.n_act)[:, :, 0]
    cfvi_V_mat = cfvi_V.reshape(mat_shape)

    rfvi_x_tra = rfvi_downward_trajectories[0].cpu().numpy()
    rfvi_u_tra = rfvi_downward_trajectories[1].cpu().numpy()
    rfvi_x_mat = rfvi_x.reshape(*mat_shape, system.n_state)
    rfvi_xx, rfvi_xy = rfvi_x_mat[:, :, 0], rfvi_x_mat[:, :, 1]
    rfvi_u_mat = rfvi_u.reshape(*mat_shape, system.n_act)[:, :, 0]
    rfvi_V_mat = rfvi_V.reshape(mat_shape)

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(left=0.07, bottom=0.09, right=1.0, top=0.93, wspace=0.1, hspace=0.125)

    ax_cfvi_val = format_space_ax(fig.add_subplot(2, 2, 1))
    ax_cfvi_val.set_title(r"Value Function - $V(x)$")
    _ = ax_cfvi_val.contourf(cfvi_xx, cfvi_xy, cfvi_V_mat, **v_plot_hyper)

    ax_rfvi_val = format_space_ax(fig.add_subplot(2, 2, 3))
    cset = ax_rfvi_val.contourf(rfvi_xx, rfvi_xy, rfvi_V_mat, **v_plot_hyper)

    ax_cfvi_val.text(x=-0.16, y=0.5, s="Continuous FVI", ha="center", va="center",
                     fontsize=12, rotation=90., transform=ax_cfvi_val.transAxes)

    ax_rfvi_val.text(x=-0.16, y=0.5, s="Robust FVI", ha="center", va="center",
                     fontsize=12, rotation=90., transform=ax_rfvi_val.transAxes)

    plt.colorbar(cset, ax=ax_cfvi_val)
    plt.colorbar(cset, ax=ax_rfvi_val)

    ax_cfvi_pi = format_space_ax(fig.add_subplot(2, 2, 2))
    ax_cfvi_pi.set_title(r"Policy - $\pi(x)$")
    _ = ax_cfvi_pi.contourf(cfvi_xx, cfvi_xy, cfvi_u_mat, **pi_plot_hyper)

    ax_rfvi_pi = format_space_ax(fig.add_subplot(2, 2, 4))
    cset = ax_rfvi_pi.contourf(rfvi_xx, rfvi_xy, rfvi_u_mat, **pi_plot_hyper)

    plt.colorbar(cset, ax=ax_cfvi_pi)
    plt.colorbar(cset, ax=ax_rfvi_pi)

    for i in range(n_plot):
        cfvi_xi_tra = add_nan(cfvi_x_tra[:, i, :, 0], system.wrap_i)
        rfvi_xi_tra = add_nan(rfvi_x_tra[:, i, :, 0], system.wrap_i)

        ax_cfvi_val.plot(cfvi_xi_tra[:, 0], cfvi_xi_tra[:, 1], c="k", alpha=0.25)
        ax_cfvi_pi.plot(cfvi_xi_tra[:, 0], cfvi_xi_tra[:, 1], c="k", alpha=0.25)

        ax_rfvi_val.plot(rfvi_xi_tra[:, 0], rfvi_xi_tra[:, 1], c="k", alpha=0.25)
        ax_rfvi_pi.plot(rfvi_xi_tra[:, 0], rfvi_xi_tra[:, 1], c="k", alpha=0.25)

    fig = plt.figure(figsize=(8, 5))
    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.97, wspace=0.1, hspace=0.3)

    ax_xp = format_time_ax(fig.add_subplot(3, 1, 1), 1)
    ax_xv = format_time_ax(fig.add_subplot(3, 1, 2), 2)
    ax_u = format_time_ax(fig.add_subplot(3, 1, 3), 3)

    t = np.linspace(0, duration, cfvi_x_tra.shape[0])
    for i in range(n_plot):
        cfvi_xi_tra = add_nan(np.concatenate((cfvi_x_tra[:, i, :, 0], t[:, np.newaxis]), axis=-1), system.wrap_i)
        rfvi_xi_tra = add_nan(np.concatenate((rfvi_x_tra[:, i, :, 0], t[:, np.newaxis]), axis=-1), system.wrap_i)

        ax_xp.plot(cfvi_xi_tra[:, -1], cfvi_xi_tra[:, 0], c="b", alpha=0.25)
        ax_xv.plot(cfvi_xi_tra[:, -1], cfvi_xi_tra[:, 1], c="b", alpha=0.25)
        ax_u.plot(t, cfvi_u_tra[:, i, 0, 0], c="b", alpha=0.25)

        ax_xp.plot(rfvi_xi_tra[:, -1], rfvi_xi_tra[:, 0], c="r", alpha=0.25)
        ax_xv.plot(rfvi_xi_tra[:, -1], rfvi_xi_tra[:, 1], c="r", alpha=0.25)
        ax_u.plot(t, rfvi_u_tra[:, i, 0, 0], c="r", alpha=0.25)

    legend = [
        Line2D([0], [0], lw=2, marker="", color="b", markersize=0, label="Continuous FVI"),
        Line2D([0], [0], lw=2, marker="", color="r", markersize=0, label="Robust FVI"),
    ]
    ax_xp.legend(handles=legend, bbox_to_anchor=(1.0, 1.0), loc='upper right', ncol=6, framealpha=0., labelspacing=1.0)
    plt.show()