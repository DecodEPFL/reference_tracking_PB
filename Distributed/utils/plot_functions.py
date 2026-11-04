import torch, os
from scipy.stats import multivariate_normal # TODO: use something compatible with tensors
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import imageio

# one color per agent (tab10 order, so agents 1 and 2 keep blue and orange); cycles beyond 10 agents
AGENT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def agent_color(i):
    return AGENT_COLORS[i % len(AGENT_COLORS)]


def plot_formation_edges(ax, x, t, formation_pairs, **kwargs):
    """Draw a segment between the agents of every pair (i, j) in formation_pairs at time t."""
    style = dict(color='k', linewidth=0.8, alpha=0.6, zorder=5)
    style.update(kwargs)
    for i, j in formation_pairs:
        ax.plot(
            [x[t, 4*i].detach().cpu(), x[t, 4*j].detach().cpu()],
            [x[t, 4*i+1].detach().cpu(), x[t, 4*j+1].detach().cpu()],
            **style
        )


def drag_field(patches, lims, n=200):
    """Drag scale s(p) = 1 + sum_k w_k exp(-||p - c_k||^2 / (2 sigma_k^2)) of a sticky patch layout
    (tensor of 4 * n_patches entries (cx, cy, w, sigma), or (n_patches, 4)) on an n x n grid over
    lims = (xmin, xmax, ymin, ymax). Returns the numpy grids xx, yy, s."""
    pt = torch.as_tensor(patches).detach().cpu().double().reshape(-1, 4).numpy()
    xx, yy = np.meshgrid(np.linspace(lims[0], lims[1], n), np.linspace(lims[2], lims[3], n))
    s = np.ones_like(xx)
    for cx, cy, w, sigma in pt:
        s += w * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return xx, yy, s


def plot_drag_patches(ax, patches, lims, s_max=None, colorbar=True):
    """Shade the sticky patches of a layout (see drag_field) on ax, from s = 1.25 up to s_max
    (default: the peak of the layout), with a colorbar of the drag scale."""
    xx, yy, s = drag_field(patches, lims)
    top = max(float(s.max()), 1.5) if s_max is None else float(s_max)
    levels = np.linspace(1.25, max(top, 1.5), 9)
    cs = ax.contourf(xx, yy, np.clip(s, None, levels[-1]), levels=levels, cmap='YlOrBr', alpha=0.45, zorder=1,
                     extend='neither')
    ax.contour(xx, yy, s, levels=[1.25], colors='saddlebrown', linewidths=0.5, alpha=0.6, zorder=1)
    if colorbar:
        cb = ax.figure.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, format='%.1f')
        cb.ax.set_title('drag', fontsize=9)    # a side label would be cut off the fixed-size frames
    return cs


def draw_obstacles(ax, xx, yy, zz, z_min, z_max, transparent=False):
    """Obstacle background (Gaussian densities zz on the grid xx, yy, indexed [x, y]) in greys.
    transparent=True (sticky patches drawn behind): black with the opacity that gives the same grey over a
    white background, fully transparent away from the obstacles; drawn as a smoothly interpolated image,
    which keeps its transparency in PDFs too (gouraud shading does not)."""
    if not transparent:
        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')
        return
    from matplotlib.colors import ListedColormap
    colors = np.zeros((256, 4))
    colors[:, 3] = 1.0 - plt.get_cmap('Greys')(np.linspace(0, 1, 256))[:, 0]   # grey g over white = black at 1 - g
    ax.imshow(zz.T, origin='lower', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
              cmap=ListedColormap(colors), vmin=z_min, vmax=z_max, interpolation='bilinear', zorder=1)


def plot_trajectories(
    x, xbar, n_agents, save_folder, text="", save=True, filename='', T=100,
    dots=False, circles=False, axis=True, min_dist=1, f=5,
    obstacle_centers=None, obstacle_covs=None,
    formation_pairs=None, formation_every=None, patches=None,
):
    """formation_pairs: optional list of agent pairs (i, j) whose segments are drawn
    every formation_every steps up to T (default: at 0 and T only).
    patches: optional sticky patch layout of the rollout (see drag_field), shaded over the arena."""
    filename = 'trajectories.pdf' if filename == '' else filename

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # sticky patches behind the obstacles
    if patches is not None:
        top = max(x.reshape(-1, 4)[:, 1].max().item(), xbar.reshape(-1, 4)[:, 1].max().item())
        plot_drag_patches(ax, patches, (-3, 7, -3, min(15.0, max(7.0, top + 1))))
    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        # background up to 1 above the highest robot or target (y in [7, 15])
        top = max(x.reshape(-1, 4)[:, 1].max().item(), xbar.reshape(-1, 4)[:, 1].max().item())
        y_max = min(15.0, max(7.0, top + 1))
        yy, xx = np.meshgrid(np.linspace(-3, y_max, round(10 * (y_max + 3))), np.linspace(-3, 7, 100))
        zz = xx * 0
        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                mean=center.detach().clone().cpu().numpy().flatten()
            )
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

        draw_obstacles(ax, xx, yy, zz, z_min, z_max, transparent=patches is not None)

    ax.set_title(text)
    for i in range(n_agents):
        ax.plot(
            x[:T+1,4*i].detach().cpu(), x[:T+1,4*i+1].detach().cpu(),
            color=agent_color(i), linewidth=1
        )
        ax.plot(
            x[T:,4*i].detach().cpu(), x[T:,4*i+1].detach().cpu(),
            color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
        )
    for i in range(n_agents):
        ax.plot(
            x[0,4*i].detach().cpu(), x[0,4*i+1].detach().cpu(),
            color=agent_color(i), marker='8'
        )
        ax.plot(
            xbar[4*i].detach().cpu(), xbar[4*i+1].detach().cpu(),
            color=agent_color(i), marker='*', markersize=10
        )

    if formation_pairs:
        times = list(range(0, min(T, x.shape[0] - 1) + 1, formation_every)) if formation_every else [0]
        times.append(min(T, x.shape[0] - 1))
        for t in sorted(set(times)):
            plot_formation_edges(ax, x, t, formation_pairs)

    if dots:
        for i in range(n_agents):
            ax.plot(  
                x[:T+1,4*i].detach().cpu(), x[:T+1,4*i+1].detach().cpu(),
                color=agent_color(i), linewidth=1, marker = "x"
            )

    if circles:
        for i in range(n_agents):
            r = min_dist/2
            circle = ax.Circle(
                (x[T, 4*i].detach().cpu(), x[T, 4*i+1].detach().cpu()),
                r, color=agent_color(i), alpha=0.5, zorder=10
            )
            ax.add_patch(circle)
    ax.axes.xaxis.set_visible(axis)
    ax.axes.yaxis.set_visible(axis)
    if save:
        fig.savefig(
            os.path.join(save_folder, filename),
            format='pdf'
        )
        plt.close()
    else:
        plt.show()


def plot_traj_vs_time(t_end, n_agents, save_folder, x, u=None, text="", save=True, filename=''):
    filename = filename if filename=='' else filename+'_'
    now = datetime.now()
    formatted_date = now.strftime('%m-%d-%H:%M')
    t = torch.linspace(0,t_end-1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i].detach().cpu())
        plt.plot(t, x[:,4*i+1].detach().cpu())
    plt.xlabel(r'$t$')
    plt.title(r'$x(t)$')
    plt.subplot(1, p, 2)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i+2].detach().cpu())
        plt.plot(t, x[:,4*i+3].detach().cpu())
    plt.xlabel(r'$t$')
    plt.title(r'$v(t)$')
    plt.suptitle(text)
    if p == 3:
        plt.subplot(1, 3, 3)
        for i in range(n_agents):
            plt.plot(t, u[:, 2*i].detach().cpu())
            plt.plot(t, u[:, 2*i+1].detach().cpu())
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig(
            os.path.join(
                save_folder,
                filename+text+'_x_u.pdf'
            ),
            format='pdf'
        )
        plt.close()
    else:
        plt.show()


def save_trajectory_frames(x, xbar, n_agents, save_folder, T=100, interval=1,f=5, obstacle_centers=None, obstacle_covs=None,
                           formation_pairs=None, lims=None, robot_radius=0.5, collision_dist=1.0, patches=None):
    """formation_pairs: optional list of agent pairs (i, j) whose current segments are drawn on every frame
    (and dashed at the target). lims: optional fixed (xmin, xmax, ymin, ymax) of every frame.
    robot_radius: radius of the disc drawn for each robot; collision_dist: centre distance below which
    the frame is titled 'Collision' (the loss' min_dist, i.e. 2 * robot_radius).
    patches: optional sticky patch layout of the rollout (see drag_field), shaded on every frame."""
    os.makedirs(save_folder, exist_ok=True)

    # obstacle background, computed once and redrawn on every frame
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        y_max = 7.0 if lims is None else max(7.0, lims[3])   # background over the whole frame
        yy, xx = np.meshgrid(np.linspace(-3, y_max, round(10 * (y_max + 3))), np.linspace(-3, 7, 100))
        zz = xx * 0
        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                mean=center.detach().clone().cpu().numpy().flatten()
            )
            zz += distr.pdf(np.stack([xx, yy], axis=-1))
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

    for t in range(0, T+1, interval):
        # one self-overwriting progress line per GIF instead of one line per frame
        print(f'\rSaving frames of {os.path.basename(save_folder)}: {t}/{T}', end='', flush=True)
        fig, ax = plt.subplots(figsize=(f,f))
        # plot obstacles
        # sticky patches behind the obstacles
        if patches is not None:
            plot_drag_patches(ax, patches, lims if lims is not None else (-3, 7, -3, 7))
        if not obstacle_covs is None:
            draw_obstacles(ax, xx, yy, zz, z_min, z_max, transparent=patches is not None)
        for i in range(n_agents):
            ax.plot(
                x[:t+1, 4*i].detach().cpu(), x[:t+1, 4*i+1].detach().cpu(),
                color=agent_color(i), linewidth=1
            )
            ax.plot(
                x[t, 4*i].detach().cpu(), x[t, 4*i+1].detach().cpu(),
                color=agent_color(i), marker='o'
            )
            ax.plot(
                x[0, 4*i].detach().cpu(), x[0, 4*i+1].detach().cpu(),
                color=agent_color(i), marker='o', markerfacecolor='none'
            )
            r = robot_radius
            circle = plt.Circle(
                (x[t, 4*i].detach().cpu(), x[t, 4*i+1].detach().cpu()),
                r, color=agent_color(i), alpha=0.5, zorder=10
            )
            ax.add_patch(circle)
            ax.plot(
                xbar[4*i].detach().cpu(), xbar[4*i+1].detach().cpu(),
                color=agent_color(i), marker='*', markersize=10
            )
            
        if formation_pairs:
            plot_formation_edges(ax, x, t, formation_pairs)
            plot_formation_edges(ax, xbar.view(1, -1), 0, formation_pairs, linestyle='dashed', alpha=0.3)
        # closest pair of agents at time t
        dist = min(
            (torch.sqrt((x[t, 4*i] - x[t, 4*j])**2 + (x[t, 4*i+1] - x[t, 4*j+1])**2)
             for i in range(n_agents) for j in range(i + 1, n_agents)),
            default=float('inf'),
        )
        if dist < collision_dist:
            ax.set_title('Collision', color='red')
             
        else:
            ax.set_title(f'Time step: {t}')
        if lims is not None:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            ax.set_aspect('equal', adjustable='box')   # same scale in x and y: discs stay round, sizes true
        
        frame_filename = os.path.join(save_folder, f'frame_{t:03d}.png')
        fig.savefig(frame_filename)
        plt.close(fig)
    print()


def create_gif_from_frames(frame_folder, gif_filename, duration=0.1):
    frames = []
    for frame_file in sorted(os.listdir(frame_folder)):
        if frame_file.endswith('.png'):
            frame_path = os.path.join(frame_folder, frame_file)
            frames.append(imageio.imread(frame_path))
    imageio.mimsave(gif_filename, frames, duration=duration, loop=0)   # loop=0: repeat forever


def plot_formation_error(errors, save_folder, filename='formation_error_over_time.pdf', deadband=None, T=None,
                         text='Formation error over time'):
    """
    errors: dict label -> (mean, max) tensors of shape (T_total,), the mean and max over the
    formation edges of |d_ij(t) - D_ij|. deadband: horizontal line at eps; T: vertical line at the horizon.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, (label, (err_mean, err_max)) in enumerate(errors.items()):
        t = np.arange(err_mean.shape[0])
        ax.plot(t, err_mean.detach().cpu(), color=f'C{k}', label=label + ' (mean)')
        ax.plot(t, err_max.detach().cpu(), color=f'C{k}', linestyle='dashed', linewidth=0.8, label=label + ' (max)')
    if deadband:
        ax.axhline(deadband, color='k', linestyle='dotted', linewidth=0.8, label='deadband')
    if T is not None:
        ax.axvline(T, color='gray', linestyle='dotted', linewidth=0.8)
    ax.set_xlabel('time step')
    ax.set_ylabel(r'$|d_{ij} - D_{ij}|$')
    ax.set_title(text)
    ax.legend(fontsize=8)
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, filename), format='pdf')
    plt.close(fig)
