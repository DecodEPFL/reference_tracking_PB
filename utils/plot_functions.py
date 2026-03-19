import torch, os
from scipy.stats import multivariate_normal # TODO: use something compatible with tensors
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import imageio

def plot_trajectories(
    x, xbar, n_agents, save_folder, text="", save=True, filename='', T=100,
    dots=False, circles=False, axis=True, min_dist=1, f=5,
    obstacle_centers=None, obstacle_covs=None
):
    filename = 'trajectories.pdf' if filename == '' else filename

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        yy, xx = np.meshgrid(np.linspace(-3, 7, 100), np.linspace(-3, 7, 100))
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

        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')

    ax.set_title(text)
    colors = ['tab:blue', 'tab:orange']
    for i in range(n_agents):
        ax.plot(
            x[:T+1,4*i].detach().cpu(), x[:T+1,4*i+1].detach().cpu(),
            color=colors[i%2], linewidth=1
        )
        ax.plot(
            x[T:,4*i].detach().cpu(), x[T:,4*i+1].detach().cpu(),
            color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
        )
    for i in range(n_agents):
        ax.plot(
            x[0,4*i].detach().cpu(), x[0,4*i+1].detach().cpu(),
            color=colors[i%2], marker='8'
        )
        ax.plot(
            xbar[4*i].detach().cpu(), xbar[4*i+1].detach().cpu(),
            color=colors[i%2], marker='*', markersize=10
        )

    if dots:
        for i in range(n_agents):
            ax.plot(  
                x[:T+1,4*i].detach().cpu(), x[:T+1,4*i+1].detach().cpu(),
                color=colors[i%2], linewidth=1, marker = "x"
            )

    if circles:
        for i in range(n_agents):
            r = min_dist/2
            circle = ax.Circle(
                (x[T, 4*i].detach().cpu(), x[T, 4*i+1].detach().cpu()),
                r, color=colors[i%2], alpha=0.5, zorder=10
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


def save_trajectory_frames(x, xbar, n_agents, save_folder, T=100, interval=1,f=5, obstacle_centers=None, obstacle_covs=None):
    os.makedirs(save_folder, exist_ok=True)

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        yy, xx = np.meshgrid(np.linspace(-3, 7, 100), np.linspace(-3, 7, 100))
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

        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')



    colors = ['tab:blue', 'tab:orange']
    for t in range(0, T+1, interval):
        print(f'Saving frame {t}/{T}...')
        fig, ax = plt.subplots(figsize=(f,f))
        # plot obstacles
        if not obstacle_covs is None:
            assert not obstacle_centers is None
            yy, xx = np.meshgrid(np.linspace(-3, 7, 100), np.linspace(-3, 7, 100))
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

            ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')
        for i in range(n_agents):
            ax.plot(
                x[:t+1, 4*i].detach().cpu(), x[:t+1, 4*i+1].detach().cpu(),
                color=colors[i%2], linewidth=1
            )
            ax.plot(
                x[t, 4*i].detach().cpu(), x[t, 4*i+1].detach().cpu(),
                color=colors[i%2], marker='o'
            )
            ax.plot(
                x[0, 4*i].detach().cpu(), x[0, 4*i+1].detach().cpu(),
                color=colors[i%2], marker='o', markerfacecolor='none'
            )
            r = 0.5
            circle = plt.Circle(
                (x[t, 4*i].detach().cpu(), x[t, 4*i+1].detach().cpu()),
                r, color=colors[i%2], alpha=0.5, zorder=10
            )
            ax.add_patch(circle)
            ax.plot(
                xbar[4*i].detach().cpu(), xbar[4*i+1].detach().cpu(),
                color=colors[i%2], marker='*', markersize=10
            )
            
        dist = torch.sqrt((x[t, 4*0] - x[t, 4*1])**2 + (x[t, 4*0+1] - x[t, 4*1+1])**2)
        if dist < 1:
            ax.set_title('Collision', color='red')
             
        else:
            ax.set_title(f'Time step: {t}')
        """ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)"""
        
        frame_filename = os.path.join(save_folder, f'frame_{t:03d}.png')
        fig.savefig(frame_filename)
        plt.close(fig)


def create_gif_from_frames(frame_folder, gif_filename, duration=0.1):
    frames = []
    for frame_file in sorted(os.listdir(frame_folder)):
        if frame_file.endswith('.png'):
            frame_path = os.path.join(frame_folder, frame_file)
            frames.append(imageio.imread(frame_path))
    imageio.mimsave(gif_filename, frames, duration=duration)