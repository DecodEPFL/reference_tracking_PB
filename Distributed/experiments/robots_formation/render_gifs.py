"""
Render the trained-controller GIFs and plots of every verification scenario of a saved formation run.

    python Distributed/experiments/robots_formation/render_gifs.py <run folder> [--interval 4] [--overwrite] [--ref | --only-ref]

Rebuilds the controller from <run folder>/trained_controller.pt and its saved args, rolls out the
verification scenarios of run.py (formation_scenarios, formation_obstacles) and writes
<run folder>/gifs/CL_<scenario>_trained.gif with its frames, and the plot <run folder>/CL_<scenario>_trained.pdf.
With --ref, also the baseline GIF and plot CL_<scenario>_ref: the operator shut to zero (u = 0), the inner PI loop alone (runs made before
this baseline existed have ref GIFs of the untrained operator instead). Existing GIFs and plots are kept unless --overwrite.
Obstacles are not stored in the checkpoint: they are read from run.py as it is now.
"""

import argparse, os, sys
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(BASE_DIR))  # parent of Distributed/, for `Distributed.*` imports

from Distributed.controllers import build_controller
from Distributed.experiments.robots_formation.run import (ZeroOperatorController, formation_obstacles,
                                                           formation_scenarios, make_dataset, make_plant,
                                                           save_scenario_gif, verification_data, verification_patches)
from Distributed.utils.plot_functions import plot_trajectories
from Distributed.controllers.interconnection import make_adjacency


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('run_folder', help='Folder of a formation run, containing trained_controller.pt.')
    parser.add_argument('--interval', type=int, default=None, help="Time steps between frames (default: the run's --gif-interval).")
    parser.add_argument('--overwrite', action='store_true', help='Re-render scenarios that already have a GIF or plot.')
    parser.add_argument('--ref', action='store_true', help='Also render the baseline GIFs (operator off, u = 0).')
    parser.add_argument('--only-ref', action='store_true', help='Render only the baseline GIFs, leave the trained ones untouched.')
    cli = parser.parse_args(argv)

    ckpt = torch.load(os.path.join(cli.run_folder, 'trained_controller.pt'), map_location='cpu', weights_only=False)
    args = argparse.Namespace(**ckpt['args'])
    n_agents = args.n_agents
    state_dim = 4 * n_agents

    dataset = make_dataset(args)
    scenarios = formation_scenarios(dataset)
    obstacle_centers, obstacle_covs = formation_obstacles()
    adjacency = make_adjacency(args.formation_graph, n_agents)
    formation_pairs = dataset.plot_edges or torch.triu(((adjacency + adjacency.T) > 0).float(), diagonal=1).nonzero().tolist()

    plant = make_plant(args)    # with the sticky patches of the run, if any
    ctl = build_controller(args, plant=plant)
    ctl.load_state_dict(ckpt['controller'])

    # same verification data as run.py: nominal start (and flipped, '<scenario>_fs'), constant target formation
    names, data_verif = verification_data(dataset, scenarios, args.horizon,
                                          patches=verification_patches(args, len(scenarios)))
    patches_of = lambda k: data_verif[k, 0, 2 * state_dim:] if plant.n_patches else None
    rollouts = {}
    with torch.no_grad():
        if not cli.only_ref:
            rollouts['trained'] = plant.rollout(ctl, data_verif)[0]
        if cli.ref or cli.only_ref:
            rollouts['ref'] = plant.rollout(ZeroOperatorController(plant.in_dim), data_verif)[0]

    gif_folder = os.path.join(cli.run_folder, 'gifs')
    os.makedirs(gif_folder, exist_ok=True)
    plot_text = {'trained': 'rPB - trained controller', 'ref': 'Baseline - operator off (u = 0)'}
    for stage, x_verif in rollouts.items():
        for k, name in enumerate(names):
            xbar_k = data_verif[k, 1, state_dim:2 * state_dim]
            plot = os.path.join(cli.run_folder, 'CL_%s_%s.pdf' % (name, stage))
            if os.path.exists(plot) and not cli.overwrite:
                print(f'{name} ({stage}): {plot} exists, kept (use --overwrite to re-render)')
            else:   # as plot_scenarios of run.py
                plot_trajectories(
                    x_verif[k], xbar=xbar_k, n_agents=n_agents, save_folder=cli.run_folder,
                    filename='CL_%s_%s.pdf' % (name, stage), text=plot_text[stage], T=args.horizon * 4,
                    obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs,
                    formation_pairs=formation_pairs, formation_every=max(1, args.horizon // 5),
                    patches=patches_of(k),
                )
                print(f'{name} ({stage}): wrote {plot}')
            target = os.path.join(gif_folder, 'CL_%s_%s.gif' % (name, stage))
            if os.path.exists(target) and not cli.overwrite:
                print(f'{name} ({stage}): {target} exists, kept (use --overwrite to re-render)')
                continue
            save_scenario_gif(
                x_verif[k], xbar_k, name, stage, n_agents=n_agents, gif_folder=gif_folder,
                horizon=args.horizon, interval=cli.interval or args.gif_interval,
                obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs, formation_pairs=formation_pairs,
                min_dist=args.min_dist, patches=patches_of(k),
            )
            print(f'{name} ({stage}): wrote {target}')


if __name__ == '__main__':
    main()
