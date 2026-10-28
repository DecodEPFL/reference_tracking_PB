"""
Render the trained-controller GIFs of every verification scenario of a saved formation run.

    python Distributed/experiments/robots_formation/render_gifs.py <run folder> [--interval 4] [--overwrite] [--ref | --only-ref]

Rebuilds the controller from <run folder>/trained_controller.pt and its saved args, rolls out the
verification scenarios of run.py (formation_scenarios, formation_obstacles) and writes
<run folder>/gifs/CL_<scenario>_trained.gif with its frames. With --ref, also the baseline GIFs
CL_<scenario>_ref.gif: the operator shut to zero (u = 0), the inner PI loop alone (runs made before
this baseline existed have ref GIFs of the untrained operator instead). Existing GIFs are kept unless --overwrite.
Obstacles are not stored in the checkpoint: they are read from run.py as it is now.
"""

import argparse, os, sys
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(BASE_DIR))  # parent of Distributed/, for `Distributed.*` imports

from Distributed.controllers import build_controller
from Distributed.experiments.robots_formation.run import (ZeroOperatorController, formation_obstacles,
                                                           formation_scenarios, make_dataset, save_scenario_gif)
from Distributed.plants import RobotsSystem
from Distributed.controllers.interconnection import make_adjacency


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('run_folder', help='Folder of a formation run, containing trained_controller.pt.')
    parser.add_argument('--interval', type=int, default=None, help="Time steps between frames (default: the run's --gif-interval).")
    parser.add_argument('--overwrite', action='store_true', help='Re-render scenarios that already have a GIF.')
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
    formation_pairs = torch.triu(((adjacency + adjacency.T) > 0).float(), diagonal=1).nonzero().tolist()

    plant = RobotsSystem(linear_plant=args.linearize_plant, k=args.spring_const, n_agents=n_agents)
    ctl = build_controller(args, plant=plant)
    ctl.load_state_dict(ckpt['controller'])

    # same verification data as run.py: nominal start, constant target formation
    data_verif = torch.zeros(len(scenarios), args.horizon + 200, 2 * plant.state_dim)
    data_verif[:, 0:1, :state_dim] = dataset.x0
    for k, xbar_k in enumerate(scenarios.values()):
        data_verif[k, 1:, state_dim:] = xbar_k
    rollouts = {}
    with torch.no_grad():
        if not cli.only_ref:
            rollouts['trained'] = plant.rollout(ctl, data_verif)[0]
        if cli.ref or cli.only_ref:
            rollouts['ref'] = plant.rollout(ZeroOperatorController(plant.in_dim), data_verif)[0]

    gif_folder = os.path.join(cli.run_folder, 'gifs')
    os.makedirs(gif_folder, exist_ok=True)
    for stage, x_verif in rollouts.items():
        for k, (name, xbar_k) in enumerate(scenarios.items()):
            target = os.path.join(gif_folder, 'CL_%s_%s.gif' % (name, stage))
            if os.path.exists(target) and not cli.overwrite:
                print(f'{name} ({stage}): {target} exists, kept (use --overwrite to re-render)')
                continue
            save_scenario_gif(
                x_verif[k], xbar_k, name, stage, n_agents=n_agents, gif_folder=gif_folder,
                horizon=args.horizon, interval=cli.interval or args.gif_interval,
                obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs, formation_pairs=formation_pairs,
                min_dist=args.min_dist,
            )
            print(f'{name} ({stage}): wrote {target}')


if __name__ == '__main__':
    main()
