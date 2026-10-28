import argparse

from Centralized.utils.graphs import ADJACENCY_GRAPHS
from Centralized.experiments.robots.arg_parser import build_parser, finalize_args, print_args as print_robots_args

# Same defaults as the distributed formation experiment (Distributed/experiments/robots_formation),
# i.e. the distributed robots run pb_contextual_ssm_bounded_mlp_10_12_13_55_17 with a third robot,
# minus the distributed-only options (information pattern, agent batching). With these widths and
# no pair features the centralized M is the one the distributed M^i are size-matched to; the pair
# features (relative positions of the robots, the counterpart of the distributed neighbor signals)
# widen its context heads.
FORMATION_DEFAULTS = dict(
    n_agents=3, pair_features=True,
    num_rollouts=300, batch_size=100, epochs=200, log_epoch=20,
    ssm_param='tv', ctx_modes='mixer,gate,select,input',
    ctx_filter='exponential', ctx_filter_decay=0.96,
    alpha_u=0.05, alpha_col=4000.0, alpha_obst=15000.0, obst_radius=1.0,
    alpha_final=400.0, final_start=90, alpha_speed=130.0,
)


# argument parser
def argument_parser(argv=None):
    parser = build_parser(description="Robots formation experiment (centralized controller).")
    bool_flag = argparse.BooleanOptionalAction   # --flag / --no-flag
    parser.set_defaults(**FORMATION_DEFAULTS)

    # formation
    parser.add_argument('--formation-side', type=float, default=2.0, help='Side of the regular n-gon formation (3 robots: equilateral triangle). Must exceed --min-dist + 0.2 so the formation itself is collision free; with --std-init-plant 0.2, a side of 1.5 starts 10%% of the rollouts in collision. Default is 2.0.')
    parser.add_argument('--alpha-form', type=float, default=500.0, help='Weight of the formation loss on the inter-robot distances. 0 disables it (baseline). Default is 500.')
    parser.add_argument('--formation-deadband', type=float, default=0.1, help='No formation cost while every distance is within this of its desired value (elastic formation). Default is 0.1.')
    parser.add_argument('--formation-graph', type=str, default='complete', help='Pairs whose distance is constrained, one of %s or an explicit 0/1 matrix (symmetrised). Default is complete.' % ', '.join(ADJACENCY_GRAPHS))
    parser.add_argument('--formation-start', type=int, default=0, help='First time step penalised by the formation loss. Default is 0.')
    parser.add_argument('--target-rotation', action=bool_flag, default=True, help='Rotate every target formation by a random angle (else same orientation as the start). Default is True.')
    parser.add_argument('--save-gifs', action=bool_flag, default=True, help='Save GIFs of every verification scenario (direct, flip, offset) before and after training. Default is True.')
    parser.add_argument('--gif-interval', type=int, default=4, help='Time steps between two GIF frames. Default is 4.')

    args = finalize_args(parser.parse_args(argv))

    if args.alpha_form < 0:
        raise ValueError('--alpha-form must be >= 0.')
    if args.formation_deadband < 0:
        raise ValueError('--formation-deadband must be >= 0.')
    if args.col_av and args.formation_side <= args.min_dist + 0.2:
        print(f'--formation-side {args.formation_side} is within the collision margin --min-dist + 0.2 = {args.min_dist + 0.2}: '
              f'the formation and the collision loss conflict.')
    return args


def print_args(args):
    msg = print_robots_args(args)
    msg += '\n[INFO] Formation: regular %i-gon, side %g -- graph: %s -- alpha_form: %g -- deadband: %g -- from t = %i -- target rotation: %s' % (
        args.n_agents, args.formation_side, args.formation_graph, args.alpha_form,
        args.formation_deadband, args.formation_start, args.target_rotation)
    return msg
