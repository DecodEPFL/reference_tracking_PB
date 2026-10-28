import argparse, math, os, sys, torch

from Distributed.controllers.interconnection import ADJACENCY_GRAPHS
from Distributed.experiments.robots.arg_parser import build_parser, finalize_args, print_args as print_robots_args

# Defaults of the robots run pb_contextual_ssm_bounded_mlp_10_12_13_55_17 (distributed, neighbors),
# the starting point of the formation experiment, with a third robot.
FORMATION_DEFAULTS = dict(
    n_agents=3,
    num_rollouts=300, batch_size=100, epochs=200,   # log_epoch: 10% of the epochs
    info_pattern='neighbors', batch_agents=True,
    ssm_param='tv', ctx_modes='mixer,gate,select,input',
    ctx_filter='exponential', ctx_filter_decay=0.96,
    alpha_u=0.05, alpha_col=4000.0, alpha_obst=15000.0, obst_radius=1.0,
    alpha_final=400.0, final_start=90, alpha_speed=130.0,
)


# argument parser
def argument_parser(argv=None):
    parser = build_parser(description="Robots formation experiment.")
    bool_flag = argparse.BooleanOptionalAction   # --flag / --no-flag
    parser.set_defaults(**FORMATION_DEFAULTS)

    # formation
    parser.add_argument('--formation-side', type=float, default=2.0, help='Side of the regular n-gon formation (3 robots: equilateral triangle). Must exceed --min-dist + 0.2 so the formation itself is collision free; with --std-init-plant 0.2, a side of 1.5 starts 10%% of the rollouts in collision. Default is 2.0.')
    parser.add_argument('--formation-shape', type=str, default='regular', choices=['regular', 'isosceles', 'bowtie'], help='regular: n-gon of side --formation-side. isosceles (3 robots): apex (robot 1) at --formation-height above a horizontal base of length --formation-base (robots 2, 3); with base > gap > height it must turn its base along y to pass the gap. bowtie (5 robots): knot (robot 1) at the centre, corners (robots 2-5) on a --formation-base wide, --formation-height high rectangle; same gap condition. Default is regular.')
    parser.add_argument('--formation-base', type=float, default=4.0, help='Base of the isosceles formation, width of the bowtie. Default is 4.0.')
    parser.add_argument('--formation-height', type=float, default=1.2, help='Height (apex to base) of the isosceles formation, height of the bowtie. Default is 1.2.')
    parser.add_argument('--alpha-form', type=float, default=500.0, help='Weight of the formation loss on the inter-robot distances. 0 disables it (baseline). Default is 500.')
    parser.add_argument('--formation-deadband', type=float, default=0.1, help='No formation cost while every distance is within this of its desired value (elastic formation). Default is 0.1.')
    parser.add_argument('--formation-graph', type=str, default='complete', help='Pairs whose distance is constrained, one of %s or an explicit 0/1 matrix (symmetrised). Default is complete.' % ', '.join(ADJACENCY_GRAPHS))
    parser.add_argument('--alpha-orient', type=float, default=0.0, help='Weight of the orientation loss on the signed areas of the triangles of robots: penalises a robot crossing the line between two others (mirroring the formation) instead of rotating it. 0 disables it. Default is 0.')
    parser.add_argument('--orient-margin', type=float, default=0.5, help='Orientation loss is 0 while every triangle keeps this fraction of its formation area (with the right sign). In [0, 1]. Default is 0.5.')
    parser.add_argument('--alpha-orient-max', type=float, default=0.0, help='Weight of the worst-step orientation loss: sum over the triangles of the max over time of the orientation penalty (not time-averaged, so a robot cutting through the line between two others in a few fast steps is not cheap). 0 disables it. Default is 0.')
    parser.add_argument('--alpha-wall', type=float, default=0.0, help='Weight of the arena walls on x (--wall-x): time-averaged quadratic penalty, summed over the robots, from --wall-margin inside a wall on (1 on the wall, growing beyond). Keeps the robots from going around the obstacles. 0 disables it. Default is 0.')
    parser.add_argument('--wall-x', type=float, nargs=2, default=[-3.0, 7.0], metavar=('X_MIN', 'X_MAX'), help='x of the left and right arena walls. Default is -3 7 (the outer obstacle centres and the GIF frame).')
    parser.add_argument('--wall-margin', type=float, default=0.5, help='Distance inside each wall where the wall penalty starts. Default is 0.5.')
    parser.add_argument('--formation-start', type=int, default=0, help='First time step penalised by the formation loss. Default is 0.')
    parser.add_argument('--target-rotation', action=bool_flag, default=True, help='Rotate every target formation by a random angle (else same orientation as the start). Default is True.')
    parser.add_argument('--target-max-angle', type=float, default=math.pi, help='With --target-rotation, target angles are uniform in [-a, a] around the start orientation (radians). pi: any orientation. Default is pi.')
    parser.add_argument('--target-flip', action=bool_flag, default=False, help='Also turn half of the targets by pi (on top of the random angle): flipped targets, with the base of a wide triangle still along x. Default is False.')
    parser.add_argument('--large-rot-frac', type=float, default=0.0, help='Fraction of the targets drawn from the large rotations only (|angle| >= --large-rot-min-angle, random sign); the others stay uniform over the whole range. Applied to the train and test data alike (same distribution). 0 disables it. Default is 0.')
    parser.add_argument('--large-rot-min-angle', type=float, default=2 * math.pi / 3, help='Smallest |angle| (radians, around the start orientation) of the oversampled large rotations. Default is 2pi/3 (120 deg).')
    parser.add_argument('--save-gifs', action=bool_flag, default=True, help='Save GIFs of every verification scenario (direct, flip, offset) before and after training. Default is True.')
    parser.add_argument('--gif-interval', type=int, default=4, help='Time steps between two GIF frames. Default is 4.')
    parser.add_argument('--resume', type=str, default=None, help='Run folder (or its training_state.pt) to continue in a new folder. Every other arg is reloaded from that run, except --epochs: the total number of epochs counted from the start of that run (default: its own, to finish an interrupted run), and --log-epoch (default: 10%% of the new total if that run used the default, else its own).')

    args = finalize_args(parser.parse_args(argv))

    if args.alpha_form < 0:
        raise ValueError('--alpha-form must be >= 0.')
    if args.alpha_orient < 0:
        raise ValueError('--alpha-orient must be >= 0.')
    if args.alpha_orient_max < 0:
        raise ValueError('--alpha-orient-max must be >= 0.')
    if args.alpha_wall < 0:
        raise ValueError('--alpha-wall must be >= 0.')
    if args.alpha_wall and not (args.wall_x[0] < args.wall_x[1] and args.wall_margin > 0):
        raise ValueError('--alpha-wall needs --wall-x X_MIN X_MAX with X_MIN < X_MAX and --wall-margin > 0.')
    if not 0 <= args.orient_margin <= 1:
        raise ValueError('--orient-margin must be in [0, 1].')
    if args.formation_deadband < 0:
        raise ValueError('--formation-deadband must be >= 0.')
    if args.formation_shape == 'isosceles':
        if args.n_agents != 3:
            raise ValueError('--formation-shape isosceles needs --n-agents 3.')
        if args.formation_base <= 0 or args.formation_height <= 0:
            raise ValueError('--formation-base and --formation-height must be > 0.')
    if args.formation_shape == 'bowtie':
        if args.n_agents != 5:
            raise ValueError('--formation-shape bowtie needs --n-agents 5.')
        if args.formation_base <= 0 or args.formation_height <= 0:
            raise ValueError('--formation-base and --formation-height must be > 0.')
    if args.target_max_angle < 0:
        raise ValueError('--target-max-angle must be >= 0.')
    if not 0 <= args.large_rot_frac <= 1:
        raise ValueError('--large-rot-frac must be in [0, 1].')
    if args.large_rot_frac > 0 and not (args.target_rotation and 0 <= args.large_rot_min_angle < min(args.target_max_angle, math.pi)):
        raise ValueError('--large-rot-frac needs --target-rotation and 0 <= --large-rot-min-angle < --target-max-angle.')
    if args.formation_shape == 'regular':
        shortest = args.formation_side
    elif args.formation_shape == 'isosceles':
        shortest = min(args.formation_base, math.hypot(args.formation_base / 2, args.formation_height))
    else:   # bowtie: corner to corner, or knot to corner
        shortest = min(args.formation_base, args.formation_height, math.hypot(args.formation_base, args.formation_height) / 2)
    if args.col_av and shortest <= args.min_dist + 0.2:
        print(f'Shortest formation distance {shortest:.3g} is within the collision margin --min-dist + 0.2 = {args.min_dist + 0.2}: '
              f'the formation and the collision loss conflict.')
    if args.resume is not None:
        args = resume_args(args, argv)
    return args


def training_state_path(resume):
    """training_state.pt of a run folder (or the file itself)."""
    return os.path.join(resume, 'training_state.pt') if os.path.isdir(resume) else resume


def resume_args(args, argv=None):
    """Args of the run continued by --resume, with the total --epochs of the command line if given."""
    path = training_state_path(args.resume)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'--resume: no training state at {path} (only runs started after resuming was added save one).')
    saved = torch.load(path, map_location='cpu', weights_only=False)['args']
    argv = sys.argv[1:] if argv is None else argv
    resumed = argparse.Namespace(**{**vars(args), **saved})   # args added since that run keep their defaults
    resumed.resume = args.resume
    given = lambda flag: any(a == flag or a.startswith(flag + '=') for a in argv)
    if given('--epochs'):
        resumed.epochs = args.epochs
        # a default log frequency (10% of the epochs) follows the new total
        if saved['log_epoch'] == max(1, math.ceil(float(saved['epochs']) / 10)):
            resumed.log_epoch = max(1, math.ceil(float(resumed.epochs) / 10))
    if given('--log-epoch'):
        resumed.log_epoch = args.log_epoch
    return resumed


def print_args(args):
    msg = print_robots_args(args)
    if args.formation_shape == 'isosceles':
        shape = 'isosceles triangle, base %g, height %g' % (args.formation_base, args.formation_height)
    elif args.formation_shape == 'bowtie':
        shape = 'bowtie, width %g, height %g' % (args.formation_base, args.formation_height)
    else:
        shape = 'regular %i-gon, side %g' % (args.n_agents, args.formation_side)
    rotation = str(args.target_rotation)
    if args.target_rotation and args.target_max_angle < math.pi:
        rotation += ' (|angle| <= %.3g rad)' % args.target_max_angle
    if args.target_flip:
        rotation += ' -- half of the targets flipped by pi'
    if args.large_rot_frac > 0:
        rotation += ' -- %.0f%% of the train/test targets with |angle| >= %.0f deg' % (
            100 * args.large_rot_frac, math.degrees(args.large_rot_min_angle))
    msg += '\n[INFO] Formation: %s -- graph: %s -- alpha_form: %g -- deadband: %g -- from t = %i -- target rotation: %s' % (
        shape, args.formation_graph, args.alpha_form,
        args.formation_deadband, args.formation_start, rotation)
    if args.alpha_orient or args.alpha_orient_max:
        msg += ' -- alpha_orient: %g, alpha_orient_max: %g (margin %g)' % (
            args.alpha_orient, args.alpha_orient_max, args.orient_margin)
    if args.alpha_wall:
        msg += '\n[INFO] Arena walls: x in [%g, %g] -- alpha_wall: %g (margin %g)' % (
            args.wall_x[0], args.wall_x[1], args.alpha_wall, args.wall_margin)
    return msg
