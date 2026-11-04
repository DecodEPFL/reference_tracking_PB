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
    # training stabilisers (runs saved before them: off, see resume_args)
    col_loss='hinge', lr_schedule='cosine', lr_warmup=20, grad_clip_rel=5.0, ema_decay=0.995,
    grad_diag=20.0,   # diagnosis only, does not change the training
)
# what the runs saved before the stabilisers trained with
PRE_STABILISER_ARGS = dict(col_loss='inverse', lr_schedule='constant', grad_clip=0.0, grad_clip_rel=0.0, ema_decay=0.0)
# what the runs saved before the collision free starts and the resampling trained on
PRE_DATA_ARGS = dict(init_min_dist=0.0, resample_data=False)


def saved_run_args(saved):
    """Args of a saved run, with what it trained with for the args added since it."""
    return {'pad_agent_inputs': False, **PRE_STABILISER_ARGS, **PRE_DATA_ARGS, **saved}


# argument parser
def argument_parser(argv=None):
    parser = build_parser(description="Robots formation experiment.")
    bool_flag = argparse.BooleanOptionalAction   # --flag / --no-flag
    parser.set_defaults(**FORMATION_DEFAULTS)

    # formation
    parser.add_argument('--formation-side', type=float, default=2.0, help='Side of the regular n-gon formation (3 robots: equilateral triangle). Must exceed --min-dist + 0.2 so the formation itself is collision free; with --std-init-plant 0.2, a side of 1.5 starts 10%% of the rollouts in collision. Default is 2.0.')
    parser.add_argument('--formation-shape', type=str, default='regular', choices=['regular', 'isosceles', 'bowtie', 'double_bowtie'], help='regular: n-gon of side --formation-side. isosceles (3 robots): apex (robot 1) at --formation-height above a horizontal base of length --formation-base (robots 2, 3); with base > gap > height it must turn its base along y to pass the gap. bowtie (5 robots): knot (robot 1) at the centre, corners (robots 2-5) on a --formation-base wide, --formation-height high rectangle; same gap condition. double_bowtie (8 robots): two bowties of width --formation-base / 2 side by side, sharing the middle column of corners (robots 4, 5). Default is regular.')
    parser.add_argument('--formation-base', type=float, default=4.0, help='Base of the isosceles formation, width of the bowtie (total width of the double bowtie). Default is 4.0.')
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
    parser.add_argument('--target-x', type=float, nargs=2, default=[0.0, 4.0], metavar=('X_MIN', 'X_MAX'), help='Range of the x of the target centroids (uniform). Keep the widest target formation inside the arena walls (--wall-x). Default is 0 4.')
    parser.add_argument('--target-y', type=float, nargs=2, default=[6.0, 6.1], metavar=('Y_MIN', 'Y_MAX'), help='Range of the y of the target centroids (uniform); the verification targets are at Y_MIN. Raise it so the targets turned along y clear the obstacles. Default is 6 6.1.')
    parser.add_argument('--target-rotation', action=bool_flag, default=True, help='Rotate every target formation by a random angle (else same orientation as the start). Default is True.')
    parser.add_argument('--target-max-angle', type=float, default=math.pi, help='With --target-rotation, target angles are uniform in [-a, a] around the start orientation (radians). pi: any orientation. Default is pi.')
    parser.add_argument('--target-flip', action=bool_flag, default=False, help='Also turn half of the targets by pi (on top of the random angle): flipped targets, with the base of a wide triangle still along x. Default is False.')
    parser.add_argument('--large-rot-frac', type=float, default=0.0, help='Fraction of the targets drawn from the large rotations only (|angle| >= --large-rot-min-angle, random sign); the others stay uniform over the whole range. Applied to the train and test data alike (same distribution). 0 disables it. Default is 0.')
    parser.add_argument('--large-rot-min-angle', type=float, default=2 * math.pi / 3, help='Smallest |angle| (radians, around the start orientation) of the oversampled large rotations. Default is 2pi/3 (120 deg).')
    parser.add_argument('--start-flip-frac', type=float, default=0.0, help='Fraction of the train and validation rollouts whose start formation is turned by pi about its centre (bowtie: robots 1, 2, the left corners, start bottom right and top right), with the target turned by pi about its own centre: the same turn to make, and the same communication graph between the robot numbers, from the other side of the arena. 0 disables it. Default is 0.')
    parser.add_argument('--test-start-flip-frac', type=float, default=None, help='The same for the test rollouts; the validation and test losses are then also logged per start (nominal / flipped). E.g. --start-flip-frac 0 --test-start-flip-frac 0.5 tests the flipped starts never trained on. Default is None (--start-flip-frac).')
    parser.add_argument('--init-min-dist', type=float, default=None, help='Every pair of robots starts at least this far apart: the noise of the initial state is redrawn until it does (train, validation and test alike). 0 disables it. Default is None (--min-dist with --col-av, a collision free start; else 0).')
    parser.add_argument('--resample-data', action=bool_flag, default=False, help='Draw fresh train rollouts (same distribution) every epoch, from a seed of --random-seed and the epoch: no fixed train set to overfit. The --num-rollouts rollouts of --random-seed are then a validation set never trained on. Default is False.')
    parser.add_argument('--init-seed', type=int, default=None, help='Seed of the model initialisation and of the batch shuffling, reset after the data is drawn: the train/test data stay those of --random-seed. Use it to repeat a run on the same data. Default is None (both follow --random-seed, as the runs saved before it).')
    parser.add_argument('--save-gifs', action=bool_flag, default=True, help='Save GIFs of every verification scenario (direct, flip, offset) before and after training. Default is True.')
    parser.add_argument('--gif-interval', type=int, default=4, help='Time steps between two GIF frames. Default is 4.')
    parser.add_argument('--run-id', type=str, default=None, help='Name of the run folder (instead of the start time). If that folder already holds a training_state.pt, the run continues from it in place (pod restarted by runai), before and instead of --resume. rcp_train.sh sets it to <job name>_<job uuid> so a preempted job resumes and a resubmitted one starts over. Default is None (timestamped folder, never resumes on its own).')
    parser.add_argument('--benchmark', type=str, default=None, help='Name of a benchmark (e.g. a sweep): the run folder goes to <n>_robots/<adjacency>/benchmark/<name>/ instead of <n>_robots/<adjacency>/, to keep the runs of a sweep together. Default is None (regular run).')
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
    if args.formation_shape in ('bowtie', 'double_bowtie'):
        n = 5 if args.formation_shape == 'bowtie' else 8
        if args.n_agents != n:
            raise ValueError('--formation-shape %s needs --n-agents %i.' % (args.formation_shape, n))
        if args.formation_base <= 0 or args.formation_height <= 0:
            raise ValueError('--formation-base and --formation-height must be > 0.')
    if not (args.target_x[0] <= args.target_x[1] and args.target_y[0] <= args.target_y[1]):
        raise ValueError('--target-x and --target-y need MIN <= MAX.')
    if args.target_max_angle < 0:
        raise ValueError('--target-max-angle must be >= 0.')
    if args.test_start_flip_frac is None:
        args.test_start_flip_frac = args.start_flip_frac
    if not (0 <= args.start_flip_frac <= 1 and 0 <= args.test_start_flip_frac <= 1):
        raise ValueError('--start-flip-frac and --test-start-flip-frac must be in [0, 1].')
    if args.init_min_dist is None:
        args.init_min_dist = args.min_dist if args.col_av else 0.0
    if args.init_min_dist < 0:
        raise ValueError('--init-min-dist must be >= 0.')
    if not 0 <= args.large_rot_frac <= 1:
        raise ValueError('--large-rot-frac must be in [0, 1].')
    if args.large_rot_frac > 0 and not (args.target_rotation and 0 <= args.large_rot_min_angle < min(args.target_max_angle, math.pi)):
        raise ValueError('--large-rot-frac needs --target-rotation and 0 <= --large-rot-min-angle < --target-max-angle.')
    if args.formation_shape == 'regular':
        shortest = args.formation_side
    elif args.formation_shape == 'isosceles':
        shortest = min(args.formation_base, math.hypot(args.formation_base / 2, args.formation_height))
    elif args.formation_shape == 'bowtie':   # corner to corner, or knot to corner
        shortest = min(args.formation_base, args.formation_height, math.hypot(args.formation_base, args.formation_height) / 2)
    else:   # double bowtie: corner to corner, or knot to corner
        shortest = min(args.formation_base / 2, args.formation_height, math.hypot(args.formation_base / 2, args.formation_height) / 2)
    if args.col_av and shortest <= args.min_dist + 0.2:
        print(f'Shortest formation distance {shortest:.3g} is within the collision margin --min-dist + 0.2 = {args.min_dist + 0.2}: '
              f'the formation and the collision loss conflict.')
    if args.run_id is not None and (not args.run_id or os.sep in args.run_id):
        raise ValueError('--run-id must be a non-empty folder name.')
    if args.benchmark is not None and (not args.benchmark or os.sep in args.benchmark):
        raise ValueError('--benchmark must be a non-empty folder name.')
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
    saved = saved_run_args(saved)
    resumed = argparse.Namespace(**{**vars(args), **saved})   # args added since that run keep their defaults
    resumed.resume = args.resume
    # how to run, not what to train: from the command line
    resumed.num_threads, resumed.compile = args.num_threads, args.compile
    resumed.run_id = args.run_id     # the new run's folder, not the one of the run it continues
    resumed.benchmark = args.benchmark
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
    elif args.formation_shape == 'double_bowtie':
        shape = 'double bowtie, width %g (2 x %g), height %g' % (args.formation_base, args.formation_base / 2, args.formation_height)
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
    msg += '\n[INFO] Seeds: data %i -- init %s -- train data: %s -- initial pairs >= %g apart' % (
        args.random_seed, 'same' if args.init_seed is None else args.init_seed,
        'resampled every epoch (fixed rollouts: validation)' if args.resample_data else 'fixed (also the validation)',
        args.init_min_dist)
    if args.start_flip_frac or args.test_start_flip_frac:
        msg += ' -- starts turned by pi (target too): %.0f%% of the train/validation, %.0f%% of the test rollouts' % (
            100 * args.start_flip_frac, 100 * args.test_start_flip_frac)
    msg += '\n[INFO] Formation: %s -- graph: %s -- alpha_form: %g -- deadband: %g -- from t = %i -- target rotation: %s' % (
        shape, args.formation_graph, args.alpha_form,
        args.formation_deadband, args.formation_start, rotation)
    msg += ' -- target centroids: x in [%g, %g], y in [%g, %g]' % (*args.target_x, *args.target_y)
    if args.alpha_orient or args.alpha_orient_max:
        msg += ' -- alpha_orient: %g, alpha_orient_max: %g (margin %g)' % (
            args.alpha_orient, args.alpha_orient_max, args.orient_margin)
    if args.alpha_wall:
        msg += '\n[INFO] Arena walls: x in [%g, %g] -- alpha_wall: %g (margin %g)' % (
            args.wall_x[0], args.wall_x[1], args.alpha_wall, args.wall_margin)
    return msg
