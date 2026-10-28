import argparse, math

from Centralized.controllers.factory import CTX_FILTERS, MB_TYPES, MP_TYPES


# argument parser
def argument_parser(argv=None):
    """Parse the robots experiment arguments (build_parser + finalize_args)."""
    return finalize_args(build_parser().parse_args(argv))


def build_parser(description="Robots minimal experiment."):
    """Parser with every robots argument; other experiments extend it (add arguments, set_defaults)."""
    parser = argparse.ArgumentParser(description=description)
    bool_flag = argparse.BooleanOptionalAction   # --flag / --no-flag

    # experiment
    parser.add_argument('--random-seed', type=int, default=5, help='Random seed. Default is 5.')
    parser.add_argument('--col-av', action=bool_flag, default=True, help='Avoid collisions. Default is True.')
    parser.add_argument('--obst-av', action=bool_flag, default=True, help='Avoid obstacles. Default is True.')

    # dataset
    parser.add_argument('--horizon', type=int, default=100, help='Time horizon for the computation. Default is 100.')
    parser.add_argument('--n-agents', type=int, default=2, help='Number of agents. Default is 2.')
    parser.add_argument('--num-rollouts', type=int, default=30, help='Number of rollouts in the training data. Default is 30.')
    parser.add_argument('--std-init-plant', type=float, default=0.2, help='std of the plant initial conditions. Default is 0.2.')

    # plant
    parser.add_argument('--spring-const', type=float, default=1.0 , help='Spring constant. Default is 1.0.')
    parser.add_argument('--linearize-plant', action=bool_flag, default=False, help='Linearize plant or not. Default is False.')

    # controller: PB operator u = M_b(w, z) ⊠ M_p(w, z), z = reference positions
    parser.add_argument('--context', type=str, default='ref', choices=['ref', 'none'], help='Context z fed to the operator: the reference positions, or zeros (disturbance-only baseline). Default is ref.')
    parser.add_argument('--pair-features', action=bool_flag, default=False, help='Append to the context the relative position (dx, dy) of every pair of robots, measured from the state x_t (3 robots: 6 features). They enter the gate/mixer/select ports, not the SSM input port. Default is False.')
    parser.add_argument('--mp-type', type=str, default='contextual_ssm', choices=MP_TYPES, help='M_p factor. Default is contextual_ssm.')
    parser.add_argument('--mb-type', type=str, default='bounded_mlp', choices=MB_TYPES, help='M_b mixer (none: u = M_p directly). Default is bounded_mlp.')
    parser.add_argument('--output-amplification', type=float, default=1.0, help='Scalar gain on the operator output. Default is 20.')
    parser.add_argument('--use-w-augment', action=bool_flag, default=False, help='Feed [w, leaky integral of w] to both factors. Default is False.')
    parser.add_argument('--w-augment-decay', type=float, default=0.97, help='Leak factor of the w integral. Default is 0.97.')

    # M_b: bounded mixer
    parser.add_argument('--feat-dim', type=int, default=16, help='Rank s of the factorization (M_p output width); 1 gives the MAD policy. Default is 16.')
    parser.add_argument('--mb-hidden', type=int, default=64, help='Hidden width of M_b. Default is 64.')
    parser.add_argument('--mb-layers', type=int, default=4, help='Number of layers of M_b. Default is 4.')
    parser.add_argument('--mb-bound', type=float, default=8.0, help='Entry-wise bound of M_b. Default is 8.')
    parser.add_argument('--z-residual-gain', type=float, default=10.0, help='Gain of the context-only residual branch of M_b. Default is 10.')

    # M_p: SSM core (shared by ssm and contextual_ssm)
    parser.add_argument('--ssm-param', type=str, default='lru', choices=['lru', 'tv', 'tvc'], help='SSM parametrization; select needs tv/tvc. Default is lru.')
    parser.add_argument('--ssm-layers', type=int, default=4, help='Number of SSM blocks. Default is 4.')
    parser.add_argument('--ssm-d-model', type=int, default=32, help='SSM model width. Default is 32.')
    parser.add_argument('--ssm-d-state', type=int, default=64, help='SSM state dimension. Default is 64.')
    parser.add_argument('--ssm-ff', type=str, default='LGLU2', help='SSM feed-forward block. Default is GLU.')
    parser.add_argument('--ssm-bcd-nonlinearity', type=str, default='tanh', choices=['tanh', 'identity'], help='Nonlinearity of the B/C/D maps (tvc only). Default is tanh.')

    # M_p: contextual SSM ports
    parser.add_argument('--ctx-modes', type=str, default='mixer,input,gate', help='Comma-separated context ports of the contextual SSM among input, gate, mixer, select. Default is mixer,input,gate.')
    parser.add_argument('--ctx-select', action=bool_flag, default=False, help='Add the select port (needs --ssm-param tv/tvc). Default is False.')
    parser.add_argument('--ctx-filter', type=str, default='taper', choices=CTX_FILTERS, help='l2 filter applied to the context before the input/select ports. Default is taper.')
    parser.add_argument('--ctx-filter-ramp', type=int, default=20, help='Ramp length of the taper filter (0: None). Default is 20.')
    parser.add_argument('--ctx-filter-decay', type=float, default=0.98, help='Per-step factor rho of the exponential filter (window rho^t). Default is 0.98.')
    parser.add_argument('--ctx-filter-power', type=float, default=1.0, help='Power p of the polynomial filter (window (t+1)^-p, p > 0.5). Default is 1.')
    parser.add_argument('--ctx-mixer-bound', type=float, default=4.0, help='Norm bound of the internal mixer port. Default is 4.')
    parser.add_argument('--ctx-d-features', type=int, default=16, help='Core feature width before the internal mixer port. Default is 16.')
    parser.add_argument('--ctx-gamma', type=float, default=0.0, help='Prescribed l2 gain of the core (<=0: free). Default is 0.')
    parser.add_argument('--ctx-gate-per-channel', action=bool_flag, default=False, help='Per-channel instead of per-block gates. Default is False.')
    parser.add_argument('--ctx-z-scale', type=float, default=1.0, help='Rescale of z before the contextual ports. Default is 1.')

    # M_p: l_p context lift (ssm / ren only)
    parser.add_argument('--mp-context-lift', action=bool_flag, default=False, help='Feed [w, lift(z)] to a non-contextual M_p. Default is False.')
    parser.add_argument('--mp-context-lift-type', type=str, default='linear', choices=['identity', 'linear', 'mlp'], help='Lift map. Default is linear.')
    parser.add_argument('--mp-context-lift-dim', type=int, default=6, help='Lift output width. Default is 6.')
    parser.add_argument('--mp-context-hidden-dim', type=int, default=24, help='Hidden width of the mlp lift. Default is 24.')
    parser.add_argument('--mp-context-decay-law', type=str, default='finite', choices=['exp', 'poly', 'finite'], help='Decay law making the lift l_p. Default is finite.')
    parser.add_argument('--mp-context-decay-rate', type=float, default=0.04, help='Rate of the exp law. Default is 0.04.')
    parser.add_argument('--mp-context-decay-power', type=float, default=0.73, help='Power of the poly law. Default is 0.73.')
    parser.add_argument('--mp-context-decay-horizon', type=int, default=-1, help='Horizon of the finite law (-1: --horizon). Default is -1.')
    parser.add_argument('--mp-context-lp-p', type=float, default=2.0, help='p of the l_p space. Default is 2.')
    parser.add_argument('--mp-context-scale', type=float, default=0.25, help='Scale of the lift. Default is 0.25.')

    # M_p: legacy contractive REN
    parser.add_argument('--cont-init-std', type=float, default=0.1, help='Initialization std for REN params. Default is 0.1.')
    parser.add_argument('--dim-internal', type=int, default=8, help='Dimension of the internal state of the REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=8, help='size of the non-linear part of REN. Default is 8.')

    # loss
    parser.add_argument('--alpha-u', type=float, default=0.1/400, help='Weight of the loss due to control input "u". Default is 0.1/400.')  #TODO: 400 is output_amplification^2
    parser.add_argument('--alpha-col', type=float, default=100, help='Weight of the collision avoidance loss. Default is 100 if "col-av" is True, else None.')
    parser.add_argument('--alpha-obst', type=float, default=5e3, help='Weight of the obstacle avoidance loss. Default is 5e3 if "obst-av" is True, else None.')
    parser.add_argument('--obst-loss', type=str, default='gaussian', choices=['gaussian', 'barrier'], help='Obstacle penalty: Gaussian bump, or barrier (R/d - 1)^2 inside radius R = --obst-radius. Default is gaussian.')
    parser.add_argument('--obst-radius', type=float, default=-1, help='Cut the obstacle penalty off at this distance from each obstacle center (continuous, 0 beyond). <= 0 keeps the untruncated Gaussian; required (> 0) for the barrier. Default is -1.')
    parser.add_argument('--min-dist', type=float, default=1.0, help='Minimum distance between agents counted as a collision. Default is 1.0 if "col-av" is True, else None.')
    parser.add_argument('--alpha-x', type=float, default=100.0, help='Weight of the tracking loss on the distance of each robot to its target (Q = alpha_x * I). Default is 100.')
    parser.add_argument('--alpha-final', type=float, default=-1, help='Tracking weight replacing --alpha-x for the steps t > --final-start (terminal cost). < 0 disables it (alpha-x at every step). Default is -1.')
    parser.add_argument('--final-start', type=int, default=150, help='Last step weighted by --alpha-x; --alpha-final applies after it. Default is 150.')
    parser.add_argument('--alpha-speed', type=float, default=1.0, help='Weight of the speed penalty on [vx, vy] of every agent (Qs = alpha_speed * I). Ignored with --legacy-speed-weight. Default is 1.')
    parser.add_argument('--legacy-speed-weight', action=bool_flag, default=False, help='Weight the speed penalty with Q (as in the original code) instead of Qs. Default is False.')

    # optimizer
    parser.add_argument('--batch-size', type=int, default=5, help='Number of forward trajectories of the closed-loop system at each step. Default is 5.')
    parser.add_argument('--epochs', type=int, default=-1, help='Total number of epochs for training. Default is 1000 if collision avoidance, else 50.')
    parser.add_argument('--lr', type=float, default=-1, help='Learning rate of the M_p core. Default is 2e-3 if collision avoidance, else 5e-3.')
    parser.add_argument('--lr-ratio', type=float, default=0.5, help='Learning rate of the mixer/heads relative to --lr. Default is 0.5.')
    parser.add_argument('--log-epoch', type=int, default=-1, help='Frequency of logging in epochs. Default is 0.1 * epochs.')
    parser.add_argument('--return-best', action=bool_flag, default=True, help='Return the best model on the validation data among all logged iterations. The train data can be used instead of validation data. The Default is True.')

    return parser


def finalize_args(args):
    """Fill the defaults that depend on other args, then check the args."""
    # set default values that depend on other args
    if args.batch_size == -1:
        args.batch_size = args.num_rollouts  # use all train data

    if args.epochs == -1 or args.epochs is None:
        args.epochs = 1000 if args.col_av else 50

    if args.lr == -1 or args.lr is None:
        args.lr = 2e-3 if args.col_av else 5e-3

    if args.log_epoch == -1 or args.log_epoch is None:
        args.log_epoch = max(1, math.ceil(float(args.epochs)/10))

    # assertions and warning
    if not args.col_av:
        args.alpha_col = None
        args.min_dist = None
    if not args.obst_av:
        args.alpha_obst = None
    if args.obst_av and args.obst_loss == 'barrier' and args.obst_radius <= 0:
        raise ValueError('--obst-loss barrier needs --obst-radius > 0.')

    if args.alpha_final >= 0 and args.horizon <= args.final_start + 1:
        print(f'--alpha-final has no effect: horizon {args.horizon} has no step after --final-start {args.final_start}')

    if args.horizon > 100:
        print(f'Long horizons may be unnecessary and pose significant computation')

    return args


def print_args(args):
    msg = '\n[INFO] Dataset: n_agents: %i' % args.n_agents + ' -- num_rollouts: %i' % args.num_rollouts
    msg += ' -- std_ini: %.2f' % args.std_init_plant + ' -- time horizon: %i' % args.horizon

    msg += '\n[INFO] Plant: spring constant: %.2f' % args.spring_const + ' -- use linearized plant: ' + str(args.linearize_plant)

    msg += '\n[INFO] Controller: M_p: %s -- M_b: %s -- context: %s' % (args.mp_type, args.mb_type, args.context)
    if args.pair_features:
        msg += ' + relative positions of the robot pairs'
    if args.mb_type == 'bounded_mlp':
        msg += ' -- rank s: %i' % args.feat_dim
    if args.mp_type == 'ren':
        msg += ' -- REN dim_internal: %i -- dim_nl: %i' % (args.dim_internal, args.dim_nl)
    else:
        msg += ' -- SSM %s: %i layers, d_model %i, d_state %i' % (args.ssm_param, args.ssm_layers, args.ssm_d_model, args.ssm_d_state)
    if args.mp_type == 'contextual_ssm':
        msg += ' -- ctx ports: %s%s, filter: %s' % (args.ctx_modes, '+select' if args.ctx_select else '', args.ctx_filter)

    msg += '\n[INFO] Loss:  alpha_x: %g -- alpha_u: %.6f' % (args.alpha_x, args.alpha_u)
    if args.alpha_final >= 0:
        msg += ' -- alpha_final: %g (t > %i)' % (args.alpha_final, args.final_start)
    msg += ' -- alpha_speed: %s' % ('Q (legacy)' if args.legacy_speed_weight else '%g' % args.alpha_speed)
    msg += ' -- alpha_col: %.f' % args.alpha_col if args.col_av else ' -- no collision avoidance'
    msg += ' -- alpha_obst: %.1f' % args.alpha_obst if args.obst_av else ' -- no obstacle avoidance'
    if args.obst_av and args.obst_loss == 'barrier':
        msg += ' (barrier, radius %g)' % args.obst_radius
    elif args.obst_av and args.obst_radius > 0:
        msg += ' (cut off at radius %g)' % args.obst_radius

    msg += '\n[INFO] Optimizer: lr: %.2e (x%.2f for mixer/heads)' % (args.lr, args.lr_ratio)
    msg += ' -- batch_size: %i' % args.batch_size + ', -- return best model for validation data among logged epochs:' + str(args.return_best)

    return msg
