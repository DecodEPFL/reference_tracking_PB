import argparse, math

from Distributed_Danilo.controllers.factory import CTX_FILTERS, INFO_PATTERNS, MB_TYPES, MP_TYPES
from Distributed_Danilo.controllers.interconnection import ADJACENCY_GRAPHS


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
    parser.add_argument('--share-weights', action=bool_flag, default=False, help='Share one operator M across all agents instead of an independent M^i per agent. Default is False.')
    parser.add_argument('--batch-agents', action=bool_flag, default=False, help='With independent weights, run the N agent operators in one vectorised call (torch.func.vmap) instead of one after the other: same results up to float rounding, faster. Falls back to the loop (with a warning) if unsupported. Default is False.')
    parser.add_argument('--pad-agent-inputs', action=bool_flag, default=True, help='With --batch-agents, agents with fewer neighbors than others (unequal input sizes, e.g. an irregular --adjacency graph) get zero channels in place of the missing ones, so all agents can still be batched; the weights reading the padded channels are held at zero, so every agent computes what it would unpadded. Without it such agents run one by one. Default is True.')
    parser.add_argument('--num-threads', type=int, default=1, help='PyTorch intra-op threads. The closed-loop rollouts are many tiny ops, so extra threads only spin: keep 1 and run several jobs in parallel instead. Set OMP_NUM_THREADS / MKL_NUM_THREADS in the environment as well, they are read before the args. Default is 1.')
    parser.add_argument('--compile', action=bool_flag, default=False, help='With --batch-agents, compile the vmapped agent step with torch.compile: about 2.4x faster per training step on CPU, same results up to float rounding; the first steps pay the compilation (needs a C++ compiler). Default is False.')
    parser.add_argument('--match-centralized-size', action=bool_flag, default=True, help='With independent weights, shrink the widths of every M^i so all N operators together have at most the parameters of the centralized M. Default is True.')
    parser.add_argument('--info-pattern', type=str, default='local', choices=INFO_PATTERNS, help='Signals seen by M^i: its own w^i and z^i (local); all of w and z as the centralized M sees them (full; own agent first with --share-weights); or local plus, for each neighbor j on --adjacency, the relative position, distance and previous input u^j_{t-1} (neighbors; plus the target z^j with --neighbor-targets); or exactly the inputs of the centralized controller with --pair-features: all of w and z plus the relative position of every pair of robots measured from x_t (centralized; independent weights only); or the centralized inputs relayed over --adjacency, every robot forwarding what it receives with --relay-hop-delay steps per hop: the disturbances w^j_{t-d} of every robot j reaching it in the SSM input, and its targets z^j_{t-d} and relative positions p^j_{t-d} - p^i_{t-d} in the context, d the delay of the shortest path from j (relay; plus the inputs u^j_{t-d} with --relay-inputs). Default is local.')
    parser.add_argument('--neighbor-drag', action=bool_flag, default=False, help='Under --info-pattern neighbors with --patches, every neighbor j sends its measured drag s^j_t instead of its distance ||p^j_t - p^i_t|| (the context signals of --info-pattern relay in Distributed/). Default is False.')
    parser.add_argument('--neighbor-targets-input', action=bool_flag, default=False, help='Under --info-pattern neighbors with --neighbor-targets, the targets z^j of the neighbors also drive the input port of the contextual SSM (filtered, as exogenous inputs of the certificate), as the relayed targets of --info-pattern relay in Distributed/; otherwise only the own target does. The context becomes [z^i, z^j of the neighbors, drag^i, other neighbor signals]. Default is False.')
    parser.add_argument('--neighbor-delay', type=int, default=0, help='Under --info-pattern neighbors, steps of delay of the neighbors\' positions, drags and targets: agent i forms p^j_{t-d} - p^i_{t-d} and reads s^j_{t-d}, z^j_{t-d} (1: a one-hop message of --info-pattern relay in Distributed/); u^j_{t-1} is not delayed further. Default is 0 (measured at t).')
    parser.add_argument('--neighbor-targets', action=bool_flag, default=True, help='Under --info-pattern neighbors, every neighbor j also sends its target z^j. Default is True.')
    parser.add_argument('--relay-inputs', action=bool_flag, default=False, help='Under --info-pattern relay, every robot also relays its input u^j (context only). Default is False.')
    parser.add_argument('--relay-inputs-to', type=str, default='mb', choices=['mb', 'mixers', 'all'], help='Parts of the operator reached by the inputs relayed with --relay-inputs: the external mixer M_b only (mb); the memoryless mixers, M_b and the mixer port of the contextual SSM (mixers); or also its gate and select ports, which shape the SSM dynamics (all). With mb and mixers they never reach the SSM state. Default is mb.')
    parser.add_argument('--relay-inputs-scale', type=float, default=0.05, help='Under --info-pattern relay with --relay-inputs, factor applied to the relayed inputs u^j before they enter the context (about 3x the targets and 7-9x the offsets in the transient otherwise). Default is 0.05 (1/20).')
    parser.add_argument('--relay-max-hops', type=int, default=0, help='Under --info-pattern relay, every robot only hears from the robots at most this many hops away on --adjacency: 1 keeps its in-neighbors only, nothing is forwarded. 0: no limit (everything that reaches it). Default is 0.')
    parser.add_argument('--relay-hop-delay', type=int, default=1, help='Under --info-pattern relay, steps of delay per hop of the communication graph; 0 shares instantly (on the complete graph: the centralized inputs). Default is 1.')
    parser.add_argument('--adjacency', type=str, default='complete', help='Communication graph of --info-pattern neighbors and relay, A[i, j] = 1 if robot i receives from robot j: one of %s, or an explicit 0/1 matrix with rows separated by ";", e.g. "0,1;1,0". Default is complete.' % ', '.join(ADJACENCY_GRAPHS))
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
    parser.add_argument('--ctx-head-hidden', type=int, default=64, help='Hidden width of the mixer and gate context heads. Default is 64.')
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
    # the SSM flags above (--mp-type, --ssm-*, --ctx-*, --feat-dim, --mb-type, --match-centralized-size, ...) build
    # the M_p and M_b of --use-ssm; the REN-MAD controller ignores them (its direction networks use --mb-*).
    # --batch-agents, --compile, --pad-agent-inputs, the l_p lift and the legacy REN flags are ignored by both.

    # REN-MAD controller (controllers/ren_mad.py; defaults of dPB-microgrid/ren_mad/src/settings.py)
    parser.add_argument('--use-ssm', action=bool_flag, default=False, help='Replace the RENs and the MAD readout by the M_p and M_b of the SSM controller of Distributed/, built from the same flags (--mp-type ssm/contextual_ssm, --ssm-*, --ctx-*, --mb-type, --feat-dim, --mb-*, --match-centralized-size), with the commands u^j_{t-1} of the --adjacency neighbors (the whole factorized output) added to the velocity channels of w^i at the input of every recurrent core (same sizes as Distributed/, no gate). The network certificate (--ren-gamma, --ren-epsilon, --ren-edge-*) sets the gain of every M_p, which needs a certified --ssm-param (e.g. tv, tvc). Default is False (REN-MAD).')
    parser.add_argument('--feedback-mode', type=str, default='u_velocity', choices=['u_velocity', 'y_append', 'core_append'], help='With --use-ssm, what is fed back between the agents, summed over the --adjacency neighbors: the commands u^j (the whole factorized output, 2-D) added to the velocity channels of w^i (u_velocity); the M_p outputs y^j (s = --feat-dim channels) appended to the input of the recurrent core, M_b out of the loop (y_append); or, as the REN outputs in ren_mad, the outputs h^j of the recurrent cores (--ctx-d-features channels, before the mixer port of the contextual SSM) appended to the core input, so that the agents interconnect only through their recurrent l2 cores and every memoryless head (mixer port, M_b) stays out of the loop (core_append). Default is u_velocity.')
    parser.add_argument('--feedback-norm', type=str, default='constant', choices=['constant', 'none'], help='With --use-ssm, the fed-back commands: u^j divided by the readout bound K_j of agent j, the gain from its core output to u^j (mixer bound x mb_bound sqrt(nu s)), so the fed-back signal has the gain of a core (constant); or the raw u^j (none). The certificate is far less conservative with the normalized signals (about 177x more coupling at the same core gain for the formation defaults); --ren-edge-weight is in the units of the fed-back signal. Default is constant.')
    parser.add_argument('--ssm-init', type=str, default='free', choices=['free', 'capped'], help='With --use-ssm, initialization of the gain-capped M_p cores: as the free (uncapped) cores of Distributed/, block gains 1 and nn.Linear encoder / decoder (free), or neural_ssm\'s own capped initialization, block gains derived from the cap and orthogonal encoder / decoder (capped). Default is free (same starting network as Distributed/).')
    parser.add_argument('--ren-gamma',type=float, default=25.0, help='Prescribed l2 gain gamma_R of the network of certified RENs, from the disturbances w to the magnitudes y (the local gains follow from it and the graph). Default is 25.')
    parser.add_argument('--ren-epsilon', type=float, default=0.02, help='Margin epsilon of the network certificate. Default is 0.02.')
    parser.add_argument('--ren-state-dim', type=int, default=6, help='State dimension of every REN. Default is 6.')
    parser.add_argument('--ren-width', type=int, default=6, help='Width of the equilibrium (nonlinear) layer of every REN. Default is 6.')
    parser.add_argument('--ren-init-std', type=float, default=0.1, help='Initialization std of the REN free parameters. Default is 0.1.')
    parser.add_argument('--ren-edge-weight', type=float, default=1e-3, help='Communication weight a_ij of every edge of --adjacency: REN i receives sum_j a_ij y^j. Default is 1e-3.')
    parser.add_argument('--ren-learn-edges', action=bool_flag, default=False, help='Train the communication weights (fixed within a rollout; the certificate holds for any value). Default is False (frozen, as in the ren_mad training).')
    parser.add_argument('--ren-edge-bounds', type=float, nargs=2, default=[1e-4, 1.0], metavar=('LOWER', 'UPPER'), help='Bounds of the learned communication weights (sigmoid). Default is 1e-4 1.')

    # loss
    parser.add_argument('--alpha-u', type=float, default=0.1/400, help='Weight of the loss due to control input "u". Default is 0.1/400.')  #TODO: 400 is output_amplification^2
    parser.add_argument('--alpha-col', type=float, default=100, help='Weight of the collision avoidance loss. Default is 100 if "col-av" is True, else None.')
    parser.add_argument('--alpha-obst', type=float, default=5e3, help='Weight of the obstacle avoidance loss. Default is 5e3 if "obst-av" is True, else None.')
    parser.add_argument('--alpha-obst-max', type=float, default=0.0, help='Weight of the worst-step obstacle loss: sum over robots of the max over time of the obstacle penalty (not time-averaged, so a fast pass through an obstacle is not cheap). 0 disables it. Default is 0.')
    parser.add_argument('--obst-loss', type=str, default='gaussian', choices=['gaussian', 'barrier'], help='Obstacle penalty: Gaussian bump, or barrier (R/d - 1)^2 inside radius R = --obst-radius. Default is gaussian.')
    parser.add_argument('--col-loss', type=str, default='inverse', choices=['inverse', 'hinge'], help='Collision penalty per pair closer than --min-dist + 0.2: 1/d^2 (inverse, gradient up to ~1e8 near contact), or a squared hinge equal to it at --min-dist with a bounded gradient (hinge). Default is inverse.')
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
    parser.add_argument('--lr-schedule', type=str, default='constant', choices=['constant', 'cosine'], help='constant lrs, or a linear warmup over --lr-warmup epochs then a cosine decay to --lr-final-ratio times the lrs at --epochs. Default is constant.')
    parser.add_argument('--lr-warmup', type=int, default=0, help='Warmup epochs of the cosine schedule. Default is 0.')
    parser.add_argument('--lr-final-ratio', type=float, default=0.01, help='Final lrs of the cosine schedule relative to the initial ones. Default is 0.01.')
    parser.add_argument('--grad-clip', type=float, default=0.0, help='Clip the gradient norm to this value. 0 disables it. Default is 0.')
    parser.add_argument('--grad-clip-rel', type=float, default=0.0, help='Clip the gradient norm to this times the median of the last 1000 unclipped norms, a threshold that follows the norms as they shrink. Combined with --grad-clip, the smaller bound applies. 0 disables it. Default is 0.')
    parser.add_argument('--grad-diag', type=float, default=0.0, help='On a step whose gradient norm exceeds this times the running median, recompute the batch and log the gradient norm of each loss term, to find the term behind the spike (costs one forward and one backward per term, a few times per log period at most). 0 disables it. Default is 0.')
    parser.add_argument('--ema-decay', type=float, default=0.0, help='Keep an exponential moving average of the weights with this decay per optimizer step; validation, the best-model selection and the returned model use the averaged weights. 0 disables it. Default is 0.')
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
    if args.grad_clip < 0 or args.grad_clip_rel < 0:
        raise ValueError('--grad-clip and --grad-clip-rel must be >= 0.')
    if args.grad_diag and args.grad_diag <= 1:
        raise ValueError('--grad-diag must be 0 (off) or > 1.')
    if args.lr_warmup < 0 or not 0 < args.lr_final_ratio <= 1:
        raise ValueError('--lr-warmup must be >= 0 and --lr-final-ratio in (0, 1].')
    if not 0 <= args.ema_decay < 1:
        raise ValueError('--ema-decay must be in [0, 1).')
    if not args.col_av:
        args.alpha_col = None
        args.min_dist = None
    if args.alpha_obst_max < 0:
        raise ValueError('--alpha-obst-max must be >= 0.')
    if args.relay_inputs_scale <= 0:
        raise ValueError('--relay-inputs-scale must be > 0.')
    if args.relay_max_hops < 0:
        raise ValueError('--relay-max-hops must be >= 0 (0: no limit).')
    if args.relay_hop_delay < 0:
        raise ValueError('--relay-hop-delay must be >= 0.')
    if args.ren_gamma <= 0 or args.ren_epsilon <= 0 or args.ren_edge_weight <= 0 or args.ren_init_std <= 0:
        raise ValueError('--ren-gamma, --ren-epsilon, --ren-edge-weight and --ren-init-std must be > 0.')
    if args.ren_state_dim < 1 or args.ren_width < 1:
        raise ValueError('--ren-state-dim and --ren-width must be >= 1.')
    if args.neighbor_delay < 0:
        raise ValueError('--neighbor-delay must be >= 0.')
    if args.ren_learn_edges and not 0 < args.ren_edge_bounds[0] < args.ren_edge_weight < args.ren_edge_bounds[1]:
        raise ValueError('--ren-learn-edges needs 0 < LOWER < --ren-edge-weight < UPPER (--ren-edge-bounds).')
    if not args.obst_av:
        args.alpha_obst = None
        args.alpha_obst_max = 0.0
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

    info = args.info_pattern
    if args.info_pattern == 'neighbors':
        info += ' on %s graph%s' % (args.adjacency, ', with targets' if args.neighbor_targets else '') + (', drag instead of distance' if getattr(args, 'neighbor_drag', False) else '') + (', targets to the SSM input' if getattr(args, 'neighbor_targets_input', False) else '') + (', delayed %i step(s)' % args.neighbor_delay if getattr(args, 'neighbor_delay', 0) else '')
    elif args.info_pattern == 'relay':
        info += ' on %s graph, %i step(s) per hop' % (args.adjacency, args.relay_hop_delay)
        if args.relay_max_hops:
            info += ', at most %i hop(s)' % args.relay_max_hops
        if args.relay_inputs:
            info += ', with inputs (to %s, x%g)' % (args.relay_inputs_to, args.relay_inputs_scale)
    edges = 'edge weights %g (%s)' % (args.ren_edge_weight, 'learned in [%g, %g]' % tuple(args.ren_edge_bounds)
                                       if args.ren_learn_edges else 'fixed')
    if getattr(args, 'use_ssm', False):
        msg += '\n[INFO] Controller: distributed certified SSM network (info: %s) -- context: %s' % (info, args.context)
        what = {'y_append': 'M_p outputs y appended to the core input',
                'core_append': 'core outputs h appended to the core input'}.get(
                    getattr(args, 'feedback_mode', 'u_velocity'), 'commands fed back to the velocity channels')
        msg += ' -- %s (%s) over the %s graph: gamma_R %g, %s' % (what, 'normalized by K_j' if args.feedback_norm == 'constant' else 'raw', args.adjacency, args.ren_gamma, edges)
        msg += ' -- M_p: %s -- M_b: %s' % (args.mp_type, args.mb_type)
        if args.mb_type == 'bounded_mlp':
            msg += ' -- rank s: %i' % args.feat_dim
        msg += ' -- SSM %s: %i layers, d_model %i, d_state %i' % (args.ssm_param, args.ssm_layers, args.ssm_d_model, args.ssm_d_state)
        if args.mp_type == 'contextual_ssm':
            msg += ' -- ctx ports: %s%s, filter: %s' % (args.ctx_modes, '+select' if args.ctx_select else '', args.ctx_filter)
    else:
        msg += '\n[INFO] Controller: distributed REN-MAD (info: %s) -- context: %s' % (info, args.context)
        msg += ' -- certified RENs on the %s graph: gamma_R %g, state %i, width %i, %s' % (
            args.adjacency, args.ren_gamma, args.ren_state_dim, args.ren_width, edges)
        msg += ' -- MAD direction M_b: %i layers, hidden %i, bound %g' % (args.mb_layers, args.mb_hidden, args.mb_bound)

    msg += '\n[INFO] Loss:  alpha_x: %g -- alpha_u: %.6f' % (args.alpha_x, args.alpha_u)
    if args.alpha_final >= 0:
        msg += ' -- alpha_final: %g (t > %i)' % (args.alpha_final, args.final_start)
    msg += ' -- alpha_speed: %s' % ('Q (legacy)' if args.legacy_speed_weight else '%g' % args.alpha_speed)
    msg += ' -- alpha_col: %.f (min distance %g)' % (args.alpha_col, args.min_dist) if args.col_av else ' -- no collision avoidance'
    if args.col_av and getattr(args, 'col_loss', 'inverse') == 'hinge':
        msg += ' (hinge)'
    msg += ' -- alpha_obst: %.1f' % args.alpha_obst if args.obst_av else ' -- no obstacle avoidance'
    if args.obst_av and args.obst_loss == 'barrier':
        msg += ' (barrier, radius %g)' % args.obst_radius
    elif args.obst_av and args.obst_radius > 0:
        msg += ' (cut off at radius %g)' % args.obst_radius
    if args.obst_av and args.alpha_obst_max:
        msg += ' -- alpha_obst_max: %g' % args.alpha_obst_max

    msg += '\n[INFO] Optimizer: lr: %.2e (x%.2f for mixer/heads)' % (args.lr, args.lr_ratio)
    if getattr(args, 'lr_schedule', 'constant') == 'cosine':
        msg += ' -- cosine schedule to x%g (warmup %i epochs)' % (args.lr_final_ratio, args.lr_warmup)
    if getattr(args, 'grad_clip', 0) or getattr(args, 'grad_clip_rel', 0):
        msg += ' -- grad clip: %s' % ' & '.join(([] if not args.grad_clip else ['%g' % args.grad_clip]) +
                                                 ([] if not args.grad_clip_rel else ['%g x median' % args.grad_clip_rel]))
    if getattr(args, 'grad_diag', 0):
        msg += ' -- per-term diagnosis of the grad spikes above %g x median' % args.grad_diag
    if getattr(args, 'ema_decay', 0):
        msg += ' -- EMA of the weights: %g' % args.ema_decay
    msg += ' -- batch_size: %i' % args.batch_size + ', -- return best model for validation data among logged epochs:' + str(args.return_best)
    msg += '\n[INFO] Threads: %i' % args.num_threads

    return msg
