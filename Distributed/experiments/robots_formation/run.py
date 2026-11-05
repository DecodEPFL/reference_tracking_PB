import sys, os, copy, logging, math, subprocess, torch, time
from functools import partial
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(BASE_DIR))  # parent of Distributed/, for `Distributed.*` imports

from Distributed.config import device
from Distributed.experiments.robots_formation.arg_parser import argument_parser, print_args, training_state_path
from Distributed.plants import RobotsSystem, RobotsFormationDataset
from Distributed.plants.robots.formation_dataset import formation_state
from Distributed.plants.robots.robots_sys import append_patches, sample_friction_patches
from Distributed.utils.plot_functions import *
from Distributed.controllers import build_controller, agent_operators, ssm_core_parameters
from Distributed.controllers.interconnection import adjacency_folder, make_adjacency
from Distributed.loss_functions import RobotsFormationLoss
from Distributed.utils.assistive_functions import WrapLogger
from Distributed.utils.training import GradClipper, SpikeDiagnoser, WeightEMA, set_lr, term_grad_norms


def code_version():
    """Git commit of the repo (+dirty if uncommitted changes), None if git is unavailable."""
    try:
        git = lambda *cmd: subprocess.run(['git', '-C', BASE_DIR, *cmd], capture_output=True, text=True,
                                          check=True).stdout.strip()
        return git('rev-parse', '--short', 'HEAD') + ('+dirty' if git('status', '--porcelain', '--untracked-files=no') else '')
    except (OSError, subprocess.CalledProcessError):
        return None


def make_dataset(args):
    """Formation dataset of the parsed args (runs saved before the isosceles shape: regular n-gon, full rotation)."""
    return RobotsFormationDataset(
        random_seed=args.random_seed,
        horizon=args.horizon,
        n_agents=args.n_agents,
        side=args.formation_side,
        std_ini=args.std_init_plant,
        target_x=tuple(getattr(args, 'target_x', (0.0, 4.0))),
        target_y=tuple(getattr(args, 'target_y', (6.0, 6.1))),
        target_rotation=args.target_rotation,
        target_max_angle=getattr(args, 'target_max_angle', math.pi),
        target_flip=getattr(args, 'target_flip', False),
        shape=getattr(args, 'formation_shape', 'regular'),
        base=getattr(args, 'formation_base', None),
        height=getattr(args, 'formation_height', None),
        large_rot_frac=getattr(args, 'large_rot_frac', 0.0),
        large_rot_min_angle=getattr(args, 'large_rot_min_angle', 2 * math.pi / 3),
        init_min_dist=getattr(args, 'init_min_dist', 0.0),
        start_flip_frac=getattr(args, 'start_flip_frac', 0.0),
        test_start_flip_frac=getattr(args, 'test_start_flip_frac', None),
    )


def make_plant(args):
    """Plant of the parsed args: with --patches, sticky friction patches (runs saved before them: none)."""
    return RobotsSystem(
        x_init=None,       # same as xbar
        u_init=None,       # all zero
        linear_plant=args.linearize_plant,
        k=args.spring_const,
        n_agents=args.n_agents,
        n_patches=args.patch_n if getattr(args, 'patches', False) else 0,
    )


def sample_patches(args, num_samples, seed):
    """Sticky patch layouts of --patch-* (num_samples, 4 * patch_n), drawn from a generator of their own."""
    return sample_friction_patches(
        num_samples, n_patches=args.patch_n, x_range=tuple(args.patch_x or args.wall_x), y_range=tuple(args.patch_y),
        w_range=tuple(args.patch_w), sigma_range=tuple(args.patch_sigma),
        generator=torch.Generator().manual_seed(int(seed)),
    )


def with_patches(args, data, seed):
    """data with its own sticky patch layout per rollout (seeded) appended with --patches, else unchanged.
    The patches are drawn from generators of their own: the rollouts and the global RNG are unchanged."""
    if not getattr(args, 'patches', False):
        return data
    return append_patches(data, sample_patches(args, data.shape[0], seed))


def get_data(args, dataset):
    """Train (also the validation set) and test data of the parsed args, with their sticky patches if any."""
    train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
    return (with_patches(args, train_data, args.random_seed + 6),
            with_patches(args, test_data, args.random_seed + 7))


def verification_patches(args, num_scenarios):
    """One sticky patch layout per verification scenario (fixed by --random-seed), None without --patches."""
    if not getattr(args, 'patches', False):
        return None
    return sample_patches(args, num_scenarios, args.random_seed + 8)


def formation_scenarios(dataset):
    """Verification scenarios: name -> reference state of the target formation."""
    theta0, y = dataset.start_theta, dataset.target_y[0]
    if dataset.target_rotation and dataset.target_max_angle >= math.pi:
        # target formation centred at (2, y) translated only (direct),
        # rotated by pi (flip: the robots must turn the formation around, crossing like the robots diag scenario),
        # and shifted right and rotated by pi/2 (offset)
        return {
            'direct': formation_state((2.0, y), theta0, dataset.offsets),
            'flip': formation_state((2.0, y), theta0 + math.pi, dataset.offsets),
            'offset': formation_state((4.5, y), theta0 + math.pi / 2, dataset.offsets),
        }
    # targets restricted to |angle| <= target_max_angle (0 without rotation), possibly flipped by pi:
    # stay within the training distribution, i.e. translated only (direct), flipped (if trained on)
    # and shifted left / right with the extreme angles
    a = dataset.target_max_angle if dataset.target_rotation else 0.0
    if dataset.target_flip:
        return {
            'direct': formation_state((2.0, y), theta0, dataset.offsets),
            'flip': formation_state((2.0, y), theta0 + math.pi, dataset.offsets),
            'right': formation_state((3.5, y), theta0 + a, dataset.offsets),
        }
    return {
        'direct': formation_state((2.0, y), theta0, dataset.offsets),
        'left': formation_state((0.5, y), theta0 - a, dataset.offsets),
        'right': formation_state((3.5, y), theta0 + a, dataset.offsets),
    }


def verification_data(dataset, scenarios, horizon, patches=None):
    """Rollouts of the verification scenarios: start x0, constant target formation. With flipped starts in
    the data (start_flip_frac or test_start_flip_frac), each scenario again with its start and target
    turned by pi (dataset.flip_starts), named '<scenario>_fs', on the same sticky patches.
    patches: optional sticky patch layout of every scenario (K, 4 * n_patches), see verification_patches.
    Returns the names and the data (K, horizon + 200, 2 state_dim [+ 4 n_patches])."""
    sd = 4 * dataset.n_agents
    data = torch.zeros(len(scenarios), horizon + 200, 2 * sd)
    data[:, 0, :sd] = dataset.x0
    for k, xbar_k in enumerate(scenarios.values()):
        data[k, 1:, sd:] = xbar_k
    if patches is not None:
        data = append_patches(data, patches)
    names = list(scenarios)
    if dataset.start_flip_frac or dataset.test_start_flip_frac:
        data = torch.cat([data, dataset.flip_starts(data, torch.ones(len(names), dtype=torch.bool))])
        names += [name + '_fs' for name in names]
    return names, data


def formation_obstacles():
    """Obstacle centres and covariances used by the loss and the plots."""
    # Obstacle scenario of the robots experiment. `RobotsLoss` expects lists of tensors.
    obstacle_centers = [
        torch.tensor([[-1, 2]], device=device),
        torch.tensor([[-0, 2]], device=device),
        torch.tensor([[-2, 2.0]], device=device),
        torch.tensor([[-3, 2.0]], device=device),

        
        torch.tensor([[4, 2.0]], device=device),
        torch.tensor([[5, 2.0]], device=device),
        torch.tensor([[6, 2]], device=device),
        torch.tensor([[7, 2]], device=device),

    ]
    obstacle_covs = [torch.tensor([[0.44, 0.44]], device=device)] * len(obstacle_centers)
    return obstacle_centers, obstacle_covs


def make_loss_fn(args, dataset, xbar):
    """Training loss of the parsed args (runs saved before an arg was added: its default when off)."""
    obstacle_centers, obstacle_covs = formation_obstacles()
    eye = torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)
    return RobotsFormationLoss(
        Q=args.alpha_x * eye,
        Qs=args.alpha_speed * eye,
        alpha_u=args.alpha_u,
        xbar=xbar,
        loss_bound=None,
        sat_bound=None,
        alpha_col=args.alpha_col,
        alpha_obst=args.alpha_obst,
        alpha_obst_max=args.alpha_obst_max,
        obstacle_centers=obstacle_centers,
        obstacle_covs=obstacle_covs,
        obstacle_radius=args.obst_radius if args.obst_radius > 0 else None,
        obstacle_loss=args.obst_loss,
        collision_loss=getattr(args, 'col_loss', 'inverse'),
        Q_final=args.alpha_final * eye if args.alpha_final >= 0 else None,
        final_start=args.final_start,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=args.n_agents,
        legacy_speed_weight=args.legacy_speed_weight,
        formation_dist=dataset.distances,
        alpha_form=args.alpha_form,
        formation_adjacency=make_adjacency(args.formation_graph, args.n_agents),
        formation_deadband=args.formation_deadband,
        formation_start=args.formation_start,
        formation_offsets=dataset.offsets,
        orient_groups=dataset.orient_groups,
        alpha_orient=args.alpha_orient,
        orient_margin=args.orient_margin,
        alpha_orient_max=getattr(args, 'alpha_orient_max', 0.0),
        alpha_wall=getattr(args, 'alpha_wall', 0.0),
        wall_x=getattr(args, 'wall_x', None),
        wall_margin=getattr(args, 'wall_margin', 0.5),
    )


class ZeroOperatorController:
    """Baseline of the 'ref' plots and GIFs: the PB operator shut to zero (u = 0), i.e. the inner PI
    loop of the plant tracking the targets on its own. Same interface as PBController in rollout()."""

    def __init__(self, u_dim):
        self.u_dim = int(u_dim)

    def reset(self, x_init, u_init=None):
        pass

    def __call__(self, x_t, z_t=None, t=None, **transition_kwargs):
        return x_t.new_zeros(x_t.shape[0], 1, self.u_dim), None


GIF_LIMS = (-3, 7, -3, 8)     # (xmin, xmax, ymin, ymax) of every GIF frame, ymax raised for higher targets


def gif_lims(xbar):
    """GIF_LIMS, with ymax raised to 1 above the highest robot of the target formation xbar."""
    top = xbar.reshape(-1, 4)[:, 1].max().item() + 1.0
    return GIF_LIMS[:3] + (max(GIF_LIMS[3], math.ceil(top)),)


def save_scenario_gif(x, xbar, name, stage, n_agents, gif_folder, horizon, interval,
                      obstacle_centers, obstacle_covs, formation_pairs, min_dist=None, patches=None):
    """Frames of one closed-loop trajectory in gifs/frames_<name>_<stage>/ and gifs/CL_<name>_<stage>.gif.
    Robots are drawn with radius min_dist / 2 (collision when the discs overlap); None: radius 0.5.
    patches: optional sticky patch layout of the rollout, shaded on every frame."""
    min_dist = 1.0 if min_dist is None else float(min_dist)
    frame_folder = os.path.join(gif_folder, 'frames_%s_%s' % (name, stage))
    save_trajectory_frames(
        x, xbar, n_agents=n_agents, save_folder=frame_folder, T=horizon, interval=interval,
        obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs,
        formation_pairs=formation_pairs, lims=gif_lims(xbar),
        robot_radius=min_dist / 2, collision_dist=min_dist, patches=patches,
    )
    create_gif_from_frames(frame_folder, os.path.join(gif_folder, 'CL_%s_%s.gif' % (name, stage)))


def main():
    """
    Train and evaluate a performance boosting controller on the robots formation task: the robots
    start in a regular n-gon formation and must reach a translated and rotated copy of it while
    keeping the formation (elastically) and avoiding the obstacles and each other.

    Same pipeline as experiments/robots/run.py (plant, distributed controller, training loop),
    with the formation dataset and loss, formation metrics in the logs and formation plots/GIFs.
    All artifacts go to `experiments/robots_formation/saved_results/<n>_robots/<adjacency>/`
    (`.../<adjacency>/benchmark/<name>/` with --benchmark <name>).
    """
    # ----- parse experiment arguments -----
    args = argument_parser()
    torch.set_num_threads(args.num_threads)

    # ----- OUTPUT FOLDERS -----
    save_path = os.path.join(BASE_DIR, 'experiments', 'robots_formation', 'saved_results', '%i_robots' % args.n_agents,
                             adjacency_folder(args.adjacency))
    if args.benchmark is not None:
        save_path = os.path.join(save_path, 'benchmark', args.benchmark)
    run_name = args.run_id or 'pb_%s_%s_%s' % (args.mp_type, args.mb_type, datetime.now().strftime("%m_%d_%H_%M_%S"))
    save_folder = os.path.join(save_path, run_name)
    save_folder_gif = os.path.join(save_folder, 'gifs')

    # training state of the run to continue: weights, optimizer, epoch, best model and RNG.
    # 1. this run's own folder already has one: the pod was restarted (runai preemption), continue in place;
    # 2. else --resume: continue another run in this new folder; 3. else start from scratch.
    own_state_path = os.path.join(save_folder, 'training_state.pt')
    restarted = args.run_id is not None and os.path.isfile(own_state_path)
    resume_state = None
    if restarted:
        resume_state = torch.load(own_state_path, map_location=device, weights_only=False)
        # same command line as the first pod; args added to the code since then are not compared
        changed = sorted(k for k, v in resume_state['args'].items()
                         if k in vars(args) and k not in ('num_threads', 'compile') and vars(args)[k] != v)
        if changed:
            raise ValueError('--run-id %s: the run in %s was started with other args (%s); use another --run-id.'
                             % (args.run_id, save_folder, ', '.join(changed)))
    elif args.resume is not None:
        resume_state = torch.load(training_state_path(args.resume), map_location=device, weights_only=False)
        if args.epochs <= resume_state['epoch']:
            raise ValueError('--resume: --epochs (%i) is the total number of epochs and must exceed the %i already done.'
                             % (args.epochs, resume_state['epoch']))
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder_gif, exist_ok=True)

    # ----- SET UP LOGGER -----
    logging.basicConfig(
        filename=os.path.join(save_folder, 'log'),
        format='%(asctime)s %(message)s',
        filemode='a' if restarted else 'w',    # a restarted pod appends to the log of the run it continues
    )
    logger = logging.getLogger('perf_boost_')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)

    if restarted:
        logger.info('\n[INFO] ------------ Restarted (pod restart / preemption): continuing after epoch %i, up to epoch %i ------------'
                    % (resume_state['epoch'], args.epochs))
        if resume_state.get('code_version') != code_version():
            logger.info('[WARNING] code changed since the last training state: %s -> %s'
                        % (resume_state.get('code_version'), code_version()))
    else:
        msg = print_args(args)
        logger.info(msg)
        logger.info('[INFO] All args: %s' % vars(args))   # everything, including what the summary above leaves out
        logger.info('[INFO] Code version: %s' % code_version())
        if resume_state is not None:
            logger.info('[INFO] Resuming %s after epoch %i, up to epoch %i.' % (args.resume, resume_state['epoch'], args.epochs))
    torch.manual_seed(args.random_seed)



    # ------------ 1. Dataset ------------
    n_agents = args.n_agents
    state_dim = 4 * n_agents     # [px, py, vx, vy] per agent

    dataset = make_dataset(args)
    scenarios = formation_scenarios(dataset)

    # To disable obstacle loss entirely, pass `--no-obst-av` (recommended) rather than overriding here.

# divide to train and test
    train_data, test_data = get_data(args, dataset)     # with their sticky patches (--patches)
    train_data, test_data = train_data.to(device), test_data.to(device)
    if getattr(args, 'init_seed', None) is not None:
        torch.manual_seed(args.init_seed)    # model initialisation and batch shuffling, on the data of --random-seed


# data for plots
    t_ext = args.horizon * 4

    plot_data = test_data[250:350, :, :]
    plot_flipped = dataset.flipped_starts(plot_data).view(-1, 1)    # flipped starts keep their side
    plot_data[:, 0, :state_dim] = torch.where(plot_flipped, dataset.x0_flipped.to(device), dataset.x0.to(device))
    plot_data = plot_data.to(device)

# batch the data
    # batch shuffling from a generator of its own, reseeded every epoch from (--init-seed, epoch): the order of the
    # batches does not depend on how much of the global RNG the controller's construction drew, so two
    # controllers trained with the same seeds see the same batches (and a resumed run the same as without the stop)
    shuffle_seed = args.init_seed if getattr(args, 'init_seed', None) is not None else args.random_seed
    shuffle_gen = torch.Generator()
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=shuffle_gen)

# ------------ 2. Plant ------------
    sys = make_plant(args).to(device)     # with --patches: sticky friction patches, drag scale in the context


# ------------ 3. Controller ------------
    # u^i = output_amplification * M_b^i(w^i, z^i) ⊠ M_p^i(w^i, z^i) per agent, with context z^i = reference of agent i
    ctl = build_controller(args, plant=sys).to(device)
    rollout = partial(sys.rollout, zero_context=(args.context == 'none'))


# ------------ 4. Loss ------------
    Q = args.alpha_x * torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)    # saved with the trained model
    loss_fn = make_loss_fn(args, dataset, xbar=train_data[0, :, state_dim:2 * state_dim])
    # pairs drawn in the plots: the shape's own segments if it has them, else the constrained pairs (i < j)
    formation_pairs = dataset.plot_edges or loss_fn.formation_edges.nonzero().tolist()

    def formation_msg(x_log):
        """Formation error over the horizon, averaged over the rollouts."""
        err_mean, err_max = loss_fn.formation_error(x_log)
        return ' -- formation error: mean %.3f | max %.3f | mirrored: %.0f%%' % (
            err_mean.mean().item(), err_max.amax(1).mean().item(), 100 * loss_fn.mirrored(x_log).float().mean().item())

    def per_start_msg(data, x_log, u_log, e_log):
        """Loss and collisions of the nominal and of the flipped starts of data, when the data has flipped starts."""
        if not (dataset.start_flip_frac or dataset.test_start_flip_frac):
            return ''
        flipped = dataset.flipped_starts(data)
        parts = []
        for name, m in (('nominal', ~flipped), ('flipped', flipped)):
            if m.any():
                part = '%s starts (%i): loss %.2f' % (name, m.sum().item(), loss_fn.forward(x_log[m], u_log[m], e_log[m]).item())
                if args.col_av:
                    part += loss_fn.collision_summary(x_log[m])
                parts.append(part)
        return '\n    per start: ' + ' || '.join(parts)

# ------------ 5. Optimizer ------------
    # the entire train data; with --resample-data, never trained on (fresh rollouts every epoch)
    valid_data = train_data
    resample = getattr(args, 'resample_data', False)
    assert not (valid_data is None and args.return_best)
    # AdamW with the full lr on the recurrent core of M_p and a reduced lr on the
    # surrounding bounded mixer / context heads.
    fast_ids = {id(p) for p in ssm_core_parameters(ctl.operator)}
    fast_params = [p for p in ctl.parameters() if id(p) in fast_ids]
    slow_params = [p for p in ctl.parameters() if id(p) not in fast_ids]
    optimizer = torch.optim.AdamW(
        [{'params': fast_params, 'lr': args.lr},
         {'params': slow_params, 'lr': args.lr * args.lr_ratio}],
        weight_decay=1e-4,
    )
    base_lrs = [args.lr, args.lr * args.lr_ratio]
    clipper = GradClipper(getattr(args, 'grad_clip', 0.0), getattr(args, 'grad_clip_rel', 0.0))
    diagnoser = SpikeDiagnoser(args.grad_diag) if getattr(args, 'grad_diag', 0.0) else None
    ema = WeightEMA(ctl, args.ema_decay) if getattr(args, 'ema_decay', 0.0) else None

    def save_training_state(epoch):
        """Everything --resume needs to continue after this epoch (last weights, not the best)."""
        # written aside then renamed: a pod killed mid-write leaves the previous state intact
        torch.save({'controller': ctl.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'best_state': best_state, 'best_valid_loss': best_valid_loss,
                    'rng_state': torch.get_rng_state(), 'args': vars(args), 'code_version': code_version(),
                    'grad_clipper': clipper.state_dict(), 'ema': None if ema is None else ema.state_dict()},
                   own_state_path + '.tmp')
        os.replace(own_state_path + '.tmp', own_state_path)

# ------------ 6. Training ------------
# plot the baseline: closed loop with the operator shut to zero (u = 0), the inner PI loop alone
    logger.info('Plotting closed-loop trajectories of the baseline without the operator (u = 0)...')

    verif_names, data_verif = verification_data(dataset, scenarios, args.horizon,
                                                patches=verification_patches(args, len(scenarios)))
    patches_of = lambda data, k: data[k, 0, 2 * state_dim:] if sys.n_patches else None   # layout of rollout k
    data_verif = data_verif.to(device)

    x_verif, _, _ = sys.rollout(ZeroOperatorController(sys.in_dim), data_verif)

    total_params = sum(p.numel() for p in ctl.parameters())
    logger.info(f"Number of parameters: {total_params} (M_p core: {sum(p.numel() for p in fast_params)})")
    if ctl.operator.param_budget is not None:
        logger.info(f"Matched to the centralized M ({ctl.operator.param_budget} parameters) with widths {ctl.operator.widths}")
    if ctl.operator.input_padding is not None:
        logger.info("Agent inputs zero-padded to w %i, z %i to batch the agents (real sizes w %s, z %s); "
                    "the weights reading the padding are held at zero" % ctl.operator.input_padding)
    for i, op in enumerate(agent_operators(ctl.operator)):
        if hasattr(op.mp, 'gain_diagnostics') and args.ctx_gamma > 0:   # certified bound needs a prescribed gamma
            logger.info(f"Contextual M_p gain diagnostics ({'shared' if args.share_weights else f'agent {i}'}): {op.mp.gain_diagnostics()}")

    def plot_scenarios(x_verif, stage, text):
        for k, name in enumerate(verif_names):
            plot_trajectories(
                x_verif[k, :, :],  # remove extra dim due to batching
                xbar=data_verif[k, 1, state_dim:2 * state_dim],
                n_agents=sys.n_agents,
                save_folder=save_folder,
                filename='CL_%s_%s.pdf' % (name, stage),
                text=text,
                T=t_ext,
                obstacle_centers=loss_fn.obstacle_centers,
                obstacle_covs=loss_fn.obstacle_covs,
                formation_pairs=formation_pairs,
                formation_every=max(1, args.horizon // 5),
                patches=patches_of(data_verif, k),
            )

    def save_gifs(x_verif, stage):
        for k, name in enumerate(verif_names):
            save_scenario_gif(
                x_verif[k], data_verif[k, 1, state_dim:2 * state_dim], name, stage, n_agents=sys.n_agents,
                gif_folder=save_folder_gif, horizon=args.horizon, interval=args.gif_interval,
                obstacle_centers=loss_fn.obstacle_centers, obstacle_covs=loss_fn.obstacle_covs,
                formation_pairs=formation_pairs, min_dist=args.min_dist, patches=patches_of(data_verif, k),
            )

    plot_scenarios(x_verif, 'ref', "Baseline - operator off (u = 0)")
    if args.save_gifs:
        save_gifs(x_verif, 'ref')


    logger.info('\n------------ Begin training ------------')
    best_valid_loss, best_state, start_epoch = float('inf'), None, 0
    if resume_state is not None:
        ctl.load_state_dict(resume_state['controller'])
        optimizer.load_state_dict(resume_state['optimizer'])
        best_valid_loss, best_state = resume_state['best_valid_loss'], resume_state['best_state']
        start_epoch = resume_state['epoch'] + 1
        torch.set_rng_state(resume_state['rng_state'])     # same shuffling of the batches as without the interruption
        if resume_state.get('grad_clipper') is not None:
            clipper.load_state_dict(resume_state['grad_clipper'])
        if ema is not None and resume_state.get('ema') is not None:
            ema.load_state_dict(resume_state['ema'])
    t = time.time()
    for epoch in range(start_epoch, 1 + args.epochs):
        set_lr(optimizer, base_lrs, epoch, args)
        shuffle_gen.manual_seed(1_000_003 * int(shuffle_seed) + epoch)
        if resample:
            # seeded by the epoch: a resumed run draws the same rollouts; apart from the generators of get_data
            g = torch.Generator().manual_seed(10000 * (args.random_seed + 3) + epoch)
            fresh = with_patches(args, dataset.sample_train(args.num_rollouts, g), 2 * g.initial_seed() + 2)
            train_dataloader = DataLoader(fresh.to(device), batch_size=args.batch_size, shuffle=True, generator=shuffle_gen)
        # iterate over all data batches
        for step, train_data_batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # simulate over horizon steps
            x_log, e_log, u_log = rollout(
                controller=ctl, data=train_data_batch, train=True,
            )
            # loss of this rollout
            loss = loss_fn.forward(x_log, u_log, e_log)
            # take a step
            loss.backward()
            stepped = clipper(ctl.parameters())       # False: non-finite gradient, step skipped
            if diagnoser is not None and diagnoser.triggered(clipper):
                # same batch, same weights: the rollout is deterministic, so these are the gradients of the spike
                def spike_terms(batch=train_data_batch):
                    x_d, e_d, u_d = rollout(controller=ctl, data=batch, train=True)
                    return loss_fn.loss_terms(x_d, u_d, e_d)
                term_norms = term_grad_norms(spike_terms, ctl.parameters())
                logger.info(diagnoser.report(epoch, step, clipper, loss.item(), term_norms))
            if stepped:
                optimizer.step()
                if ema is not None:
                    ema.update(ctl)

        # print info
        if epoch % args.log_epoch == 0:
            msg = 'Epoch: %i --- train loss: %.2f' % (epoch, loss)
            terms = loss_fn.breakdown(x_log, u_log, e_log)   # last train batch; replaced by validation terms below
            x_terms = x_log

            if args.return_best:
                # rollout the controller on the valid data, with the averaged weights if any
                current_state = None if ema is None else ema.swap(ctl)
                with torch.no_grad():
                    x_log_valid, e_log_valid, u_log_valid = rollout(
                        controller=ctl, data=valid_data, train=False,
                    )
                    # loss of the valid data
                    loss_valid = loss_fn.forward(x_log_valid, u_log_valid, e_log_valid)
                    terms = loss_fn.breakdown(x_log_valid, u_log_valid, e_log_valid)
                    x_terms = x_log_valid
                    valid_per_start = per_start_msg(valid_data, x_log_valid, u_log_valid, e_log_valid)
                msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
                # compare with the best valid loss
                if loss_valid.item() < best_valid_loss:
                    best_valid_loss = loss_valid.item()
                    best_state = copy.deepcopy(ctl.state_dict())  # record state dict if best on valid
                    msg += ' (best so far)'
                if ema is not None:
                    msg += ' (EMA)'
                    ctl.load_state_dict(current_state)
            duration = time.time() - t
            msg += ' ---||--- time: %.0f s' % (duration)
            msg += '\n    %s terms: %s' % ('validation' if args.return_best else 'train batch', loss_fn.format_breakdown(terms))
            msg += formation_msg(x_terms.detach())
            msg += clipper.summary() + ' -- lr: %.2e' % optimizer.param_groups[0]['lr']
            if args.return_best:
                msg += valid_per_start
            clipper.reset_stats()
            if diagnoser is not None:
                diagnoser.reset_stats()
            logger.info(msg)
            t = time.time()
            save_training_state(epoch)   # overwritten each time: a preemption loses at most log_epoch epochs
    if start_epoch > args.epochs:
        logger.info('[INFO] Training already done in this folder: evaluating only.')
    save_training_state(args.epochs)

    # set to best seen during training
    if args.return_best:
        ctl.load_state_dict(best_state)
    elif ema is not None:
        ctl.load_state_dict(ema.state)

# ------ 7. Save and evaluate the trained model ------
# save
    # full controller (M_p, M_b, lifter, ...) + metadata to make the checkpoint self-describing;
    # reload with build_controller(argparse.Namespace(**ckpt['args']), plant).load_state_dict(ckpt['controller'])
    res_dict = {'controller': ctl.state_dict(), 'Q': Q, 'args': vars(args),
                'formation_offsets': dataset.offsets, 'formation_dist': dataset.distances}
    filename = os.path.join(save_folder, 'trained_controller' + '.pt')
    torch.save(res_dict, filename)
    logger.info('[INFO] saved trained model.')

# evaluate on the train data
    logger.info('\n[INFO] evaluating the trained controller on %i %s rollouts.' % (
        train_data.shape[0], 'validation (not trained on)' if resample else 'training'))
    with torch.no_grad():
        x_log, e_log, u_log = rollout(
            controller=ctl, data=train_data, train=False,
        )   # use the entire train data, not a batch
        # evaluate losses
        loss = loss_fn.forward(x_log, u_log, e_log)
        msg = 'Loss: %.4f' % (loss)
        msg += ' -- ' + loss_fn.format_breakdown(loss_fn.breakdown(x_log, u_log, e_log))
        msg += formation_msg(x_log)
# count collisions
    if args.col_av:
        msg += loss_fn.collision_summary(x_log)
    with torch.no_grad():
        msg += per_start_msg(train_data, x_log, u_log, e_log)
    logger.info(msg)

# evaluate on the test data
    logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
    with torch.no_grad():
        # simulate over horizon steps
        x_log, e_log, u_log = rollout(
            controller=ctl, data=test_data, train=False,
        )
        # loss
        test_loss = loss_fn.forward(x_log, u_log, e_log).item()
        msg = "Loss: %.4f" % (test_loss)
        msg += ' -- ' + loss_fn.format_breakdown(loss_fn.breakdown(x_log, u_log, e_log))
        msg += formation_msg(x_log)
# count collisions
    if args.col_av:
        msg += loss_fn.collision_summary(x_log)
    with torch.no_grad():
        msg += per_start_msg(test_data, x_log, u_log, e_log)
    logger.info(msg)


    # plot closed-loop trajectories using the trained controller
    logger.info('Plotting closed-loop trajectories using the trained controller...')
    x_log, _, u_log = rollout(ctl, plot_data)
    plot_trajectories(
        x_log[0, :, :],  # remove extra dim due to batching
        xbar=plot_data[0, 5, state_dim:2 * state_dim],
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_trained.pdf',
        text="CL - trained controller",
        T=t_ext,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
        formation_pairs=formation_pairs,
        formation_every=max(1, args.horizon // 5),
        patches=patches_of(plot_data, 0),
    )

    x_verif, _, u_verif = rollout(ctl, data_verif)
    u_verif, v_verif = u_verif.cpu(), sys.v_log.cpu()
    plot_scenarios(x_verif, 'trained', "rPB - trained controller")
    if args.save_gifs:
        save_gifs(x_verif, 'trained')

    # formation error of the verification scenarios
    err_mean, err_max = loss_fn.formation_error(x_verif)
    plot_formation_error(
        {name: (err_mean[k], err_max[k]) for k, name in enumerate(verif_names)},
        save_folder=save_folder,
        deadband=args.formation_deadband,
        T=args.horizon,
        text='Formation error over time - trained controller',
    )



#### Plot the evolution of the reference over time for the second scenario (flip, or left) ####
    evol_name = list(scenarios)[1]
    x_ref_evol = torch.zeros(1, args.horizon + 200, state_dim)
    x_ref_evol[:, :, sys.pos_idx] = u_verif[1:2].cpu()    # offset u^i of agent i on its position channels
    x_ref_evol = x_ref_evol + scenarios[evol_name]



    plot_trajectories(
        x_ref_evol[0, :, :],  # remove extra dim due to batching
        xbar=scenarios[evol_name],
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_xbar_evolution.pdf',
        text="CL - evolution of the reference",
        T=t_ext,
        dots=True,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
        patches=patches_of(data_verif, 1),
    )


# One subplot per robot
    t_axis = np.array(range(u_verif.shape[1]))
    fig, axs = plt.subplots(n_agents, 1, figsize=(10, 3.5 * n_agents), squeeze=False)
    for i, ax in enumerate(axs[:, 0]):
        ax.plot(t_axis, u_verif[1, :, 2 * i], label="dX")
        ax.plot(t_axis, u_verif[1, :, 2 * i + 1], label="dY")
        ax.set_title("Robot %i" % (i + 1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Delta ref")
        ax.legend()
        ax.grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

    plt.suptitle(
        'Performance boosting offset to the reference over time \n for the %s scenario' % evol_name,
        fontsize=13,
    )
    plt.savefig(os.path.join(save_folder, "U_over_time.pdf"))
    plt.close()


# One subplot per robot
    fig, axs = plt.subplots(n_agents, 1, figsize=(10, 3.5 * n_agents), squeeze=False)
    for i, ax in enumerate(axs[:, 0]):
        ax.plot(t_axis, v_verif[1, :, 2 * i], label="v_X")
        ax.plot(t_axis, v_verif[1, :, 2 * i + 1], label="v_Y")
        ax.set_title("Robot %i" % (i + 1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("v")
        ax.legend()
        ax.grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

    plt.suptitle('Integral variable over time \n for the %s scenario' % evol_name, fontsize=13)
    plt.savefig(os.path.join(save_folder, "V_over_time.pdf"))
    plt.close()


if __name__ == "__main__":
    main()
