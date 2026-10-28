import sys, os, copy, logging, math, torch, time
from functools import partial
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(BASE_DIR))  # parent of Distributed/, for `Distributed.*` imports

from Distributed.config import device
from Distributed.experiments.robots_formation.arg_parser import argument_parser, print_args, training_state_path
from Distributed.plants import RobotsSystem, RobotsFormationDataset
from Distributed.plants.robots.formation_dataset import formation_state
from Distributed.utils.plot_functions import *
from Distributed.controllers import build_controller, agent_operators, ssm_core_parameters
from Distributed.controllers.interconnection import adjacency_folder, make_adjacency
from Distributed.loss_functions import RobotsFormationLoss
from Distributed.utils.assistive_functions import WrapLogger


def make_dataset(args):
    """Formation dataset of the parsed args (runs saved before the isosceles shape: regular n-gon, full rotation)."""
    return RobotsFormationDataset(
        random_seed=args.random_seed,
        horizon=args.horizon,
        n_agents=args.n_agents,
        side=args.formation_side,
        std_ini=args.std_init_plant,
        target_rotation=args.target_rotation,
        target_max_angle=getattr(args, 'target_max_angle', math.pi),
        target_flip=getattr(args, 'target_flip', False),
        shape=getattr(args, 'formation_shape', 'regular'),
        base=getattr(args, 'formation_base', None),
        height=getattr(args, 'formation_height', None),
        large_rot_frac=getattr(args, 'large_rot_frac', 0.0),
        large_rot_min_angle=getattr(args, 'large_rot_min_angle', 2 * math.pi / 3),
    )


def formation_scenarios(dataset):
    """Verification scenarios: name -> reference state of the target formation."""
    theta0 = dataset.start_theta
    if dataset.target_rotation and dataset.target_max_angle >= math.pi:
        # target formation centred at (2, 6) translated only (direct),
        # rotated by pi (flip: the robots must turn the formation around, crossing like the robots diag scenario),
        # and shifted right and rotated by pi/2 (offset)
        return {
            'direct': formation_state((2.0, 6.0), theta0, dataset.offsets),
            'flip': formation_state((2.0, 6.0), theta0 + math.pi, dataset.offsets),
            'offset': formation_state((4.5, 6.0), theta0 + math.pi / 2, dataset.offsets),
        }
    # targets restricted to |angle| <= target_max_angle (0 without rotation), possibly flipped by pi:
    # stay within the training distribution, i.e. translated only (direct), flipped (if trained on)
    # and shifted left / right with the extreme angles
    a = dataset.target_max_angle if dataset.target_rotation else 0.0
    if dataset.target_flip:
        return {
            'direct': formation_state((2.0, 6.0), theta0, dataset.offsets),
            'flip': formation_state((2.0, 6.0), theta0 + math.pi, dataset.offsets),
            'right': formation_state((3.5, 6.0), theta0 + a, dataset.offsets),
        }
    return {
        'direct': formation_state((2.0, 6.0), theta0, dataset.offsets),
        'left': formation_state((0.5, 6.0), theta0 - a, dataset.offsets),
        'right': formation_state((3.5, 6.0), theta0 + a, dataset.offsets),
    }


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


class ZeroOperatorController:
    """Baseline of the 'ref' plots and GIFs: the PB operator shut to zero (u = 0), i.e. the inner PI
    loop of the plant tracking the targets on its own. Same interface as PBController in rollout()."""

    def __init__(self, u_dim):
        self.u_dim = int(u_dim)

    def reset(self, x_init, u_init=None):
        pass

    def __call__(self, x_t, z_t=None, t=None, **transition_kwargs):
        return x_t.new_zeros(x_t.shape[0], 1, self.u_dim), None


GIF_LIMS = (-3, 7, -3, 8)     # fixed (xmin, xmax, ymin, ymax) of every GIF frame


def save_scenario_gif(x, xbar, name, stage, n_agents, gif_folder, horizon, interval,
                      obstacle_centers, obstacle_covs, formation_pairs, min_dist=None):
    """Frames of one closed-loop trajectory in gifs/frames_<name>_<stage>/ and gifs/CL_<name>_<stage>.gif.
    Robots are drawn with radius min_dist / 2 (collision when the discs overlap); None: radius 0.5."""
    min_dist = 1.0 if min_dist is None else float(min_dist)
    frame_folder = os.path.join(gif_folder, 'frames_%s_%s' % (name, stage))
    save_trajectory_frames(
        x, xbar, n_agents=n_agents, save_folder=frame_folder, T=horizon, interval=interval,
        obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs,
        formation_pairs=formation_pairs, lims=GIF_LIMS,
        robot_radius=min_dist / 2, collision_dist=min_dist,
    )
    create_gif_from_frames(frame_folder, os.path.join(gif_folder, 'CL_%s_%s.gif' % (name, stage)))


def main():
    """
    Train and evaluate a performance boosting controller on the robots formation task: the robots
    start in a regular n-gon formation and must reach a translated and rotated copy of it while
    keeping the formation (elastically) and avoiding the obstacles and each other.

    Same pipeline as experiments/robots/run.py (plant, distributed controller, training loop),
    with the formation dataset and loss, formation metrics in the logs and formation plots/GIFs.
    All artifacts go to `experiments/robots_formation/saved_results/<n>_robots/<adjacency>/`.
    """
    # ----- parse experiment arguments -----
    args = argument_parser()
    torch.set_num_threads(args.num_threads)
    # training state of the run to continue: weights, optimizer, epoch, best model and RNG
    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(training_state_path(args.resume), map_location=device, weights_only=False)
        if args.epochs <= resume_state['epoch']:
            raise ValueError('--resume: --epochs (%i) is the total number of epochs and must exceed the %i already done.'
                             % (args.epochs, resume_state['epoch']))

    # ----- SET UP LOGGER / OUTPUT FOLDERS -----
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(BASE_DIR, 'experiments', 'robots_formation', 'saved_results', '%i_robots' % args.n_agents,
                             adjacency_folder(args.adjacency))

    save_folder = os.path.join(save_path, 'pb_%s_%s_%s' % (args.mp_type, args.mb_type, now))
    save_folder_gif = os.path.join(save_folder, 'gifs')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder_gif, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_folder, 'log'),
        format='%(asctime)s %(message)s',
        filemode='w',
    )
    logger = logging.getLogger('perf_boost_')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)

    msg = print_args(args)
    logger.info(msg)
    if resume_state is not None:
        logger.info('[INFO] Resuming %s after epoch %i, up to epoch %i.' % (args.resume, resume_state['epoch'], args.epochs))
    torch.manual_seed(args.random_seed)



    # ------------ 1. Dataset ------------
    n_agents = args.n_agents
    state_dim = 4 * n_agents     # [px, py, vx, vy] per agent

    dataset = make_dataset(args)
    scenarios = formation_scenarios(dataset)
    obstacle_centers, obstacle_covs = formation_obstacles()

    # To disable obstacle loss entirely, pass `--no-obst-av` (recommended) rather than overriding here.

# divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
    train_data, test_data = train_data.to(device), test_data.to(device)


# data for plots
    t_ext = args.horizon * 4

    plot_data = test_data[250:350, :, :]
    plot_data[:, 0, :state_dim] = dataset.x0.detach()
    plot_data = plot_data.to(device)

# batch the data
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant ------------
    plant_input_init = None     # all zero
    plant_state_init = None    # same as xbar
    sys = RobotsSystem(
        x_init=plant_state_init,
        u_init=plant_input_init,
        linear_plant=args.linearize_plant,
        k=args.spring_const,
        n_agents=n_agents,
    ).to(device)


# ------------ 3. Controller ------------
    # u^i = output_amplification * M_b^i(w^i, z^i) ⊠ M_p^i(w^i, z^i) per agent, with context z^i = reference of agent i
    ctl = build_controller(args, plant=sys).to(device)
    rollout = partial(sys.rollout, zero_context=(args.context == 'none'))


# ------------ 4. Loss ------------
    Q = args.alpha_x * torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)
    Qs = args.alpha_speed * torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)
    formation_adjacency = make_adjacency(args.formation_graph, n_agents)
    loss_fn = RobotsFormationLoss(
        Q=Q,
        Qs=Qs,
        alpha_u=args.alpha_u,
        xbar=train_data[0, :, state_dim:],
        loss_bound=None,
        sat_bound=None,
        alpha_col=args.alpha_col,
        alpha_obst=args.alpha_obst,
        alpha_obst_max=args.alpha_obst_max,
        obstacle_centers=obstacle_centers,
        obstacle_covs=obstacle_covs,
        obstacle_radius=args.obst_radius if args.obst_radius > 0 else None,
        obstacle_loss=args.obst_loss,
        Q_final=args.alpha_final * torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device) if args.alpha_final >= 0 else None,
        final_start=args.final_start,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=sys.n_agents,
        legacy_speed_weight=args.legacy_speed_weight,
        formation_dist=dataset.distances,
        alpha_form=args.alpha_form,
        formation_adjacency=formation_adjacency,
        formation_deadband=args.formation_deadband,
        formation_start=args.formation_start,
        formation_offsets=dataset.offsets,
        alpha_orient=args.alpha_orient,
        orient_margin=args.orient_margin,
        alpha_orient_max=getattr(args, 'alpha_orient_max', 0.0),
        alpha_wall=getattr(args, 'alpha_wall', 0.0),
        wall_x=getattr(args, 'wall_x', None),
        wall_margin=getattr(args, 'wall_margin', 0.5),
    )
    formation_pairs = loss_fn.formation_edges.nonzero().tolist()    # constrained pairs (i < j), for the plots

    def formation_msg(x_log):
        """Formation error over the horizon, averaged over the rollouts."""
        err_mean, err_max = loss_fn.formation_error(x_log)
        return ' -- formation error: mean %.3f | max %.3f | mirrored: %.0f%%' % (
            err_mean.mean().item(), err_max.amax(1).mean().item(), 100 * loss_fn.mirrored(x_log).float().mean().item())

# ------------ 5. Optimizer ------------
    valid_data = train_data      # use the entire train data for validation
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

    def save_training_state(epoch):
        """Everything --resume needs to continue after this epoch (last weights, not the best)."""
        torch.save({'controller': ctl.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'best_state': best_state, 'best_valid_loss': best_valid_loss,
                    'rng_state': torch.get_rng_state(), 'args': vars(args)},
                   os.path.join(save_folder, 'training_state.pt'))

# ------------ 6. Training ------------
# plot the baseline: closed loop with the operator shut to zero (u = 0), the inner PI loop alone
    logger.info('Plotting closed-loop trajectories of the baseline without the operator (u = 0)...')

    data_verif = torch.zeros(len(scenarios), args.horizon + 200, 2 * sys.state_dim, device=device)
    data_verif[:, 0:1, :state_dim] = dataset.x0
    for k, xbar_k in enumerate(scenarios.values()):
        data_verif[k, 1:, state_dim:] = xbar_k.to(device)

    x_verif, _, _ = sys.rollout(ZeroOperatorController(sys.in_dim), data_verif)

    total_params = sum(p.numel() for p in ctl.parameters())
    logger.info(f"Number of parameters: {total_params} (M_p core: {sum(p.numel() for p in fast_params)})")
    if ctl.operator.param_budget is not None:
        logger.info(f"Matched to the centralized M ({ctl.operator.param_budget} parameters) with widths {ctl.operator.widths}")
    for i, op in enumerate(agent_operators(ctl.operator)):
        if hasattr(op.mp, 'gain_diagnostics') and args.ctx_gamma > 0:   # certified bound needs a prescribed gamma
            logger.info(f"Contextual M_p gain diagnostics ({'shared' if args.share_weights else f'agent {i}'}): {op.mp.gain_diagnostics()}")

    def plot_scenarios(x_verif, stage, text):
        for k, (name, xbar_k) in enumerate(scenarios.items()):
            plot_trajectories(
                x_verif[k, :, :],  # remove extra dim due to batching
                xbar=xbar_k,
                n_agents=sys.n_agents,
                save_folder=save_folder,
                filename='CL_%s_%s.pdf' % (name, stage),
                text=text,
                T=t_ext,
                obstacle_centers=loss_fn.obstacle_centers,
                obstacle_covs=loss_fn.obstacle_covs,
                formation_pairs=formation_pairs,
                formation_every=max(1, args.horizon // 5),
            )

    def save_gifs(x_verif, stage):
        for k, (name, xbar_k) in enumerate(scenarios.items()):
            save_scenario_gif(
                x_verif[k], xbar_k, name, stage, n_agents=sys.n_agents, gif_folder=save_folder_gif,
                horizon=args.horizon, interval=args.gif_interval,
                obstacle_centers=loss_fn.obstacle_centers, obstacle_covs=loss_fn.obstacle_covs,
                formation_pairs=formation_pairs, min_dist=args.min_dist,
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
    t = time.time()
    for epoch in range(start_epoch, 1 + args.epochs):
        # iterate over all data batches
        for train_data_batch in train_dataloader:
            optimizer.zero_grad()
            # simulate over horizon steps
            x_log, e_log, u_log = rollout(
                controller=ctl, data=train_data_batch, train=True,
            )
            # loss of this rollout
            loss = loss_fn.forward(x_log, u_log, e_log)
            # take a step
            loss.backward()
            optimizer.step()

        # print info
        if epoch % args.log_epoch == 0:
            msg = 'Epoch: %i --- train loss: %.2f' % (epoch, loss)
            terms = loss_fn.breakdown(x_log, u_log, e_log)   # last train batch; replaced by validation terms below
            x_terms = x_log

            if args.return_best:
                # rollout the current controller on the valid data
                with torch.no_grad():
                    x_log_valid, e_log_valid, u_log_valid = rollout(
                        controller=ctl, data=valid_data, train=False,
                    )
                    # loss of the valid data
                    loss_valid = loss_fn.forward(x_log_valid, u_log_valid, e_log_valid)
                    terms = loss_fn.breakdown(x_log_valid, u_log_valid, e_log_valid)
                    x_terms = x_log_valid
                msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
                # compare with the best valid loss
                if loss_valid.item() < best_valid_loss:
                    best_valid_loss = loss_valid.item()
                    best_state = copy.deepcopy(ctl.state_dict())  # record state dict if best on valid
                    msg += ' (best so far)'
            duration = time.time() - t
            msg += ' ---||--- time: %.0f s' % (duration)
            msg += '\n    %s terms: %s' % ('validation' if args.return_best else 'train batch', loss_fn.format_breakdown(terms))
            msg += formation_msg(x_terms.detach())
            logger.info(msg)
            t = time.time()
            if epoch % (4 * args.log_epoch) == 0:
                save_training_state(epoch)
    save_training_state(args.epochs)

    # set to best seen during training
    if args.return_best:
        ctl.load_state_dict(best_state)

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
    logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
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
    logger.info(msg)


    # plot closed-loop trajectories using the trained controller
    logger.info('Plotting closed-loop trajectories using the trained controller...')
    x_log, _, u_log = rollout(ctl, plot_data)
    plot_trajectories(
        x_log[0, :, :],  # remove extra dim due to batching
        xbar=plot_data[0, 5, state_dim:],
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_trained.pdf',
        text="CL - trained controller",
        T=t_ext,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
        formation_pairs=formation_pairs,
        formation_every=max(1, args.horizon // 5),
    )

    x_verif, _, u_verif = rollout(ctl, data_verif)
    u_verif, v_verif = u_verif.cpu(), sys.v_log.cpu()
    plot_scenarios(x_verif, 'trained', "rPB - trained controller")
    if args.save_gifs:
        save_gifs(x_verif, 'trained')

    # formation error of the verification scenarios
    err_mean, err_max = loss_fn.formation_error(x_verif)
    plot_formation_error(
        {name: (err_mean[k], err_max[k]) for k, name in enumerate(scenarios)},
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
