import sys, os, copy, logging, torch, time
from functools import partial
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(BASE_DIR))  # parent of Distributed/, for `Distributed_Danilo.*` imports

from Distributed_Danilo.config import device
from Distributed_Danilo.experiments.robots.arg_parser import argument_parser, print_args
from Distributed_Danilo.plants import RobotsSystem, RobotsDataset
from Distributed_Danilo.plants.robots.robots_dataset import scenario_xbar
from Distributed_Danilo.utils.plot_functions import *
from Distributed_Danilo.controllers import build_controller, controller_tag, core_parameters
from Distributed_Danilo.controllers.interconnection import adjacency_folder
from Distributed_Danilo.loss_functions import RobotsLoss
from Distributed_Danilo.utils.assistive_functions import WrapLogger
from Distributed_Danilo.utils.training import GradClipper, SpikeDiagnoser, WeightEMA, set_lr, term_grad_norms


def main():
    """
    Train and evaluate a performance boosting controller on the robots reference-tracking task.

    This script is intentionally self-contained: it generates synthetic rollouts/references,
    trains by backpropagating through closed-loop rollouts, and writes all artifacts (logs, plots,
    checkpoints) under `experiments/robots/saved_results/<n>_robots/<adjacency>/`.
    """
    # ----- parse experiment arguments -----
    args = argument_parser()
    torch.set_num_threads(args.num_threads)

    # ----- SET UP LOGGER / OUTPUT FOLDERS -----
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', '%i_robots' % args.n_agents,
                             adjacency_folder(args.adjacency))

    save_folder = os.path.join(save_path, '%s_%s' % (controller_tag(args), now))
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
    torch.manual_seed(args.random_seed)



    # ------------ 1. Dataset ------------
    n_agents = args.n_agents
    state_dim = 4 * n_agents     # [px, py, vx, vy] per agent
    # verification scenarios (2 agents: diag [0, 4, 0, 0, 4, 4, 0, 0], direct [5, 4, 0, 0, -1, 4, 0, 0],
    # center [0.5, 4, 0, 0, 1.5, 4, 0, 0])
    xbar_train = scenario_xbar('diag', n_agents)
    xbar_verif2 = scenario_xbar('direct', n_agents)
    xbar_verif3 = scenario_xbar('center', n_agents)

    # Obstacle scenarios used in experiments. `RobotsLoss` expects lists of tensors.
    # The final assignment below is the one used (the intermediate ones are kept as quick presets).
    obstacle_centers = [
        torch.tensor([[-0.5, 0]], device=device),
        torch.tensor([[0.5, 0.0]], device=device),
    ]
    obstacle_centers = [
        torch.tensor([[-1.5, 2]], device=device),
        torch.tensor([[-0.5, 2]], device=device),
        torch.tensor([[0.5, 2.0]], device=device),
        torch.tensor([[3.5, 2]], device=device),
        torch.tensor([[4.5, 2.0]], device=device),
        torch.tensor([[5.5, 2.0]], device=device),
    ]


    """obstacle_centers = [
        torch.tensor([[-1, 2]], device=device),
        torch.tensor([[1, 2.0]], device=device),
        torch.tensor([[3, 2]], device=device),
        torch.tensor([[5, 2.0]], device=device),
        torch.tensor([[7, 2.0]], device=device),
        torch.tensor([[-3, 2.0]], device=device),
    ]"""

    obstacle_covs = [torch.tensor([[0.44, 0.44]], device=device)] * len(obstacle_centers)

    # To disable obstacle loss entirely, pass `--no-obst-av` (recommended) rather than overriding here.

    dataset = RobotsDataset(
        random_seed=args.random_seed,
        horizon=args.horizon,
        x_bar=xbar_verif2,
        std_ini=args.std_init_plant,
        n_agents=n_agents,
    )

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
    loss_fn = RobotsLoss(
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
        collision_loss=args.col_loss,
        Q_final=args.alpha_final * torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device) if args.alpha_final >= 0 else None,
        final_start=args.final_start,
        min_dist=args.min_dist if args.col_av else None,
        n_agents=sys.n_agents,
        legacy_speed_weight=args.legacy_speed_weight,
    )
 
# ------------ 5. Optimizer ------------
    valid_data = train_data      # use the entire train data for validation
    assert not (valid_data is None and args.return_best)
    # AdamW with the full lr on the recurrent core of M_p and a reduced lr on the
    # surrounding bounded mixer / context heads.
    fast_ids = {id(p) for p in core_parameters(ctl.operator)}
    fast_params = [p for p in ctl.parameters() if id(p) in fast_ids]
    slow_params = [p for p in ctl.parameters() if id(p) not in fast_ids]
    optimizer = torch.optim.AdamW(
        [{'params': fast_params, 'lr': args.lr},
         {'params': slow_params, 'lr': args.lr * args.lr_ratio}],
        weight_decay=1e-4,
    )
    base_lrs = [args.lr, args.lr * args.lr_ratio]
    clipper = GradClipper(args.grad_clip, args.grad_clip_rel)
    diagnoser = SpikeDiagnoser(args.grad_diag) if getattr(args, 'grad_diag', 0.0) else None
    ema = WeightEMA(ctl, args.ema_decay) if args.ema_decay else None
 
# ------------ 6. Training ------------
# plot closed-loop trajectories before training the controller
    logger.info('Plotting closed-loop trajectories before training the controller...')
    x_log, _, u_log = rollout(ctl, plot_data)


    data_verif = torch.zeros(3, args.horizon + 200, 2 * sys.state_dim, device=device)

    data_verif[:, 0:1, :state_dim] = dataset.x0
    data_verif[0:1, 1:, state_dim:] = xbar_train.to(device)
    data_verif[1:2, 1:, state_dim:] = xbar_verif2.to(device)
    data_verif[2:3, 1:, state_dim:] = xbar_verif3.to(device)

    x_verif, _, u_verif = rollout(ctl, data_verif)

    total_params = sum(p.numel() for p in ctl.parameters())
    logger.info(f"Number of parameters: {total_params} (l2 cores and gains: {sum(p.numel() for p in fast_params)}, "
                f"readouts and heads: {sum(p.numel() for p in slow_params)})")
    if ctl.operator.param_budget is not None:
        logger.info(f"Matched to the centralized M ({ctl.operator.param_budget} parameters) with widths {ctl.operator.widths}")
    if ctl.operator.input_padding is not None:
        logger.info("Agent inputs zero-padded to w %i, z %i to batch the agents (real sizes w %s, z %s); "
                    "the weights reading the padding are held at zero" % ctl.operator.input_padding)
    logger.info(f"Network certificate at initialization: {ctl.operator.certify()}")

    plot_trajectories(
        x_verif[0, :, :],  # remove extra dim due to batching
        xbar=xbar_train,
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_diag_ref.pdf',
        text="CL - before training",
        T=t_ext,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
    )

    plot_trajectories(
        x_verif[1, :, :],  # remove extra dim due to batching
        xbar=xbar_verif2,
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_direct_ref.pdf',
        text="CL - before training",
        T=t_ext,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
    )

    plot_trajectories(
        x_verif[2, :, :],  # remove extra dim due to batching
        xbar=xbar_verif3,
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_center_ref.pdf',
        text="CL - before training",
        T=t_ext,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
    )


    logger.info('\n------------ Begin training ------------')
    best_valid_loss = 1e6
    t = time.time()
    for epoch in range(1 + args.epochs):
        set_lr(optimizer, base_lrs, epoch, args)
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
            msg += clipper.summary() + ' -- lr: %.2e' % optimizer.param_groups[0]['lr']
            clipper.reset_stats()
            if diagnoser is not None:
                diagnoser.reset_stats()
            logger.info(msg)
            t = time.time()

    # set to best seen during training
    if args.return_best:
        ctl.load_state_dict(best_state)
    elif ema is not None:
        ctl.load_state_dict(ema.state)

# ------ 7. Save and evaluate the trained model ------
# save
    # full controller (M_p, M_b, lifter, ...) + metadata to make the checkpoint self-describing;
    # reload with build_controller(argparse.Namespace(**ckpt['args']), plant).load_state_dict(ckpt['controller'])
    res_dict = {'controller': ctl.state_dict(), 'Q': Q, 'args': vars(args)}
    filename = os.path.join(save_folder, 'trained_controller' + '.pt')
    torch.save(res_dict, filename)
    logger.info('[INFO] saved trained model.')
    logger.info('[INFO] Network certificate of the trained controller: %s' % ctl.operator.certify())

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
    )

    x_verif, _, u_verif = rollout(ctl, data_verif)
    u_verif, v_verif = u_verif.cpu(), sys.v_log.cpu()
    plot_trajectories(
        x_verif[0, :, :],  # remove extra dim due to batching
        xbar=xbar_train,
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_diag_trained.pdf',
        text="rPB - trained controller",
        T=t_ext,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
    )

    plot_trajectories(
        x_verif[1, :, :],  # remove extra dim due to batching
        xbar=xbar_verif2,
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_direct_trained.pdf',
        text="rPB - trained controller",
        T=t_ext,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
    )

    plot_trajectories(
        x_verif[2, :, :],  # remove extra dim due to batching
        xbar=xbar_verif3,
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_center_trained.pdf',
        text="CL - trained controller",
        T=t_ext,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
    )



#### Plot the evolution of the reference over time for the diagonal scenario ####
    x_ref_evol = torch.zeros(1, args.horizon + 200, state_dim)
    x_ref_evol[:, :, sys.pos_idx] = u_verif[0:1].cpu()    # offset u^i of agent i on its position channels
    x_ref_evol = x_ref_evol + xbar_train



    plot_trajectories(
        x_ref_evol[0, :, :],  # remove extra dim due to batching
        xbar=xbar_train,
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='CL_xbar_evolution.pdf',
        text="CL - evolution of the reference",
        T=t_ext,
        dots=True,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
    )


# One subplot per robot (2 robots: a 2x1 grid of size (10, 7))
    t_axis = np.array(range(u_verif.shape[1]))
    fig, axs = plt.subplots(n_agents, 1, figsize=(10, 3.5 * n_agents), squeeze=False)
    for i, ax in enumerate(axs[:, 0]):
        ax.plot(t_axis, u_verif[2, :, 2 * i], label="dX")
        ax.plot(t_axis, u_verif[2, :, 2 * i + 1], label="dY")
        ax.set_title("Robot %i" % (i + 1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Delta ref")
        ax.legend()
        ax.grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

    plt.suptitle(
        'Performance boosting offset to the reference over time \n for the diagonal scenario',
        fontsize=13,
    )
    plt.savefig(os.path.join(save_folder, "U_over_time.pdf"))
    plt.close()


# One subplot per robot
    fig, axs = plt.subplots(n_agents, 1, figsize=(10, 3.5 * n_agents), squeeze=False)
    for i, ax in enumerate(axs[:, 0]):
        ax.plot(t_axis, v_verif[0, :, 2 * i], label="v_X")
        ax.plot(t_axis, v_verif[0, :, 2 * i + 1], label="v_Y")
        ax.set_title("Robot %i" % (i + 1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("v")
        ax.legend()
        ax.grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

    plt.suptitle('Integral variable over time \n for the diagonal scenario', fontsize=13)
    plt.savefig(os.path.join(save_folder, "V_over_time.pdf"))
    plt.close()


if __name__ == "__main__":
    main()
