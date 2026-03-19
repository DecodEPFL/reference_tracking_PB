import sys, os, logging, torch, time
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController
from loss_functions import RobotsLoss
from utils.assistive_functions import WrapLogger


# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')

save_folder = os.path.join(save_path, 'perf_boost_'+now)
save_folder_gif = os.path.join(save_folder, 'gifs')
os.makedirs(save_folder)

logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args)
logger.info(msg)
torch.manual_seed(args.random_seed)



# ------------ 1. Dataset ------------
xbar_train = torch.tensor([0, 4, 0, 0, 4, 4 ,0 ,0 ])
xbar_verif2 = torch.tensor([5, 4, 0, 0, -1, 4 ,0 ,0 ])
xbar_verif3 = torch.tensor([0.5, 4, 0, 0, 1.5, 4 ,0 ,0 ])
x_init = torch.tensor([7, 4, 0, 0, -3., 4 ,0 ,0 ])


#### Specific obstacle scenarios for the paper ####
obstacle_centers = [
                torch.tensor([[-0.5, 0]], device=device),
                torch.tensor([[0.5, 0.0]], device=device),
            ]

obstacle_centers = [
                torch.tensor([[0.5, 2]], device=device),
                torch.tensor([[1, 2.0]], device=device),
                torch.tensor([[3, 2]], device=device),
                torch.tensor([[3.5, 2.0]], device=device),

            ]

### Get six obstacles ###
obstacle_centers = [
                torch.tensor([[-1, 2]], device=device),
                torch.tensor([[1, 2.0]], device=device),
                torch.tensor([[3, 2]], device=device),
                torch.tensor([[5, 2.0]], device=device),
                torch.tensor([[7, 2.0]], device=device),
                torch.tensor([[-3, 2.0]], device=device),
            ]

obstacle_covs = [
    torch.tensor([[0.05, 0.05]], device=device)
            ] * len(obstacle_centers)

#obstacle_centers = None
#obstacle_covs = None

dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, x_bar=xbar_verif2, std_ini=args.std_init_plant, n_agents=2)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
train_data, test_data = train_data.to(device), test_data.to(device)


# data for plots
t_ext = args.horizon * 4
n_agents = 2

plot_data = test_data[250:350,:,:]
plot_data[:, 0, :8] = dataset.x0.detach()

plot_data = plot_data.to(device)

# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant ------------
plant_input_init = None     # all zero
plant_state_init = None    # same as xbar
sys = RobotsSystem(
    x_init=plant_state_init,
    u_init=plant_input_init, linear_plant=args.linearize_plant, k=args.spring_const, n_agents=n_agents
).to(device)


# ------------ 3. Controller ------------
ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init, output_init=sys.u_init,
    dim_internal=args.dim_internal, dim_nl=args.dim_nl,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)


# ------------ 4. Loss ------------
Q = 100*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)  
Qs = 1*torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)    
loss_fn = RobotsLoss(
    Q=Q,Qs = Qs, alpha_u=args.alpha_u, xbar=train_data[0,:,8:],
    loss_bound=None, sat_bound=None,
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,obstacle_centers=obstacle_centers,obstacle_covs=obstacle_covs,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=sys.n_agents if args.col_av else None,
)
 
# ------------ 5. Optimizer ------------
valid_data = train_data      # use the entire train data for validation
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
 
# ------------ 6. Training ------------
# plot closed-loop trajectories before training the controller
logger.info('Plotting closed-loop trajectories before training the controller...')
x_log, _, u_log = sys.rollout(ctl, plot_data)


data_verif = torch.zeros(3, args.horizon+200, 16)

data_verif[:, 0:1, :8] = \
    dataset.x0 
data_verif[0:1, 1:, 8:] = \
    xbar_train
data_verif[1:2, 1:, 8:] = \
    xbar_verif2
data_verif[2:3, 1:, 8:] = \
    xbar_verif3

x_verif, _, u_verif = sys.rollout(ctl, data_verif)

total_params = sum(p.numel() for p in ctl.parameters())
print(f"Number of parameters: {total_params}")

plot_trajectories(
    x_verif[0, :, :], # remove extra dim due to batching
    xbar=xbar_train, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_diag_ref.pdf',
    text="CL - before training", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)

plot_trajectories(
    x_verif[1, :, :], # remove extra dim due to batching
    xbar=xbar_verif2, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_direct_ref.pdf',
    text="CL - before training", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)

plot_trajectories(
    x_verif[2, :, :], # remove extra dim due to batching
    xbar=xbar_verif3, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_center_ref.pdf',
    text="CL - before training", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)


logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e6
t = time.time()
for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()
        # simulate over horizon steps
        x_log, e_log, u_log = sys.rollout(
            controller=ctl, data=train_data_batch, train=True,
        )
        # loss of this rollout
        loss = loss_fn.forward(x_log, u_log,e_log)
        # take a step
        loss.backward()
        optimizer.step()

    # print info
    if epoch%args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.2f'% (epoch, loss)

        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, e_log_valid, u_log_valid = sys.rollout(
                    controller=ctl, data=valid_data, train=False,
                )
                # loss of the valid data
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid,e_log_valid)
            msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
            # compare with the best valid loss
            if loss_valid.item()<best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params_ren = ctl.get_parameters_as_vector()  # record state dict if best on valid
                best_params_mlp = ctl.get_mlp_parameters()
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % (duration)
        logger.info(msg)
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl.set_parameters_as_vector(best_params_ren)
    ctl.set_mlp_parameters(best_params_mlp)

# ------ 7. Save and evaluate the trained model ------
# save
res_dict = ctl.c_ren.state_dict()
# TODO: append args
res_dict['Q'] = Q
filename = os.path.join(save_folder, 'trained_controller'+'.pt')
torch.save(res_dict, filename)
logger.info('[INFO] saved trained model.')

# evaluate on the train data
logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
with torch.no_grad():
    x_log, e_log, u_log = sys.rollout(
        controller=ctl, data=train_data, train=False,
    )   # use the entire train data, not a batch
    # evaluate losses
    loss = loss_fn.forward(x_log, u_log,e_log)
    msg = 'Loss: %.4f' % (loss)
# count collisions
if args.col_av:
    num_col = loss_fn.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# evaluate on the test data
logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
with torch.no_grad():
    # simulate over horizon steps
    x_log, e_log, u_log = sys.rollout(
        controller=ctl, data=test_data, train=False,
    )
    # loss
    test_loss = loss_fn.forward(x_log, u_log,e_log).item()
    msg = "Loss: %.4f" % (test_loss)
# count collisions
if args.col_av:
    num_col = loss_fn.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)


# count collisions
if args.col_av:
    num_col = loss_fn.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# plot closed-loop trajectories using the trained controller
logger.info('Plotting closed-loop trajectories using the trained controller...')
x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    xbar=plot_data[0,5,8:], n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_trained.pdf',
    text="CL - trained controller", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)

x_verif, _, u_verif = sys.rollout(ctl, data_verif)
v_verif = sys.v_log
plot_trajectories(
    x_verif[0, :, :], # remove extra dim due to batching
    xbar=xbar_train, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_diag_trained.pdf',
    text="rPB - trained controller", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)

plot_trajectories(
    x_verif[1, :, :], # remove extra dim due to batching
    xbar=xbar_verif2, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_direct_trained.pdf',
    text="rPB - trained controller", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)

plot_trajectories(
    x_verif[2, :, :], # remove extra dim due to batching
    xbar=xbar_verif3, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_center_trained.pdf',
    text="CL - trained controller", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)



#### Plot the evolution of the reference over time for the diagonal scenario ####
x_ref_evol = torch.zeros(1,args.horizon+200,8)
x_ref_evol[:,:,0:2] = u_verif[0:1,:,0:2]
x_ref_evol[:,:,4:6] = u_verif[0:1,:,2:4]

x_ref_evol = x_ref_evol + xbar_train



plot_trajectories(
   x_ref_evol[0,:,:], # remove extra dim due to batching
    xbar=xbar_train, n_agents=sys.n_agents,
    save_folder=save_folder, filename='CL_xbar_evolution.pdf',
    text="CL - evolution of the reference", T=t_ext, dots = True,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)


# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 7))
axs[0].plot(np.array(range(u_verif.shape[1])), u_verif[2,:,0],label = "dX")
axs[0].plot(np.array(range(u_verif.shape[1])), u_verif[2,:,1],label = "dY")
axs[0].set_title("Robot 1")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Delta ref")
axs[0].legend()
axs[0].grid()

axs[1].plot(np.array(range(u_verif.shape[1])), u_verif[2,:,2],label = "dX")
axs[1].plot(np.array(range(u_verif.shape[1])), u_verif[2,:,3],label = "dY")
axs[1].set_title("Robot 2")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Delta ref")
axs[1].legend()
axs[1].grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

plt.suptitle(f'Performance boosting offset to the reference over time \n for the diagonal scenario', fontsize=13)
plt.savefig(os.path.join(save_folder, "U_over_time.pdf"))
plt.close()


# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 7))
axs[0].plot(np.array(range(u_verif.shape[1])), v_verif[0,:,0],label = "v_X")
axs[0].plot(np.array(range(u_verif.shape[1])), v_verif[0,:,1],label = "v_Y")
axs[0].set_title("Robot 1")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("v")
axs[0].legend()
axs[0].grid()

axs[1].plot(np.array(range(u_verif.shape[1])), v_verif[0,:,2],label = "v_X")
axs[1].plot(np.array(range(u_verif.shape[1])), v_verif[0,:,3],label = "v_Y")
axs[1].set_title("Robot 2")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("v")
axs[1].legend()
axs[1].grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

plt.suptitle(f'Integral variable over time \n for the diagonal scenario', fontsize=13)
plt.savefig(os.path.join(save_folder, "V_over_time.pdf"))
plt.close()
