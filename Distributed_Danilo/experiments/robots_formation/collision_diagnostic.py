"""Where do the trained controllers collide? Re-runs trained runs on their own train and test rollouts and
relates the collisions to the target of each rollout (rotation, position) and to how well the training set
covers it.

Per run: the logged train/test collision summary (check it against the log), the collision rate and the
pair-steps per colliding rollout by target rotation, train vs test, the share of the test collision loss
from the worst rollouts and the agent pairs that collide.
Over the runs (they must share --random-seed, i.e. the same data): how many runs collide on each test
rollout, and the rollouts most of them collide on. A rollout every size collides on is hard for the data,
not for one model.
Coverage: distance of each test target to the nearest train target, on (cos, sin) of the rotation and the
centre scaled to the target box; compared between the test rollouts with and without collisions.

Saves <out>/rollouts.csv (one row per run and rollout) and <out>/collision_diagnostic.pdf.
Usage: python collision_diagnostic.py <run folder> [<run folder> ...] [--out folder] [--no-compile]
"""
import argparse, csv, math, os, sys, statistics
import torch

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Distributed_Danilo.config import device
from Distributed_Danilo.experiments.robots_formation.arg_parser import argument_parser, saved_run_args
from Distributed_Danilo.experiments.robots_formation.run import get_data, make_dataset, make_loss_fn, make_plant
from Distributed_Danilo.plants import RobotsSystem
from Distributed_Danilo.controllers import build_controller

ANGLE_BINS = (0, 60, 120, 150, 165, 180.01)     # |rotation of the target| in degrees
WORST = (5, 10, 25)                             # share of the test collision loss from the worst k rollouts


def load_run(folder, compile_):
    """Args (saved over the current defaults), plant and trained controller of a run folder."""
    ckpt = torch.load(os.path.join(folder, 'trained_controller.pt'), map_location=device, weights_only=False)
    # args added since that run: what it trained with (as --resume), else the current defaults
    args = argparse.Namespace(**{**vars(argument_parser([])), **saved_run_args(ckpt['args'])})
    if compile_ is not None:
        args.compile = compile_
    plant = make_plant(args).to(device)    # with the sticky patches of the run, if any
    ctl = build_controller(args, plant=plant).to(device)
    ctl.load_state_dict(ckpt['controller'])
    return args, plant, ctl


def target_pose(xbar, offsets, start_theta):
    """Rotation (relative to the start, wrapped to [-180, 180) degrees) and centre of each target: (S,), (S, 2).
    start_theta: angle of the start formation, a float or one per target (S,) (flipped starts: + pi)."""
    n = offsets.shape[0]
    pos = xbar.reshape(xbar.shape[0], n, 4)[..., :2]                          # (S, n, 2)
    q = offsets - offsets.mean(0)
    p = pos - pos.mean(1, keepdim=True)
    cross = (q[None, :, 0] * p[..., 1] - q[None, :, 1] * p[..., 0]).sum(1)
    dot = (q[None] * p).sum((1, 2))
    theta = torch.atan2(cross, dot)                                           # labelled agents: no symmetry
    c, s = torch.cos(theta), torch.sin(theta)
    m = offsets.mean(0)
    center = pos.mean(1) - torch.stack([c * m[0] - s * m[1], s * m[0] + c * m[1]], -1)
    rel = torch.remainder(torch.rad2deg(theta - torch.as_tensor(start_theta, dtype=theta.dtype)) + 180, 360) - 180
    return rel, center


def coverage_features(rel_deg, center, dataset):
    """(cos, sin) of the rotation and the centre scaled to [0, 1] over the target box: (S, 4)."""
    r = torch.deg2rad(rel_deg)
    (x0, x1), (y0, y1) = dataset.target_x, dataset.target_y
    return torch.stack([torch.cos(r), torch.sin(r), (center[:, 0] - x0) / (x1 - x0), (center[:, 1] - y0) / (y1 - y0)], -1)


@torch.no_grad()
def per_rollout(args, plant, ctl, loss_fn, data):
    """Collision and formation metrics of every rollout of data (dict of (S,) tensors) and the colliding pairs."""
    x, e, u = plant.rollout(ctl, data, train=False, zero_context=(args.context == 'none'))
    d2 = loss_fn.get_pairwise_distance_sq(x.reshape(*x.shape, 1))           # (S, T, n, n)
    col = (0.0001 < d2) & (d2 < loss_fn.min_dist ** 2)
    n = d2.shape[-1]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=d2.device)
    t_idx = torch.arange(x.shape[1], device=x.device).view(1, -1).expand(x.shape[0], -1)
    any_t = col.flatten(2).any(-1)                                            # (S, T)
    first = torch.where(any_t, t_idx, x.shape[1]).amin(1)
    first = torch.where(any_t.any(1), first, torch.full_like(first, -1))     # -1: no collision
    err_mean, _ = loss_fn.formation_error(x)
    metrics = {
        'collision_loss': loss_fn.loss_terms(x, u, e)['collision'].flatten(),
        'pair_steps': col.sum((1, 2, 3)) / 2,
        'first_col_t': first,
        'min_dist': d2[..., off_diag].amin(-1).amin(-1).clamp(min=0).sqrt(),
        'form_err': err_mean.mean(1),
        'mirrored': loss_fn.mirrored(x),
    }
    pairs = torch.triu(col.any(1), diagonal=1)                                # (S, n, n): pair collided at some step
    return {k: v.float().cpu() for k, v in metrics.items()}, pairs.cpu()


def bin_table(rel_deg, m, label):
    """Collision rate and pair-steps per colliding rollout by |rotation|."""
    rows = []
    a = rel_deg.abs()
    for lo, hi in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:]):
        sel = (a >= lo) & (a < hi)
        k = int(sel.sum())
        hit = sel & (m['pair_steps'] > 0)
        h = int(hit.sum())
        rows.append('    %-5s |rot| %3d-%3d deg: %4d rollouts, %5.1f%% collide, %5.1f pair-steps per colliding one, '
                    'collision loss %7.1f per rollout' % (
                        label, lo, min(hi, 180), k, 100 * h / max(k, 1),
                        m['pair_steps'][hit].mean().item() if h else 0.0,
                        m['collision_loss'][sel].mean().item() if k else 0.0))
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='Run folders (with trained_controller.pt).')
    parser.add_argument('--out', default=None, help='Output folder. Default: collision_diagnostic/ next to the first run.')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=None,
                        help='torch.compile the agents (default: as the run was trained).')
    parser.add_argument('--num-threads', type=int, default=int(os.environ.get('CPUS', 4)))
    parser.add_argument('--top', type=int, default=15, help='Test rollouts listed as the hardest. Default 15.')
    opts = parser.parse_args()
    torch.set_num_threads(opts.num_threads)
    out = opts.out or os.path.join(os.path.dirname(os.path.abspath(opts.runs[0].rstrip('/'))), 'collision_diagnostic')
    os.makedirs(out, exist_ok=True)

    data_key, dataset, results = None, None, {}
    for folder in opts.runs:
        name = os.path.basename(os.path.abspath(folder.rstrip('/')))
        args, plant, ctl = load_run(folder, opts.compile)
        key = (args.random_seed, args.num_rollouts, args.horizon, args.n_agents, args.formation_shape,
               args.large_rot_frac, args.init_min_dist, args.start_flip_frac, args.test_start_flip_frac,
               (args.patch_n, *args.patch_w, *args.patch_sigma, *(args.patch_x or args.wall_x), *args.patch_y)
               if args.patches else None)
        if data_key is not None and key != data_key:
            sys.exit('%s was trained on other data (seed, rollouts, horizon, agents, shape, large-rot, init-min-dist, '
                     'start flips, sticky patches): %s vs %s'
                     % (name, key, data_key))
        if dataset is None:
            data_key, dataset = key, make_dataset(args)
            train_data, test_data = get_data(args, dataset)     # with their sticky patches, if any
            train_data, test_data = train_data.to(device), test_data.to(device)
            sd = 4 * args.n_agents
            # rotations relative to the start of each rollout (flipped starts: start_theta + pi)
            flipped = {s: dataset.flipped_starts(d).cpu() for s, d in (('train', train_data), ('test', test_data))}
            pose = {s: target_pose(d[:, 1, sd:2 * sd].cpu(), dataset.offsets, dataset.start_theta + math.pi * flipped[s].float())
                    for s, d in (('train', train_data), ('test', test_data))}
            feats = {s: coverage_features(*pose[s], dataset) for s in pose}
            nn_dist = torch.cdist(feats['test'], feats['train']).amin(1)       # (S_test,)
            x0 = torch.where(flipped['test'].view(-1, 1), dataset.x0_flipped, dataset.x0).to(device)
            noise = (test_data[:, 0, :sd] - x0).norm(dim=1).cpu()
        loss_fn = make_loss_fn(args, dataset, xbar=train_data[0, :, 4 * args.n_agents:8 * args.n_agents])
        res = {s: per_rollout(args, plant, ctl, loss_fn, d) for s, d in (('train', train_data), ('test', test_data))}
        results[name] = res

        print('\n==== %s' % name)
        for s in ('train', 'test'):
            m = res[s][0]
            print('  %-5s collisions: %d/%d rollouts (%d pair-steps) -- collision loss %.2f -- formation error %.3f'
                  % (s, int((m['pair_steps'] > 0).sum()), len(m['pair_steps']), int(m['pair_steps'].sum()),
                     m['collision_loss'].mean(), m['form_err'].mean()))
        for s in ('train', 'test'):
            print(bin_table(pose[s][0], res[s][0], s))
        cl = res['test'][0]['collision_loss'].sort(descending=True).values
        print('  test collision loss from the worst %s rollouts: %s of the total' % (
            '/'.join(map(str, WORST)), ' / '.join('%.0f%%' % (100 * cl[:k].sum() / max(cl.sum(), 1e-12)) for k in WORST)))
        pairs = res['test'][1].sum(0)
        top_pairs = sorted(((int(pairs[i, j]), i, j) for i in range(pairs.shape[0]) for j in range(i + 1, pairs.shape[0])),
                           reverse=True)
        print('  test pairs that collide (rollouts): ' + ', '.join('%d-%d: %d' % (i, j, c) for c, i, j in top_pairs if c))
        hit = res['test'][0]['pair_steps'] > 0
        if hit.any() and (~hit).any():
            print('  coverage: nearest train target, median %.3f for the colliding test rollouts vs %.3f for the others'
                  % (nn_dist[hit].median(), nn_dist[~hit].median()))

    # ---- over the runs ----
    names = list(results)
    hits = torch.stack([results[n]['test'][0]['pair_steps'] > 0 for n in names]).sum(0)    # (S_test,) runs colliding
    steps = torch.stack([results[n]['test'][0]['pair_steps'] for n in names]).sum(0)
    rel, center = pose['test']
    print('\n==== over the %d runs (test rollouts)' % len(names))
    for k in range(len(names) + 1):
        c = int((hits == k).sum())
        if c:
            print('  collide in %d run(s): %3d rollouts -- median |rot| %5.1f deg -- median nearest train target %.3f'
                  % (k, c, rel[hits == k].abs().median(), nn_dist[hits == k].median()))
    order = sorted(range(len(hits)), key=lambda i: (-int(hits[i]), -float(steps[i])))[:opts.top]
    print('\n  hardest test rollouts (most runs colliding, then most pair-steps):')
    print('  %5s %5s %10s %8s %16s %10s %8s' % ('idx', 'runs', 'pair-steps', 'rot', 'centre', 'nn train', '|x0 noise|'))
    for i in order:
        print('  %5d %5d %10d %7.1f° (%6.2f, %5.2f) %10.3f %8.2f' % (
            i, hits[i], steps[i], rel[i], center[i, 0], center[i, 1], nn_dist[i], noise[i]))

    # ---- CSV ----
    with open(os.path.join(out, 'rollouts.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        cols = list(next(iter(results.values()))['test'][0])
        w.writerow(['run', 'split', 'idx', 'rot_deg', 'center_x', 'center_y', 'nn_train'] + cols)
        for n in names:
            for s in ('train', 'test'):
                m = results[n][s][0]
                r, c = pose[s]
                for i in range(len(r)):
                    w.writerow([n, s, i, '%.2f' % r[i], '%.3f' % c[i, 0], '%.3f' % c[i, 1],
                                '%.4f' % nn_dist[i] if s == 'test' else ''] + ['%.4g' % m[k][i] for k in cols])

    # ---- plot ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    tr_rel, tr_c = pose['train']
    for ax, (j, lab) in zip(axes[:2], ((0, 'target centre x'), (1, 'target centre y'))):
        ax.scatter(tr_rel, tr_c[:, j], s=8, c='0.6', marker='x', label='train targets')
        clean = hits == 0
        ax.scatter(rel[clean], center[clean, j], s=10, facecolors='none', edgecolors='tab:blue', lw=0.6, label='test, no collision')
        sc = ax.scatter(rel[~clean], center[~clean, j], s=12 + 4 * hits[~clean].float(), c=hits[~clean],
                        cmap='autumn_r', vmin=1, vmax=len(names), edgecolors='k', lw=0.4, label='test, collision')
        ax.set_xlabel('target rotation (deg)')
        ax.set_ylabel(lab)
        ax.set_xlim(-180, 180)
    fig.colorbar(sc, ax=axes[1], label='runs colliding')
    axes[0].legend(fontsize=8, loc='upper left')
    ax = axes[2]
    ax.scatter(nn_dist, steps / len(names), s=10, c=hits, cmap='autumn_r', edgecolors='k', lw=0.3)
    ax.set_xlabel('distance to the nearest train target')
    ax.set_ylabel('pair-steps in collision (mean over runs)')
    fig.suptitle('Test collisions vs target (%d runs: %s)' % (len(names), ', '.join(names)), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'collision_diagnostic.pdf'))
    print('\nsaved %s and %s' % (os.path.join(out, 'rollouts.csv'), os.path.join(out, 'collision_diagnostic.pdf')))


if __name__ == '__main__':
    main()
