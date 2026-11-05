"""Compare the training runs of a benchmark folder from their logs (e.g. the stabiliser ablation of sweep_stabilisers.sh).

Runs are grouped by arm: the run folder name without its trailing _<uuid> and -s<seed> (stab-hinge-s5_1a2b3c4d -> stab-hinge).
The collision term is left out of the compared losses: its scale depends on --col-loss, so the arms are compared on
the loss without it, the collision counts, the formation error and the mirrored fraction.

Usage: python compare_runs.py <benchmark folder> [--plot out.pdf]
"""
import argparse, os, re, statistics

EPOCH = re.compile(r'Epoch: (\d+) --- train loss: ([\d.naninf]+) ---\|\|--- validation loss: ([\d.naninf]+)')
TERMS = re.compile(r'(\w+): ([\d.naninf]+)')
FORM = re.compile(r'formation error: mean ([\d.]+) \| max ([\d.]+) \| mirrored: (\d+)%')
COLL = re.compile(r'Collisions: (\d+)/(\d+) rollouts \((\d+) pair-steps\)')
GRAD = re.compile(r'grad norm: median ([\d.e+-]+) \| max ([\d.e+-]+)(?: \| clipped (\d+)/(\d+))?')


def terms_without_collision(line):
    """Sum of the loss terms of a breakdown line, collision left out."""
    # 'validation terms: <terms> -- formation error ...' or 'Loss: <total> -- <terms> -- formation error ...'
    part = line.split(' -- ')[1] if 'Loss: ' in line else line.split(' -- ')[0].split('terms: ')[-1]
    return sum(float(v) for k, v in TERMS.findall(part) if k != 'collision')


def parse_log(path):
    lines = open(path).read().splitlines()
    run = {'epochs': [], 'val': [], 'val_nocol': [], 'mirrored': [], 'clipped': 0, 'steps': 0, 'test': None}
    for i, line in enumerate(lines):
        m = EPOCH.search(line)
        if m and i + 1 < len(lines):
            terms = lines[i + 1]
            run['epochs'].append(int(m.group(1)))
            run['val'].append(float(m.group(3)))
            run['val_nocol'].append(terms_without_collision(terms))
            f = FORM.search(terms)
            run['mirrored'].append(int(f.group(3)) if f else float('nan'))
            g = GRAD.search(terms)
            if g and g.group(3):
                run['clipped'] += int(g.group(3))
                run['steps'] += int(g.group(4))
        if 'evaluating the trained controller on' in line and 'test rollouts' in line:
            res = next((l for l in lines[i + 1:i + 3] if 'Loss: ' in l), None)
            if res:
                f, c = FORM.search(res), COLL.search(res)
                run['test'] = {'nocol': terms_without_collision(res),
                               'form_err': float(f.group(1)) if f else float('nan'),
                               'mirrored': int(f.group(3)) if f else float('nan'),
                               'col_rollouts': '%s/%s' % (c.group(1), c.group(2)) if c else '0',
                               'col_steps': int(c.group(3)) if c else 0}
    return run


def worst_spike(values):
    """Largest ratio of a logged value to the best one logged before it (1: never went up)."""
    best, worst = float('inf'), 1.0
    for v in values:
        if best < float('inf') and best > 0:
            worst = max(worst, v / best)
        best = min(best, v)
    return worst


def arm_of(name):
    return re.sub(r'-s\d+(?=-|$)', '', re.sub(r'_[0-9a-f]{8}$', '', name))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder')
    parser.add_argument('--plot', default=None, help='Save the validation curves to this file (needs matplotlib).')
    args = parser.parse_args()

    runs = {}
    for name in sorted(os.listdir(args.folder)):
        path = os.path.join(args.folder, name, 'log')
        if os.path.isfile(path):
            runs[name] = parse_log(path)

    header = '%-28s %6s %10s %8s %7s %8s | %10s %9s %7s %8s %6s' % (
        'run', 'epoch', 'best val*', 'at ep', 'spike*', 'clipped',
        'test loss*', 'collided', 'pairs', 'form err', 'mirr')
    print('* without the collision term; spike: worst ratio of the validation loss to its best value before\n')
    print(header + '\n' + '-' * len(header))
    for name, r in runs.items():
        if not r['val']:
            print('%-28s no logged epoch yet' % name)
            continue
        best = min(range(len(r['val_nocol'])), key=r['val_nocol'].__getitem__)
        clipped = '%d/%d' % (r['clipped'], r['steps']) if r['steps'] else '-'
        t = r['test']
        test = ('%10.1f %9s %7d %8.3f %5d%%' % (t['nocol'], t['col_rollouts'], t['col_steps'], t['form_err'], t['mirrored'])
                if t else '%10s' % 'running')
        print('%-28s %6d %10.1f %8d %7.1f %8s | %s' % (
            name, r['epochs'][-1], r['val_nocol'][best], r['epochs'][best], worst_spike(r['val_nocol']), clipped, test))

    # per arm, over the seeds
    arms = {}
    for name, r in runs.items():
        if r['test']:
            arms.setdefault(arm_of(name), []).append(r)
    if any(len(v) > 1 for v in arms.values()):
        print('\nper arm (median over the finished seeds):')
        for arm, rs in arms.items():
            print('%-20s %d seeds -- test loss* %.1f -- form err %.3f -- collided pair-steps %d -- worst spike %.1f' % (
                arm, len(rs), statistics.median(r['test']['nocol'] for r in rs),
                statistics.median(r['test']['form_err'] for r in rs),
                statistics.median(r['test']['col_steps'] for r in rs),
                statistics.median(worst_spike(r['val_nocol']) for r in rs)))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = sorted({arm_of(n) for n in runs})
        colors = {a: plt.cm.tab10(i % 10) for i, a in enumerate(names)}
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        seen = set()
        for name, r in runs.items():
            arm = arm_of(name)
            label = None if arm in seen else arm
            seen.add(arm)
            ax1.plot(r['epochs'], r['val_nocol'], color=colors[arm], label=label, lw=1.2)
            ax2.plot(r['epochs'], r['mirrored'], color=colors[arm], lw=1.2)
        ax1.set_yscale('log')
        ax1.set_ylabel('validation loss (no collision term)')
        ax2.set_ylabel('mirrored (%)')
        ax2.set_xlabel('epoch')
        ax1.legend()
        fig.tight_layout()
        fig.savefig(args.plot)
        print('\nsaved', args.plot)


if __name__ == '__main__':
    main()
