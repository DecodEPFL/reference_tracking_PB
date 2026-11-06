"""Summary plots of the relay vs Danilo benchmark (sweep_vs_danilo.sh + sweep_core_append.sh), from the run logs.

Methods: relay (Distributed/), and Danilo's certified SSM network with the neighbors' commands fed back raw (raw),
normalized (norm), or the neighbors' core outputs appended to the core input (core_append, a = 0.1). One column per
SSM size point (d_model:d_state), seeds 1-10, then 11-30 (finished runs only).

Figures (written to --out, as .pdf and .png):
  test_loss_box     final test loss (500 test rollouts, every term) per method, one box per point, seeds as dots
  val_loss_curves   validation loss over training: median over the seeds, band = interquartile range, light band = min-max
  test_breakdown    test loss per term: mean over the seeds (bars), seeds as dots; log scale

Usage: python plot_benchmark.py [--out DIR]
"""
import argparse, glob, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
GRAPH = "saved_results/5_robots/matrix_01111-00100-01000-10001-10010/benchmark"
DIST = os.path.join(REPO, "Distributed/experiments/robots_formation", GRAPH)
DANILO = os.path.join(REPO, "Distributed_Danilo/experiments/robots_formation", GRAPH)

# method -> (label, run folder glob without the point/seed suffix); fixed categorical order and colors
METHODS = [
    ("relay", "Relay", os.path.join(DIST, "vs_danilo_600x200_fs50_e2500_patches", "vs-relay")),
    ("core", "Core append\n(a = 0.1)", os.path.join(DANILO, "core_append_600x200_fs50_e2500_a0.1_patches", "ca-core-a0p1")),
    ("norm", "Normalized u\nfeedback", os.path.join(DANILO, "vs_danilo_600x200_fs50_e2500_patches", "vs-norm")),
    ("raw", "Raw u\nfeedback", os.path.join(DANILO, "vs_danilo_600x200_fs50_e2500_patches", "vs-raw")),
]
COLORS = {"relay": "#2a78d6", "core": "#eb6834", "norm": "#1baf7a", "raw": "#eda100"}
POINTS = [("8", "16"), ("14", "27")]
TERMS = ["formation", "orientation_max", "collision", "orientation", "obstacle_max", "obstacle",
         "speed", "final", "input", "tracking", "wall"]

EPOCH = re.compile(r"Epoch: (\d+) --- train loss: ([\d.naninf]+) ---\|\|--- validation loss: ([\d.naninf]+)")
TERM = re.compile(r"(\w+): ([\d.e+-]+)")


def parse(path):
    lines = open(path).read().splitlines()
    epochs, val = [], []
    test = None
    for i, line in enumerate(lines):
        m = EPOCH.search(line)
        if m:
            epochs.append(int(m.group(1)))
            val.append(float(m.group(3)))
        if "test rollouts" in line and i + 1 < len(lines) and "Loss:" in lines[i + 1]:
            head, terms = lines[i + 1].split(" -- ")[:2]
            test = {"total": float(head.split("Loss:")[1])}
            test.update({k: float(v) for k, v in TERM.findall(terms)})
    return np.array(epochs), np.array(val), test


def load():
    data = {}  # (method, point) -> list of runs
    for key, _, prefix in METHODS:
        for dm, ds in POINTS:
            runs = []
            for d in sorted(glob.glob(f"{prefix}-dm{dm}-ds{ds}-s*-e2500_*")):
                log = os.path.join(d, "log")
                if os.path.exists(log):
                    ep, val, test = parse(log)
                    if test is not None:
                        runs.append({"name": os.path.basename(d), "epochs": ep, "val": val, "test": test})
            data[key, (dm, ds)] = runs
            print(f"{key:5s} dm{dm}:ds{ds}: {len(runs)} finished runs")
    return data


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def point_title(p, data=None):
    title = f"SSM d_model {p[0]}, d_state {p[1]}"
    if data is not None:
        counts = sorted({len(data[k, p]) for k, _, _ in METHODS})
        title += f" ({'-'.join(map(str, counts))} seeds per method)"
    return title


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def test_loss_box(data, out):
    fig, axes = plt.subplots(1, len(POINTS), figsize=(5.2 * len(POINTS), 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, p in zip(axes, POINTS):
        vals = [[r["test"]["total"] for r in data[k, p]] for k, _, _ in METHODS]
        bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="#0b0b0b", linewidth=1.6),
                        whiskerprops=dict(color="#52514e"), capprops=dict(color="#52514e"))
        for patch, (k, _, _) in zip(bp["boxes"], METHODS):
            patch.set(facecolor=COLORS[k] + "55", edgecolor=COLORS[k], linewidth=1.5)
        for j, (v, (k, _, _)) in enumerate(zip(vals, METHODS)):
            ax.scatter(j + 1 + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=COLORS[k],
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.annotate(f"{np.median(v):.0f}", (j + 1.33, np.median(v)), va="center", fontsize=8.5, color="#52514e")
        ax.set_xticks(range(1, len(METHODS) + 1), [lbl for _, lbl, _ in METHODS], fontsize=9)
        ax.set_yscale("log")
        ax.set_title(point_title(p, data), fontsize=10.5, color="#0b0b0b")
        style(ax)
    axes[0].set_ylabel("Test loss (500 rollouts, log scale)", fontsize=10)
    fig.suptitle("Final test loss per seed (median labelled)", fontsize=12, x=0.08, ha="left")
    fig.tight_layout()
    save(fig, out, "test_loss_box")


def val_loss_curves(data, out):
    fig, axes = plt.subplots(1, len(POINTS), figsize=(5.6 * len(POINTS), 4.2), sharey=True)
    for ax, p in zip(axes, POINTS):
        for k, lbl, _ in METHODS:
            runs = data[k, p]
            n = min(len(r["epochs"]) for r in runs)
            ep = runs[0]["epochs"][:n]
            V = np.stack([r["val"][:n] for r in runs])
            q0, q25, q50, q75, q100 = np.nanpercentile(V, [0, 25, 50, 75, 100], axis=0)
            ax.fill_between(ep, q0, q100, color=COLORS[k], alpha=0.08, linewidth=0)
            ax.fill_between(ep, q25, q75, color=COLORS[k], alpha=0.22, linewidth=0)
            ax.plot(ep, q50, color=COLORS[k], linewidth=2, label=lbl.replace("\n", " "))
        ax.set_yscale("log")
        ax.set_ylim(1e2, 3e4)        # the first epochs and a few spikes go above: clipped
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_title(point_title(p, data), fontsize=10.5)
        style(ax)
    axes[0].set_ylabel("Validation loss (log scale)", fontsize=10)
    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Validation loss during training: median, interquartile range (dark band), min-max (light band)",
                 fontsize=12, x=0.06, ha="left")
    fig.tight_layout()
    save(fig, out, "val_loss_curves")


def test_breakdown(data, out):
    fig, axes = plt.subplots(len(POINTS), 1, figsize=(11, 3.6 * len(POINTS)), sharex=True)
    n_m = len(METHODS)
    width = 0.8 / n_m
    x = np.arange(len(TERMS))
    for ax, p in zip(axes, POINTS):
        for j, (k, lbl, _) in enumerate(METHODS):
            T = np.array([[r["test"].get(t, np.nan) for t in TERMS] for r in data[k, p]])
            pos = x - 0.4 + width * (j + 0.5)
            ax.bar(pos, np.nanmean(T, axis=0), width * 0.88, color=COLORS[k], label=lbl.replace("\n", " "), zorder=2)
            for c in range(len(TERMS)):
                ax.scatter(np.full(T.shape[0], pos[c]), T[:, c], s=6, color="#0b0b0b", alpha=0.45, zorder=3,
                           linewidth=0)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Test loss term\n(symlog)", fontsize=10)
        ax.set_title(point_title(p, data), fontsize=10.5, loc="left")
        style(ax)
    axes[-1].set_xticks(x, [t.replace("_", "\n") for t in TERMS], fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, ncol=n_m, loc="upper right")
    fig.suptitle("Test loss per term: mean over the seeds (bars), individual seeds (dots)", fontsize=12, x=0.06,
                 ha="left")
    fig.tight_layout()
    save(fig, out, "test_breakdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(DIST, "vs_danilo_summary"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load()
    test_loss_box(data, args.out)
    val_loss_curves(data, args.out)
    test_breakdown(data, args.out)
    print("figures in", args.out)
