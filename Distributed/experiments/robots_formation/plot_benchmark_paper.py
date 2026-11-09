"""Paper figures for the relay vs Danilo benchmark at one SSM size (default d_model 14, d_state 27).

Writes two independent PDFs, without titles (axis labels + legend only):
  <OUT>/test_loss_box_dm14_ds27.pdf     final test loss per seed, one box per method
  <OUT>/val_loss_curves_dm14_ds27.pdf   validation loss over training: median, IQR band, (optional) min-max band

Everything meant to be tweaked is in the CONFIG block below.
Run from anywhere: python plot_benchmark_paper.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================== CONFIG
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
GRAPH = "saved_results/5_robots/matrix_01111-00100-01000-10001-10010/benchmark"
DIST = os.path.join(REPO, "Distributed/experiments/robots_formation", GRAPH)
DANILO = os.path.join(REPO, "Distributed_Danilo/experiments/robots_formation", GRAPH)

D_MODEL, D_STATE = 14, 27
OUT = os.path.join(DIST, "vs_danilo_summary")

# (key, label shown in the figure, run folder prefix, color). Order = order in the figures.
METHODS = [
    ("relay", "Relay", os.path.join(DIST, "vs_danilo_600x200_fs50_e2500_patches", "vs-relay"), "#2a78d6"),
    ("core", "Core append (a = 0.1)", os.path.join(DANILO, "core_append_600x200_fs50_e2500_a0.1_patches", "ca-core-a0p1"), "#eb6834"),
    ("norm", "Normalized u feedback", os.path.join(DANILO, "vs_danilo_600x200_fs50_e2500_patches", "vs-norm"), "#1baf7a"),
    ("raw", "Raw u feedback", os.path.join(DANILO, "vs_danilo_600x200_fs50_e2500_patches", "vs-raw"), "#eda100"),
]

FONTSIZE = 10
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 9
LEGEND_BELOW = True          # legend centered under the axes (2 columns); False: inside, at *_LEGEND_LOC
LEGEND_NCOL = 2

# box plot
BOX_FIGSIZE = (5.2, 4.0)
BOX_YLABEL = "Test loss"
BOX_LOG_Y = True
BOX_SHOW_SEEDS = True        # individual seeds as jittered dots
BOX_SHOW_MEDIAN_TEXT = True  # median value written next to each box
BOX_TICK_LABELS = False      # True: method names under the boxes; False: blank x axis + legend
BOX_LEGEND_LOC = "upper left"

# validation curves
CURVE_FIGSIZE = (5.6, 4.0)
CURVE_XLABEL = "Epoch"
CURVE_YLABEL = "Validation loss"
CURVE_LOG_Y = True
CURVE_YLIM = (1e2, 3e4)      # None for automatic; the first epochs and a few spikes exceed 3e4
CURVE_SHOW_IQR = True        # 25-75 % band
CURVE_SHOW_MINMAX = False    # min-max band (busy with 4 methods)
CURVE_LEGEND_LOC = "upper right"
# ====================================================================

EPOCH = re.compile(r"Epoch: (\d+) --- train loss: ([\d.naninf]+) ---\|\|--- validation loss: ([\d.naninf]+)")
TEST = re.compile(r"test rollouts\.?\s*\n[^\n]*?Loss: ([\d.]+|nan|inf)")


def parse(path):
    text = open(path).read()
    epochs, val = [], []
    for m in EPOCH.finditer(text):
        epochs.append(int(m.group(1)))
        val.append(float(m.group(3)))
    test = TEST.findall(text)
    return np.array(epochs), np.array(val), (float(test[-1]) if test else None)


def load():
    data = {}
    for key, _, prefix, _ in METHODS:
        runs = []
        for d in sorted(glob.glob(f"{prefix}-dm{D_MODEL}-ds{D_STATE}-s*-e2500_*")):
            log = os.path.join(d, "log")
            if os.path.exists(log):
                ep, val, test = parse(log)
                if test is not None:  # finished runs only
                    runs.append({"epochs": ep, "val": val, "test": test})
        data[key] = runs
        print(f"{key:5s} dm{D_MODEL}:ds{D_STATE}: {len(runs)} finished runs")
    return data


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def legend(ax, loc, handles=None):
    kw = dict(frameon=False, fontsize=LEGEND_FONTSIZE)
    if handles is not None:
        kw["handles"] = handles
    if LEGEND_BELOW:
        below = -0.04 if ax.get_xlabel() == "" and not ax.get_xticks().size else -0.14
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, below), ncol=LEGEND_NCOL, **kw)
    else:
        ax.legend(loc=loc, **kw)


def box_figure(data):
    fig, ax = plt.subplots(figsize=BOX_FIGSIZE)
    rng = np.random.default_rng(0)
    vals = [[r["test"] for r in data[k]] for k, _, _, _ in METHODS]
    bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#0b0b0b", linewidth=1.6),
                    whiskerprops=dict(color="#52514e"), capprops=dict(color="#52514e"))
    for j, (patch, v, (_, label, _, color)) in enumerate(zip(bp["boxes"], vals, METHODS)):
        patch.set(facecolor=color + "55", edgecolor=color, linewidth=1.5, label=label)
        if BOX_SHOW_SEEDS:
            ax.scatter(j + 1 + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=color,
                       edgecolor="white", linewidth=0.6, zorder=3)
        if BOX_SHOW_MEDIAN_TEXT:
            ax.annotate(f"{np.median(v):.0f}", (j + 1.33, np.median(v)), va="center",
                        fontsize=TICK_FONTSIZE - 0.5, color="#52514e")
    if BOX_TICK_LABELS:
        ax.set_xticks(range(1, len(METHODS) + 1), [m[1] for m in METHODS], fontsize=TICK_FONTSIZE)
    else:
        ax.set_xticks([])
    if BOX_LOG_Y:
        ax.set_yscale("log")
    ax.set_ylabel(BOX_YLABEL, fontsize=FONTSIZE)
    style(ax)
    legend(ax, BOX_LEGEND_LOC, handles=bp["boxes"])
    fig.tight_layout()
    return fig


def curve_figure(data):
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    for k, label, _, color in METHODS:
        runs = data[k]
        n = min(len(r["epochs"]) for r in runs)
        ep = runs[0]["epochs"][:n]
        V = np.stack([r["val"][:n] for r in runs])
        q0, q25, q50, q75, q100 = np.nanpercentile(V, [0, 25, 50, 75, 100], axis=0)
        if CURVE_SHOW_MINMAX:
            ax.fill_between(ep, q0, q100, color=color, alpha=0.08, linewidth=0)
        if CURVE_SHOW_IQR:
            ax.fill_between(ep, q25, q75, color=color, alpha=0.22, linewidth=0)
        ax.plot(ep, q50, color=color, linewidth=2, label=label)
    if CURVE_LOG_Y:
        ax.set_yscale("log")
    if CURVE_YLIM is not None:
        ax.set_ylim(*CURVE_YLIM)
    ax.set_xlabel(CURVE_XLABEL, fontsize=FONTSIZE)
    ax.set_ylabel(CURVE_YLABEL, fontsize=FONTSIZE)
    style(ax)
    legend(ax, CURVE_LEGEND_LOC)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = load()
    tag = f"dm{D_MODEL}_ds{D_STATE}"
    for name, fig in [(f"test_loss_box_{tag}", box_figure(data)), (f"val_loss_curves_{tag}", curve_figure(data))]:
        path = os.path.join(OUT, f"{name}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)
