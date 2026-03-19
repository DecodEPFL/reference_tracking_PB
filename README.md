# Performance Boosting Controllers (Reference Tracking)

This repository contains a small, self-contained PyTorch research codebase to train and evaluate **performance boosting controllers** for a simple multi-robot reference tracking problem with optional **collision** and **obstacle** avoidance.

The main runnable entrypoint is `experiments/robots/run.py`, which:
- generates a synthetic dataset of rollouts + references,
- instantiates the robot plant (`RobotsSystem`) and the controller (`PerfBoostController`),
- trains the controller by backpropagating through closed-loop rollouts,
- saves plots and a trained controller checkpoint to `experiments/robots/saved_results/`.

## Quickstart

### 1) Create an environment and install deps

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Run a training

From the repo root:

```bash
python experiments/robots/run.py
```

Useful overrides (examples):

```bash
# shorter run
python experiments/robots/run.py --epochs 50 --num-rollouts 10 --horizon 50

# disable collision avoidance
python experiments/robots/run.py --no-col-av

# disable obstacle avoidance
python experiments/robots/run.py --no-obst-av

# force CPU (by disabling CUDA visibility)
CUDA_VISIBLE_DEVICES="" python experiments/robots/run.py
```

### 3) Outputs

Each run creates a timestamped folder under:
- `experiments/robots/saved_results/perf_boost_<MM_DD_HH_MM_SS>/`

Typical outputs include:
- `log`: training log file
- `trained_controller.pt`: trained controller weights (REN state_dict + metadata)
- `CL_*_ref.pdf`: closed-loop trajectories before training
- `CL_*_trained.pdf`: closed-loop trajectories after training
- additional diagnostic PDFs (e.g., reference evolution, signals over time)

## Repository map (what each file/folder does)

### Top-level
- **`config.py`**: selects the PyTorch device (`cuda:0` if available, else CPU).
- **`setup.py`**: minimal packaging metadata (not required to run; `requirements.txt` is the main dependency entrypoint).
- **`README.md`**: this file.

### `experiments/robots/`
- **`run.py`**: main experiment script (dataset → plant/controller/loss → training → evaluation/plots/saving).
- **`arg_parser.py`**: CLI flags for `run.py` (epochs, horizon, losses, etc.).
- **`saved_results/`**: generated artifacts (ignored by git).

### `plants/`
- **`costum_dataset.py`**: dataset base class (disk caching + train/test split utilities).
- **`robots/robots_dataset.py`**: synthetic dataset for the robots experiment (initial states + per-rollout references).
- **`robots/robots_sys.py`**: robot plant dynamics + closed-loop rollout helper used for training.

### `controllers/`
- **`PB_controller.py`**: `PerfBoostController` that wraps a contractive REN + a small MLP.
- **`contractive_ren.py`**: `ContractiveREN` implementation (stability/contraction-constrained recurrent model).
- **`MLP.py`**: simple feed-forward network used inside `PerfBoostController`.
- **`__init__.py`**: controller exports.

### `loss_functions/`
- **`lq_loss.py`**: finite-horizon LQ loss base used by experiments.
- **`robots_loss.py`**: robots-specific loss terms (tracking, control effort, collision, obstacles).
- **`__init__.py`**: loss exports.

### `utils/`
- **`assistive_functions.py`**: small utilities (tensor conversion, logging wrapper).
- **`plot_functions.py`**: plotting helpers for trajectories and diagnostics.

## Notes for collaborators

- **Python version**: this repo targets Python 3.10+.
- **Reproducibility**: `--random-seed` controls dataset generation and PyTorch seed.
- **Performance**: training backprops through closed-loop rollouts; GPU helps but CPU runs are supported.

The base case
