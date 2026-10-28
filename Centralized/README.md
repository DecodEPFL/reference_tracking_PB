# Performance Boosting Controllers (Reference Tracking)

This repository contains a small, self-contained PyTorch research codebase to train and evaluate **performance boosting controllers** for a simple multi-robot reference tracking problem with optional **collision** and **obstacle** avoidance.

The PB operator is factorized as

```
u_t = output_amplification · M_b(ŵ_t, z_t) ⊠ M_p(ŵ_t, z_t)
```

where `ŵ_t` is the reconstructed disturbance, `z_t = x̄_t` is the context (the reference positions), `M_p` is a recurrent l2-stable operator (a deep SSM, optionally a **contextual SSM** that injects `z` through selectable ports) and `M_b` is a bounded memoryless mixer.

The main runnable entrypoint is `experiments/robots/run.py`, which:
- generates a synthetic dataset of rollouts + references,
- instantiates the robot plant (`RobotsSystem`) and the controller (`PBController`, built from the CLI flags by `controllers/factory.py`),
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
python -c "from neural_ssm import ContextualDeepSSM"   # needs the pinned neural_ssm, older releases lack it
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

# choose the operator (see "Operator configurations" below)
python experiments/robots/run.py --mp-type contextual_ssm --mb-type none --ctx-modes input,gate

# force CPU (by disabling CUDA visibility)
CUDA_VISIBLE_DEVICES="" python experiments/robots/run.py
```

### 3) Outputs

Each run creates a timestamped folder under:
- `experiments/robots/saved_results/pb_<mp-type>_<mb-type>_<MM_DD_HH_MM_SS>/`

Typical outputs include:
- `log`: training log file
- `trained_controller.pt`: `{'controller': state_dict, 'Q', 'args'}` — the full controller; rebuild it with
  `build_controller(argparse.Namespace(**ckpt['args']), plant, z_dim=plant.in_dim).load_state_dict(ckpt['controller'])`
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
- **`robots/robots_sys.py`**: robot plant dynamics (inner PI loop tracking `x̄ + u`), `nominal_dynamics` for the PB reconstruction, and the closed-loop rollout used for training.

### `controllers/`
- **`PB_controller.py`**: `PBController`: step-wise disturbance reconstruction `ŵ_t = x_t − f_nom(x_{t−1}, u_{t−1}; v_{t−1}, x̄_t)` followed by `u_t = M(ŵ_t, z_t)`.
- **`operators.py`**: `OperatorBase`, `FactorizedOperator` (`M_b ⊠ M_p`), `boxtimes_timewise`, `WIntegralAugmenter`, `as_bt`.
- **`ssm_operators.py`**: `MpDeepSSM` and `MpContextualSSM` (wrapping `neural_ssm.ContextualDeepSSM`), and `ContextRescale`.
- **`bounded_mlp.py`**: bounded mixers `BoundedMLPOperator` (`nu × s`) and `DiagonalBoundedMLPOperator` (rPB).
- **`context_lifting.py`**: `LpContextLifter`, which feeds an l_p-decaying lift of `z` to a non-contextual `M_p`.
- **`contractive_ren.py`**: `ContractiveREN`, kept as the legacy `M_p` (`--mp-type ren`).
- **`factory.py`**: `build_operator` / `build_controller` from the CLI flags, plus validation of flag combinations.
- **`__init__.py`**: controller exports.

### `loss_functions/`
- **`lq_loss.py`**: finite-horizon LQ loss base used by experiments.
- **`robots_loss.py`**: robots-specific loss terms (tracking, control effort, collision, obstacles).
- **`__init__.py`**: loss exports.

### `tests/`
- **`test_operators.py`**: every `M_p × M_b` combination and context-port subset builds, rolls out, backpropagates, resets and reloads.
- **`test_reconstruction.py`**: `ŵ` matches the true disturbance exactly, including across the reference jump.

### `utils/`
- **`assistive_functions.py`**: small utilities (tensor conversion, logging wrapper).
- **`plot_functions.py`**: plotting helpers for trajectories and diagnostics.

## Notes for collaborators

- **Python version**: this repo targets Python 3.10+.
- **Reproducibility**: `--random-seed` controls dataset generation and PyTorch seed.
- **Performance**: training backprops through closed-loop rollouts; GPU helps but CPU runs are supported.

## Operator configurations

| Configuration | Flags | How the context `z` enters |
|---|---|---|
| Factorized PB | `--mp-type ssm --mb-type bounded_mlp` | only through `M_b` (add `--mp-context-lift` to also feed `[ŵ, lift(z)]` to `M_p`) |
| MAD | `... --mb-type bounded_mlp --feat-dim 1` | `M_b` (direction) |
| rPB | `--mb-type diagonal` | `M_b = diag(g(ŵ, z))` |
| M_p only | `--mp-type ssm --mb-type none --mp-context-lift` | lifted `z` into `M_p` |
| Fused contextual SSM | `--mp-type contextual_ssm --mb-type none` | the ports of `--ctx-modes` |
| Contextual SSM ⊠ mixer (default) | `--mp-type contextual_ssm --mb-type bounded_mlp` | the `M_p` ports and `M_b` |
| Legacy REN | `--mp-type ren --mb-type bounded_mlp` | `M_b` |
| Disturbance-only baseline | any of the above with `--context none` | `z = 0` |

Contextual SSM ports (`--ctx-modes`, comma-separated; see `MpContextualSSM`):
- `input`: `z`, made l2 by `--ctx-filter` (default `taper`, ramp `--ctx-filter-ramp`), is concatenated to `ŵ`. This acts as a driver and works even when `ŵ ≈ 0`.
- `gate`: per-block (or `--ctx-gate-per-channel`) sigmoid gates on the SSM/FF branches.
- `mixer`: an internal bounded matrix (`--ctx-mixer-bound`) on the core features. This acts as a router and vanishes with `ŵ`.
- `select` (`--ctx-select`): `z` conditions the per-step dynamics. It needs `--ssm-param tv|tvc`.

A certified l2 gain (`--ctx-gamma > 0`) needs a certified core and feed-forward block, e.g. `--ssm-param tv --ssm-ff LGLU2`.

## Changes with respect to the REN × MLP version

- **Disturbance reconstruction:** it now uses the reference of the current transition. The previous controller used `x̄_{t−1}`, which leaked the reference step into `ŵ`.
- **Speed penalty:** it is weighted by `Qs`. It used to be weighted by `Q`; `--legacy-speed-weight` restores that.
- **Checkpoint:** it now contains the whole controller. Previously the MLP weights were not saved.
- **Boolean flags:** they use `--flag/--no-flag`, the rollout is device-agnostic, and the per-agent indices follow `n_agents`.
- **Optimizer:** AdamW with the full `--lr` on the recurrent core of `M_p` and `--lr × --lr-ratio` on the mixer and context heads.
