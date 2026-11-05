# Distributed REN-MAD benchmark (copy of `Distributed/`)

This folder is a copy of `Distributed/` in which only the controller changes: the contextual SSM operator is
replaced by the distributed **REN-MAD** controller of Saccani, Massai and Ferrari-Trecate, *Distributed
Performance Boosting through Neural Control for Plug-and-Play Networks* (ported from
`dPB-microgrid/ren_mad`). Plant, datasets, losses, training loop, plots and command lines are the same, so
runs of both folders compare directly. Imports use the `Distributed_Danilo.` package.

```
v^i_t = [ w^i_t ; sum_j a_ij y^j_t ]        REN i hears the outputs of its --adjacency neighbors (no relaying)
y^i_{t+1} = R^i(v^i)_t                       gain-certified REN, one-step output buffer
u^i_t = |y^i_t| * d^i(w^i_t, c^i_t)          MAD readout; d^i = bounded M_b of Distributed/ with rank 1
```

The local gains are computed from the interconnection so that the network of RENs has l2 gain at most
`--ren-gamma` from w to y for any parameter value (certificate logged at the start and end of every run).
The context `c^i` is the one of `Distributed/`: own target (and measured drag with `--patches`), plus the
neighbor signals with `--info-pattern neighbors`. Supported: `--info-pattern local | neighbors`, independent
weights. The SSM flags are still parsed and ignored. Code: `controllers/ren_mad.py`, `controllers/factory.py`;
tests: `tests/test_ren_mad.py`. REN-MAD flags: `--ren-gamma`, `--ren-epsilon`, `--ren-state-dim`,
`--ren-width`, `--ren-init-std`, `--ren-edge-weight`, `--ren-learn-edges`, `--ren-edge-bounds`.

### `--use-ssm`: the same certified network with the SSM operators of `Distributed/`

```
v^i_t = w^i_t + E_v sum_j a_ij u^j_{t-1} / K_j    neighbors' normalized commands added to the velocity channels of w^i
y^i_t = M_p^i(v^i_t, c^i_t),   u^i_t = M_b^i(w^i_t, c^i_t) ⊠ y^i_t     M_p, M_b, context: as in Distributed/
```

The only change with respect to the SSM controller of `Distributed/` is this feedback: M_p, M_b and their
sizes are the same (no input channel is added), and the mixer / gate heads and M_b read the pure w^i. Because
the outputs are fed back, stability needs the network certificate (`--ren-gamma`, `--ren-epsilon`,
`--ren-edge-*`): it sets local gains gamma_i for the maps v^i -> u^i / K_i. K_j = mixer bound x |M_b|
(|M_b| = mb_bound sqrt(nu s)) is the readout bound of agent j, so with `--feedback-norm constant` (default)
every core runs with the prescribed gain gamma_i itself; with `--feedback-norm none` (raw u^j) it runs with
gamma_i / K_i. The certificate bounds the local gains roughly as gamma_i^2 x (incoming weight) <= 1, quadratic
in the gain and linear in the coupling, so it allows far more coupling with the normalized signals (about
177x at the same core gain for the formation defaults). `--ren-edge-weight` is in the units of the fed-back
signal; the guarantee is `||u|| <= max_j K_j gamma_R ||e||`.
`--ssm-param` must be certified (e.g. `tv`, `tvc`; not `lru`). With the `input` context port, the filtered
targets count as exogenous inputs (e = [w; filtered targets]).
The SSM controller of `Distributed/` has no such loop (the neighbors only reach the context), so its operators
only need to be l2 stable, with any finite gain: compare it as it is (free gain) against `--use-ssm` here,
with the same `--ssm-*`, `--ctx-*`, `--mb-*`, `--feat-dim` and `--match-centralized-size`. The capped cores
start from the free initialization (`--ssm-init free`). As the REN realization, the certified encoder /
decoder of every core (spectral caps, decoder attenuation to its gain) are computed once per rollout, and
`--batch-agents` / `--compile` run the agents through the `AgentBatch` of `Distributed/` (same per-step cost:
about 2.0 s vs 1.9 s per step batched, 0.7 s for both compiled, 5 robots, horizon 150, batch 50). Code: `controllers/ssm_network.py`; tests:
`tests/test_ssm_network.py`.

```bash
python Distributed_Danilo/experiments/robots_formation/run.py <the same flags as Distributed/>             # REN-MAD
python Distributed_Danilo/experiments/robots_formation/run.py --use-ssm <the same flags as Distributed/>   # SSM network
```

Runs are saved under `Distributed_Danilo/experiments/<experiment>/saved_results/` as `renmad_<time>` or
`ssmnet_<mp-type>_<mb-type>_<time>`.

The rest of this README is the one of `Distributed/` (paths relative to this folder); its SSM-specific
sections do not apply here.

---

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
