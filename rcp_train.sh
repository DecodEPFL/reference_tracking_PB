#!/bin/bash
# Launch a robots_formation run in the RCP image (runai), non-interactive.
# Usage: bash rcp_train.sh <run.py args>     e.g. ... --epochs 200 --compile
# Set CPUS to the --cpu of the runai job (default 4).
set -euo pipefail
cd /home/kirsch/distributed_cPB

# pinned neural_ssm (+ jax) installed with pip --user; the image disables the user site, so add it by hand
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages
# 1 compute thread (--num-threads sets torch's own), compile workers = the job's cores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=${CPUS:-4}

# preflight: fail now rather than train on the slow path
command -v g++ >/dev/null || { echo "preflight: g++ not found (needed by --compile)"; exit 1; }
python - <<'EOF'
import torch, neural_ssm, neural_ssm.ssm.selective_cells as sc
print("preflight: torch", torch.__version__, "| neural_ssm", neural_ssm.__file__)
assert hasattr(sc, "_resolve_runtime_state"), "wrong neural_ssm (not the pinned commit): agents would not be batched"
torch.compile(lambda x: torch.sin(x) + 1)(torch.randn(3))
print("preflight: torch.compile OK")
EOF

# under runai, name the run folder after the job: a pod restarted by runai (preemption) has the same JOB_UUID
# and continues from its last training state, a resubmitted job gets a new one and starts over.
# Put first so that a --run-id given on the command line wins.
if [[ -n "${JOB_UUID:-}" ]]; then
    set -- --run-id "${JOB_NAME}_${JOB_UUID:0:8}" "$@"
fi

# a fallback to the per-agent loop is only a warning in factory.py: make it an error here
exec python -W "error:--batch-agents disabled" Distributed/experiments/robots_formation/run.py "$@"
