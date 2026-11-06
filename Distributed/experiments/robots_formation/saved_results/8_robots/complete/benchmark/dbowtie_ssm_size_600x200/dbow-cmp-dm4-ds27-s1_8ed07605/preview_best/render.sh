#!/bin/bash
# One-off: render GIFs of the best-so-far controller of a still-running run (best_state of training_state.pt).
set -euo pipefail
cd /home/kirsch/distributed_cPB
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages OMP_NUM_THREADS=2
RUN=Distributed/experiments/robots_formation/saved_results/8_robots/complete/benchmark/dbowtie_ssm_size_600x200/dbow-cmp-dm4-ds27-s1_8ed07605
python - "$RUN" <<'PY'
import sys, torch
run = sys.argv[1]
s = torch.load(run + '/training_state.pt', map_location='cpu', weights_only=False)
print('last epoch', s['epoch'], '| best valid loss', s['best_valid_loss'])
args = dict(s['args']); args['compile'] = False
torch.save({'controller': s['best_state'], 'args': args}, run + '/preview_best/trained_controller.pt')
PY
python Distributed/experiments/robots_formation/render_gifs.py $RUN/preview_best --ref --overwrite
