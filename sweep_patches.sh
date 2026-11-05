#!/bin/bash
# The fs50 sweep of the leader graph (5_robots/matrix_01111-00100-01000-10001-10010/benchmark/
# ssm_size_600x200_fs50_e2500) again on sticky friction patches (--patches, default --patch-* layout):
# same grid, seeds, data and losses, every rollout on its own drag terrain.
# Results: .../matrix_01111-00100-01000-10001-10010/benchmark/ssm_size_600x200_fs50_e2500_patches/
# Usage: bash sweep_patches.sh             submit every point, for every seed
#        DRY=1 bash sweep_patches.sh       only print the commands
#        bash sweep_patches.sh 4:8 14:27   only these d_model:d_state points (other options: see sweep_ssm.sh)
#        EPOCHS=5 SEEDS=1 bash sweep_patches.sh 4:8
#                                          sanity check: one short job (...-e5-pt, results in ..._fs50_e5_patches)
set -euo pipefail
cd "$(dirname "$0")"
ADJ=irregular FLIP=0.5 EPOCHS=${EPOCHS:-2500} PATCHES=1 exec bash sweep_ssm.sh "$@"
