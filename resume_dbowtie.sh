#!/bin/bash
# Continue the 3 runs of the 8 robots double bowtie patches sweep (bowtie graph, dbowtie_ssm_size_600x200_patches)
# whose validation loss was still going down at epoch 2000: one runai job each, up to EPOCHS in total.
# --resume reloads every arg of the run (weights, optimizer, epoch, best model and RNG from its training_state.pt)
# and continues from its last weights (epoch 2000) in a new folder; --epochs is the total counted from the start.
# Results: .../8_robots/matrix_01100000-.../benchmark/dbowtie_ssm_size_600x200_patches_e<EPOCHS>/
# Usage: bash resume_dbowtie.sh          submit the 3 jobs
#        DRY=1 bash resume_dbowtie.sh    only print the commands
#        EPOCHS=3500 bash resume_dbowtie.sh
set -euo pipefail
cd "$(dirname "$0")"
CPUS=${CPUS:-2}
EPOCHS=${EPOCHS:-3000}
SWEEP=Distributed/experiments/robots_formation/saved_results/8_robots/matrix_01100000-10100000-11011000-00101100-00110100-00011011-00000101-00000110/benchmark/dbowtie_ssm_size_600x200_patches
BENCH="dbowtie_ssm_size_600x200_patches_e${EPOCHS}"
RUNS=(
    dbow-btg-dm14-ds27-s2-pt_4e0ce493
    dbow-btg-dm8-ds8-s1-pt_aeab962e
    dbow-btg-dm4-ds8-s3-pt_52d90fe9
)

for run in "${RUNS[@]}"; do
    [[ -f "$SWEEP/$run/training_state.pt" ]] || { echo "no training state in $SWEEP/$run" >&2; exit 1; }
    ${DRY:+echo} runai submit --name "${run%%_*}-e${EPOCHS}" \
        --image registry.rcp.epfl.ch/test_v1_kirsch/image_ssm:v1.0 \
        --gpu 0 --cpu "$CPUS" -e CPUS="$CPUS" \
        --existing-pvc claimname=sci-sti-gft-scratch,path=/scratch \
        --existing-pvc claimname=home,path=/home/kirsch \
        --backoff-limit 0 \
        -- bash /home/kirsch/distributed_cPB/rcp_train.sh \
        --resume "$SWEEP/$run" --epochs "$EPOCHS" --benchmark "$BENCH" --num-threads 1 --compile
done
