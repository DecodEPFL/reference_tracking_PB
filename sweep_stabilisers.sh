#!/bin/bash
# Ablation of the training stabilisers on the 5 robots bowtie, relay on the irregular graph
# 01111-00100-01000-10001-10010 (run bowtie-irregular_246db6d3, which blew up at epoch ~900),
# one runai job per (arm, seed), all in parallel. Arms:
#   none    all stabilisers off (the setting of bowtie-irregular_246db6d3)
#   all     all on (the formation defaults)
#   clip    only the gradient clipping (5 x the running median of the norms)
#   cosine  only the lr schedule (20 epochs warmup, cosine to 0.01 x lr)
#   hinge   only the bounded collision penalty
#   ema     only the EMA of the weights (0.995 per step)
# Results: Distributed/experiments/robots_formation/saved_results/5_robots/matrix_01111-00100-01000-10001-10010/benchmark/stabilisers/
# Compare: python Distributed/experiments/robots_formation/compare_runs.py <that folder>
# Usage: bash sweep_stabilisers.sh                 submit every arm, seed 5
#        DRY=1 bash sweep_stabilisers.sh           only print the commands
#        bash sweep_stabilisers.sh none all        submit only these arms
#        SEEDS="5 6 7" bash sweep_stabilisers.sh   several seeds per arm (a blow-up is a chance event: one seed is weak evidence)
#        CPUS=4 bash sweep_stabilisers.sh          cores per job (default 2)
set -euo pipefail
CPUS=${CPUS:-2}
SEEDS=${SEEDS:-5}

# args of bowtie-irregular_246db6d3, logged every 50 epochs instead of 200 to see the spikes
BASE="--n-agents 5 --info-pattern relay --adjacency 0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0
 --horizon 150 --epochs 2000 --lr 2e-3 --log-epoch 50
 --alpha-x 0.1 --alpha-col 10000 --alpha-speed 1 --alpha-obst-max 3000 --min-dist 0.5
 --formation-shape bowtie --alpha-form 400 --alpha-orient 200 --alpha-orient-max 1000
 --alpha-wall 1000 --large-rot-frac 0.6 --num-threads 1 --compile --benchmark stabilisers"

# every stabiliser off; an arm turns its own back on (the later flag wins)
OFF="--grad-clip 0 --grad-clip-rel 0 --lr-schedule constant --col-loss inverse --ema-decay 0"
declare -A ARMS=(
    [none]="$OFF"
    [all]="--grad-clip-rel 5 --lr-schedule cosine --lr-warmup 20 --lr-final-ratio 0.01 --col-loss hinge --ema-decay 0.995"
    [clip]="$OFF --grad-clip-rel 5"
    [cosine]="$OFF --lr-schedule cosine --lr-warmup 20 --lr-final-ratio 0.01"
    [hinge]="$OFF --col-loss hinge"
    [ema]="$OFF --ema-decay 0.995"
)

SELECTED=("$@")
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=(none all clip cosine hinge ema)
fi

for arm in "${SELECTED[@]}"; do
    [[ -n "${ARMS[$arm]+x}" ]] || { echo "unknown arm: $arm (one of ${!ARMS[*]})"; exit 1; }
    for seed in $SEEDS; do
        # BASE and the arm flags are split on spaces on purpose; the ; of the adjacency matrix stays inside the arg
        ${DRY:+echo} runai submit --name "stab-${arm}-s${seed}" \
            --image registry.rcp.epfl.ch/test_v1_kirsch/image_ssm:v1.0 \
            --gpu 0 --cpu "$CPUS" -e CPUS="$CPUS" \
            --existing-pvc claimname=sci-sti-gft-scratch,path=/scratch \
            --existing-pvc claimname=home,path=/home/kirsch \
            --backoff-limit 0 \
            -- bash /home/kirsch/distributed_cPB/rcp_train.sh $BASE ${ARMS[$arm]} --random-seed "$seed"
    done
done
