#!/bin/bash
# Benchmark: Danilo's certified SSM network (Distributed_Danilo/, --use-ssm) with --feedback-mode core_append: the
# outputs h^j of the neighbors' recurrent cores (before the mixer port, --ctx-d-features channels), summed over the
# in-neighbors with the edge weights a_ij, are appended to the input of every recurrent core (instead of the commands
# u^j added to the velocity channels of w^i, as in the raw / norm arms of sweep_vs_danilo.sh). The agents interconnect
# only through their l2 cores; the mixer port and M_b stay out of the loop (K_j = 1, --feedback-norm is moot).
#
# Same setting as sweep_vs_danilo.sh (leader graph, sticky friction patches, 50% flipped starts, 2500 epochs, 600 fresh
# rollouts per epoch in batches of 200, same M_p / M_b sizes, same context flags), except the edge weight:
#   a = --ren-edge-weight 0.1 (fixed).
#
# Arms (one runai job per arm, point and seed):
#   core   --feedback-mode core_append (default)
#   yapp   --feedback-mode y_append    (the M_p outputs y^j appended instead; optional)
#
# Randomness: the seed k is both --random-seed and --init-seed (same data and batch order per seed as the other
# benchmarks).
#
# Results: <code>/experiments/robots_formation/saved_results/5_robots/matrix_01111-00100-01000-10001-10010/
#          benchmark/core_append_600x200_fs50_e2500_a0.1_patches/
# Usage: bash sweep_core_append.sh                       arm core, points 14:27 and 8:16, seeds 1..10 (20 jobs)
#        DRY=1 bash sweep_core_append.sh                 only print the commands
#        ARMS="core yapp" bash sweep_core_append.sh      both arms
#        SEEDS="1 2" bash sweep_core_append.sh 8:16      these seeds, this point
#        EDGE=3e-2 bash sweep_core_append.sh             another edge weight (own benchmark folder)
#        EPOCHS=5 SEEDS=1 bash sweep_core_append.sh 8:16 sanity check
set -euo pipefail
CPUS=${CPUS:-2}
SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10}
ARMS=${ARMS:-core}
EPOCHS=${EPOCHS:-2500}
EDGE=${EDGE:-0.1}
ROLLOUTS=600 BATCH=200 FLIP=0.5
POINTS=("$@")
[[ ${#POINTS[@]} -eq 0 ]] && POINTS=(14:27 8:16)

TAG="${ROLLOUTS}x${BATCH}"
BENCH="core_append_${TAG}_fs50_e${EPOCHS}_a${EDGE}_patches"
BASE="--n-agents 5 --adjacency 0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0
 --horizon 150 --epochs $EPOCHS --lr 2e-3 --log-epoch 50
 --num-rollouts $ROLLOUTS --batch-size $BATCH --resample-data
 --col-loss hinge --grad-clip-rel 5 --lr-schedule constant --ema-decay 0
 --alpha-x 0.1 --alpha-col 10000 --alpha-speed 1 --alpha-obst-max 3000 --min-dist 0.5
 --formation-shape bowtie --alpha-form 400 --alpha-orient 200 --alpha-orient-max 1000
 --alpha-wall 1000 --large-rot-frac 0.6 --start-flip-frac $FLIP --num-threads 1 --compile --patches
 --no-match-centralized-size --mp-type contextual_ssm --mb-type bounded_mlp --ssm-param tv
 --mb-hidden 27 --ctx-d-features 7 --ctx-head-hidden 27 --benchmark $BENCH
 --info-pattern neighbors --use-ssm --feedback-norm constant --ren-edge-weight $EDGE
 --neighbor-drag --neighbor-targets-input --neighbor-delay 1"
EDGE_TAG=${EDGE//./p}                       # job names: no dots

for arm in $ARMS; do
    case "$arm" in
        core) ARM="--feedback-mode core_append" ;;
        yapp) ARM="--feedback-mode y_append" ;;
        *) echo "ARMS must be among core yapp, got $arm" >&2; exit 1 ;;
    esac
    for p in "${POINTS[@]}"; do for seed in $SEEDS; do
        dm=${p%%:*} ds=${p##*:}
        ${DRY:+echo} runai submit --name "ca-${arm}-a${EDGE_TAG}-dm${dm}-ds${ds}-s${seed}-e${EPOCHS}" \
            --image registry.rcp.epfl.ch/test_v1_kirsch/image_ssm:v1.0 \
            --gpu 0 --cpu "$CPUS" -e CPUS="$CPUS" -e CODE=Distributed_Danilo \
            --existing-pvc claimname=sci-sti-gft-scratch,path=/scratch \
            --existing-pvc claimname=home,path=/home/kirsch \
            --backoff-limit 0 \
            -- bash /home/kirsch/distributed_cPB/rcp_train.sh $BASE $ARM \
            --ssm-d-model "$dm" --ssm-d-state "$ds" --random-seed "$seed" --init-seed "$seed"
    done; done
done
