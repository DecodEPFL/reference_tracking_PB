#!/bin/bash
# Benchmark: relay (Distributed/) vs Danilo's certified SSM network (Distributed_Danilo/, --use-ssm), on the
# leader graph with sticky friction patches: the setting of sweep_patches.sh (irregular graph, 50% flipped starts,
# 2500 epochs, 600 fresh rollouts per epoch in batches of 200), same M_p / M_b sizes.
#
# Arms (one runai job per arm, point and seed):
#   relay   Distributed/: --info-pattern relay (the centralized inputs relayed over the graph, free core gains)
#   raw     Distributed_Danilo/ --use-ssm --feedback-norm none:     neighbors' raw commands u^j fed back
#   norm    Distributed_Danilo/ --use-ssm --feedback-norm constant: neighbors' commands u^j / K_j fed back
#   both Danilo arms: edge weight a = --ren-edge-weight 3e-2 (fixed), --info-pattern neighbors (context of the
#   in-neighbors, one hop: the certified network does not relay), --ren-gamma / --ren-epsilon at their defaults,
#   --neighbor-drag (every neighbor sends its measured drag instead of its distance: per neighbor j the context holds
#   p^j - p^i, s^j, u^j_{t-1}, z^j, the signals of the relay plus u^j_{t-1}), --neighbor-targets-input
#   (the neighbors' targets drive the SSM input port with the own target, as the relayed targets do) and
#   --neighbor-delay 1 (the neighbors' positions, drags and targets of t - 1, as a one-hop relayed message;
#   u^j_{t-1} as before).
# Same SSM in every arm: --mp-type contextual_ssm, --mb-type bounded_mlp, --ssm-param tv, 4 layers, d_model:d_state
# of the point, --feat-dim 16, --mb-hidden 27, --ctx-d-features 7, --ctx-head-hidden 27, same context ports.
#
# Randomness: the seed k is both --random-seed and --init-seed. Every seed has its own data (train/validation/test,
# per-epoch resampled rollouts and patch layouts, all from generators of --random-seed and the epoch), and for a given
# seed every arm draws the same data and the same batch order (generator of --init-seed and the epoch); the seed also
# sets the model initialisation (the arms' models differ, so their weights do too).
#
# Results: <code>/experiments/robots_formation/saved_results/5_robots/matrix_01111-00100-01000-10001-10010/
#          benchmark/vs_danilo_600x200_fs50_e2500_patches/   (relay in Distributed/, raw and norm in Distributed_Danilo/)
# Usage: bash sweep_vs_danilo.sh                       every arm, points 14:27 and 8:16, seeds 1..10 (60 jobs)
#        DRY=1 bash sweep_vs_danilo.sh                 only print the commands
#        ARMS="raw norm" bash sweep_vs_danilo.sh       only these arms
#        SEEDS="1 2" bash sweep_vs_danilo.sh 8:16      these seeds, this point
#        EPOCHS=5 SEEDS=1 bash sweep_vs_danilo.sh 8:16 sanity check (jobs ...-e5, results in vs_danilo_..._e5_patches)
set -euo pipefail
CPUS=${CPUS:-2}
SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10}
ARMS=${ARMS:-relay raw norm}
EPOCHS=${EPOCHS:-2500}
EDGE=${EDGE:-3e-2}
ROLLOUTS=600 BATCH=200 FLIP=0.5
POINTS=("$@")
[[ ${#POINTS[@]} -eq 0 ]] && POINTS=(14:27 8:16)

TAG="${ROLLOUTS}x${BATCH}"
BENCH="vs_danilo_${TAG}_fs50_e${EPOCHS}_patches"
BASE="--n-agents 5 --adjacency 0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0
 --horizon 150 --epochs $EPOCHS --lr 2e-3 --log-epoch 50
 --num-rollouts $ROLLOUTS --batch-size $BATCH --resample-data
 --col-loss hinge --grad-clip-rel 5 --lr-schedule constant --ema-decay 0
 --alpha-x 0.1 --alpha-col 10000 --alpha-speed 1 --alpha-obst-max 3000 --min-dist 0.5
 --formation-shape bowtie --alpha-form 400 --alpha-orient 200 --alpha-orient-max 1000
 --alpha-wall 1000 --large-rot-frac 0.6 --start-flip-frac $FLIP --num-threads 1 --compile --patches
 --no-match-centralized-size --mp-type contextual_ssm --mb-type bounded_mlp --ssm-param tv
 --mb-hidden 27 --ctx-d-features 7 --ctx-head-hidden 27 --benchmark $BENCH"

for arm in $ARMS; do
    case "$arm" in
        relay) CODE=Distributed        ARM="--info-pattern relay" ;;
        raw)   CODE=Distributed_Danilo ARM="--info-pattern neighbors --use-ssm --feedback-norm none --ren-edge-weight $EDGE --neighbor-drag --neighbor-targets-input --neighbor-delay 1" ;;
        norm)  CODE=Distributed_Danilo ARM="--info-pattern neighbors --use-ssm --feedback-norm constant --ren-edge-weight $EDGE --neighbor-drag --neighbor-targets-input --neighbor-delay 1" ;;
        *) echo "ARMS must be among relay raw norm, got $arm" >&2; exit 1 ;;
    esac
    for p in "${POINTS[@]}"; do for seed in $SEEDS; do
        dm=${p%%:*} ds=${p##*:}
        ${DRY:+echo} runai submit --name "vs-${arm}-dm${dm}-ds${ds}-s${seed}-e${EPOCHS}" \
            --image registry.rcp.epfl.ch/test_v1_kirsch/image_ssm:v1.0 \
            --gpu 0 --cpu "$CPUS" -e CPUS="$CPUS" -e CODE="$CODE" \
            --existing-pvc claimname=sci-sti-gft-scratch,path=/scratch \
            --existing-pvc claimname=home,path=/home/kirsch \
            --backoff-limit 0 \
            -- bash /home/kirsch/distributed_cPB/rcp_train.sh $BASE $ARM \
            --ssm-d-model "$dm" --ssm-d-state "$ds" --random-seed "$seed" --init-seed "$seed"
    done; done
done
