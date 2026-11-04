#!/bin/bash
# Sweep of the SSM size (d_model x d_state) of the 8 robots double bowtie, relay, one runai job per point and
# init seed, all in parallel. Same setup as the 5 robots ssm_size_600x200 sweep (sweep_ssm.sh, context runs):
# fresh train rollouts every epoch, 600 of them in batches of 200, the 600 fixed rollouts of --random-seed 5 as
# validation, collision free starts, bounded collision penalty and relative gradient clipping, no lr schedule
# nor EMA, the other widths fixed at those of the 5 robots sweep.
# Changed for the double bowtie (2 bowties of width 3, height 1.2, sharing the middle column of robots 4 and 5):
#   targets centred in x [1, 3] (the flat 6 wide target stays off the walls), y [7, 7.1] (the target turned
#   along y reaches y = 10 and clears the obstacles), same horizon 150 and arrival from t = 90 (final cost),
#   formation weights rescaled to the 5 robots balance: 28 constrained pairs instead of 10 (alpha_form 400 -> 150),
#   16 orientation triangles, the 8 of each bowtie (alpha_orient 200 -> 100, alpha_orient_max 1000 -> 500).
# Agent inputs are padded (default): on the bowtie graph the robots have 2 to 4 neighbours; on the complete
# graph they all have 7 and the padding changes nothing.
# Graphs (ADJ):
#   complete  every robot receives from every other one (default)
#   bowtie    the 11 drawn links of the double bowtie without the outer rectangle, both ways: the vertical sides
#             1-2, 4-5, 7-8 and each knot (3, 6) to its 4 corners (robots numbered from 1)
# Results: Distributed/experiments/robots_formation/saved_results/8_robots/<complete | matrix_01100000-...>/benchmark/dbowtie_ssm_size_600x200/
# Compare: python Distributed/experiments/robots_formation/compare_runs.py <that folder>
# Usage: bash sweep_dbowtie.sh                   complete graph, every point, for every seed
#        ADJ=bowtie bash sweep_dbowtie.sh        the bowtie graph
#        DRY=1 bash sweep_dbowtie.sh             only print the commands
#        bash sweep_dbowtie.sh 4:8 14:27         submit only these d_model:d_state points
#        SEEDS="1 2" bash sweep_dbowtie.sh       init seeds (default 1 2 3)
#        EPOCHS=500 bash sweep_dbowtie.sh        epochs (default 2000, as the 5 robots sweep)
#        CPUS=4 bash sweep_dbowtie.sh            cores per job (default 2)
set -euo pipefail
CPUS=${CPUS:-2}
SEEDS=${SEEDS:-1 2 3}
EPOCHS=${EPOCHS:-2000}
ADJ=${ADJ:-complete}
ROLLOUTS=600 BATCH=200 DATA_SEED=5

case "$ADJ" in
    complete) ADJACENCY="complete" GRAPH="cmp" ;;
    bowtie) ADJACENCY="0,1,1,0,0,0,0,0;1,0,1,0,0,0,0,0;1,1,0,1,1,0,0,0;0,0,1,0,1,1,0,0;0,0,1,1,0,1,0,0;0,0,0,1,1,0,1,1;0,0,0,0,0,1,0,1;0,0,0,0,0,1,1,0"
            GRAPH="btg" ;;
    *) echo "ADJ must be complete or bowtie, got $ADJ" >&2; exit 1 ;;
esac

TAG="${ROLLOUTS}x${BATCH}"
BASE="--n-agents 8 --info-pattern relay --adjacency $ADJACENCY --horizon 150 --epochs $EPOCHS --lr 2e-3 --log-epoch 50
 --num-rollouts $ROLLOUTS --batch-size $BATCH --random-seed $DATA_SEED --resample-data
 --col-loss hinge --grad-clip-rel 5 --lr-schedule constant --ema-decay 0
 --alpha-x 0.1 --alpha-col 10000 --alpha-speed 1 --alpha-obst-max 3000 --min-dist 0.5
 --formation-shape double_bowtie --formation-base 6 --formation-height 1.2 --target-x 1 3 --target-y 7 7.1
 --alpha-form 150 --alpha-orient 100 --alpha-orient-max 500
 --alpha-wall 1000 --large-rot-frac 0.6 --num-threads 1 --compile
 --no-match-centralized-size --mb-hidden 27 --ctx-d-features 7 --ctx-head-hidden 27
 --dim-internal 3 --dim-nl 3 --benchmark dbowtie_ssm_size_$TAG"

POINTS=("$@")
if [[ ${#POINTS[@]} -eq 0 ]]; then
    for dm in 4 8 14; do for ds in 8 16 27; do POINTS+=("$dm:$ds"); done; done
fi

for p in "${POINTS[@]}"; do for seed in $SEEDS; do
    dm=${p%%:*} ds=${p##*:}
    ${DRY:+echo} runai submit --name "dbow-${GRAPH}-dm${dm}-ds${ds}-s${seed}" \
        --image registry.rcp.epfl.ch/test_v1_kirsch/image_ssm:v1.0 \
        --gpu 0 --cpu "$CPUS" -e CPUS="$CPUS" \
        --existing-pvc claimname=sci-sti-gft-scratch,path=/scratch \
        --existing-pvc claimname=home,path=/home/kirsch \
        --backoff-limit 0 \
        -- bash /home/kirsch/distributed_cPB/rcp_train.sh $BASE \
        --ssm-d-model "$dm" --ssm-d-state "$ds" --init-seed "$seed"
done; done
