#!/bin/bash
# Sweep of the SSM size (d_model x d_state) of the 5 robots bowtie, relay on a cycle
# (run pb_contextual_ssm_bounded_mlp_10_28_16_04_13), one runai job per point and init seed, all in parallel.
# The other widths are fixed at the values --match-centralized-size gave that run.
# Data: fresh train rollouts every epoch (--resample-data), ROLLOUTS of them in batches of BATCH; the
# ROLLOUTS fixed rollouts of --random-seed DATA_SEED are the validation set (never trained on), the
# 500 test rollouts the same for every job. Every run starts collision free (--init-min-dist = --min-dist).
# The seed (--init-seed) only changes the initialisation and the batch shuffling.
# Stabilisers: the two best arms of the stabiliser ablation (benchmark/stabilisers on the irregular graph),
# the bounded collision penalty and the relative gradient clipping; the cosine schedule and the EMA are off
# (worse alone, and the all-on arm got a seed stuck mirrored). Logged every 50 epochs to see the spikes.
# Results: Distributed/experiments/robots_formation/saved_results/5_robots/cycle/benchmark/ssm_size_<ROLLOUTS>x<BATCH>[_fixed]/
# Compare: python Distributed/experiments/robots_formation/compare_runs.py <that folder>
# Usage: bash sweep_ssm.sh                 submit every point, for every seed
#        DRY=1 bash sweep_ssm.sh           only print the commands
#        bash sweep_ssm.sh 4:8 14:27       submit only these d_model:d_state points
#        SEEDS="1 2" bash sweep_ssm.sh     init seeds (default 1 2 3)
#        ROLLOUTS=900 BATCH=300 bash sweep_ssm.sh
#                                          train rollouts per epoch and batch size (default 600 and 200: the 3
#                                          steps per epoch of the first sweep, each on 2x the rollouts; the
#                                          time per epoch grows about with ROLLOUTS, check it on the first log)
#        RESAMPLE=0 bash sweep_ssm.sh      one fixed train set instead (also the validation set)
#        DATA_SEED=5 bash sweep_ssm.sh     seed of the validation/test data (default 5)
#        CPUS=4 bash sweep_ssm.sh          cores per job (default 2: training is single threaded,
#                                          the extra cores only speed up torch.compile at the start)
#        NOCTX=1 bash sweep_ssm.sh         no context: a plain SSM fed only its reconstructed w^i and the
#                                          relayed w^j of the others, u = M_p directly (no M_b, no mixers or
#                                          context ports); same grid, results in ssm_size_<TAG>[_fixed]_noctx
#        ADJ=irregular bash sweep_ssm.sh   same sweep on the irregular graph of the stabiliser ablation
#                                          (--adjacency 0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0) instead of
#                                          the cycle: jobs ssm-irr-..., results in 5_robots/matrix_01111-.../benchmark/
#                                          (agent inputs padded, as in the stabiliser ablation)
#        MATCH=4:8 bash sweep_ssm.sh       no context at the size of the context run d_model:d_state (implies
#                                          NOCTX=1): for every d_model of MATCH_DMS (default 4 8 14 20 31), the
#                                          d_state whose plain SSM has the closest total number of parameters;
#                                          results in ssm_size_<TAG>[_fixed]_noctx_match<dm>x<ds>
#        FLIP=0.5 bash sweep_ssm.sh        half of the train/validation/test starts turned by pi (--start-flip-frac,
#                                          target turned too): jobs ...-fs50, results in ssm_size_<TAG>[...]_fs50
#        EPOCHS=2500 bash sweep_ssm.sh     epochs (default 2000); other than 2000: jobs ...-e2500, results in ..._e2500
#        PATCHES=1 bash sweep_ssm.sh       sticky friction patches (--patches, default --patch-* layout:
#                                          centres x in [-1, 5], y in [-1, 6]): every rollout its own drag terrain, log
#                                          drag scale in the context; jobs ...-pt,
#                                          results in ..._patches
#        MATCH=all bash sweep_ssm.sh       a no context twin of every context point: same d_state, the d_model
#                                          with the closest number of parameters (job ssm-noctx-m<dm>x<ds>-...
#                                          for the twin of dm:ds); results in ssm_size_<TAG>[_fixed]_noctx_matched
set -euo pipefail
CPUS=${CPUS:-2}
SEEDS=${SEEDS:-1 2 3}
DATA_SEED=${DATA_SEED:-5}
ROLLOUTS=${ROLLOUTS:-600}
BATCH=${BATCH:-200}
RESAMPLE=${RESAMPLE:-1}
NOCTX=${NOCTX:-0}
ADJ=${ADJ:-cycle}
MATCH=${MATCH:-}
MATCH_DMS=${MATCH_DMS:-4 8 14 20 31}
FLIP=${FLIP:-0}
EPOCHS=${EPOCHS:-2000}
PATCHES=${PATCHES:-0}
[[ -n "$MATCH" ]] && NOCTX=1
if (( ROLLOUTS % BATCH )); then
    echo "BATCH=$BATCH does not divide ROLLOUTS=$ROLLOUTS: the last batch of every epoch would be smaller" >&2
    exit 1
fi

case "$ADJ" in
    # the cycle keeps the unpadded inputs of the first sweeps (all agents have the same inputs, padding changes
    # nothing); the irregular graph needs the default padding, its agents have different input sizes
    cycle) ADJACENCY="cycle" GRAPH="" PAD="--no-pad-agent-inputs" ;;
    irregular) ADJACENCY="0,1,1,1,1;0,0,1,0,0;0,1,0,0,0;1,0,0,0,1;1,0,0,1,0" GRAPH="-irr" PAD="" ;;
    *) echo "ADJ must be cycle or irregular, got $ADJ" >&2; exit 1 ;;
esac

TAG="${ROLLOUTS}x${BATCH}"
if [[ "$RESAMPLE" == 1 ]]; then BENCH="ssm_size_$TAG" DATA="--resample-data"; else BENCH="ssm_size_${TAG}_fixed" DATA=""; fi
if [[ "$NOCTX" == 1 ]]; then
    BENCH="${BENCH}_noctx" NAME="ssm-noctx"
    if [[ "$MATCH" == all ]]; then BENCH="${BENCH}_matched"
    elif [[ -n "$MATCH" ]]; then BENCH="${BENCH}_match${MATCH/:/x}" NAME="${NAME}-m${MATCH/:/x}"; fi
    OPERATOR="--mp-type ssm --mb-type none --context none"
else
    NAME="ssm"
    OPERATOR="--mb-hidden 27 --ctx-d-features 7 --ctx-head-hidden 27"
fi

SUFFIX=""
if [[ "$FLIP" != 0 ]]; then
    pct=$(awk -v f="$FLIP" 'BEGIN { if (f < 0 || f > 1) exit 1; printf "%d", f * 100 + 0.5 }') \
        || { echo "FLIP must be in [0, 1], got $FLIP" >&2; exit 1; }
    SUFFIX="${SUFFIX}-fs${pct}" BENCH="${BENCH}_fs${pct}"
fi
if [[ "$EPOCHS" != 2000 ]]; then SUFFIX="${SUFFIX}-e${EPOCHS}" BENCH="${BENCH}_e${EPOCHS}"; fi
TERRAIN=""
if [[ "$PATCHES" == 1 ]]; then SUFFIX="${SUFFIX}-pt" BENCH="${BENCH}_patches" TERRAIN="--patches"; fi

BASE="--n-agents 5 --info-pattern relay --adjacency $ADJACENCY --horizon 150 --epochs $EPOCHS --lr 2e-3 --log-epoch 50
 --num-rollouts $ROLLOUTS --batch-size $BATCH --random-seed $DATA_SEED $DATA
 --col-loss hinge --grad-clip-rel 5 --lr-schedule constant --ema-decay 0
 --alpha-x 0.1 --alpha-col 10000 --alpha-speed 1 --alpha-obst-max 3000 --min-dist 0.5
 --formation-shape bowtie --alpha-form 400 --alpha-orient 200 --alpha-orient-max 1000
 --alpha-wall 1000 --large-rot-frac 0.6 --start-flip-frac $FLIP --num-threads 1 --compile $PAD $TERRAIN
 --no-match-centralized-size $OPERATOR
 --dim-internal 3 --dim-nl 3 --benchmark $BENCH"

POINTS=("$@") TWINS=()
# number of parameters of the context runs (5 agents, from their logs) of the 600x200 sweep
declare -A CTX_PARAMS=([4:8]=67080 [4:16]=79720 [4:27]=97100 [8:8]=73180 [8:16]=89020 [8:27]=110800
                       [14:8]=84730 [14:16]=105370 [14:27]=133750)
# plain SSM, 5 agents (fits the 9 runs of ssm_size_600x200_noctx)
noctx_params() { echo $(( 5 * (8 * $1 * $1 + 26 * $1 + 12 + 20 * $2 * ($1 + 1)) )); }
if [[ "$MATCH" == all ]]; then
    POINTS=()
    for dm in 4 8 14; do for ds in 8 16 27; do
        target=${CTX_PARAMS[$dm:$ds]} m=1
        while (( $(noctx_params $(( m + 1 )) $ds) <= target )); do m=$(( m + 1 )); done
        (( target - $(noctx_params $m $ds) > $(noctx_params $(( m + 1 )) $ds) - target )) && m=$(( m + 1 ))
        echo "twin of $dm:$ds ($target): d_model $m, d_state $ds, $(noctx_params $m $ds) parameters" >&2
        POINTS+=("$m:$ds") TWINS+=("-m${dm}x${ds}")
    done; done
elif [[ -n "$MATCH" ]]; then
    target=${CTX_PARAMS[$MATCH]:?"MATCH=$MATCH: no context run of that size, pick one of ${!CTX_PARAMS[*]}"}
    POINTS=()
    for dm in $MATCH_DMS; do
        num=$(( target / 5 - 8 * dm * dm - 26 * dm - 12 )) den=$(( 20 * (dm + 1) ))
        ds=$(( (num + den / 2) / den ))
        (( ds >= 1 )) || { echo "MATCH=$MATCH: d_model $dm alone is already larger than $target parameters" >&2; exit 1; }
        echo "d_model $dm, d_state $ds: $(noctx_params $dm $ds) parameters (target $target)" >&2
        POINTS+=("$dm:$ds")
    done
elif [[ ${#POINTS[@]} -eq 0 ]]; then
    for dm in 4 8 14; do for ds in 8 16 27; do POINTS+=("$dm:$ds"); done; done
fi

for i in "${!POINTS[@]}"; do for seed in $SEEDS; do
    p=${POINTS[$i]} twin=${TWINS[$i]:-}
    dm=${p%%:*} ds=${p##*:}
    ${DRY:+echo} runai submit --name "${NAME}${GRAPH}${twin}-dm${dm}-ds${ds}-s${seed}-${TAG}${DATA:+-rs}${SUFFIX}" \
        --image registry.rcp.epfl.ch/test_v1_kirsch/image_ssm:v1.0 \
        --gpu 0 --cpu "$CPUS" -e CPUS="$CPUS" \
        --existing-pvc claimname=sci-sti-gft-scratch,path=/scratch \
        --existing-pvc claimname=home,path=/home/kirsch \
        --backoff-limit 0 \
        -- bash /home/kirsch/distributed_cPB/rcp_train.sh $BASE \
        --ssm-d-model "$dm" --ssm-d-state "$ds" --init-seed "$seed"
done; done
