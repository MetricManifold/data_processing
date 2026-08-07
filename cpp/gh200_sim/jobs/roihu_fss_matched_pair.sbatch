#!/bin/bash
# ---------------------------------------------------------------------------
# FSS matched pair, following manuscript Sec. "Aging" EXACTLY.
#
#   "the control and soft runs use the same random PLACEMENT and so begin from
#    identical cell positions.  The control run is all-normal throughout, while
#    in the soft run cell 0 is soft FROM THE OUTSET, equilibrating and aging at
#    gamma_c = 0.35.  BOTH ARE EQUILIBRATED SEPARATELY for 8 tau at v_A = 0 and
#    then aged at v_A = 0.01 for 208 tau.  The two runs also share the polarity
#    seed, so once aging begins their cells reorient at the same times and
#    angles."
#
# So per seed there are FOUR runs, not three:
#     eq_ctrl   all-normal,        8 tau,  v_A = 0
#     eq_soft   cell 0 gamma=0.35, 8 tau,  v_A = 0     <-- soft during eq too
#     prod_ctrl resume eq_ctrl,  208 tau,  v_A = 0.01
#     prod_soft resume eq_soft,  208 tau,  v_A = 0.01
#
# Equilibrating all-normal and flipping gamma at production start is NOT the
# same experiment: the soft cell would enter aging carrying its stiff-cell shape
# and relax during the early window, contaminating D_eff with a transient.
#
# SHARED between branches: --seed (placement) and --polarity-seed (reorientation
# sequence). DIFFERENT: gamma of cell 0, and therefore the equilibration.
#
# Checkpoints every 2e7 steps so a walltime kill is resumable -- production is
# ~15 h per branch and there is no automatic requeue here.
# ---------------------------------------------------------------------------
#SBATCH --job-name=fss_pair
#SBATCH --account=project_2019216
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:gh200:2
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out

set -o pipefail

BIN=${BIN:-$HOME/gh200_sim/build/cell_gh200}
OUT=${OUT:-/scratch/project_2019216/$USER/fss_pair_${NCELL:-800}_$SLURM_JOB_ID}

NCELL=${NCELL:-800}
RADIUS=49
RHO=0.90
TAU=10000
VA=0.01
EQ_T=${EQ_T:-80000}          # 8 tau,   v_A = 0
PROD_T=${PROD_T:-2080000}    # 208 tau, v_A = 0.01
SEED=${SEED:-1}              # placement, shared by both branches
POLSEED=${POLSEED:-90001}    # reorientation sequence, shared by both branches
SAMPLES=${SAMPLES:-20000}
CKPT_EVERY=${CKPT_EVERY:-20000000}

# one soft cell of N -> cancer-fraction rounds to exactly 1
CF=$(awk -v n="$NCELL" 'BEGIN{printf "%.10f", 1.0/n}')

mkdir -p "$OUT"
echo "output   : $OUT"
echo "protocol : N=$NCELL rho=$RHO  eq=${EQ_T} (8tau, v_A=0)  prod=${PROD_T} (208tau)"
echo "shared   : seed=$SEED (placement)  polarity-seed=$POLSEED (reorientation)"
echo "soft     : cancer-fraction=$CF -> exactly 1 cell at gamma=0.35, soft from the outset"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo

# --- stage 1: two SEPARATE equilibrations, in parallel on the two GPUs ------
for br in ctrl soft; do
  cf=0.0; [ "$br" = soft ] && cf=$CF
  g=0;    [ "$br" = soft ] && g=1
  d="$OUT/eq_$br"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$g "$BIN" \
      --N $NCELL --radius $RADIUS --rho $RHO \
      --t-end "$EQ_T" --v-A 0 --tau $TAU \
      --cancer-fraction "$cf" --gamma-cancer 0.35 \
      --seed "$SEED" --polarity-seed "$POLSEED" \
      --print-interval 2000000 \
      --checkpoint-interval "$CKPT_EVERY" --checkpoint-dir "$d" \
      > "$d/run.log" 2>&1 &
done
wait
echo "=== equilibration done ==="
for br in ctrl soft; do
  c="$OUT/eq_$br/checkpoint.bin"
  [ -s "$c" ] || { echo "[fatal] missing $c"; exit 3; }
  printf "  %-5s %s\n" "$br" "$(ls -la "$c" | awk '{print $5" B"}')"
  grep -h "alarms" "$OUT/eq_$br/run.log" | tail -1
done

# --- stage 2: aging, each branch resuming its OWN equilibration ------------
echo
echo "=== production (208 tau, v_A=$VA) ==="
for br in ctrl soft; do
  g=0; [ "$br" = soft ] && g=1
  d="$OUT/prod_$br"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$g "$BIN" \
      -c "$OUT/eq_$br/checkpoint.bin" \
      --t-end "$PROD_T" --v-A "$VA" \
      --trajectory-samples "$SAMPLES" --print-interval 20000000 \
      --checkpoint-interval "$CKPT_EVERY" --checkpoint-dir "$d" \
      --out "$d/trajectory.txt" > "$d/run.log" 2>&1 &
done
wait

echo
echo "=== completion ==="
for br in ctrl soft; do
  d="$OUT/prod_$br"
  n=$(( ($(wc -l < "$d/trajectory.txt" 2>/dev/null || echo 3) - 3) / NCELL ))
  printf "  %-5s frames=%-7s %s\n" "$br" "$n" "$(grep -h 'alarms' "$d/run.log" | tail -1)"
done
echo
echo "Analyse with:  cell_analyze check $OUT/prod_soft"
echo "Then diff the two tagged-cell D_eff at lag 8 tau."
