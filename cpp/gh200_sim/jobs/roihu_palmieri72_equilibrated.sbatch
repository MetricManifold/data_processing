#!/bin/bash
# ---------------------------------------------------------------------------
# Palmieri 72-cell validation, WITH equilibration -- supersedes
# roihu_palmieri72_200tau.sbatch, which ran production straight from t = 0.
#
# WHY THE OLD SCRIPT WAS WRONG, AND WHICH WAY IT WAS WRONG
#
# Job 431339 ran 0 -> 200 tau with v_A = 0.01 on from the first step. Cells
# therefore spent the opening of the trajectory relaxing out of their random
# placement -- rounding up, unjamming -- and that relaxation is displacement
# that lands in the MSD. Call it D_transient. It is common to both branches,
# so what gets measured is
#
#     ratio_obs = (D_soft + D_transient) / (D_ctrl + D_transient)
#
# which for D_transient > 0 is strictly CLOSER TO 1 than the true
# D_soft/D_ctrl. The bias is toward the null. Job 431339 returned a paired
# ratio of 1.112 +/- 0.077 against a Palmieri value near 1.5, and the missing
# equilibration is a mechanism that pushes in exactly that direction, so that
# comparison could not distinguish "the effect is smaller than Palmieri's"
# from "the protocol diluted it".
#
# Pairing does NOT rescue this. Sharing the placement seed and the polarity
# seed cancels seed-specific noise, which is why the error bar was tight, but
# a COMMON ADDITIVE offset survives pairing untouched -- it is in numerator
# and denominator alike. A tight error bar around a diluted centre.
#
# THE PROTOCOL HERE (identical to roihu_fss_matched_pair.sbatch, so the N=72
# and N=800 points are on one protocol and the comparison is meaningful):
#
#     eq_ctrl    all-normal,          8 tau,  v_A = 0
#     eq_soft    cell 0 gamma = 0.35, 8 tau,  v_A = 0    <-- soft during eq too
#     prod_ctrl  resume eq_ctrl,    208 tau,  v_A = 0.01
#     prod_soft  resume eq_soft,    208 tau,  v_A = 0.01
#
# Cell 0 is soft FROM THE OUTSET in the soft branch. Equilibrating all-normal
# and flipping gamma at production start is a different experiment: the soft
# cell would enter aging carrying its stiff-cell shape and relax during the
# early window, reintroducing a transient in one branch only -- which biases
# the ratio UP rather than down. Neither is acceptable.
#
# SEED COUNT. 431339's four seeds gave per-seed ratios 1.073, 1.243, 0.910,
# 1.223: a per-seed SD of 0.155. That spread is the quantity of interest here,
# because the open question is whether a single seed can land near 1.5 by
# chance. 16 seeds put the SEM at 0.155/4 = 0.039 and give the spread itself
# ~18% precision, enough to state the single-seed probability rather than
# guess at it.
#
# Analysis with rust/cell_analyze, never an ad-hoc script.
# ---------------------------------------------------------------------------
#SBATCH --job-name=pf72_eq
#SBATCH --account=project_2019216
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=20:00:00
# 16 seeds x 2 branches = 32 processes, all launched at once and spread 8 to a
# GPU. 72 CTAs per run means one run alone leaves 60 of the 132 SMs idle, so the
# packing is what makes this affordable; the cost is that each run is ~4x its
# solo wall time. Estimated ~9 h; 20 h is margin, and every run checkpoints.
#SBATCH --output=%x_%j.out

set -o pipefail                    # NOT set -u: the CSC module init trips it

# Alarms compiled out (-DPF_NO_ALARMS=ON): 35% faster, verified bitwise
# identical to the alarms-on build on a 32,927,376 B state dump.
BIN=${BIN:-$HOME/gh200_sim/build_noalarm/cell_gh200}
OUT=${OUT:-/scratch/project_2019216/$USER/pf72_eq_$SLURM_JOB_ID}

NCELL=72; RADIUS=49; RHO=0.90; TAU=10000; VA=0.01
EQ_T=${EQ_T:-80000}                # 8 tau,   v_A = 0
PROD_T=${PROD_T:-2080000}          # 208 tau, v_A = 0.01 (absolute t_end)
SAMPLES=${SAMPLES:-20000}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"}
CANCER_FRAC=$(awk -v n=$NCELL 'BEGIN{printf "%.10f", 1.0/n}')   # exactly 1 cell
PER_GPU=${PER_GPU:-2}              # 72 CTAs/run on 132 SMs -> 2 fills the GPU
NGPU=4
CKPT_EVERY=${CKPT_EVERY:-20000000}

mkdir -p "$OUT"
echo "binary  : $BIN"
echo "output  : $OUT"
echo "protocol: N=$NCELL rho=$RHO  eq=$EQ_T (8tau, v_A=0)  prod=$PROD_T (208tau)"
echo "soft    : cancer-fraction=$CANCER_FRAC -> exactly 1 cell, gamma=0.35, soft from outset"
echo "seeds   : [$SEEDS]"
echo "shared  : --seed and --polarity-seed both set per seed, identical across branches"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if [ "$PER_GPU" -gt 1 ]; then
    nvidia-cuda-mps-control -d && echo "MPS daemon started (${PER_GPU} replicas/GPU)"
    trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null; rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"' EXIT
fi

t0=$(date +%s)

# --- stage 1: two SEPARATE equilibrations per seed, v_A = 0 ----------------
echo "=== stage 1: equilibration (8 tau, v_A = 0) ==="
i=0
for seed in $SEEDS; do
  for cond in ctrl soft; do
    cf=0.0; [ "$cond" = soft ] && cf=$CANCER_FRAC
    g=$(( i / PER_GPU % NGPU )); i=$((i+1))
    d="$OUT/eq_${cond}/seed_${seed}"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=$g "$BIN" \
        --N $NCELL --radius $RADIUS --rho $RHO \
        --t-end "$EQ_T" --v-A 0 --tau $TAU \
        --cancer-fraction "$cf" --gamma-cancer 0.35 \
        --seed "$seed" --polarity-seed "$seed" \
        --print-interval 2000000 \
        --checkpoint-interval "$CKPT_EVERY" --checkpoint-dir "$d" \
        > "$d/run.log" 2>&1 &
  done
done
wait
echo "--- equilibration done ---"
missing=0
for seed in $SEEDS; do for cond in ctrl soft; do
  c="$OUT/eq_${cond}/seed_${seed}/checkpoint.bin"
  [ -s "$c" ] || { echo "[fatal] missing $c"; missing=1; }
done; done
[ "$missing" -eq 0 ] || exit 3

# --- stage 2: aging, each branch resuming ITS OWN equilibration ------------
echo
echo "=== stage 2: aging (208 tau, v_A = $VA) ==="
i=0
for seed in $SEEDS; do
  for cond in ctrl soft; do
    g=$(( i / PER_GPU % NGPU )); i=$((i+1))
    d="$OUT/prod_${cond}/seed_${seed}"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=$g "$BIN" \
        -c "$OUT/eq_${cond}/seed_${seed}/checkpoint.bin" \
        --t-end "$PROD_T" --v-A "$VA" \
        --trajectory-samples "$SAMPLES" --print-interval 20000000 \
        --checkpoint-interval "$CKPT_EVERY" --checkpoint-dir "$d" \
        --out "$d/trajectory.txt" > "$d/run.log" 2>&1 &
  done
done
wait
t1=$(date +%s)

echo
echo "=== completion ==="
awk -v a=$t0 -v b=$t1 -v n=$NGPU \
  'BEGIN{printf "wall %.2f h -> %.1f GPU-hours on %d GPUs -> %.0f BU\n",
         (b-a)/3600, (b-a)/3600*n, n, (b-a)/3600*n*200}'
echo "run dirs: $(ls -d "$OUT"/prod_*/seed_* 2>/dev/null | wc -l)  (expect $(( $(echo $SEEDS | wc -w) * 2 )))"
grep -h "alarms" "$OUT"/prod_*/seed_*/run.log 2>/dev/null | sort | uniq -c
echo "frames per run:"
for d in "$OUT"/prod_*/seed_*; do
  printf "  %-30s %s\n" "${d#$OUT/}" \
    "$(( ($(wc -l < "$d/trajectory.txt" 2>/dev/null || echo 3) - 3) / NCELL ))"
done
echo
echo "Analyse:  cell_analyze check $OUT/prod_soft/seed_1"
echo "Ratio  :  paired per seed, then averaged -- see make_fig3_canonical.py"
