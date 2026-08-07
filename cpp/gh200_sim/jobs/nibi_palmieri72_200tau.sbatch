#!/bin/bash
# ---------------------------------------------------------------------------
# Palmieri (2015) reproduction at N=72, run to 200 tau, on Nibi H100.
#
# PURPOSE: validate the FUSE-1R solver against the published result. The target
# is the soft/control ratio of the TAGGED-CELL effective diffusion coefficient,
# expected ~1.5. Matched pairs: each seed is run twice, once with cell 0 softened
# (gamma = 0.35) and once all-normal, sharing the seed so the initial packing and
# the polarity stream are identical. The ratio is then formed PER SEED and
# averaged -- the same protocol make_fig3_canonical.py uses.
#
# 200 tau = 2e6 time units at dt = 0.01 = 2e8 steps per run.
#
# NO CHECKPOINTING YET. If this hits the walltime the run is lost, so the time
# request is deliberately generous. That is the top item on the feature-gap list.
#
# Analysis is done with rust/cell_analyze -- the trajectory.txt written here is
# the legacy 12-column format it reads natively. Do not post-process with
# ad-hoc scripts.
# ---------------------------------------------------------------------------
#SBATCH --job-name=pf72_200tau
#SBATCH --account=def-mkarttu_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=11:00:00
#SBATCH --output=%x_%j.out

set -o pipefail
module load cuda/12.6 2>/dev/null

BIN=${BIN:-$HOME/gh200_sim/build/cell_gh200}
OUT=${OUT:-$HOME/scratch/pf72_200tau_$SLURM_JOB_ID}

NCELL=72
RADIUS=49
RHO=0.90
TAU=10000
VA=0.01
TEND=${TEND:-2000000}          # 200 tau
# 20000 frames = one every 10000 steps = every 0.01 tau. This is set by MSD
# resolution, not by file size: at 400 frames the first available lag is 0.5 tau,
# so the ballistic regime (t << tau) and the ballistic->diffusive crossover are
# entirely unresolved and only the plateau is visible. 20000 frames resolve two
# decades below tau. Cost is ~158 MB/run and a sync every 10000 steps (~0.4 s of
# compute), i.e. negligible against the 2.2 h run.
SAMPLES=${SAMPLES:-20000}
SEEDS=${SEEDS:-"1 2 3 4"}
CANCER_FRAC=0.0139             # 1 cell of 72

mkdir -p "$OUT"
echo "binary  : $BIN"
echo "output  : $OUT"
echo "protocol: N=$NCELL rho=$RHO t_end=$TEND (=$((TEND/TAU)) tau) seeds=[$SEEDS]"
echo "          matched pairs, soft = 1 cell at gamma=0.35, ctrl = all gamma=1"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

g=0
for seed in $SEEDS; do
  for cond in ctrl soft; do
    cf=0.0
    [ "$cond" = soft ] && cf=$CANCER_FRAC
    d="$OUT/${NCELL}c_${cond}/seed_${seed}"
    mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=$g "$BIN" \
        --N $NCELL --radius $RADIUS --rho $RHO \
        --t-end "$TEND" --v-A $VA --tau $TAU \
        --cancer-fraction "$cf" --gamma-cancer 0.35 \
        --seed "$seed" --trajectory-samples "$SAMPLES" \
        --print-interval 20000000 \
        --out "$d/trajectory.txt" > "$d/run.log" 2>&1 &
    g=$(( (g + 1) % 8 ))
  done
done
wait

echo
echo "=== completion ==="
ls -d "$OUT"/${NCELL}c_*/seed_* | wc -l | xargs echo "run dirs:"
grep -h "alarms" "$OUT"/${NCELL}c_*/seed_*/run.log 2>/dev/null | sort | uniq -c
grep -h "ALARMS SET" "$OUT"/${NCELL}c_*/seed_*/run.log 2>/dev/null | sort | uniq -c
echo "trajectories written:"; ls -1 "$OUT"/${NCELL}c_*/seed_*/trajectory.txt 2>/dev/null | wc -l

cat <<EOF

Analyse with cell_analyze (NOT a scratch script):

  cell_analyze check $OUT/72c_soft/seed_1
  cell_analyze study <study.toml> --data-dir $OUT

with a study TOML whose discovery pattern is "{N}c_{cond}/seed_{seed}", computing
msd_palmieri and pairing soft/ctrl. The headline number is the per-seed ratio of
d_eff from msd_t_cell at lag 8 tau -- expected ~1.5.
EOF
