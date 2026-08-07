#!/bin/bash
# ---------------------------------------------------------------------------
# Palmieri (2015) reproduction at N=72 to 200 tau, on Roihu GH200 (gpumedium).
#
# WHY gpumedium AND NOT gputest: gputest is free but capped at 15 min and one
# running job, and this needs ~3 h. Nibi's queue is deep, so paying a little BU
# here is faster than waiting there.
#
# COST: 8 runs on 4 GPUs, 2 per GPU under MPS (measured 1.67x at N=72), so
#   wall  ~ 2 x 2.61 h / 1.67  = 3.1 h
#   GPU-h = 4 x 3.1            = 12.5 GPU-hours
#   ~1250 BU IF the rate matches Mahti's 100 BU/GPU-h -- the Roihu rate is
#   documented nowhere on the cluster, so derive it by differencing
#   `csc-projects` across this job (see jobs/roihu_bu_usage.sh).
# Running 1-per-GPU instead would cost 20.8 GPU-h, so MPS saves ~40% of the BU
# as well as wall time.
#
# BUDGET CONTEXT (csc-projects, 2026-08-02): project_2019216 has 173,383 GPU BU
# remaining of 651,000 -- that budget is a 400k top-up against a ~220k deficit,
# so the headroom is smaller than the budget line suggests. Scale campaigns
# against 173k, not 651k.
#
# SCRATCH: output goes to /scratch/project_2019216 (this project, 250G, 180d
# cleanup). Do NOT write to /scratch/project_2017848 -- that is RoihuPilot_ELASTIC,
# a different project this account merely has access to, with a 90d cleanup.
#
# PROTOCOL: matched pairs. Each seed is run twice -- once with cell 0 softened
# to gamma=0.35, once all-normal -- sharing the seed so the packing and the
# polarity stream are identical. The ratio is formed PER SEED and then averaged,
# which is what make_fig3_canonical.py does and what cancels seed-specific noise.
#
# 20000 trajectory frames = one every 0.01 tau. Set by MSD resolution: at 400
# frames the shortest available lag is 0.5 tau and the ballistic regime is
# entirely invisible. Do not reduce this to save disk.
#
# Analysis with rust/cell_analyze, never an ad-hoc script.
# ---------------------------------------------------------------------------
#SBATCH --job-name=pf72_200tau
#SBATCH --account=project_2019216
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out

set -o pipefail                    # NOT set -u: the CSC module init trips it

BIN=${BIN:-$HOME/gh200_sim/build/cell_gh200}
OUT=${OUT:-/scratch/project_2019216/$USER/pf72_200tau_$SLURM_JOB_ID}

NCELL=72; RADIUS=49; RHO=0.90; TAU=10000; VA=0.01
TEND=${TEND:-2000000}              # 200 tau
SAMPLES=${SAMPLES:-20000}          # every 0.01 tau
SEEDS=${SEEDS:-"1 2 3 4"}
CANCER_FRAC=0.0139                 # 1 cell of 72
PER_GPU=${PER_GPU:-2}

mkdir -p "$OUT"
echo "binary  : $BIN"
echo "output  : $OUT"
echo "protocol: N=$NCELL rho=$RHO t_end=$TEND (=$((TEND/TAU)) tau) seeds=[$SEEDS]"
echo "          matched pairs; soft = 1 cell at gamma=0.35, ctrl = all gamma=1"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if [ "$PER_GPU" -gt 1 ]; then
    nvidia-cuda-mps-control -d && echo "MPS daemon started (${PER_GPU} replicas/GPU)"
    trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null; rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"' EXIT
fi

t0=$(date +%s)
i=0
for seed in $SEEDS; do
  for cond in ctrl soft; do
    cf=0.0; [ "$cond" = soft ] && cf=$CANCER_FRAC
    g=$(( i / PER_GPU % 4 )); i=$((i+1))
    d="$OUT/${NCELL}c_${cond}/seed_${seed}"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=$g "$BIN" \
        --N $NCELL --radius $RADIUS --rho $RHO \
        --t-end "$TEND" --v-A $VA --tau $TAU \
        --cancer-fraction "$cf" --gamma-cancer 0.35 \
        --seed "$seed" --trajectory-samples "$SAMPLES" \
        --print-interval 20000000 \
        --checkpoint-interval 20000000 --checkpoint-dir "$d" \
        --out "$d/trajectory.txt" > "$d/run.log" 2>&1 &
  done
done
wait
t1=$(date +%s)

echo
echo "=== completion ==="
awk -v a=$t0 -v b=$t1 'BEGIN{printf "wall %.2f h -> %.1f GPU-hours on 4 GPUs\n",(b-a)/3600,(b-a)/3600*4}'
echo "run dirs: $(ls -d "$OUT"/${NCELL}c_*/seed_* | wc -l)"
grep -h "alarms" "$OUT"/${NCELL}c_*/seed_*/run.log 2>/dev/null | sort | uniq -c
echo "frames per run:"
for d in "$OUT"/${NCELL}c_*/seed_*; do
  printf "  %-28s %s\n" "${d#$OUT/}" "$(( ($(wc -l < "$d/trajectory.txt" 2>/dev/null || echo 3) - 3) / NCELL ))"
done
echo
echo "Next: cell_analyze study <toml> --data-dir $OUT   (pattern {N}c_{cond}/seed_{seed})"
echo "Then: jobs/roihu_bu_usage.sh   to confirm what this actually cost."
