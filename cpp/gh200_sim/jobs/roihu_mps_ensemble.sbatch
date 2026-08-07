#!/bin/bash
# ---------------------------------------------------------------------------
# Roihu GH200: ensemble of independent replicas, several per GPU, under MPS.
#
# WHY MPS AND NOT MIG
#   MIG is Disabled on Roihu (nvidia-smi reports mig.mode.current = Disabled,
#   compute mode Default), so it is not available. It would not help anyway:
#   MIG hard-partitions SMs, so a replica on a 1/4 slice gets ~33 SMs and its
#   CTAs simply queue 4x deeper. The gain here comes from filling the TEMPORAL
#   gaps a single replica leaves -- wave tails, load imbalance between cells of
#   different shape classes, and the kernel-boundary drain between steps -- and
#   only a shared, un-partitioned GPU can do that.
#
#   Note the gain is NOT from two CTAs sharing an SM: k_step asks for 213,504 B
#   of shared memory against the SM's 233,472 B, so exactly one CTA is resident
#   per SM no matter which process it came from. MPS interleaves at CTA
#   granularity in time, it does not co-schedule.
#
# WHEN IT PAYS (measured, N=132, 1 GH200, aggregate steps/s):
#     K=1  3832      K=2  6483 (1.69x)      K=4  8663 (2.26x)      K=8  8663
#   Saturates at 4. A single MPS process is 0.91x, i.e. MPS costs ~9% when it
#   has nothing to share with -- do not enable it for a lone large run.
#
#   End-to-end with THIS script, full node, 16 replicas (4 GPUs x 4), N=72,
#   t_end=2000, identical in every other respect:
#     MPS on   30.5 s      MPS off   50.8 s      ->  1.67x
#   16/16 replicas clean in both cases. That 1.67x is the number to plan an
#   N=72 seed ensemble against, not the 2.26x single-GPU figure above.
#
#     N <= ~200   4 replicas/GPU   -> up to 2.26x
#     N ~  400    4 replicas/GPU   -> ~1.10x  (one replica already fills the SMs)
#     N >= ~1000  1 replica/GPU    -> concurrency buys nothing
#
# BILLING: gputest is unbilled but capped at 15 min and one job. Production
# belongs on gpumedium, which DOES consume the GPU BU allocation.
# ---------------------------------------------------------------------------
#SBATCH --job-name=pf_ensemble
#SBATCH --account=project_2019216
#SBATCH --partition=gputest          # <-- change to gpumedium for production
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=00:14:00
#SBATCH --output=pf_ensemble_%j.out

set -o pipefail                      # NOT set -u: the CSC module init trips it

BIN=${BIN:-$HOME/gh200_sim/build/cell_gh200}
OUTROOT=${OUTROOT:-/scratch/project_2019216/$USER/ensemble_$SLURM_JOB_ID}

NGPU=${NGPU:-4}                      # GPUs on this node
PER_GPU=${PER_GPU:-4}                # replicas per GPU (see table above)
NCELL=${NCELL:-72}
RADIUS=${RADIUS:-49}
RHO=${RHO:-0.90}
TEND=${TEND:-10000}                  # 1 tau at tau=1e4
VA=${VA:-0.01}
TAU=${TAU:-10000}
SAMPLES=${SAMPLES:-200}
CANCER_FRAC=${CANCER_FRAC:-0.0}
GAMMA_CANCER=${GAMMA_CANCER:-0.35}

mkdir -p "$OUTROOT"
echo "binary   : $BIN"
echo "output   : $OUTROOT"
echo "ensemble : $NGPU GPUs x $PER_GPU replicas = $((NGPU*PER_GPU)) runs"
echo "physics  : N=$NCELL R=$RADIUS rho=$RHO t_end=$TEND cancer=$CANCER_FRAC"

# --- MPS daemon, one per node ---------------------------------------------
# Per-job pipe/log dirs, or concurrent jobs on a shared node collide.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

USE_MPS=${USE_MPS:-1}
if [ "$PER_GPU" -gt 1 ] && [ "$USE_MPS" = "1" ]; then
    nvidia-cuda-mps-control -d && echo "MPS daemon started"
    trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null; rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"' EXIT
else
    echo "MPS not started (PER_GPU=$PER_GPU, USE_MPS=$USE_MPS)"
fi

# --- launch ----------------------------------------------------------------
t0=$(date +%s.%N)
for ((g = 0; g < NGPU; ++g)); do
  for ((r = 0; r < PER_GPU; ++r)); do
    seed=$(( 1000 + g * PER_GPU + r ))
    d="$OUTROOT/gpu${g}_rep${r}_seed${seed}"
    mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=$g "$BIN" \
        --N "$NCELL" --radius "$RADIUS" --rho "$RHO" \
        --t-end "$TEND" --v-A "$VA" --tau "$TAU" \
        --cancer-fraction "$CANCER_FRAC" --gamma-cancer "$GAMMA_CANCER" \
        --seed "$seed" --trajectory-samples "$SAMPLES" \
        --print-interval 1000000 \
        --out "$d/trajectory.txt" > "$d/run.log" 2>&1 &
  done
done
wait
t1=$(date +%s.%N)

# --- report ----------------------------------------------------------------
runs=$((NGPU * PER_GPU))
awk -v a="$t0" -v b="$t1" -v r="$runs" 'BEGIN{
  w = b - a
  printf "\nwall %.1f s for %d replicas = %.1f s/replica effective\n", w, r, w/r
  printf "GPU-hours (node) %.4f   per replica %.5f\n", w*4/3600, w*4/3600/r
}'

ok=$(grep -l "alarms: all clear\|no fatal flags" "$OUTROOT"/*/run.log 2>/dev/null | wc -l)
echo "replicas finishing clean: $ok / $runs"
grep -h "ALARMS SET" "$OUTROOT"/*/run.log 2>/dev/null | sort | uniq -c
echo "trajectories:"; ls -1 "$OUTROOT"/*/trajectory.txt 2>/dev/null | wc -l
