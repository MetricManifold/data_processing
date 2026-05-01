#!/bin/bash
#SBATCH --account=rrg-mkarttu-ab-phase_cpu
#SBATCH --job-name=cpu_ref_2tau_v2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=/scratch/ssilber/cpu_ref_validate/cell_sim_logs/%x_%j.out
#SBATCH --error=/scratch/ssilber/cpu_ref_validate/cell_sim_logs/%x_%j.err

set -euo pipefail
mkdir -p /scratch/ssilber/cpu_ref_validate/cell_sim_logs
module load StdEnv/2023 rust/1.91.0
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

BIN=/scratch/ssilber/cpu_ref_validate/src_repo/rust/cpu_ref/target/release/cpu_ref
OUT=/scratch/ssilber/cpu_ref_validate/run_2tau_v2

mkdir -p $OUT

echo "[$(date)] starting on $(hostname), threads=$RAYON_NUM_THREADS"
$BIN \
  --ic /scratch/ssilber/cpu_ref_validate/ic_run/checkpoint.bin \
  --v-a 0.01 \
  --tau 10000 \
  --t-end 20000 \
  --dt 0.01 \
  --save-every 5000 \
  --polarity-seed 12345 \
  --out $OUT/traj.npz \
  --trajectory $OUT/trajectory.txt \
  --events $OUT/events.txt \
  --log-every 50000 \
  --threads $SLURM_CPUS_PER_TASK
echo "[$(date)] done"
