#!/bin/bash
# ---------------------------------------------------------------------------
# Nibi: submit a long run as a CHAIN of <3 h legs, to exploit the walltime bin.
#
# THE FINDING THIS EXISTS FOR (measured over 400 of your historical GPU jobs):
#
#     partition          bin      jobs   median wait
#     gpubase_bygpu_b1   <= 3 h    244        0.21 h   <-- 13 minutes
#     gpubase_bygpu_b2   <=12 h     91       51.13 h
#     gpubase_bygpu_b3   <=24 h     60      132.94 h
#     gpubase_bygpu_b4   <= 3 d      3      179.08 h
#
#   and by requested walltime, 1 GPU:
#     3 h ->  0.05 h median wait
#     4 h -> 36.60 h median wait      <-- the cliff is EXACTLY at 3 h
#    12 h -> 174.65 h
#
# One extra hour of requested walltime costs ~700x the queue wait. Slurm routes
# on the REQUESTED limit, not the actual runtime, so a 16 h job asking for 16 h
# lands in b3 and waits ~5.5 days. Six 2h59m legs land in b1 and wait ~13 min
# each. Verified by submitting 2:59:00 and 3:01:00 probes: they were routed to
# b1 and b2 respectively.
#
# GPU count matters too (1-2 GPUs schedule fast, 4 -> ~1 h, 8 -> ~5.7 h), so
# each leg takes exactly ONE GPU and branches run as independent chains.
#
# This is only possible because the solver checkpoints and resumes. Each leg
# resumes the previous leg's checkpoint and appends to the same trajectory.txt
# (the writer opens in append mode and emits the header only when the file is
# empty), so the chain produces one continuous trajectory.
#
# usage:
#   nibi_chain_3h.sh <outdir> <eq_checkpoint> <t_end_total> <n_legs> [extra sim args...]
# ---------------------------------------------------------------------------
set -o pipefail

OUT=${1:?outdir}; EQ=${2:?eq checkpoint}; T_END=${3:?absolute t_end}; LEGS=${4:?n legs}
shift 4
EXTRA="$@"

BIN=${BIN:-$HOME/gh200_sim/build/cell_gh200}
ACCOUNT=${ACCOUNT:-def-mkarttu_gpu}
WALL=${WALL:-02:59:00}          # MUST stay under 3 h or it falls into b2
GPUS=${GPUS:-h100:1}            # 1 GPU: fastest to schedule

mkdir -p "$OUT"
[ -s "$EQ" ] || { echo "[fatal] no equilibration checkpoint at $EQ"; exit 2; }

# t at the end of the equilibration, so legs divide the REMAINING span evenly.
T0=$(python3 -c "
import struct,sys
f=open('$EQ','rb'); d=f.read(44)
print(struct.unpack_from('<d', d, 12)[0])" 2>/dev/null || echo 0)
SPAN=$(python3 -c "print(($T_END - $T0)/$LEGS)")

echo "chain: $LEGS legs x $WALL, 1 GPU each, t $T0 -> $T_END (span $SPAN per leg)"
echo "  each leg lands in gpubase_bygpu_b1 (median wait 0.21 h)"

dep=""
for k in $(seq 1 "$LEGS"); do
  t_leg=$(python3 -c "print(min($T_END, $T0 + $SPAN*$k))")
  # Leg 1 resumes the equilibration; later legs resume the previous leg.
  src="$EQ"; [ "$k" -gt 1 ] && src="$OUT/checkpoint.bin"
  jid=$(sbatch $dep \
      --account="$ACCOUNT" --time="$WALL" --gpus-per-node="$GPUS" \
      --cpus-per-task=8 --mem=32G \
      -J "leg${k}" -o "$OUT/leg%a_%j.out" \
      --wrap "$BIN -c '$src' --t-end $t_leg \
              --checkpoint-interval 5000000 --checkpoint-dir '$OUT' \
              --out '$OUT/trajectory.txt' $EXTRA" \
      2>&1 | grep -oE '[0-9]+$')
  echo "  leg $k -> job $jid   t_end=$t_leg"
  dep="--dependency=afterok:$jid"
done
echo
echo "Chain submitted. Each leg becomes eligible only when the previous succeeds,"
echo "then queues in b1. Watch with: squeue -u \$USER -o '%.12i %.8j %.9T %R'"
