#!/bin/bash
# ---------------------------------------------------------------------------
# Roihu BU meter.
#
# THE AUTHORITATIVE SOURCE IS `csc-projects`, which lives in the csc-tools
# module and is NOT on the default PATH -- that is why it looks absent:
#
#     module load manual/general/aarch64/csc-tools
#     # or just:
#     export PATH=/appl/soft/manual/general/aarch64/csc-tools/bin:$PATH
#
# It reports Budget / Used / Remain for CPU, GPU and Storage BU per project.
#
# Do NOT use Slurm's `billing` TRES as a BU proxy: TRESBillingWeights is unset
# on every Roihu GPU partition, so `billing` merely counts CPUs (a real
# gpumedium job reports billing=32, cpu=32, gres/gpu:gh200=4). It has no GPU
# term at all and will understate GPU cost by orders of magnitude.
#
# This script pairs the authoritative balance with GPU-hours from sacct, so you
# can (a) see the balance and (b) DERIVE the BU-per-GPU-hour rate empirically,
# which is not documented anywhere on the cluster.
#
# usage:  roihu_bu_usage.sh [since]          # default: today
#         roihu_bu_usage.sh 2026-07-29
# ---------------------------------------------------------------------------
set -o pipefail
SINCE="${1:-$(date +%Y-%m-%d)}"
CSC_BIN=/appl/soft/manual/general/aarch64/csc-tools/bin
[ -d "$CSC_BIN" ] && export PATH="$CSC_BIN:$PATH"

echo "=================== BU balance (authoritative) ==================="
if command -v csc-projects >/dev/null 2>&1; then
    csc-projects 2>/dev/null | grep -vE "not accessible|^-+$" | sed '/^$/d'
else
    echo "  csc-projects not found; is the csc-tools module path still $CSC_BIN ?"
fi

echo
echo "=================== GPU-hours from sacct since $SINCE ==================="
# gputest is unbilled; every other GPU partition draws down GPU BU.
sacct -u "$USER" -S "$SINCE" -X -P -n \
      --format=JobID,Partition,State,ElapsedRaw,AllocTRES 2>/dev/null |
awk -F'|' '
function is_free(p) { return p == "gputest" }
{
    part=$2; secs=$4+0; tres=$5; ngpu=0
    if (match(tres, /gres\/gpu[^=]*=[0-9]+/)) {
        s=substr(tres, RSTART, RLENGTH); sub(/.*=/,"",s); ngpu=s+0
    }
    if (ngpu == 0) next
    njob[part]++; gs[part] += secs*ngpu
    if (is_free(part)) free_gs += secs*ngpu; else paid_gs += secs*ngpu
}
END {
    printf "  %-14s %7s %14s %12s\n", "partition","jobs","GPU-hours","billed?"
    for (p in gs)
        printf "  %-14s %7d %14.4f %12s\n", p, njob[p], gs[p]/3600.0,
               (is_free(p) ? "no (free)" : "YES")
    printf "\n  BILLED : %.4f GPU-hours\n", paid_gs/3600.0
    printf "  free   : %.4f GPU-hours\n", free_gs/3600.0
    if (paid_gs > 0)
        printf "\n  To DERIVE the rate: note GPU BU \"Used\" above, run a known job,\n" \
               "  re-run this, and divide the BU delta by the GPU-hour delta.\n"
}'

echo
echo "NOTE: the same account is shared, so sacct cannot tell whose jobs are whose."
echo "      Filter by JobID if you need to attribute consumption to one campaign."
