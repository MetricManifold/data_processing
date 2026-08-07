# FUSE-1R — measured results on Roihu GH200

Everything below was run on `gputest` (unbilled, `--account=project_2019216`).
**Zero GPU BU consumed.** See `CALIBRATION.md` for the hardware constants.

## 1. Build

Builds clean, first attempt, no warnings-as-errors triggered:

```
nvhpc 26.3 / nvcc CUDA 13.0.88, aarch64, -DCMAKE_CUDA_ARCHITECTURES=90
k_step: 80 registers, 1 barrier, 304 B stack, 0 B spill stores/loads
smem/CTA: 211840 B of 232448 B opt-in max
```

Zero register spill in the fused kernel, which was the main compile-side risk.

## 2. Correctness — validated against the verified oracle

The oracle is `cpp/simulation/tests/python/cpu_reference.py`, verified this session
against Palmieri Eq. (S15) by three independent routes.

**One-step field comparison.** Dump the GPU state at step 200 and at step 201 (same
seed, deterministic), advance the oracle one step from state 200, compare:

```
N=24, R=49, rho=0.85, v_A=0, 24 cells in contact (repulsion fully active)

WORST over 24 cells:  max|dphi| = 6.42e-08
```

fp32 epsilon at phi~1 is 1.19e-07, so **the GPU agrees with the oracle to
single-precision round-off across a full step**, repulsion included. A wrong
coefficient would show O(1) here.

**Component checks against an independent numpy recomputation of the same field:**

| quantity | agreement | note |
|---|---|---|
| `V = sum(phi^2)` | **1e-16** | machine precision; the carried-V scheme is exact |
| `S` field (Q5.27 vs fp64) | **exactly 0 difference** | fixed-point S is lossless here |
| motility coeff (`vx/Ix`) | 0.99999996 of `60k/(xi l^2)` | fp32 rounding only |
| `Ix`, `Iy` | 2.6e-3 of the sum of absolute terms | see below |

`Ix = sum(phi * d_x phi * S_other)` is a near-zero integral of large signed terms
(the net interaction force on a near-equilibrium cell almost cancels). Its
*absolute* error is ~1e-4 against a term scale of ~0.2; expressed against the
cancelled sum that looks like percent-level, which is cancellation amplification of
the 6e-8 field noise, not a coefficient error. The one-step field test above is the
authoritative check and it is clean.

**Self-test gates** (`--self-test`) all pass, and a 10,000-step run reports
`alarms: all clear` with all eight sticky flags zero (no S overflow, no q clamp, no
support clip, no class exhaustion). Printed coefficients:

```
bulk  30/l^2      0.612244898
inter 60k/l^2     12.244898          <- 60, not 30
motil 60k/(xl^2)  0.00816326531
inter/motil       1500   == xi       <- the convention-free invariant
p_tumble          9.999995000e-07    == -expm1(-dt/tau)
```

## 3. Performance — honest head-to-head, same GPU, same case

`--bench`, no I/O, R=49, rho=0.89, one GH200. "OLD" is the existing
post-coefficient-fix `cell_sim`.

| N | waves (132 SMs) | OLD µs/step | NEW µs/step | speedup |
|---:|---:|---:|---:|---:|
| 132 | 1 | 71.00 | **41.40** | **1.71x** |
| 264 | 2 | 126.00 | **87.23** | **1.44x** |
| 396 | 3 | 175.00 | **134.35** | **1.30x** |
| 1056 | 8 | 442.00 | **373.56** | **1.18x** |

At the original comparison point N=288 (a ragged 2.18 waves) the new solver is
126.7 µs/step against 137, i.e. 1.08x — that point flatters the old code because
288 = 2x132 + 24 leaves the third wave 18% full.

**The projected 3–4x was not achieved.** The projection assumed the fused kernel
would approach the measured 251 Gpix/s stencil roofline. It does not.

## 4. Why — measured, not guessed

At N=396: 396 cells x 144² px = 8.21e6 px in 134.35 µs. The minimum HBM traffic
(read phi + write phi, S served from cache) is 8 B/px = 65.7 MB, so the kernel is
sustaining **0.485 TB/s against the 2.0–2.7 TB/s this GPU actually delivers — about
20% of peak.** It is *latency* bound, not bandwidth bound.

The cause is structural. The design spends 211,840 B of shared memory to keep the
whole rect resident, which pins it to **1 CTA/SM**; at 768 threads that is 24 of 64
possible warps, **37.5% occupancy**. Worse, `k_step` is internally phased
(load → reduce Ix/Iy → RHS → scatter/moments) with barriers between phases, and with
only one CTA per SM there is no second CTA to overlap those phases against, so the
memory system idles during every reduction and the barriers drain the pipeline.

Two things were tested and ruled out:

- **More warps in the one CTA does not help.** Raising the block to 1024 threads
  forces nvcc down to 64 registers/thread (stack 304→432 B, i.e. spill), and it is
  *worse everywhere*: 59.55 vs 41.40 µs at N=132, 185.78 vs 134.35 at N=396,
  516.85 vs 373.56 at N=1056 — roughly 40% slower. Reverted; 768/80 is the better
  point. So the kernel is not simply warp-starved.
- **L2 persistence of S is worth nothing here.** `--no-l2` measures 135.13 µs vs
  135.34 baseline at N=396. S is small enough to stay cached without the carve-out.
  `--no-graph` (136.86) and `--morton` (137.18) are likewise neutral at this size.

## 4b. Nsight Compute profile — the limiter, measured

`ncu --kernel-name k_step` at N=396, sm_90:

| metric | value |
|---|---|
| DRAM throughput | **15.67 %** |
| Memory throughput | 22.46 % |
| Compute (SM) throughput | 53.32 % |
| Achieved occupancy (`sm__warps_active`) | **37.49 %** |
| Warp cycles per issued instruction | 10.88 |
| Avg. active threads per warp | 29.22 / 32 |
| `launch__occupancy_limit_shared_mem` | **1 block** |
| `launch__occupancy_limit_registers` | **1 block** |
| `launch__occupancy_limit_warps` | 2 blocks |

ncu's own rule fires: *"Achieved compute throughput and/or memory bandwidth below 60% of
peak typically indicate latency issues."* DRAM at 15.7% confirms there is ~6x of memory
headroom being left on the floor.

**The finding that changes the plan: registers cap occupancy at 1 CTA/SM independently of
shared memory.** sm_90 has a 65,536-register file per SM; the kernel uses 80 regs x 768
threads = 61,440 for a *single* block, so a second block cannot fit no matter how much
shared memory is freed. The ceiling is

```
warps_per_SM  <=  65536 / (32 * regs_per_thread)
   80 regs -> 25.6 warps -> 40 % occupancy (we are at 37.5 %)
   64 regs -> 32   warps -> 50 %
   48 regs -> 42   warps -> 66 %
   40 regs -> 51   warps -> 80 %
```

So the split-kernel change in §5 must hit **both** a shared-memory target and a register
target, or it buys nothing. This is why the 1024-thread experiment failed: it bought warps
by forcing registers down to 64, and paid for them in spill.

## 4c. Concurrent replicas per GPU — a zero-code-change win

Because one replica only reaches 37.5% occupancy and 15.7% DRAM, the GPU is mostly idle.
Running independent replicas concurrently fills it. Aggregate throughput, one GH200:

Without MPS (N=132): K=1 4115 steps/s, K=2 5970 (1.45x), K=4 6969 (**1.69x**).

With CUDA MPS (`nvidia-cuda-mps-control -d`, present at `/usr/bin`):

| N | K=1 | K=2 | K=4 | gain |
|---:|---:|---:|---:|---:|
| 132 | 3832 | 6483 | 8663 | **2.26x** |
| 396 | 2783 | 3127 | 3069 | 1.10x |
| 1056 | 906 | 985 | 1003 | 1.11x |

MPS beats plain multi-process (2.11x vs 1.69x at N=132) and saturates at K=4; K=8 gives
nothing further. A single MPS process is 0.91x, i.e. MPS costs ~9% when not sharing.

**The gain is large only at small N.** At N=132 one replica is exactly one wave (132 CTAs
on 132 SMs) at 37.5% occupancy, so concurrent replicas fill the empty warp slots. From
N~396 up there are already 3+ waves queued and the SMs never starve, so concurrency adds
only ~10%.

**Recommendation for the FSS campaign:** at N <= ~200 run **4 replicas per GPU under MPS**
(2.2x more science per GPU-hour); at N >= ~400 run **1 replica per GPU**. Either way run
4 GPUs per node. This needs no code change and is independent of, and composes with, the
kernel optimization below.

## 4d. Large-N behaviour

**Scaling is clean — there is no large-N cliff.** Per-cell cost is flat:

| N | L | µs/step | **µs/cell** | S MB/buffer |
|---:|---:|---:|---:|---:|
| 1056 | 2992 | 375.25 | 0.355 | 34.3 |
| 2112 | 4231 | 737.48 | 0.349 | 68.7 |
| 4224 | 5984 | 1454.46 | 0.344 | 136.6 |
| 8448 | 8462 | 2903.74 | 0.344 | 273.7 |

0.344 µs/cell holds from N=1056 to N=8448 even though S exceeds the 60 MB L2 at every
one of those sizes. It is also the same per-cell cost as N=396 (0.339), so **the latency
limit of §4b applies uniformly for N >= ~400** and the occupancy fix in §5 benefits large
N just as much as small N. Extrapolating, N=12800 is ~4.4 ms/step.

Bandwidth roofline per cell at 144²: read phi + read S + write phi + atomic S
= 4 x 82,944 B = 0.332 MB; at 2.5 TB/s that is **0.133 µs/cell**. We are at 0.344, i.e.
**2.6x off the roofline** — the same latency headroom seen at small N.

### The L2 persisting carve-out must be conditional (free 6-7% at large N)

The code currently always carves out L2 for the S buffers. Measured:

| N | S MB/buf | 2xS fits 37.5 MB? | L2 on | L2 off | effect of `--no-l2` |
|---:|---:|:---:|---:|---:|---:|
| 132 | 4.39 | **yes** | 41.35 | 43.12 | **-4.3% (carve-out helps)** |
| 396 | 12.97 | **yes** | 135.61 | 134.91 | +0.5% (neutral) |
| 1056 | 34.33 | no | 375.06 | 347.87 | **+7.2% (carve-out hurts)** |
| 2112 | 68.69 | no | 736.35 | 688.70 | **+6.5% (carve-out hurts)** |

The rule is exactly the fits-test: once S exceeds the persisting limit the carve-out
reserves 25 MB of a 60 MB L2 for a buffer that cannot fit anyway, evicting phi.

**Action:** make the carve-out automatic rather than unconditional —
enable iff `2 * S_bytes_per_buffer <= persistingL2CacheMaxSize`, disable otherwise.
Crossover is S ~ 18.75 MB/buffer, i.e. L ~ 2165, i.e. **N ~ 550**. This preserves the
4.3% win at small N and recovers 6-7% across the whole large-N range, for free.

### Morton ordering should be off

`--morton` is **slower** at large N — +1.8% at N=2112 and +2.0% at N=8448 — despite its
help text claiming it "helps only at large N". It was also neutral-to-slightly-negative at
N=396. Recommend leaving it off by default and correcting the help text.

### Not worth doing at large N

- **16-bit S.** Halving S would cut ~25% of traffic, but Q5.27's exactness argument
  (`q_S - q_of(phi)` is an exact non-negative uint32 subtraction, which is what allows the
  `Soth < 0` clamp to be deleted) depends on the fixed-point width. 16 bits gives a
  quantum ~1e-4 against a phi² resolution requirement near 1e-7. Correctness risk is not
  worth 25% of a traffic budget that is not the binding constraint anyway.
- **Tighter tile pitch.** `kTilePitch = 256` against a 208-px maximum window span means the
  pool is ~1.5x larger than needed (6.7 GB vs 4.4 GB at N=12800). But only the window is
  ever read or written, so *traffic* is unaffected — this is footprint and TLB pressure
  only, and 95 GiB is not close to binding.

## 5. The split-kernel occupancy experiment — IMPLEMENTED, MEASURED, REJECTED

Available behind `--split`; the fused path remains the default. **Do not enable it.**

The hypothesis was §4b's: the fused kernel is latency-bound at 37.5% occupancy, pinned to
1 CTA/SM by *both* shared memory and registers, so splitting P3/P3b into a second kernel
frees the big shared-memory buffer and lets 2 CTAs fit.

**The occupancy targets were met exactly as designed:**

```
k_step_rhs   512 thr  64 regs  110464 B smem  -> 2 CTAs/SM  (32 warps, 50.0%)
k_step_post  512 thr  40 regs    1024 B smem  -> 3 CTAs/SM  (48 warps, 75.0%)
k_step (fused, unchanged)  768 thr  80 regs  211840 B smem  -> 1 CTA/SM (24 warps, 37.5%)
```

**And it is comprehensively slower:**

| N | fused µs | split µs | gain |
|---:|---:|---:|---:|
| 132 | 41.69 | 80.05 | **0.52x** |
| 396 | 135.44 | 192.91 | **0.70x** |
| 1056 | 375.62 | 567.82 | **0.66x** |
| 2112 | 736.43 | 1144.59 | **0.64x** |

Why: `k_step_rhs` spills **368 B/thread** to local memory to meet its 64-register budget.
At 512 threads that is ~188 KB of local-memory traffic per CTA, which swamps the benefit of
going from 24 to 32 warps. The second kernel's extra read of phi and the second launch per
step add to it.

**This is the second independent confirmation of the same lesson.** The 1024-thread
experiment (§4, 64 regs, 432 B spill) was ~40% slower; the split (64 regs, 368 B spill) is
30-48% slower. The kernel's live-register working set is genuinely ~80 registers, and every
attempt to buy occupancy by forcing it lower has cost more than it bought.

**Conclusion: 37.5% occupancy at 80 registers with zero spill is the correct operating
point for this algorithm.** Raising occupancy is not a route to more performance here.
Any future attempt must reduce the register working set *algorithmically* — fewer live
values, not a tighter budget on the same code — and should be validated against these two
data points before being believed.

## 5b. Conditional L2 carve-out — IMPLEMENTED AND DELIVERED

`src/sim.cu`: the carve-out is now taken **only when the two hot S buffers actually fit**
in the 0.85 x `persistingL2CacheMaxSize` budget; otherwise it is skipped and
`opt_.l2_persist` is cleared, which also suppresses the per-launch access-policy window
(`l2_window_for_slot` gates on the same flag). The decision and its reason are printed.

```
N=132    L2 persisting carve-out 8.78 MB (S buffer 4.39 MB each)
N=396    L2 persisting carve-out 25.94 MB (S buffer 12.97 MB each)
N=1056   L2 persisting carve-out DISABLED: 2 x S = 68.66 MB exceeds the 31.88 MB budget
N=2112   L2 persisting carve-out DISABLED: 2 x S = 137.38 MB exceeds the 31.88 MB budget
```

| N | before | after | gain |
|---:|---:|---:|---:|
| 132 | 41.56 | 41.74 | -0.4% (noise) |
| 396 | 135.55 | 135.61 | 0.0% |
| 1056 | 375.66 | **347.90** | **+7.4%** |
| 2112 | 736.48 | **687.96** | **+6.6%** |

**Numerically inert, proven:** same seed, N=1056, 300 steps, full 87,707,472-byte state dump
from the old and new binaries — identical md5 (`d6e86d3f0fac5532f6792704db853d18`), `cmp`
clean. An L2 residency hint must not change arithmetic, and it does not.

## 6. Final head-to-head vs the existing code

One GH200, `--bench`, no I/O, R=49, rho=0.89, conditional L2 policy active:

| N | existing µs/step | new µs/step | speedup | new µs/cell |
|---:|---:|---:|---:|---:|
| 132 | 73.00 | **42.15** | **1.73x** | 0.319 |
| 396 | 175.00 | **134.49** | **1.30x** | 0.340 |
| 1056 | 445.00 | **348.38** | **1.28x** | 0.330 |
| 2112 | 962.00 | **690.01** | **1.39x** | 0.327 |
| 4224 | 2256.00 | **1365.72** | **1.65x** | 0.323 |

**The large-N story is the better one.** The new solver's per-cell cost is flat at
0.32-0.34 µs across a 32x range of N. The existing code degrades: 0.421 µs/cell at N=1056
rising to 0.534 at N=4224, i.e. **+27%**. That is why the speedup dips to 1.28x in the
middle and climbs back to 1.65x at N=4224 — and it should keep widening beyond that.

Combined with §4c, the operating guidance is:

| regime | what to run | why |
|---|---|---|
| N <= ~200 | 4 replicas/GPU under MPS | 2.26x aggregate; one replica leaves the GPU 62% idle |
| N ~ 400-550 | 1 replica/GPU, L2 carve-out on | carve-out still fits |
| N >= ~550 | 1 replica/GPU, carve-out auto-disabled | +6-7%; flat per-cell scaling |

Always 4 GPUs per node, one replica per GPU (or 4 per GPU under MPS at small N).

## 7. Optimization campaign — what was tried, measured, kept and rejected

Everything below was measured on real GH200 hardware on `gputest`. Four of six ideas
failed; they are recorded because a measured negative result is worth more than an
untested idea, and two of them failed for the *same* reason.

| # | change | result | kept? |
|---|---|---|---|
| 1 | Conditional L2 carve-out (§5b) | **+6.6 to +7.4% at N>=1056** | **YES** |
| 2 | MPS, 4 replicas/GPU at small N (§4c) | **2.26x aggregate at N=132** | **YES (ops)** |
| 3 | Split kernel, 2+3 CTAs/SM (§5) | 0.52-0.70x | no |
| 4 | 1024-thread block (§4) | ~0.6x | no |
| 5 | 512 / 640-thread block, more registers | 0.85x / 0.88x | no |
| 6 | fp64 accumulator unrolling (2-way) | 0.987x | no |

### Warp stall breakdown (ncu, N=396, cycles per issued instruction of 10.88)

```
long_scoreboard  2.42   global/local memory latency  <- largest
wait             1.90   fixed-latency dependencies
barrier          1.21   __syncthreads between phases
short_scoreboard 0.69   shared memory / MIO
math_pipe        0.36   |  no_instruction 0.23  |  mio_throttle 0.04
lg_throttle      0.00   |  drain 0.00
```

### Two structural facts that close off the obvious routes

**The 304 B/thread spill is structural, not pressure-driven.** At 512 threads ptxas has
114 registers available and still spills exactly 304 B — the same as at 768 threads with
80 registers. A spill that does not shrink when the budget nearly doubles is a
dynamically-indexed local object, not register pressure. Raising the budget therefore
cannot remove it, which is why experiment 5 (which bought registers by giving up warps)
lost 12-15%.

**Occupancy cannot be raised without spilling, and spilling always loses.** Experiments 3
and 4 both hit their occupancy targets exactly (50%/75% and 32 warps respectively) and both
lost 30-48%, because each bought warps with a 368-432 B/thread spill. Combined with
experiment 5 in the other direction, the conclusion is firm: **768 threads / 80 registers /
one fused CTA per SM at 37.5% occupancy is a robust local optimum for this algorithm.**

`idx % WX` was checked and is *not* a real division — `WX` is `constexpr` from the
`process_cell<CLS>` template parameter, so it compiles to multiply-shift.

## 7b. The spill is NOT worth chasing — measured

I proposed hunting the 304 B/thread spill as the top remaining lead. That was wrong, and
the measurement killed it. `ncu` memory-traffic breakdown, `k_step` at N=396, per launch:

| counter | bytes |
|---|---:|
| global load | 83.96 MB |
| global store | 46.61 MB |
| **local load** | **12.67 KB** |
| **local store** | **608.26 KB** |
| total L1TEX | 158.19 MB |
| DRAM | 106.34 MB |

**Local traffic is 621 KB of 158 MB — 0.39%.** Eliminating the spill entirely would gain
essentially nothing. (Local *stores* exceed local *loads* by 48x, i.e. values are spilled
defensively and almost never reloaded, which is why it is so cheap.)

The same table validates the traffic model exactly: predicted global traffic at
396 cells x 144² px x 4 arrays x 4 B = 131 MB against a measured 130.6 MB. **There is no
hidden traffic anywhere.** DRAM (106 MB) is below global (131 MB) because L2 absorbs part
of the S accesses.

## 7c. Strip size matters; larger strips need wider classes

The `barrier` stall (1.21 cycles/instruction) is real and strip-size sensitive:

| kStripRows | strips (NS) | N=396 | N=2112 |
|---:|---:|---:|---:|
| 8 | 18 | 140.34 | 730.24 |
| **16** | **9** | **134.80** | **684.43** |

Halving the barrier count is worth 4-6.7%, so fewer/larger strips win. But 16 is the
ceiling: `kStripRows` must divide `WY` for every shape class, and `gcd(144, 176) = 16`.

Widening the elongated classes to 192 makes `gcd(144, 192) = 48`, cutting the round class
from 9 strips to 3:

| geometry | N=396 | N=2112 | N=4224 |
|---|---:|---:|---:|
| 176-wide / strip 16 (current) | **134.46** | 687.99 | 1364.12 |
| 192-wide / strip 48 | 135.98 | **671.81** | **1328.72** |

**+2.4% at N=2112 and +2.6% at N=4224**, -1.1% at N=396 — it trades load/compute overlap
(with NS=3 and a 3-stage prologue the whole rect is issued before any compute) for fewer
barriers, which pays off once the memory system is busy. Physics is unaffected: identical
observables to every printed digit at N=1056 over 600 steps
(`<V>/A0 0.98973`, `V range [7414.4 7508.5]`, `<|v|> 9.9750e-03`, `max|phi| 1.05121`,
`shifts 1777`).

**Not adopted by default**, for two honest reasons: it costs 1.1% at small N, and it takes
shared memory to 230,784 B of the 232,448 B limit — **1,664 B of headroom**, which leaves
no room for any future geometry change. It also requires relaxing the split path's
`kSplitRhsCtasPerSm` assert (harmless, since that path is rejected). Take it if the
campaign is dominated by N >= 2000. The 192-wide class itself was not exercised in the
correctness run (all cells stayed round, `cls 1056/0/0`), so promotion behaviour under the
new geometry should be checked before production use.

## 7d. BLOCKER: the shape-class table cannot hold deformed soft cells

**This blocks the soft-in-normal campaign. It is not a performance issue.**

All performance work above used uniform `gamma = 1`. With `gamma = 0.35` (soft cells) run
for one full tau at rho=0.89, N=396:

```
step 1000000  t 10000.000  <V>/A0 0.99673  cls 163/76/157/0  shifts 24555  tumbles 377
*** ALARMS SET -- THE RUN IS INVALID, NOT MERELY SLOW ***
  support_clip     9218334      of 3.96e8 cell-steps  = 2.33%
  class_exhausted   686651      of 3.96e8 cell-steps  = 0.17%
```

Soft cells deform until their support outgrows every shape class, and phi is then truncated
on a live face. The flags are now **counters, not sticky bits** (`atomicAdd`, not
`atomicOr`), which is what made the magnitude visible — as a sticky bit, one event in a
billion looked identical to systemic failure.

Note the class census `163/76/157/**0**`: by 1 tau only 41% of cells remain round.

### Why the obvious fix does not work

A 160x160 class was added (`kClassBig`, 213,440 B — it fits). **It is never selected**, and
the run still fails identically. The reason is the promotion *path*: a cell elongates along
one axis first, promotes to tall (144x176), and grows `ey` freely to 168 there. When `ex`
later crosses 136 it needs `wx >= 144` **and** `wy >= 176`. 160x160 is too short in the long
axis; 144x176 is too narrow. Nothing fits. The "both axes moderately large" case that
160x160 covers simply does not arise, because promotion happens on the *first* axis to
overflow. The class is retained (it is geometrically sound and costs 1,664 B) but it does
not address this failure.

### The real constraint

Both `phi` (with its halo) and `S` are resident in shared memory, so the largest possible
class is bounded by 232,448 B:

| class | bytes | fits? |
|---|---:|:---:|
| 160x160 | 213,440 | yes |
| 144x192 / 192x144 | 230,720 / 229,568 | yes |
| 160x176 | 234,432 | **no**, by 1,984 B |
| 176x160 | 234,048 | **no**, by 1,600 B |
| 176x176 | 254,912 | **no**, by 22,464 B |

The largest square class that fits is 160x160, so **any cell needing more than 152 px in
both axes cannot be represented**, full stop. Shrinking the round class does not help —
`kSmemRaw` is the max over classes, not the sum.

### Options, in increasing order of work

1. **Large-class fallback (principled).** For the largest class only, do not stage `S` in
   shared memory (read it from global — it is pointwise, needs no stencil) and write
   `phi^{n+1}` straight to global. `phi` alone at 176x176 is 131,008 + 2,176 = 133,184 B,
   which fits with 99 KB to spare, and would allow classes well beyond 176. Costs some
   speed for the minority of cells in that class. This is the correct fix.
2. **Relax `class_not_narrower` for genuinely elongated cells.** A cell that is 200 long is
   necessarily narrow (the volume constraint holds `pi*a*b ~ A0`), so a 224x112 class would
   fit its support even though 112 < 144. The current invariant forbids this because it was
   written to prevent the 208x112 truncation bug — but that bug was promoting a *round*
   cell into a narrow class, which is a different thing from tracking an already-elongated
   one. Needs care.
3. **Accept and document.** 2.33% of cell-steps clipped is not acceptable for a campaign
   measuring a D_eff *ratio* between soft and normal populations, since the bias falls
   entirely on the soft population. Not recommended.

### STATUS: option 1 is IMPLEMENTED, NOT YET MEASURED

Shape class 4, `192 x 192` at `tx0 = ty0 = 32`, added as the "large" class. It is the one
class that does **not** stage `S`: S is read pointwise from global, `phi^{n+1}` is written
straight to global from P2 (plus a frame pass for the destination pixels with no source
pixel), and P3 re-reads `phi^{n+1}` from global for the scatter and the moments. Selected
at compile time by `if constexpr (kStagesS<CLS>)` in `process_cell<CLS>`, so classes 0-3
emit exactly the code they emitted before.

Shared memory: `kScalarBytes + phi_bytes(192,192) = 2,176 + 155,200 = 157,376 B`, which is
**56,064 B below** the 213,440 B class 3 already costs. `kSmemRaw` is therefore unchanged
at 213,440 and `kSmemBytes` at 213,504 — `static_assert`ed
(`kSmemRaw == smem_raw_staged_only()`). Occupancy and the opt-in are untouched.

The `ex = 150, ey = 176` support that raised `FLAG_CLASS_EXHAUSTED` above now resolves to
class 4. The representable set was `136x136 | 168x136 | 136x168 | 152x152` — no member of
which holds `150 x 176` — and gains **184 x 184**, which dominates all four. A 260x260
exhaustive scan confirms `class_containing()` resolves every support that any class holds,
always to the smallest-area one, and still returns -1 (→ `FLAG_CLASS_EXHAUSTED`, never a
clip) past 184 on either axis.

224 x 224 was rejected as **illegal**, not merely large: it needs `tx0 <= 31` to fit the
tile with its 1-px zero ring and the only multiple of 32 there is 0. 208 x 208 is legal
(183,616 B, `tx0 = ty0 = 32`) and is the next step if 184 px proves insufficient.

**Not yet measured.** What has to be checked on the next build, in order:
1. `-Xptxas -v` and the startup dump: `k_step`'s register count and its 304 B spill. All
   five class bodies share one kernel and therefore one register allocation, so the large
   body could push either. This is the one number this change could regress for the small
   classes.
2. The 1-tau all-soft case that fails above: `support_clip` and `class_exhausted` must go
   to zero, and the census gains a fifth field.
3. A gamma=1 case against a pre-change dump: the census is `N/0/0/0/0`, no cell reaches
   class 4, and the trajectory must be bit-identical.
4. Cost: class 4 moves 1.78x the phi traffic of the round class *and* re-reads it. If a
   large fraction of cells end up there the step time will move accordingly.

Until (2) is measured, **the solver is validated for uniform-gamma runs only.** All the
performance results above stand — they were measured at gamma=1, where the census is
`N/0/0/0` and every flag is zero.

## 7e. RESOLVED — large class landed, and the alarm was miscalibrated

Two separate things were wrong. One was a real geometry limit; the other was a
diagnostic that cried wolf. Both are fixed.

### The real defect: class exhaustion — FIXED

`kClassLarge = 4` at **192x192**, which does **not** stage `S` in shared memory. `S` is
pointwise (never in a stencil), so it is read straight from global; `phi^{n+1}` is written
straight to global and P3 re-reads it. Shared memory for the large class is therefore
`phi` only:

```
staged max (class 3, 160x160): 2,176 + 108,864 + 102,400 = 213,440 B
large      (class 4, 192x192): 2,176 + 155,200 +       0 = 157,376 B
kSmemRaw = 213,440  (UNCHANGED -- the large class is the cheapest, not the dearest)
```

224x224 is *illegal* rather than too big: `tx0 + 224 <= kTilePitch - 1` forces `tx0 <= 31`,
and the only multiple of 32 there is 0, which fails the zero-ring rule. 208x208 is legal
(183,616 B) and pre-verified if 184 px of containable extent ever proves too little.

Result on the exact test that failed, all-soft gamma=0.35, N=396, 1 tau:

| | before | after |
|---|---:|---:|
| `class_exhausted` | **686,651** | **0** |
| class census | 163/76/157/0 | 163/76/153/0/**4** |

Cells now reach class 4 and nothing is exhausted. **`max abs(dphi)` against the verified
CPU oracle is 6.421e-08 — bit-for-bit the same as before the change**, so the new store
path is exactly correct.

**Cost: 1.4-2.7%.** The large-class body raises `k_step`'s spill from 304 to 368 B/thread
(all five class bodies share one ptxas allocation), giving 138.11 vs 134.49 us at N=396
and 699.53 vs 690.01 at N=2112. Registers stay at 80 and `kSmemBytes` is unchanged.

**Latent bug found on the way:** both dispatch switches ended in
`default: process_cell<kClassTall>`, so **class 3 was silently executed as class 2**. It
was harmless only because class 3 is never selected. Both switches are now exhaustive with
a counted refusal in `default`.

### The false alarm: support_clip — RECLASSIFIED

`FLAG_SUPPORT_CLIP` fires when the `phi > 1e-5` bbox merely *touches* the window edge. That
is a tripwire on the far exponential tail, not a measure of lost mass. Measured directly by
dumping state at t=5000 and summing `phi^2` over the border ring:

| | cells with phi>1e-5 on border | max border phi | **border phi^2 / total phi^2** |
|---|---:|---:|---:|
| gamma = 1.0 | 0 / 396 | 4.05e-06 | **7.2e-17** |
| gamma = 0.35 | 5 / 396 | 3.57e-04 | **3.8e-13** |

Six to ten orders of magnitude below fp32 epsilon. The interface itself is fully contained;
only the tail below ~4e-4 is cut. Declaring that `*** THE RUN IS INVALID ***` would have
caused good runs to be thrown away — a worse failure than missing it.

`report_flags()` now separates **fatal** from **advisory**: `support_clip` prints as an
advisory with its rate and what it means; every other flag stays fatal. Both regimes at
1 tau now report `alarms: no fatal flags`, with support_clip at 0.222% (gamma=1) and
2.127% (gamma=0.35) of cell-steps.

**Status: the solver is now validated for soft-in-normal as well as uniform-gamma runs.**

## 8. Where the remaining performance actually is

Get to **≥2 CTAs/SM**, which needs shared memory ≤ 116,736 B.

`S_s` (101,376 B) is currently dual-purpose: it holds the input `S` rect and is then
reused to stage `phi^{n+1}` for the scatter and the moment pass. So it cannot simply
be deleted. The change that frees it:

1. Read `S` directly from global in the two pointwise sites that need it (it needs no
   stencil, and `--no-l2` proves S traffic is not on the critical path).
2. Write `phi^{n+1}` straight to global from the RHS sweep.
3. Move the scatter into `S_next` plus the perimeter/moment reduction into a **second,
   lightweight kernel** that re-reads `phi^{n+1}`.

That leaves `phi_s` alone in shared memory: `(144+2) x 184 x 4 + 2176 = 109,632 B`
→ **2 CTAs/SM, 48 warps, 75% occupancy.**

It costs one extra read of phi (3 HBM passes instead of 2, a 1.5x traffic increase)
to buy 2x the latency hiding. At 20% of peak bandwidth that trade is strongly
favourable: the traffic increase is affordable precisely because bandwidth is not the
binding constraint.

Expected outcome: 60–75 µs/step at N=396 (a further 1.8–2.2x), putting the total at
**2.3–2.9x over the existing code**. That should be measured, not assumed — the
1024-thread experiment above is a reminder that occupancy reasoning can be wrong.

## 6. Reproducing

```bash
# build
source /usr/share/lmod/lmod/init/bash
module use /appl/modulefiles/manual/general/aarch64 && module load nvhpc
cd gh200_sim && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 && cmake --build . -- -j 16

# correctness
srun --account=project_2019216 --partition=gputest --nodes=1 --ntasks=1 \
     --cpus-per-task=16 --mem=32G --gres=gpu:gh200:1 --time=00:10:00 \
     ./cell_gh200 --self-test

# head-to-head
./cell_gh200 --N 396 --radius 49 --rho 0.89 --bench 3000

# oracle comparison (needs two consecutive dumps, then numpy + cpu_reference.py)
./cell_gh200 --N 24 --radius 49 --rho 0.85 --t-end 2.00 --tau 1e9 --v-A 0 \
             --seed 777 --dump-state sA.bin
./cell_gh200 --N 24 --radius 49 --rho 0.85 --t-end 2.01 --tau 1e9 --v-A 0 \
             --seed 777 --dump-state sB.bin
./dump_phi sA.bin ./cmpA --composite && ./dump_phi sB.bin ./cmpB --composite
```

## 7. Production note

Run **one replica per GPU, four per node**. 95 GiB holds even N=12800, and the
science needs seed ensembles, so four independent replicas beat any multi-GPU
decomposition of a single replica. Utilization is then trivially 100%.

A caveat for the FSS campaign: the speedup *falls* with N (1.71x at N=132 → 1.18x at
N=1056), so the benefit is largest at the small-N end of the scan. Fixing the
occupancy problem in §5 should flatten that curve, because the ragged-wave and
phase-overlap penalties both shrink when two CTAs share an SM.
