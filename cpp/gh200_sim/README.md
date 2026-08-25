# FUSE-1R — GH200 phase-field cell engine

The normal path uses one CTA per cell with the active field in shared memory.
A second sparse kernel handles only cells that outgrow those classes, using the
same fixed per-cell tile through global memory. Both compute the exact
(non-lagged) interaction velocity for the current step.

Physics: Palmieri et al. 2015, *Sci Rep* **5**:11745, Eq. (S15), dimensionless
units, mobility `M = 1/2`, explicit Euler, periodic BCs, `h = dx = dy = 1`.

---

## 1. Build

```bash
source /usr/share/lmod/lmod/init/bash
module use /appl/modulefiles/manual/general/aarch64
module load nvhpc                       # nvhpc 26.3 -> nvcc CUDA 13.0.88
# alternative: export MODULEPATH=/appl/modulefiles
#              module load spack/aarch64/v2026_03/gcc/15.2.0/cuda/13.0.2

mkdir -p build && cd build
cmake /path/to/cpp/gh200_sim -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90        # or 90a for Hopper-only PTX
cmake --build . -- -j 8
```

The default layout uses a `288 x 288` per-cell tile, a `224 x 224` largest
shared-phi class, and a rare `286 x 286` global-memory fallback at offset
`(1,1)`. The outer tile pixels remain the stencil's zero ring. Configure
`-DPF_EXTENDED_SUPPORT_LAYOUT=OFF` only for the compact `256/208/254` layout.

Products: `cell_gh200` (solver) and `dump_phi` (state-dump converter).

**Gate the build on the register report.** `-Xptxas -v` is on by default; the
performance model assumes the 85-register budget holds with **zero spill
stores**:

```bash
cmake --build . -- -j 8 2>&1 | grep -E 'k_step|spill'
```

`cell_gh200` also prints `k_step`'s register count and local-memory (spill) size
at startup, so this is checkable without the build log.

Never `-use_fast_math`. `-fmad=true` is passed explicitly so the FMA-contraction
decisions the golden-hash regression depends on are recorded in the build rather
than inherited from a default. Host code is portable C++17 for aarch64
Neoverse-V2: no x86 intrinsics, no `__builtin_ia32_*`, no SSE/AVX, and no
`-march=native`.

## 2. Run

```bash
# production case
./cell_gh200 --N 288 --radius 49 --rho 0.90 --dt 0.01 --t-end 100 \
             --lambda 7 --kappa 10 --mu 1 --xi 1500 --tau 1e4 --v-A 1e-2 \
             --gamma 1.0 --seed 1234 --out traj.csv

# soft-in-normal contrast (the science): 20% of cells at gamma = 0.35
./cell_gh200 --N 288 --gamma 1.0 --gamma-cancer 0.35 --cancer-fraction 0.2

# validation gates, no GPU work needed beyond init
./cell_gh200 --self-test --t-end 0

# benchmark (no I/O)
./cell_gh200 --N 288 --radius 49 --rho 0.89 --bench 5000
```

The domain is square by construction: `L = ceil(sqrt(N*A0/rho))`, and `Nx == Ny`
and `dx == dy == 1` are asserted at startup — the run refuses to start otherwise,
with a message explaining why (the 9-point Laplacian, the `1/(2h)` gradients and
the `dA = 1` quadrature are all hard-coded for `h = 1`).

### Validation-only dual centroids

`--dual-centroid-out <path>` writes independently recomputed `phi`- and
`phi^2`-weighted periodic centroids on the same frames as `--out`. It is an
opt-in validation sidecar, not a production observable; absent the option,
there is no added allocation, launch, or file operation and the legacy
trajectory formatter is unchanged. Exact definitions, output columns,
overhead scope, and the remaining GH200 gates are in
[`DUAL_CENTROID_VALIDATION.md`](DUAL_CENTROID_VALIDATION.md).

### Benchmarking on Roihu

`gputest` is free but serial (`MaxJobs=1`, `MaxSubmitJobs=2`, 15:00 walltime), so
**sweep the whole A/B matrix inside one binary invocation in one job** — never as
a job array.

```bash
srun --account=project_2019216 --partition=gputest \
     --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G \
     --gres=gpu:gh200:1 --time=00:15:00 \
     bash -c '
       for f in "" "--split" "--no-graph" "--morton" "--no-l2" "--morton --no-l2"; do
         echo "=== $f ==="; ./cell_gh200 --N 288 --radius 49 --rho 0.89 --bench 3000 $f
       done'
```

Use `--account=project_2019216`: `project_2017848` has `MaxSubmitJobs=0` on
`gputest`. Never `gpumedium`/`gpularge` — they consume the 400k GPU BU
allocation. Always pass `--mem`. Never freestyle sbatch parameters between
related runs.

Multi-GPU is **one independent replica per GH200**, four per node, each pinned to
its own Grace NUMA domain: zero inter-GPU traffic, no NCCL, no halo exchange,
perfect scaling. There is no domain decomposition of a single replica, and there
must not be: a cell straddling a slab boundary no longer fits in one CTA, which
kills the CTA-local exact `Ix/Iy` reduction the whole design rests on.

## 3. Comparing against the CPU oracle

`cpp/simulation/tests/python/cpu_reference.py` is the verified oracle. Export a
state and convert it:

```bash
./cell_gh200 --N 2 --radius 20 --rho 0.05 --t-end 1.0 --tau 1e9 \
             --dump-state state.bin
./dump_phi state.bin ./cmp --composite
```

`./cmp` then holds `meta.json`, `cells.csv`, `origins.npy`, `classes.npy`,
`phi_%05d.npy` (and `phi_all.npy` when every cell shares a shape class). Paint
each rect into a full periodic domain exactly as
`cpu_reference.cells_from_checkpoint` does:

```python
import json, numpy as np
from cpu_reference import CPUParams, CPUCell, integrate

m   = json.load(open("cmp/meta.json"))
org = np.load("cmp/origins.npy")
L   = m["Nx"]
cells = []
for n in range(m["num_cells"]):
    rect = np.load(f"cmp/phi_{n:05d}.npy").astype(np.float64)   # (wy, wx)
    gx0, gy0 = org[n]
    full = np.zeros((L, L))
    ys = (gy0 + np.arange(rect.shape[0])) % L
    xs = (gx0 + np.arange(rect.shape[1])) % L
    full[np.ix_(ys, xs)] = rect
    cells.append(CPUCell(phi=full))
p = CPUParams(Nx=L, Ny=L, dt=m["dt"], lambd=m["lambda"], kappa=m["kappa"],
              mu=m["mu"], xi=m["xi"], target_radius=m["target_radius"])
ref = integrate(cells, p, 1)      # oracle advances one step from the same state
```

**Do not expect bit-exactness against `tests/python/fixtures/cpu_ref_2tau`.**
Q5.27 fixed-point `S`, fp64 reductions, Philox, and the removal of the
`Soth < 0` clamp all move low-order bits. Re-baseline those fixtures in a
deliberate, documented commit and treat *"violates an invariant"* as the bug
signal, never *"differs from the old fixture"*. `--dump-state` writes `Ix`, `Iy`,
`V`, `Cx`, `Cy` and the bbox alongside `phi` precisely so the comparison can be
made term by term instead of on the field alone.

## 4. Physics, and where it lives

Every coefficient in the tree comes from `include/params.cuh`. Nothing else
spells out 30, 60 or 120.

```
dphi_n/dt = gamma_n * lap(phi_n)
          - bulk_coeff(lambda) * gamma_n * phi(1-phi)(1-2phi)     # 30 gamma/l^2
          + volume_coeff(mu,A0) * (A0 - V_n) * phi                # 2 mu/A0
          - interaction_coeff(kappa,lambda) * phi_n * S_other      # 60 kappa/l^2
          - (vx dphi/dx + vy dphi/dy)

v_n = v_A p_hat_n + motility_coeff(kappa,xi,lambda)
                    * integral(phi_n grad(phi_n) S_other dA)       # 60 kappa/(xi l^2)
```

The repulsion coefficient is **60**`*kappa/lambda^2`, not 30: Eq. (10)'s
interaction free energy sums over **ordered** pairs, so
`dF_int/dphi_n = 120 kappa/lambda^2`, and `x(-1/2)` gives 60. A factor-of-2 error
here survived eight months in the previous codebase. The invariant that catches
it is convention-free:

```
interaction_coeff / motility_coeff == xi
```

`motility_coeff` is *defined* as `interaction_coeff / xi`, so the invariant holds
structurally; it is additionally `static_assert`ed at four parameter points
(1e-12 relative) and re-checked at startup against the actual CLI values.
`--self-test` prints all four coefficients and the ratio.

Run-and-tumble uses `p = -expm1(-dt/tau)`, computed **in double on the host** and
passed to the kernel as a `double`. `1.0f - expf(-dt/tau)` in fp32 returns
`1.013279e-06` instead of `9.999995e-07` at `tau = 1e4, dt = 0.01` — a +1.33%
bias that turns `tau_eff = 10000` into 9869. `--self-test` gate 4a demonstrates
both numbers. The polarity is stored as an angle and `p_hat = (cos, sin)` is
recomputed every step, so `|p_hat| == 1` by construction with no renormalisation
drift and no RNG state to checkpoint (Philox4x32-10 keyed on
`(seed, global_id, step)`).

Per-cell `gamma` and per-cell `v_A` are real fields of `CellState`, not globals —
`--gamma-cancer` / `--cancer-fraction` set the stiffness contrast and
`--v-A-sigma` sets lognormal motility disorder.

### No silent clamps

`S_other` is computed as an **exact uint32 subtraction** `q_S - q_of(phi_n)`.
Step *n* reads exactly the phi buffer that step *n−1* scattered from, with the
same `q_of` and the same rect→global map, so the result is provably
non-negative — the old `if (Soth < 0) Soth = 0;` is deleted, not ported. If the
subtraction ever does go negative that is a broken invariant, so it is
**counted** in `FLAG_S_NEGATIVE` and reported, never floored away. Q5.27
overflow, the `phi^2 <= 4` bound, non-finite `phi`, and non-positive carried
`V` are likewise fatal. Touching a class boundary is advisory; a support that
outgrows the shared classes moves into the fixed-tile fallback.

## 5. Design rationale (what the code is doing and why)

### 5.1 Fixed windows, zero-cost rebind

The default phi pool is `[N][288][288]` floats. The rect window inside a tile is
fixed per shape class:

| class | WX x WY | TX0, TY0 | smem/CTA | staged in smem | holds extent up to |
|---|---|---|---:|---|---|
| 0 round | 144 x 144 | 64, 64 | 173,888 | phi + S | 136 x 136 |
| 1 wide  | 176 x 144 | 32, 64 | 211,008 | phi + S | 168 x 136 |
| 2 tall  | 144 x 176 | 64, 32 | 211,776 | phi + S | 136 x 168 |
| 3 big   | 160 x 160 | 32, 32 | 213,440 | phi + S | 152 x 152 |
| 4 large | 224 x 224 | 32, 32 | **211,904** | **phi only** | 216 x 216 |
| 5 fallback | 286 x 286 | 1, 1 | **2,176** | global phi + S | 278 x 278 guarded; 286 physical |

The **short** side of the elongated classes is 144, never less: a class change
must not shrink the window on the axis that did *not* trigger it. The
destination is therefore chosen by **containment on both axes**
(`class_containing()` in `kernels.cu`), not by comparing `ex` to `ey` — which
also makes wide↔tall a legal one-step transition. A support beyond class 4
enters class 5. If it later consumes the ordinary eight-pixel margin, output is
retained and the condition is reported for review; boundary-touching dynamics
may be clipped and are not claimed to be lossless.

Class 4 is the one that breaks the shared-memory ceiling, and it does it by
**not staging S**. Every other class keeps both `phi_s` (with its halo ring) and
`S_s` resident, which caps the largest class at 160 x 160 — and 160 x 160 is
never actually selected, because promotion happens on the *first* axis to
overflow, so a deforming cell needs `wx >= 144` **and** `wy >= 176`
simultaneously and 176 x 176 needs 254,912 B (see RESULTS.md 7d). For class 4
only, `process_cell<CLS>` compiles a second body — selected by
`if constexpr (kStagesS<CLS>)`, never by a runtime branch — in which

* S is read **pointwise from global** (it never enters a stencil, so no staging
  is required for correctness; the same word, same rotation slot and same
  rect→global map, so the Q5.27 exact-subtraction argument is unchanged);
* `phi^{n+1}` is written **straight to global** from the P2 sweep, through the
  same shifted-store map P3 applies in the staged path, plus a frame pass that
  zeroes the destination pixels with no source pixel — together they write every
  destination pixel exactly once;
* P3 **re-reads `phi^{n+1}` from global** for the S scatter and the
  V/Cx/Cy/perim/bbox moments. That extra HBM read is what the large path costs.

Class 4 remains below the 213,440 B class-3 maximum, so `kSmemRaw` and the normal
path's 1 CTA/SM occupancy do not move. Class 5 stages neither field; its
286-pixel interior leaves rows and columns 0 and 287 as the stencil ring and
uses the existing tile allocation rather than increasing it.

Recentring is applied by *reading shared memory at a shifted index* during the
store, so a rebind costs **0 extra HBM traffic** and this invariant is
unconditional:

> **I1** — outside the current class's window a tile is exactly `0.0f`.

I1 is what lets the CTA **synthesise** the 1-pixel stencil ring (580 floats)
instead of loading a halo: phi is read exactly `WX*WY` per cell per step. The
only place a tile is ever zeroed is a genuine shape-class change, where the new
window does not contain the old one.

Rect sizing (`--self-test` gate 5 recomputes all of it): with
`k = sqrt(7.5)/lambda = 0.391230`, the phi tail decays as `exp(-2kd)` and phi²
as `exp(-4kd)`. A jammed cell's hexagonal Voronoi circumradius is
`sqrt(A0/(3 sqrt3/2)) = 53.88` px; phi² drops below half a Q5.27 quantum 12.40 px
past that; plus 1 px of stencil gives a required half-width of 67.3. `hw = 72`
clears it with 4.72 px of drift margin, and gate 5 checks that half-width on
**both axes of every class**, not just the round one. (The frequently quoted
`exp(-0.404 d)` tail belongs to a different profile normalisation and is wrong
by 1.94x here, which is what made W=160 look necessary.)

### 5.2 The step, in phases

| phase | what |
|---|---|
| C | grid-stride non-temporal zero of `S[(step+2)%3]`, issued first so it retires under the first cell's load latency |
| P0 | thread 0: tumble, coefficients, recentring shift, shape-class hysteresis; all threads: the zero ring |
| P1 | 3-stage `cp.async` strip pipeline (16 rows/strip) + the exact `Ix/Iy` integral over **this** step's phi and S |
| P1b | fp64 warp butterfly → 24 slots → fixed-order serial sum → `v_n` |
| P2 | RHS sweep, 3x3 rolling window in registers (3 shared reads/row, not 9); `phi^{n+1}` staged back into `S_s` |
| P3 | shifted store (`st.global.cs`) + fused S scatter for step *n+1* + `V`, `Cx`, `Cy`, bbox for step *n+1* |
| P3b | fp64 fixed-order reduce + integer bbox/`|phi|max` reduce; thread 0 writes `CellState` |

Four things are free in P3: `V(phi^{n+1})` is accumulated during the store that
was happening anyway (the volume reduction pass ceases to exist, and `V` is
consistent with the field actually stored); the S scatter for the next step is
fused into the store (`if (q)` skips zero-quantum pixels, and adding 0 is a
no-op, so the skip is bit-exact); the recentring shift is a shared-memory index
offset; the moments and bbox are byproducts.

Overwriting `S_s[y][x]` immediately after reading it is safe because S is used
**pointwise**, never in a stencil, so no other thread reads that slot.

For the **large class** (4) the phases shift: there is no `S_s`, so P1 stages
phi only, P2 reads S from global and writes `phi^{n+1}` straight to global with
a plain store (not `st.global.cs` — P3 re-reads those bytes one `__syncthreads()`
later, so evict-first is the wrong hint), and P3 re-reads the stored frame
instead of the staged rect. The class-change tile zeroing moves ahead of P2
there, because in that path P2 *is* the store. Everything else — the physics,
the reduction order, the flags, the hysteresis — is the same code.

For the **fallback class** (5), `k_step` deliberately skips the cell and
`k_step_fallback` processes it immediately afterward in the same stream. The
input-parity `cls_written` value selects cells, preventing a class-4 cell that
has just promoted from being advanced twice. Phi and S are read from global
memory; the equations, run-and-tumble key, S rotation/scatter, moments, bbox,
and checkpoint representation are unchanged.

### 5.3 Shared memory

```
red_s   double[24][8]                          1536 B  (slots 4..7 aliased as int[8])
bcast_s 128 words                               512 B
        (reserved)                              128 B
phi_s   float (WY+2) x phi_pitch(WX)
S_s     uint32 WY x WX               (absent for the large class)
```

| class | phi_s | S_s | total |
|---|---|---|---|
| 0 | 146 x 152 x 4 = 88,768 | 82,944 | 173,888 |
| 1 | 146 x 184 x 4 = 107,456 | 101,376 | 211,008 |
| 2 | 178 x 152 x 4 = 108,224 | 101,376 | 211,776 |
| 3 | 162 x 168 x 4 = 108,864 | 102,400 | **213,440** |
| 4 | 226 x 232 x 4 = 209,728 | — | 211,904 |
| 5 | — | — | 2,176 scalar/reduction region |

`kSmemRaw` is the **max** over classes, not the sum: 213,440 B, set by class 3.
Requested (rounded to 128 B): **213,504 B**, opted in with
`cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`. The
sm_90 per-**block** opt-in maximum is **232,448 B**
(`cudaDevAttrMaxSharedMemoryPerBlockOptin`); 233,472 B is the per-**SM** figure
and must not be budgeted against. That leaves 18,944 B of deliberate margin at
91.9% of the true cap, `static_assert`ed in `params.cuh`.

Note the last row: the largest *window* is the **cheapest** class after the
round one, because it is the only one that does not stage S. That is the whole
trick, and `params.cuh` `static_assert`s that adding it did not move `kSmemRaw`.

`phi_s` carries a **4-float** left pad rather than the 1 float the stencil needs:
16 B `cp.async` requires every copied row's destination to be 16 B aligned, and a
1-float pad leaves it at +4 B. That is the only place this implementation's
shared-memory budget exceeds the paper design (by 3,328 B).

Every warp-wide shared access is 32 consecutive `x` at one `y`, so it hits 32
distinct banks and is conflict-free regardless of pitch.

This is 1 CTA/SM, deliberately. Little's law wants 19.7 KB in flight per SM; the
3-stage pipeline carrying 16 phi rows + 16 S rows per stage holds 55.3 KB — 2.8x
the requirement. Warp count is **not** the source of memory-level parallelism
here, the async pipeline is.

### 5.4 Q5.27 and the triple buffer

`S = sum_m phi_m^2` is stored as `q = round(phi^2 * 2^27)`: range `[0, 32)`,
quantum 7.45e-9, per-contribution clamp at `phi^2 <= 4` so eight saturated
contributors are still needed to wrap. Integer addition is associative and
commutative, so the scatter's ordering is irrelevant to the result — that is what
makes a *deterministic* single-pass scatter possible at all. (bf16 would put
~2e-3 relative error on `S_other`, whose integral **is** the dominant velocity
term, and CUDA has no 16-bit integer `atomicAdd`.)

Three buffers, rotating: step *n* reads `S[n%3]`, scatters into `S[(n+1)%3]`,
zero-fills `S[(n+2)%3]`. Two are provably insufficient — step *m* would have to
zero the buffer it is reading. There is no `cudaMemsetAsync` anywhere in the step
loop.

### 5.5 One launch per step

The work cursor is a monotone 64-bit counter with **two slots** alternating with
the phi parity: the kernel resets the slot the *next* launch will use, so there
is no reset kernel and the kernel boundary supplies the only ordering needed. The
device step counter works the same way (read the slot the previous launch wrote,
publish into the other), so the value every CTA sees is stable for the whole
kernel and cadence predicates are device-side.

That makes the step loop launch-invariant, so a **6-step CUDA graph body**
(`lcm(2 phi parities, 3 S rotation slots)`) can be captured once and replayed
forever, with each node baking its own pointers, parities and
`cudaLaunchAttributeAccessPolicyWindow`. At ~30 µs/step a 2–4 µs launch would be
a 7–13% tax. `--no-graph` disables it for A/B measurement.

Cooperative launch is **not** used — not because of shared-memory limits (132
persistent CTAs at 192 KB each is legal on sm_90) but because every reduction
here is an integral over ONE cell, which fits in ONE CTA, so `grid.sync()` would
have nothing to synchronise.

### 5.6 Determinism

| quantity | precision | order |
|---|---|---|
| `Ix, Iy` | fp64 | fixed per-thread pixel order → warp butterfly → 24-slot serial ascending sum |
| `V, Cx, Cy, perim` | fp64 | same |
| bbox, `|phi|max` | int32 | min/max: associative, commutative, exact |
| S accumulation | uint32 Q5.27 | order-irrelevant |

The thread→pixel map and the reduction tree depend only on `(WX, WY, blockDim)`,
all compile-time constants, so bit-reproducibility survives a change of SM count,
of cell ordering, and of work-cursor interleaving. It does **not** survive a
change of nvcc version, `--use_fast_math`, or FMA-contraction decisions: record
`nvcc --version` and the `(WX, WY, blockDim)` triple in every checkpoint and keep
a 1000-step golden hash regression.

`blockDim` being in that triple is why `--split` (below) is a **different
trajectory**, not a faster route to the same bits.

### 5.7 `--split`: the higher-occupancy path

`k_step` is pinned to **1 CTA/SM by two independent limits**, and Nsight Compute
confirms both: `launch__occupancy_limit_shared_mem = 1` (211,840 of 233,472 B)
*and* `launch__occupancy_limit_registers = 1` (80 regs × 768 threads = 61,440 of
65,536). Freeing only one of them changes nothing. `--split` frees both:

| | fused `k_step` | `--split` |
|---|---|---|
| kernels/step | 1 | 2 (`k_step_rhs`, `k_step_post`) |
| threads/CTA | 768 | 512 |
| smem/CTA | 211,840 B | 110,464 B (rhs) / 1,024 B static (post) |
| reg budget | 85 (`__launch_bounds__(768,1)`) | 64 (rhs), 42 (post) |
| target CTAs/SM | 1 | 2 (rhs), 3 (post) |
| target occupancy | 37.5% | 50% (rhs), 75% (post) |
| HBM passes over phi | 2 | 3 |

`k_step_rhs` keeps P0/P1/P1b/P2 and drops `S_s` from shared memory: `S` needs no
stencil, so it is read pointwise from global at the two sites that use it, and
`phi^{n+1}` is stored straight to global through the same shifted-store /
recentring map the fused P3 applies. `k_step_post` re-reads `phi^{n+1}` (the one
extra HBM pass) and does the `S` scatter, the moments, the bbox and the fp64
reductions.

The S triple-buffer rotation and the exact `q_S - q_of(phi_n)` subtraction are
unchanged: `k_step_post` scatters from exactly the buffer `k_step_rhs` wrote, and
the next step's `k_step_rhs` reads exactly that buffer, so no parity fix-up was
needed. All eight sticky flags fire identically; there is no new clamp.

Two things to know before using it:

* **It is not bit-comparable with the fused path.** 512 threads means the fp64
  reductions run over 16 warp slots instead of 24, so the two paths differ in the
  last ulp and drift apart. Each is individually bitwise reproducible. Never
  compare a `--split` dump against a fused dump step for step.
* **`perim` is the one quantity computed differently** (not merely rounded
  differently): the fused P3 takes ∇φ from the staged *source* rect, so on the
  window border it can use a pixel that is never stored; `--split` uses the
  stored frame with the I1 zero ring. Full-moment steps only, 1-px border only,
  ~1e-5 absolute against a perimeter of order 2πR.

Startup prints the achieved registers, spill bytes and CTAs/SM for each kernel
from `cudaFuncGetAttributes` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
and warns if the target was missed — check that, and the `-Xptxas -v` spill line,
before believing any `--bench` number. Forcing occupancy by spilling has already
been measured *slower* on this kernel (the 1024-thread experiment: 64 regs,
432 B stack, ~40% slower everywhere). `kSplitRhsCtasPerSm = 1` in `params.cuh`
restores the unconstrained register allocation for an A/B.

CUDA graph capture is fully supported: the 6-step body just holds 12 kernel nodes
instead of 6, since the period comes from the argument rotation, not the launch
count.

### 5.8 L2 and Morton

`cudaLimitPersistingL2CacheSize` is set from the device's *queried*
`persistingL2CacheMaxSize` (the two calibration sources for this machine disagree
on whether L2 is 50 or 60 MB, so the code asks and never assumes). The window
covers the read buffer, plus the scatter buffer when the pair is contiguous; the
clear-ahead buffer is write-only streaming and is deliberately **not** pinned —
that is what leaves L2 room for phi. phi stores use `st.global.cs` (evict-first)
so 48 MB/step of single-use phi cannot evict the pinned S.

`--morton` sorts the visit order by the Morton code of each cell's COM every 6
steps (aligned to the graph body) so concurrently resident CTAs touch a compact
region of S. It is worth ~14% at N=12800 where L2 pinning is impossible and
essentially nothing at N=288, so it is **off by default** and is an A/B flag.
The sort is a single-CTA bitonic sort over `(morton << 32 | index)` in shared
memory, which caps it at N ≤ 16384 (it falls back to identity order above that,
with a warning).

## 6. Validation gates

`--self-test` runs gates 1–5; the rest need a run.

1. `lap(x^2) == 2` exactly, McLellan weights sum to 0 exactly. *(self-test)*
2. `interaction/motility == xi`; repulsion numerator is 60 = 2x30. *(self-test,
   plus four `static_assert`s and a startup check)*
3. Interaction-velocity **sign** on a two-cell fixture at four separations: cell
   *n* left of *m* ⇒ `Ix < 0` ⇒ `vx < 0` ⇒ *n* moves away. *(self-test)*
4. `p = -expm1(-dt/tau)` exactly, with the `1-expf` bias printed for comparison;
   plus a Philox uniformity gate at `p = 1e-3`. *(self-test)*
5. Rect sizing and the shared-memory budget: the required support half-width is
   checked against **both axes of every shape class**, so a class that would
   truncate phi on promotion fails the gate. *(self-test)*
6. Invariant I1: `max|phi|` outside the window `== 0.0f` for every tile.
   *(`--strict`)*
7. Carried `V` vs recomputed `V` agrees to 1e-6 relative. *(`--strict`)*
8. `max(S) < 32`; all `d_flags` zero. *(`--strict`, and reported at exit always)*
9. Determinism: two runs bit-identical; a 1-CTA run bit-identical to a 132-CTA
   run (proves the reduction order is SM-count independent); restart across a
   checkpoint bit-identical. *(compare `--dump-state` outputs)*
10. `red.global.add.u32` microbenchmark **before** trusting any headline: the
    fused scatter issues 7.4 M atomics/step at N=288 and 265 M at N=12800. If
    sustained atomic bandwidth is a small fraction of read bandwidth, the
    single-scatter premise is what breaks, not the rest of the design.
    (`tools/gh200_probe.cu` already measures 385 Gatomic/s L2-resident,
    244 Gatomic/s at L=4096, i.e. atomics are not the bottleneck.)

```bash
./cell_gh200 --N 288 --t-end 10 --strict --verify-every 512
```

## 7. Expected performance, and what to measure

Traffic at N=288, R=49, rho=0.89 (L=1563, P=1568, W=144): phi read 23.89 MB +
phi write 23.89 + S read 23.89 cold / 9.80 with Morton + S atomic 19.61 + S clear
9.80 + scalars 0.11 = **101.2 MB cold / 87.1 MB with Morton** per step.
`B_rebind = B_reduce = B_memset = 0`.

Two rooflines, and they disagree, so quote both:

* Against the paper design's 4 TB/s HBM figure: 25.3 µs cold, 21.8 µs with
  Morton; with the 1.375x wave-quantisation factor at N=288 (288/132 = 2.18 → 3
  waves = 72.7%), **28–33 µs/step**.
* Against this machine's **measured** 9-point-stencil rate of 251 Gpix/s
  (`CALIBRATION.md`, one read + one write per pixel), the same accounting gives
  **35–45 µs/step**, and anything under 25 µs is claiming to beat a measured
  roofline.

Baselines to beat, all N=288 R=49 rho=0.89, no I/O: **0.137 ms/step** on this
same GH200 with the existing 3-kernel code (`scatter_S`, `reduce_mb`, `rhs_mb`),
0.239 ms/step on an RTX 4090, 0.866 ms/step on an H100 1g.10gb MIG slice. So the
honest claim to test is **3–4x the existing GH200 code**.

Wave quantisation is irreducible at N=288 without splitting a cell (which
forfeits the exact CTA-local velocity) and it vanishes with scale: 97% at N=1152,
100% at N=12800 (97 full waves). At N=12800 the footprint is phi 6.71 GB + S
1.29 GB ≈ 8.0 GB of 95 GB, i.e. four independent replicas per node with 11x
headroom.

Verify with `ncu`, do not trust the model:

```bash
ncu --metrics dram__bytes.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
dram__sectors_read.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    ./cell_gh200 --N 288 --bench 20 --no-graph
```

`l1tex__data_bank_conflicts_pipe_lsu_mem_shared` must be ≈ 0. The shared-memory
sweep is ~12.7 accesses/px (6 in P1, 5.67 in P2, 1 in P3); at N=12800 that is
~454 µs of the ~1130 µs memory time, which is why the P2 rolling window is
mandatory and why the perimeter is computed only on `--full-moment` steps. A
naive 9-load stencil plus all eight moments every step would be ~600 µs and a
genuine co-bottleneck.

Deferred optimisations, in priority order, each measured before adoption:
per-row circular span mask on the S load (101.2 → 96.4 MB, bit-exact, +5%);
`cp.async.bulk.tensor.3d` for the phi load/store **only** (never for S, whose
periodic wrap TMA's zero-fill OOB mode cannot express); an `sx==sy==0` fast path
fusing P2 and P3 (+5%); W=128 after a long-run diff against W=192 (−21%).

## 8. Files

| file | role |
|---|---|
| `include/params.cuh` | every physics coefficient, tile/rect geometry, Q5.27, the flag enum, startup validation |
| `include/kernels.cuh` | `CellState`, `StepArgs`, Philox, dump format, kernel declarations |
| `include/sim.cuh` | host-side `Sim` interface |
| `include/validation_centroid.cuh` | validation sidecar record and read-only launch interface |
| `src/kernels.cu` | `k_step` and the init/verify/observability kernels |
| `src/validation_centroid.cu` | opt-in `phi`/`phi^2` sampling-frame reduction |
| `src/sim.cu` | allocation, init, launch sequence, L2 policy, graph capture, dump |
| `src/main.cu` | CLI and the self-test validation gates |
| `scripts/compare_dual_centroids.py` | host-only aligned legacy/independent-`phi^2` agreement gate |
| `tools/dump_phi.cu` | `--dump-state` binary → `.npy` + CSV for the CPU oracle |
| `tools/gh200_probe.cu` | the hardware probe behind `CALIBRATION.md` |
| `CALIBRATION.md` | measured GH200/Roihu numbers; supersedes spec sheets |
