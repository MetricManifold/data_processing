# Multi-GPU performance prompt

You are diagnosing a multi-GPU CUDA simulation that **runs slower at G=2
than G=1**. Find the cause. Do not redesign the algorithm; the
correctness has been verified separately. The work is to identify what
makes the multi-GPU step ~3× more expensive per step at production size
than the single-GPU step ÷ 2.

## Summary

`cpp/simulation/` is a CUDA phase-field cell simulator. Each cell `n`
carries a `phi_n(x,y)` field on a `TILE_T × TILE_T` (T=320) buffer. All
cells couple only through a shared global `S(x,y) = Σ phi_n²`. Per
step, three kernels run: `k_polar` (per-cell), `k_scatter_S` (writes
`phi²` into `S`), `k_evolve_l1` (reads `S` and `phi`, writes `phi_out`).
Every 8 steps, a `k_rebind` runs.

A 1D-along-y slab decomposition was added. Each rank owns a horizontal
strip of the global grid plus a halo of `H = TILE_T/2 - 1 = 159` rows
on each side. After scatter, neighbour ranks exchange a `2H × Nx` band
via NCCL `Send/Recv`, then a `launch_halo_add` kernel folds the
neighbour's contribution into the local `S`. Cells migrate between
ranks at rebind cadence (every 8 steps) when their COM crosses a slab
boundary.

The implementation lives in:

- [src/sim.cu](cpp/simulation/src/sim.cu) — `Simulation::step_pre_reduce`,
  `Simulation::step_post_reduce`, `Simulation::migrate_cells`, and
  `run_multi_gpu` (the orchestrator).
- [src/kernels.cu](cpp/simulation/src/kernels.cu) — `k_scatter_S`,
  `k_evolve_l1`, `k_rebind`, `launch_halo_add`,
  classify/pack/unpack/compact migration kernels.
- [src/multi_gpu.cu](cpp/simulation/src/multi_gpu.cu) — NCCL wrappers.
- [include/types.cuh](cpp/simulation/include/types.cuh) — `slab_local_y`
  helper.

## Measurements (Narval A100 SXM-NV4, NCCL 2.29.7, NCCL_P2P_LEVEL=NVL)

All numbers from `B1.5` binary (`build_mg`), v_A=0.01, ρ=0.90, R=49.

| Config | wall | ms/step |
|---|---:|---:|
| N=1152 G=1 | 0.504 s | 1.008 |
| N=1152 G=2 | 1.188 s | **2.375** |
| N=4608 G=1 | 1.047 s | 5.233 |
| N=4608 G=2 | 1.215 s | **6.077** |

Ideal G=2 would be ~half of G=1: 0.504 ms/step at N=1152, 2.6 ms/step
at N=4608. Actual is significantly worse. **The constant gap of about
0.85 ms/step at N=4608 grows to a 1.4 ms/step gap at N=1152**, meaning
there is both a constant overhead floor AND a per-cell-cost difference.

## Likely contributors (you should rank by impact)

1. **Multi-thread orchestrator + NCCL group calls.** Each step issues
   `start_barrier.wait()`, then per-rank thread does `step_pre_reduce`,
   `mg_group_start/end` for halo send/recv, two `launch_halo_add`
   kernels, `step_post_reduce`, `end_barrier.wait()`. Two `cudaEventSync`
   -level barriers at minimum.
2. **Loss of CUDA Graph capture.** Single-GPU path
   ([sim.cu:`Simulation::step()`](cpp/simulation/src/sim.cu)) captures
   `polar + scatter + evolve` into a graph and replays it. Multi-GPU path
   uses individual launches and cannot capture across NCCL by default.
3. **Migration on rebind.** Even with all-zero counts the path runs
   `k_classify_migrants` + 3× `cudaMemcpyAsync(D2H)` + `cudaStreamSynchronize`
   + an NCCL i32 group exchange + 2× `D2H` + `cudaStreamSynchronize`
   before the fast-exit branch fires. That is two host-side
   syncs every 8 steps.
4. **Halo exchange data volume.** Per step at N=4608: each direction
   sends `2H × Nx × 4B = 318 × 6215 × 4 = ~7.9 MB`, two directions, two
   ranks, so each rank moves about 16 MB/step. NVLink-NV4 ~200 GB/s
   sustained should make this ~80 µs/rank. Should not cause a 0.85 ms
   floor by itself.
5. **`launch_halo_add` kernel** runs after the NCCL recv; small kernel
   on a contiguous `2H × Nx` strip. Probably tens of µs.
6. **Implicit per-rank serialization** if the kernel launches end up
   sharing the GPU's main scheduler queue rather than running
   concurrently. NVLink P2P doesn't necessarily mean concurrent
   kernel execution across ranks.
7. **Scatter-side atomic contention** on boundary rows. Both ranks'
   cells now write `atomicAdd` to overlapping global rows; on a single
   GPU these don't actually conflict (different memory) but maybe the
   work distribution changed and SMs see worse occupancy.

## What to deliver

1. A **profiling plan** that decomposes the 6.077 ms/step at N=4608 G=2
   into a budget: pre-reduce, halo NCCL, halo-add kernel,
   post-reduce, migration overhead, host-side barriers/syncs, anything
   else. Use `nsys` recipe and the metrics to extract.
2. **Identify the dominant cost** with the smallest experiment that can
   prove it (changing one variable at a time).
3. **One or two concrete fixes** with predicted impact. Examples
   that have been considered but not committed:
   - Replace dual `mg_send_recv_f32` with a single `ncclAllReduce` on
     each band (NCCL is faster on reduction ops than P2P + local kernel).
   - Use CUDA Graph capture for multi-GPU, including the NCCL collectives
     (NCCL >=2.11 supports it).
   - Move migration off the per-rebind synchronous path: do classify
     non-blocking and only sync if any rank actually has migrants
     (would need a lightweight all-reduce of `(n_up + n_down)` int).
   - Eliminate one of the host-side barriers if the data dependency
     allows.
4. **Honest assessment** of whether the architecture can ever reach a
   1.5×+ speedup at G=2 with a 1D slab + halo. If not, what would.

## Constraints

- Single-GPU performance must remain bit-identical to current.
- The captured-graph fast path on G=1 ([sim.cu:`Simulation::step()`])
  is the regression baseline.
- HALO_H is fixed by `TILE_T/2-1` and cannot be reduced. We tested at
  TILE_T=320; smaller T is not on the table.
- The user's actual target is **weak scaling**: N cells/GPU constant as
  G grows. So we need to know whether the per-step cost stays roughly
  constant as N and G grow proportionally, not whether strong scaling
  works.

Output: written analysis with concrete next experiments. No code
required, but pseudocode for the experiments is welcome.
