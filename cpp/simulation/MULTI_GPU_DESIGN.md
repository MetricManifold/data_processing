# Multi-GPU Plan for cell_sim — Design Document

**Author:** Copilot, drafted 2026-05-08
**Status:** Proposal. No code in this document. Tear it apart before implementation.

---

## 1. Why this document exists

The current multi-GPU implementation in [src/sim.cu](src/sim.cu) (data-parallel cells, replicated `S(x,y)`, NCCL all-reduce per step) was a quick first pass. Measured on Narval 4× A100 NVLink-NV4 with the threaded orchestrator fix:

| Config | ms/step | vs G=1 |
|---|---:|---:|
| n4608 g1 | 5.22 | 1.00× |
| n4608 g2 | 5.20 | 1.00× (no speedup) |
| n4608 g4 | **15.40** | **0.34× (3× slower)** |

The G=4 case shows the all-reduce is taking ~14 ms — far above what NVLink should permit. NCCL_DEBUG output revealed NCCL chose the InfiniBand network plugin over NVLink P2P even though all four GPUs are on one node with NV4 links between every pair. A pending sbatch (`60640677`) tests whether `NCCL_P2P_LEVEL=NVL` fixes that. **That result is independent of this design** — even if the env var recovers ~3 ms back, the architecture below addresses the larger weak-scaling problem.

The user's actual scaling target is weak-scaling:
> "If it takes 3 hours for 1600 cells, scale to 6400 cells across 4 GPUs in approximately the same time."

The current architecture cannot satisfy that target regardless of NCCL tuning, because the all-reduce buffer scales with `Nx·Ny ∝ N_cells/ρ`, while per-rank compute is fixed at `(N_cells / G) · TILE_AREA`. Doubling N at fixed G doubles both, but doubling N at fixed cells/rank (the weak-scaling case the user wants) leaves per-rank compute flat while doubling the all-reduce. Asymptotically, communication dominates.

A different decomposition is needed.

---

## 2. What the simulation actually requires

Reading the kernels in [src/kernels.cu](src/kernels.cu) without assumptions:

| Kernel | Reads | Writes | Per-cell or grid? |
|---|---|---|---|
| `k_polar` | `theta_arr`, `rng_states` | `theta_arr`, `polar_x`, `polar_y` | per-cell, no spatial extent |
| `k_scatter_S` | `phi[n]`, `origin[n]`, `rect[n]` | `S` (atomic add into pixel `(gx0+lx) mod L`) | per-cell tile, atomic scatter |
| `k_evolve_l1` reduce pass | `phi[n]`, `S` (at every pixel of cell n's tile) | per-cell scalars | per-cell tile |
| `k_evolve_l1` rhs pass | `phi[n]`, `S`, broadcast scalars | `phi_out[n]` | per-cell tile |
| `k_rebind` | `phi[n]`, second moments | `phi_out[n]` (shifted), `origin[n]`, `rect[n]` | per-cell tile, may also shift global coords |

Crucial property: **each cell's kernel work touches only its own `phi` tile and the `Nx·Ny` global pixels covered by that tile**. Cell `n` reads `S` at `(gx0+lx, gy0+ly) mod L` for `(lx, ly) ∈ [0, TILE_T)²`. With `TILE_T = 320`, that's a 320 × 320 patch of the global grid — a small window.

Cell-to-cell coupling is **only through `S`**. There is no neighbour list. Two cells "interact" if and only if their tiles overlap a common pixel.

This is the property that makes domain decomposition feasible: if cell `n` and cell `m` are far apart (their tiles do not overlap), they share no `S` pixels, so they could be on different ranks with no data exchange between them.

---

## 3. Numbers to keep in mind

Per the cell-simulation instructions, these are the verified production sizes at confluence ρ = 0.89, R = 49:

| N (cells) | L = ⌈√(N·π·R²/ρ)⌉ | `S` size (Nx·Ny·4B) | Phi pool (2·N·TILE_AREA·4B) |
|---:|---:|---:|---:|
| 288 | 1562 | 9.3 MB | 235 MB |
| 1152 | 3124 | 37 MB | 944 MB |
| 4608 | 6249 | 149 MB | 3.78 GB |
| **12800** *(target)* | **10412** | **414 MB** | **10.5 GB** |

NVLink-NV4 (Narval A100) sustained bandwidth ≈ 200 GB/s per pair. NVLink between H100 SXM5 ≈ 900 GB/s. Effective NCCL ring-allreduce bandwidth is roughly ¾ of pair bandwidth.

`TILE_T = 320` is the cell tile size. At N=12800 that means tiles cover `320² / L² = 0.94 %` of the domain each on average — a cell only "sees" about 1 % of `S`.

---

## 4. Candidate architectures

### Option A — Replicated `S`, all-reduce every step (current)

**What it is:** every rank holds the full `Nx·Ny` `S`. Each scatters its cells' `phi²` into its local copy. NCCL all-reduce sum across ranks. Each evolves its cells, reading from the now-identical `S`.

**Pros**
- Already implemented and correct.
- Zero cell migration logic needed.
- Memory cost is the same as single-GPU on every rank for `S`; phi pool divides by G.

**Cons**
- All-reduce buffer is `O(Nx·Ny)` per step. At N=12800 that is **414 MB** on every step — even at NVLink's theoretical 800 GB/s it is ~0.5 ms; on Narval's 200 GB/s effective it is ~2 ms; on PCIe fallback (what we are seeing) it is ~16 ms.
- `S` storage is replicated, not divided — VRAM doesn't actually scale with G for the largest buffer.
- Worst at exactly the case we care about (large N, small per-rank work). Strong scaling collapses; weak scaling has a hard ceiling.

**Best case if perfectly tuned (NVLink, GPUDirect, optimal NCCL):** at N=12800 G=4, allreduce ~2 ms, per-rank compute ~12 ms → ~85 % efficiency. **At N=1152 G=4, allreduce ~0.2 ms vs per-rank compute ~0.25 ms → 50 % efficiency at best.** The architecture inherently caps strong scaling.

---

### Option B — Spatial domain decomposition with cell ownership and tile overlap (the user's proposal)

**What it is:** partition the global `Nx × Ny` grid into G slabs (1D for now: each rank owns a horizontal strip of height `Ny / G`). Each rank stores only its own slab of `S` (size `Nx · Ny / G + halo`). Cells whose **tile** lies entirely inside a slab are owned by that slab's rank. Cells whose tile **spans** a boundary need their tile available on both sides.

The user's key insight, restated precisely:
> *"once a cell becomes owned by one of the GPUs, [no per-step communication of that cell's data is needed]. The domain decomposition needs to have a sufficiently large boundary because when it owns a cell, it's not really a perfect cutoff — you have a messy boundary, and there needs to be sufficient overlap where a cell can be on one boundary or the other in totality."*

This is exactly correct. The "sufficient overlap" is the **halo**. Concretely:

- Each rank owns the rows `[y_lo, y_hi)` of `S`, plus halo rows `[y_lo - H, y_lo)` above and `[y_hi, y_hi + H)` below. Halo height H must satisfy: any cell whose COM is inside `[y_lo, y_hi)` has its full `TILE_T × TILE_T` tile within `[y_lo - H, y_hi + H)`. So `H ≥ TILE_T / 2` suffices, plus the `R/2` margin used in `k_rebind`. **`H = 192` (= TILE_T / 2 + 32 alignment slack) is the safe value.**
- A cell is "owned" by the rank whose slab contains its COM. Ownership transfers when the COM crosses a boundary — but only at rebind boundaries (every 8 steps), so migration is rare.
- `k_scatter_S` writes phi² of every owned cell into the owner's `S` slab, including into halo rows when the cell tile extends past the slab boundary.
- `k_evolve_l1` reads `S` only at pixels covered by the cell tile, all of which are inside the owner's slab + halo. **No cross-rank read of `S`.**
- After scatter (before evolve), each rank exchanges its halo rows with the neighbour ranks. Both ranks need `S` to include contributions from the other side's cells whose tiles reach into the halo.

**Halo exchange data volume per step:** for G ranks splitting `Ny` into G slabs along y, each interior boundary exchanges `Nx · H · 4B` bytes in each direction. With `H = 192` and N=12800 (`L = 10412`):

- Bytes per boundary, one direction: `10412 · 192 · 4 = 7.6 MB`.
- 2 boundaries per interior rank, 2 directions: `~30 MB/rank/step` total.
- Compare to current all-reduce: `414 MB/step` per rank.
- **~14× reduction in per-step communication.**

For G=2 there is one boundary; for G=4 (1D slab) there are 3 internal boundaries. Stays bounded.

**Pros**
- Communication scales as `O(L · H)`, not `O(L²)`. Halo grows with √N, not N.
- VRAM for `S` divides by G (rank holds slab + 2H rows ≈ `(L/G + 2H) · L · 4B`).
- Phi pool divides by G as before.
- Weak scaling is achievable: doubling N at fixed cells/rank doubles `L` linearly, halo grows linearly with `L` — but per-rank compute also grows linearly, so the ratio is preserved.
- **This is the standard approach for spatial PDE codes on multi-node clusters.** It's known to scale.

**Cons**
- New code paths for: cell migration, halo exchange, slab-aware scatter (cells near boundaries write into halo of neighbour rank).
- Cells whose tiles span a boundary contribute to both ranks' `S`, so the halo exchange has to be a sum-reduce of phi² in the halo region (not just a copy). This is exactly an all-reduce on the halo strip — but it's a small strip, not the whole grid.
- Rebind crossing a slab boundary is the trickiest case: the cell needs to be transferred from rank A to rank B, and rank B must be ready to scatter it next step. Cell migration logic is O(few cells per rebind) — bounded.
- Load balance: cells are not uniformly distributed at all times. A pile-up on one boundary (typical in confluent jamming) makes one rank carry more cells than another. We would need to either accept the imbalance or repartition slab boundaries periodically (1D allows trivial repartitioning by sliding the boundary).

**Best case (well-tuned):** at N=12800 G=4 H=192:
- Halo exchange `30 MB / step / rank` ≈ 0.15 ms on NVLink.
- Per-rank compute `(N/G) · ms/cell ≈ 3200 · ~0.005 ms ≈ 16 ms`.
- **Efficiency ≈ 95 %.** Weak-scaling target is achievable.

---

### Option C — 2D domain decomposition

Same as B but tiled in both x and y. For G = `Gx × Gy`, halo exchange perimeter scales as `(L/Gx) + (L/Gy)` per rank, total roughly `2L/√G` per rank vs `2L/G` for 1D. **2D is strictly worse than 1D for small G** (G ≤ 4) because the halo perimeter is the same length but split into more pieces with more handshakes. 2D only wins when G ≥ 16 or so. Not relevant for our 4-GPU target. Skip until later.

---

### Option D — Replicated `S` but overlapped allreduce + compute

Keep Option A but pipeline: while step k's allreduce is in flight, start step k+1's compute on local `S`. This requires the simulation to tolerate a 1-step staleness in `S` — every cell sees `S` from one step ago. This is a **physics approximation** that may be acceptable depending on cell speeds.

**Pros**
- No structural code change. Just stream/event reordering.
- Fully hides allreduce when compute > allreduce.

**Cons**
- Still has the strong-scaling ceiling of Option A.
- Physics approximation: the sim is no longer bit-equivalent to single-GPU. We would need to validate that observables (MSD, jamming transition) are unchanged by the 1-step lag.
- Worth ~30-40 % gain at best (in the regime where compute and allreduce are roughly equal), not the 4× the user wants.

---

### Option E — Per-cell ghost copies between ranks instead of halo

Each cell knows which ranks' slabs its tile overlaps. Instead of a halo strip exchange, each boundary cell pushes its phi² contribution directly to the relevant rank's `S`.

This is functionally equivalent to Option B's halo exchange, but expressed as cell-level scatter instead of strip-level reduce. It would be appealing if cells were very sparse (few cells overlap boundaries, so per-cell scatter is cheap). At ρ = 0.89 a meaningful fraction of cells touch any given y-row, so it's roughly the same cost as the strip exchange — but with worse memory access patterns. **Skip in favor of B.**

---

## 5. Recommendation: Option B, 1D slab decomposition along y

Based on:
1. The user's articulated requirement of a halo for cells that span boundaries.
2. The asymptotic scaling math: O(L·H) halo vs O(L²) all-reduce.
3. The fact that 2D doesn't help at G ≤ 4.
4. The fact that it generalises to multi-node trivially (a halo exchange between adjacent ranks is the same code whether the ranks are NVLink peers or InfiniBand-connected nodes).

**Reject** Option A as the long-term architecture. **Reject** Option D because it doesn't address the real scaling issue. **Defer** Option C until G > 4 is on the menu.

---

## 6. Implementation plan for Option B

### 6.1 New constants

```
HALO_H        = 192      // = TILE_T / 2 + 32-pixel alignment slack
G_max         = 8        // upper bound on rank count we plan to support
```

`HALO_H` is the strip thickness in pixels. Verified: with `TILE_T = 320` and `bbox_align = 16`, no cell tile can extend more than `TILE_T / 2 = 160` pixels past its COM, so `HALO_H = 192` provides a 32-pixel safety margin even when COM is exactly at a slab boundary.

### 6.2 Slab partition

For `G` ranks splitting the y axis:
```
slab_lo[g]  = ⌊g · Ny / G⌋
slab_hi[g]  = ⌊(g+1) · Ny / G⌋
slab_full_lo[g] = slab_lo[g] - HALO_H        (with periodic wrap)
slab_full_hi[g] = slab_hi[g] + HALO_H
```

Rank g stores `S` for rows `[slab_full_lo[g], slab_full_hi[g])`, total `(Ny/G + 2·HALO_H) · Nx` floats.

**At N=12800 G=4:** `(2603 + 384) · 10412 · 4B ≈ 124 MB` per rank vs the current 414 MB replicated. ~3.3× memory reduction for `S` alone.

### 6.3 Cell ownership

A cell `n` is owned by rank `g` iff `cy_n ∈ [slab_lo[g], slab_hi[g])`. Owner is updated only at rebind events (every `REBIND_EVERY = 8` steps).

Boundary-crossing cells at rebind: scan the COMs after rebind, build a migration list per rank `(cell_id, src_rank, dst_rank)`, and execute a one-time data shuffle:
- copy phi tile from src `phi_in` to dst `phi_in`
- copy origin, rect, gamma_cell, v_A_cell, polar_theta, polar_x, polar_y, tgt_radius, rng_state, velocities_x/y
- update src and dst num_cells

**Migration frequency estimate:** cells move ≤ a few pixels per rebind interval (8 dt = 0.08 time units, max v ≈ 0.5 px/t). A boundary-crossing cell only flips ownership when its COM crosses one row. With `slab_height ≈ Ny / G` and ~3000 px/slab, less than 1 cell out of 100 crosses per rebind. **~30-50 cells migrate per rebind for N=4608, G=4.** Migration cost: `TILE_AREA · 4B ≈ 410 KB / cell` × 30 ≈ 12 MB per rebind = trivial.

### 6.4 Halo exchange

After `k_scatter_S` writes phi² into local `S` (including halo strips), but before `k_evolve_l1` reads `S`, neighbour ranks exchange and sum the halo strips:

```
for each pair of adjacent ranks (g, g+1):
    g sends rows [slab_hi[g] - HALO_H, slab_hi[g] + HALO_H) to g+1
    g+1 sends rows [slab_lo[g+1] - HALO_H, slab_lo[g+1] + HALO_H) to g
    each rank adds the received strip to its local S in the same rows
```

This is **a small all-reduce on the halo strip, not a full-grid all-reduce.** Implemented as paired `ncclSend` / `ncclRecv` (NCCL P2P primitive) plus a local kernel that adds the received buffer to `S`. NCCL has an `ncclAllReduce` primitive that operates on a contiguous slice; we can either call it on the strip or do `ncclSend`/`ncclRecv` plus a tiny kernel.

**Why both rows are exchanged in each direction:** rank g's cells whose tiles extend past `slab_hi[g]` write into rows `[slab_hi[g], slab_hi[g] + HALO_H)`; those writes belong to rank g+1's interior, so g+1 must add them. Symmetrically, rank g+1's cells whose tiles extend before `slab_lo[g+1]` write into rows `[slab_lo[g+1] - HALO_H, slab_lo[g+1])` — those writes belong to rank g's interior, so g must add them.

Periodic wrapping: rank G-1 and rank 0 are also neighbours (the y axis wraps). Same exchange.

### 6.5 Kernel changes

Most kernels are unchanged because they already operate on per-cell tiles and read `S` at `(global_x, global_y)`. The only difference is each rank's `S` is now indexed in **slab-local** coordinates, not global.

Pseudocode for the index translation:
```
slab_local_y = (global_y - slab_full_lo[g]) mod Ny
S[slab_local_y * Nx + global_x]   // not S[global_y * Nx + global_x]
```

Affected kernels: `k_scatter_S`, `k_evolve_l1` reduce pass, `k_evolve_l1` rhs pass, `k_initial_velocity`. The change is mechanical.

`k_rebind` is unchanged (operates only on the cell's own tile).

### 6.6 Phases of work

| Phase | What lands | Validation |
|---|---|---|
| **B0** Spike | Single-rank build with slab indexing (G=1, slab = full grid). Confirms index translation is right with no comm. | All existing tests pass; ms/step within 5 % of pre-change. |
| **B1** G=2 happy path | Halo exchange between two ranks; cell migration logic; static slab boundary. No periodic wrap (top rank's bottom and bottom rank's top do still wrap, but no third rank to worry about). | n4608 g2 produces same checkpoint hash as g1. ms/step beats 5.20 (current g2 number). |
| **B2** G=4 + periodic | Generalize to G>2; confirm rank G-1 ↔ rank 0 wrap is correct. | n4608 g4 < 5 ms/step. n12800 g4 weak-scaling against single-GPU n3200 baseline. |
| **B3** Migration polish | Robust migration at rebind, including cells that cross by more than one slab in one step (shouldn't happen physically but defensive). | Long-run n12800 g4 = 1000 tau without mass loss or NaN. |
| **B4** Multi-node | Replace intra-node halo exchange with NCCL across nodes. Should be a no-op since NCCL handles transport. | n51200 g16 across 4 nodes. |

Each phase is a separate commit. Don't proceed to the next until tests are green and ms/step is in range.

### 6.7 What goes in the source tree

- `include/multi_gpu.cuh` — keep, extend with `MgSlab` struct (slab bounds, halo extents, neighbour rank ids).
- `src/multi_gpu.cu` — keep NCCL wrapper, add halo-exchange primitive (`mg_halo_exchange_sum`).
- `src/sim.cu` — `Simulation` gains `slab_full_lo`, `slab_full_hi`, `slab_lo`, `slab_hi`. `cells.S` becomes the slab-local buffer. Cell migration in `step()` after rebind.
- `src/kernels.cu` — change `S` indexing in 4 kernels. Mechanical.
- `tests/python/test_multi_gpu.py` — new file. Required tests:
  - `test_g1_matches_no_mg` — G=1 slab build matches the single-GPU build bit-for-bit.
  - `test_halo_strip_correct` — synthetic two-rank scatter, verify halo sum equals what a single-rank scatter produces.
  - `test_migration_preserves_phi` — force a cell to cross a boundary, verify checkpoint matches single-GPU run.
  - `test_g4_n4608_no_nan` — full N=4608 G=4 run, t=10, no NaN, mass conservation within 1 %.

### 6.8 What this plan does not yet decide

These are decisions to make at implementation time with you, not now:
1. Whether to use `ncclSend`/`ncclRecv` + a local add kernel, or `ncclAllReduce` over the halo strip viewed as a contiguous buffer. Both work; the second is simpler.
2. Whether to use one stream per direction (split halo into top-send/top-recv/bottom-send/bottom-recv with separate streams to allow overlap with compute) or a single stream that serializes them. Start simple, optimize if profiling shows it matters.
3. Cell migration: synchronous (block step loop until done) or asynchronous (overlap with next step's polar). Start synchronous; rebind already breaks the captured-graph fast path.

---

## 7. What to do before writing any of this

1. Wait for `60640677` (NCCL_P2P_LEVEL=NVL test). If it shows the current architecture can hit ~85 % efficiency at the user's target, the scope of work is much smaller — we just set the env var and add 1-step pipelining (Option D) for the remaining gap. Domain decomp becomes a future-work item.
2. If NCCL fix does not recover speed, this plan is the next step.
3. Either way, the user reviews this document and either approves Option B or steers us elsewhere.

---

## 8. Risks I am not minimizing

- **Migration can be wrong in subtle ways.** A cell mid-rebind that lands exactly on a slab boundary, or a cell whose tile straddles the boundary at exactly the moment ownership flips, are corner cases. We will need exhaustive tests.
- **Load imbalance is real.** Confluent jammed states tend to have density gradients. If one rank ends up with 1.5× the cells of another, it sets the step rate and we get 67 % efficiency from imbalance alone. Phase B1 will measure this; if it is bad, we add periodic boundary repartitioning (slide boundaries to equalize cell counts).
- **The halo size is a contract that everything depends on.** If a kernel ever lets a cell's tile extend beyond `HALO_H` past the COM (e.g. because of a bug in the rebind half-width clamp), we get silent data corruption at slab boundaries. We need an assertion in the rebind kernel that catches this.
- **All checkpoints are now per-rank.** We need a sane gather-and-write story for trajectory and VTK. Phase A (current) deferred this; Phase B1 should not.
