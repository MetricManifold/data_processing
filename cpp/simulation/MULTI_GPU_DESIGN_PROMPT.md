# Multi-GPU domain decomposition — design prompt

You are designing a multi-GPU strategy for a CUDA phase-field cell
simulation. **Design the algorithm from scratch.** Do not assume any
particular decomposition is right; start from the data dependencies
and work forward.

Your output is a written design — pseudocode, math, and prose. No code
required. Be explicit about correctness arguments and why each choice is
made.

---

## 1. The simulation, precisely

Each "cell" `n` carries a scalar phase field `phi_n(x, y)` defined on a
fixed `TILE_T × TILE_T = 320 × 320` square buffer that follows the cell
through the simulation. The whole population shares a single periodic
global domain `Nx × Ny` (square). Cell COMs land at the centre of their
tile after every `REBIND_EVERY = 8` steps (the rebind kernel re-centres
the tile on the cell's COM).

The fields touched by every step:

| Buffer | Size | Description |
|---|---|---|
| `phi[n]` | TILE_AREA = 102400 floats per cell | each cell's phi tile |
| `S(x,y)` | Nx·Ny floats (global) | `S = Σ_n phi_n²(x,y)` |
| origin[n] | 2 ints | global (x,y) of the tile's (0,0) corner |
| rect[n] | 4 ints | active sub-rectangle inside the tile (cell extent) |
| polar, gamma, v_A, tgt_R | scalars per cell | physics/per-cell constants |
| rng_state | curandState per cell | for run-and-tumble polarity |

**Hard guarantee**: `phi_n(x,y) = 0` for any pixel outside the tile's
active rect. The rebind kernel clamps the rect's half-width to
`hwmax = TILE_T/2 - 1 = 159 px`. Therefore phi_n is zero for
`|x - cx_n| > 159` or `|y - cy_n| > 159`.

**Per step, the ordered kernels that touch the field**:
1. `k_polar` — one thread per cell; updates polarity (no spatial extent).
2. `k_scatter_S` — one CTA per cell; for every pixel in the cell's
   active rect, atomicAdd `phi²` into the global `S` at
   `(origin.x + lx, origin.y + ly) mod (Nx, Ny)`.
3. `k_evolve_l1` (two passes per cell) — read `phi[n]` + read `S` at
   the same pixels, compute the PDE update, write `phi_out[n]`.
4. Every 8 steps: `k_rebind` — recompute origin from new COM, shift the
   tile, recompute rect.

The PDE is local (5- and 9-point stencils on phi tile + per-pixel `S`
read). **Cells couple only through `S`.** No neighbour list. Two cells
that don't share any tile pixels share no data.

## 2. Target sizes

Standard production parameters: cell radius `R = 49`, packing fraction
ρ = 0.89, so `Ny = ceil(sqrt(N · π · R² / ρ))`.

| N (cells) | Ny | `S` size | Phi pool (2× double-buffer) |
|---:|---:|---:|---:|
| 288 | 1562 | 9 MB | 235 MB |
| 1152 | 3124 | 37 MB | 944 MB |
| 4608 | 6249 | 149 MB | 3.78 GB |
| **12800** *(target)* | **10412** | **414 MB** | **10.5 GB** |

Hardware: NVIDIA A100 SXM (Narval, NVLink NV4 between every pair) and
H100 SXM (Nibi, NVLink with NVSwitch). NCCL 2.29.7 available. `G ≤ 4`
GPUs per node typical.

## 3. The actual goal — weak-scaling

The user's specification:
> "If 1600 cells take 3 hours on 1 GPU, then 6400 cells should take
> approximately 3 hours on 4 GPUs."

That is, **per-rank load held constant, doubling N doubles G, wall-time
flat**. This is the contract.

Not strong scaling (fixed N, smaller wall as G grows).

## 4. What was tried and failed

A first attempt used data-parallel cells with a fully-replicated `S`
field on every rank, plus a per-step `ncclAllReduce` on the entire
Nx·Ny `S` to keep every rank's copy in sync. It works correctly but
collapses to ~0% efficiency at G≥2 because the all-reduce work scales
with `Nx·Ny`, while per-rank compute scales with `N/G`. Doubling N at
fixed N/G doubles compute per rank but also doubles the all-reduce —
the ratio is constant in N, so weak-scaling is impossible by design.

A second attempt (slab decomposition along the y axis, halo bands of
thickness `2 × HALO_H`, NCCL send/recv between adjacent ranks, cell
migration when COM crosses a slab boundary) compiled and ran, but
produced a ~6% volume deficit (cells failed to inflate fully) at G=2
relative to G=1. Suspected cause: the boundary band that one rank sends
to its neighbour didn't cover the same global rows that the neighbour
recv'd into, so contributions were misaligned at the boundary.
Performance was also bad (~6× slower than G=1 at small N).

## 5. What you need to deliver

1. **Decomposition choice** with explicit rationale: 1D slabs, 2D tiles,
   replicated-grid+pipelining, something else? Address why each rejected
   option fails the weak-scaling contract.

2. **Halo / overlap region math**, derived from first principles:
   - The exact rows / regions each rank must own.
   - The exact rows / regions each rank must hold partial copies of.
   - The exact data exchanged each step, as a function of `(L, G, R, TILE_T)`.
   - Pseudocode showing both sides of the exchange agreeing on the
     same global pixel ranges.

3. **Cell ownership and migration policy**:
   - When does a cell become owned by a different rank?
   - What needs to be transferred (phi tile + scalars + RNG)?
   - At what cadence (every step? every rebind? on demand?)
   - Bound on the per-step migration data (pessimistic and typical).

4. **Per-step communication budget** at the user's target (N=12800, G=4,
   ρ=0.89, R=49). Compare to NVLink and NCCL bandwidth numbers. Is
   weak-scaling actually achievable? If yes, what is the predicted
   efficiency? If not, what would have to change in the simulation?

5. **Correctness audit** — explicit invariants the runtime must hold
   (e.g. "for every pixel y, the per-rank-sum of S at y over all ranks
   that hold y must equal the true global S at y"). Show how the
   exchange protocol enforces them.

6. **What you would do that's different from "1D slab + halo + rebind
   migration"**, and why.

## 6. Constraints

- The single-GPU code path must remain bit-identical to the current one
  (a regression suite checks N=72/N=1152 wall-clock times).
- The cell sim is the only target — you do not need to design a generic
  multi-GPU framework. Specialise.
- Disk I/O (trajectory, checkpoint) is currently per-rank; that is
  acceptable. Output post-processing handles concatenation.
- No assumption that ranks are co-located on one node — design must
  generalise to multi-node InfiniBand (though G ≤ 4 single-node is the
  immediate target).

Output as much detail as you would need yourself to implement it
correctly in two weeks. Show the math. Defend the choices.
