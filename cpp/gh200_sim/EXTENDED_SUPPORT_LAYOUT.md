# GH200 extended-support candidate

## Status

**CPU/static candidate only; not production-ready.** No GPU job has been run or
submitted for this change. The option is off by default, and an enabled build
prints `EXTENDED CANDIDATE -- GPU UNVALIDATED` in its run log.

The `N=800` Palmieri-parameter soft branch failed closed after a transient field
support exceeded the compact representation's 200-pixel limit. Its retained
checkpoint has a unique `113 x 200` boundary candidate, but polling does not
identify the trigger or its exact extent. Because that extent was measured in a
208-pixel window, it was in `201..208`; the candidate below can represent every
such first trigger. It does not prove that later support cannot exceed its new
216-pixel limit.

## Representation change

Configure with `-DPF_EXTENDED_SUPPORT_LAYOUT=ON` to select:

| invariant | compact default | extended candidate |
|---|---:|---:|
| native per-cell tile | `256 x 256` | `288 x 288` |
| terminal phi-only class | `208 x 208 @ (32,32)` | `224 x 224 @ (32,32)` |
| support capacity after 8-pixel slack | 200 | 216 |
| terminal-class raw shared memory | 183,616 B | 211,904 B |
| fused launch shared memory | 213,504 B | 213,504 B |

The 288/224 pair retains 32-pixel (128-byte) source alignment, a one-pixel zero
ring, 16-row strips, the existing five class IDs, and the same fused-kernel
dispatch. The terminal class remains below the staged `160 x 160` class's
213,440-B raw footprint by 1,536 B, so it does not change the launch request.
Supports above 216 still set the always-on fatal `class_exhausted` flag; there is
no clipping or truncation fallback.

The checkpoint format already stores `tile_t`, so its version remains v8. The
reader's exact path is now tile-edge agnostic: if every nonzero source pixel is
inside a native class at its canonical offset, it copies every stored float and
preserves its global coordinate. Thus a compatible 256-pixel checkpoint can be
loaded losslessly into the 288-pixel candidate. Otherwise the pre-existing
support-centering path remains, measures the largest discarded tail, and
refuses any value above `kSupportEps`.

The static audit also follows `tile_t` through the Rust `cell_analyze` reader,
checkpoint merger, and CPU-reference reader. All three size v7/v8 records from
the value stored in the file; the analyzer's 500,000-float sanity cap admits a
288-squared field (82,944 floats).

No model coefficient, physical parameter, update equation, `dt`, Philox
counter/key, polarity stream, tumble comparison, or checkpoint ABI changes.
The CPU contract compiles both layouts from the same headers and requires their
Palmieri parameters, `-expm1(-dt/tau)` bits, Philox result, uniform draw,
polarity draw, and checkpoint layout to be identical.

## Source-authority gate

The tracked `cpp/gh200_sim` directory is **not** the complete source authority
for the live executable. Roihu's pinned archive is
`/scratch/project_2019216/stevsilb/gh200_hardened_class208_20260814/upload/source.tar.gz`,
SHA-256 `f23a18430d079902aaacc4ed862db99db72b3ce75fb7ca19c21a44ba2aaa2c65`.
Its checked core files match the local read-only
`scratch/combined_validation_source_20260807` mirror. The tracked tree differs
in CMake, parameter selection, simulator/main code, and omits the pinned
initializer, dual-centroid sources, tests, and tools. In particular, tracked
CMake names `tools/dump_phi.cu`, which is absent; do not fabricate it or treat a
partial build as evidence.

Before any GPU validation:

1. Start from a fresh copy of the complete pinned archive (never edit the
   provenance mirror), and record a full input manifest.
2. Port the layout pair, compile-time assertions, tile-edge-agnostic exact
   checkpoint helper, and CPU contract into that complete tree. Replace the
   pinned edge-only `PF_LARGE_CLASS_EDGE` selector with one atomic layout
   selector; never permit the illegal `tile=256,class=224` combination.
3. Diff against the pinned archive and prove that only representation,
   checkpoint placement, tests, logging, and build selection changed. Preserve
   `include/kernels.cuh` byte-for-byte and verify both run-and-tumble blocks in
   `src/kernels.cu` are unchanged.
4. Freeze the candidate source archive, build log, executable hash, compiler,
   CUDA version, and all test outputs before requesting GPU time.

## Validation gates

CPU/compile-only (zero GPU BU):

- Run `python -m pytest cpp/gh200_sim/tests/test_support_layout_contract.py`.
- Configure and compile the **complete** source twice, compact and extended,
  for `sm_90`; require no spills, the same 213,504-B fused launch request, and
  unchanged register limits. Never use `-use_fast_math`.
- Reject the candidate if the full-source diff or checkpoint-format hash is not
  clean, even if the small contract passes.

Local evidence on 2026-08-15: all four CPU tests passed. Direct `sm_90`
compile-only builds of the tracked extended `kernels.cu`, `checkpoint.cu`,
`sim.cu`, and `main.cu` also succeeded with CUDA 12.8/MSVC 19.44; compact and
extended kernels retained 80/64/40 registers for fused/RHS/post. Ptxas reported
compact versus extended stack/spill-store/spill-load bytes of `368/16/16`
versus `368/8/8` (fused), `432/176/248` versus `432/124/244` (RHS), and
`24/24/24` versus `24/24/24` (post). Both layouts therefore spill under this
local toolchain and do **not** satisfy the production zero-spill gate. Full
CMake generation failed exactly at the pre-existing missing
`tools/dump_phi.cu`. These results prove parsing/code generation only, not a
complete or acceptable build.

Free-queue GPU gate (only after confirming that queue is uncharged):

- Load both accepted `N=800` equilibration checkpoints into the extended build;
  require 800/800 lossless repacks, zero discarded amplitude, identical
  POLR/GAMA/VA_A/RADI values, and clean fatal flags at step zero.
- Save a 288-pixel checkpoint, reload it, and require bitwise state identity.
  Compare uninterrupted versus restart trajectories at 1, 10, and 100 steps,
  including theta, tumble counters, Philox decisions, class IDs, fields, and
  moments.
- Run compact and extended builds from the same checkpoint. Require bitwise
  identity while all cells remain in classes 0--3. Once the terminal class is
  entered, use an independent 224-window CPU one-step oracle; wider support may
  legitimately expose field values the compact representation could not retain.
- Exercise a synthetic 201--208-pixel support: compact must refuse it, extended
  must process it with no fatal/advisory support flag and no value above
  `kSupportEps` on the destination boundary.

Paid gate (requires a new written plan and explicit authorization):

- Run the `N=800` soft engineering branch from the accepted passive checkpoint
  past the old failure boundary, then preferably to the full 200-tau production
  endpoint. Require no fatal flags, no support above 216, restart parity, and a
  terminal hash receipt. At the old throughput, one branch is roughly 3.6k GPU
  BU to 200 tau before candidate overhead; budget 4--5k until measured.
- Only after that passes should a fresh matched control/soft pair be considered
  (roughly 7.2k GPU BU at old throughput; reserve 8--10k until benchmarked).
  The failed diagnostic checkpoint must not be resumed or admitted.

## Residual risks

- The new limit is finite; a later `>216` event remains possible.
- The terminal shared-memory margin is only 1,536 B below the staged maximum.
  Static arithmetic does not replace ptxas resource and occupancy evidence.
- The wider tile increases two phi buffers by about 106 MiB at `N=800` and
  213 MiB at `N=1600`; a checkpoint grows by about 53 and 106 MiB respectively.
  HBM capacity is ample, but cache/TLB and checkpoint-I/O costs need measurement.
- Exact checkpoint placement proves stored-state fidelity, not equivalence of
  subsequent GPU arithmetic. Only the oracle, restart, and long-run gates can
  establish that.
