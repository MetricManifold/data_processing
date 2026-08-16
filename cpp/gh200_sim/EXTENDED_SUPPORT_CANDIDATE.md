# Extended-support engineering candidate

## Verdict

**Compile/CPU-test qualified only; not production-ready and not deployed.** No
GPU command, Slurm command, job submission, or live-campaign write was used for
this candidate. GPU validation remains a separate authorization gate.

The candidate was ported into a fresh extraction of the complete production
source archive at
`/scratch/project_2019216/stevsilb/gh200_extended_support_candidate_20260815`.
The authority archive is
`/scratch/project_2019216/stevsilb/gh200_hardened_class208_20260814/upload/source.tar.gz`,
SHA-256 `f23a18430d079902aaacc4ed862db99db72b3ce75fb7ca19c21a44ba2aaa2c65`.
The extracted, unmodified `cpp/` manifest hash was
`6ac4103aa556f7bf825b31b3d5b3fd523faa691bba25a019fe596495365fa15c`.

## Representation-only change

`PF_EXTENDED_SUPPORT_LAYOUT` is `OFF` by default. Omitting it configures the
same compact `tile=256,class=208` geometry as the pinned production build.
Enabling it selects only the coupled `tile=288,class=224` pair:

| invariant | compact default | extended candidate |
|---|---:|---:|
| per-cell tile | 256 x 256 | 288 x 288 |
| terminal phi-only class | 208 x 208 @ (32,32) | 224 x 224 @ (32,32) |
| support capacity after 8-pixel slack | 200 | 216 |
| terminal raw/aligned shared memory | 183,616 / 183,680 B | 211,904 / 211,968 B |
| fused launch shared memory | 213,504 B | 213,504 B |
| margin below staged raw maximum | 29,824 B | 1,536 B |

The coupled selector prevents the illegal `tile=256,class=224` combination.
All five class IDs, classes 0--3, terminal origin, 32-pixel row alignment,
zero-ring rule, strip geometry, dispatch, and fatal `class_exhausted` behavior
are unchanged. Supports above 216 still fail closed; there is no clipping
fallback.

No equation, physical parameter, integration step, Philox counter/key,
polarity stream, tumble comparison, stochastic ordering, or checkpoint ABI was
changed. The pinned files containing the kernel declarations, both kernel
bodies, simulator control flow, CLI, and v8 ABI remain byte-identical. The
compact candidate's normalized SASS hashes for fused/RHS/post are also exactly
equal to the pinned class-208 executable:

- fused: `573cd9699f5a5c93936ddbd198a948d031216a931ce3f2bcf7a89c34936032b3`
- RHS: `824de0b0393e297b657fb1321ecac2221b6f783a2a69685a45ab08b7606cde5a`
- post: `39b709853d2cc3bec489517997702bda6a172d357139ad042c3ff492c7c7e98f`

## v8 checkpoint contract

The file already stores `tile_t`; v8 is unchanged. The production repacker now
uses a shared constexpr rule to accept a foreign tile exactly whenever its
support and every nonzero source value lie in a native class at the canonical
offset. It then copies stored floats without moving their global coordinates.
The CPU test includes the production `checkpoint.cu` implementation directly:

- compact: a 256 tile with 200-pixel support is exact; 208 is refused;
- extended: the same 256 tile loads exactly into a 288 tile, preserving
  sub-threshold tails; both 200- and 208-pixel supports are accepted;
- discarded amplitude is exactly zero on accepted cases.

The writer continues to emit `tile_t=kTilePitch`, so extended output records
288 explicitly. `FixedPrefix`, `RankTrailer`, `CellRecordHeader`, and
`SimParamsV8` remain 44/12/32/144 bytes and format version remains 8.

## Exact Roihu compile evidence

Both builds used the login node only, sequentially, with:

- `nvhpc/26.3`, `nvc++ 26.3-0`, CUDA `13.1`, nvcc `V13.1.115`;
- Release, `sm_90`, C++17, `-O3 -DNDEBUG`, `-fmad=true`, `-lineinfo`,
  `-Xptxas=-v`, and no fast-math;
- identical generated flags; their `flags.make` files differ only in
  `PF_EXTENDED_SUPPORT_LAYOUT=0` versus `1`.

Ptxas comparison:

| kernel | compact registers; stack/store/load | extended registers; stack/store/load | delta |
|---|---:|---:|---:|
| fused | 80; 368/16/16 B | 80; 368/8/8 B | spills decrease 8/8 B |
| RHS | 64; 432/176/248 B | 64; 432/124/244 B | stores -52 B, loads -4 B |
| post | 40; 24/24/24 B | 40; 24/24/24 B | unchanged |

The earlier absolute zero-spill gate was invalid for this source lineage: the
pre-existing pinned production class-208 build itself has exactly the compact
spill counts above under the same Roihu toolchain. Windows reproduced those
counts; they were not an MSVC artifact. The correct gate is therefore a clean,
identical-toolchain comparison against pinned compact. Extended adds no
register, stack, spill, shared-memory-launch, or occupancy regression and in
fact reduces the recorded spill traffic.

Both clean CMake builds completed and both CTest runs passed 7/7 CPU tests.
The no-option configure independently recorded
`PF_EXTENDED_SUPPORT_LAYOUT:BOOL=OFF` and printed compact 256/208.

## Build receipts

- compact `cell_gh200`: `6ca4e43a50ae44cb856a6010c5e21ab53e299a0321f7d147c840d642aa570b2c`
- extended `cell_gh200`: `a307b743d4c9bced5e13c4c5ea1054330c20e8e27354b2c86ebaea4632bcddd4`
- compact build log: `d9db45e8fe65450948d3c1842550e99f782dfa39e75ade5ed7cf7f03b82dec9d`
- extended build log: `80901b0aef01e30a479c5c86e9a195f14f14b69e5e1ad5b46832b88ad778b232`
- compact CTest log: `5be4b08af7709f91db36243a0947306b9f6d168c45602c821f3c94b9dc6c84fc`
- extended CTest log: `cd52fe5721175e153881785ebb1347c0b689f351e483ddda91ea63101aa80cdd`
- toolchain receipt: `7df917ce1f8a7ab4e94fba4b1029ef986d590b3291925d86dc363da203d09ea4`

## Remaining production gate

Do not deploy or resume a campaign with this executable yet. A separately
authorized, unbilled GPU validation must still establish load/save/reload
identity, uninterrupted-versus-restart parity, compact-versus-extended parity
while classes remain 0--3, a one-step independent oracle after terminal-class
entry, and a synthetic 201--208 support case. Only after those pass should a
paid N=800 branch be considered. The failed diagnostic checkpoint must not be
treated as an accepted production restart.
