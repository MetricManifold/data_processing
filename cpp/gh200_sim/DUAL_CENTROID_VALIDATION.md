# Validation-only dual-centroid output

## Scope

`--dual-centroid-out <path>` adds a separate, append-only sidecar on exactly
the frames selected for the legacy `--out` trajectory. It exists to measure the
method discrepancy between `phi`- and `phi^2`-weighted periodic centroids. The
option requires `--out`, refuses `--bench`, and refuses to share the legacy
trajectory path.

When the option is absent there is no validation-buffer allocation, validation
kernel launch, or sidecar file operation. `TrajPackedCell`, `k_pack_traj`, and
the legacy header/row formatter are unchanged; the source slice containing the
legacy open/append functions retains its `c130ce95` SHA-256
`d2ea89782cf8d01d32584c62abb6bcd292f23bc8d9d8db85525155c73b3f8fd0`.
Portable SHA-256 values for every implementation/test file and the unchanged
solver-kernel invariants are frozen in
[`DUAL_CENTROID_VALIDATION_HASHES.md`](DUAL_CENTROID_VALIDATION_HASHES.md).

In the composed GH200 overlay this option coexists with the published-centre
initializer/pairing path and the 192/208 terminal-class selector. The manifest
therefore hashes the composed sources. Its kernel invariant projects the
selector's single comment-only edit back to the base spelling before requiring
the complete `c130ce95` kernel-source hash; no solver token is waived.

Example (validation only; not a production authorization):

```bash
./cell_gh200 -c checkpoint.bin --t-end 2082500 \
  --trajectory-interval 100 --out legacy_trajectory.txt \
  --dual-centroid-out dual_centroids.txt --no-final-checkpoint
```

## Exact semantics

At a trajectory boundary for completed step `s`, the stream first sees the
current buffer `phi[s % 2]`. A separate one-CTA-per-cell kernel reads the active
shape-class window and the current rect origin; all simulation pointers are
`const`. It writes only a dedicated 64-byte-per-cell mapped host buffer.

For active-window coordinates `(a,b)` and raw stored values `phi[a,b]`, it
computes in double precision and fixed warp-index reduction order

```text
S1  = sum(phi)       X1 = sum(a*phi)       Y1 = sum(b*phi)
S2  = sum(phi^2)     X2 = sum(a*phi^2)     Y2 = sum(b*phi^2)
```

No clipping, thresholding, absolute value, or smoothing is applied. The
periodic centroids use the same unambiguous local lift as the solver's existing
moments:

```text
x_phi  = wrap(gx0 + X1/S1, L)    y_phi  = wrap(gy0 + Y1/S1, L)
x_phi2 = wrap(gx0 + X2/S2, L)    y_phi2 = wrap(gy0 + Y2/S2, L)
```

This is not a circular mean over global pixels: each cell's localized rect is
the periodic lift, so a support crossing `x=0` stays contiguous. `S1` and `S2`
are unscaled lattice sums (`dx=dy=1` in this engine). The `phi^2` result is
independently recomputed from the same current field rather than copied from
`CellState`; it can therefore audit the legacy centroid and volume.

Output columns are

```text
time cell_id x_phi y_phi x_phi2_scan y_phi2_scan sum_phi sum_phi2_scan valid_phi valid_phi2_scan
```

The `_scan` suffix is deliberate: these `phi^2` values are recomputed from the
current field with the validation kernel. They are not copied from the
single-precision legacy trajectory staging record.

Coordinates and sums use `%.17g`; time retains the legacy six-decimal
presentation. A weight is valid only when its sum and both first moments are
finite and its sum is positive. An invalid centroid is written as `nan` and
its flag as zero. The file is flushed after each complete frame, matching the
legacy crash-recovery policy.

## Overhead boundary

The hot per-step graph and both solver kernels are untouched. Only an enabled
sampling frame adds:

- one read of each active-window `float`;
- six fp64 moment reductions per cell;
- one 64-byte mapped record per cell; and
- ten text columns per cell plus a flush.

Cost therefore scales with saved frames, not integration steps. Kernel-only
`--bench` intentionally excludes this output and is rejected when the option
is present; the relevant performance measurement is paired end-to-end wall
time at the intended trajectory cadence.

## Required GH200 gates before scientific use

These remain unmeasured and require a separately authorized free `gputest`
session; none was run while preparing this path.

1. Archive source, binary, compiler, ptxas register/spill/local-memory report,
   command, checkpoint, and output hashes.
2. Run identical short checkpoint resumes with the option off/on. Require
   bitwise-identical final simulator state and bitwise-identical legacy
   trajectory bytes. The sidecar must have exactly the same `(time,cell_id)`
   keys as the legacy file.
3. Independently recompute both definitions from the saved current fields,
   including a cell crossing each periodic boundary. On every aligned
   `(time,cell_id)`, require the periodic coordinate distance and absolute
   volume difference between `*_phi2_scan` and the legacy `x/y/volume` to be
   at most `2*ulp_binary32(legacy_value) + 5.1e-7`. This is the frozen gate for
   the legacy float staging plus six-decimal text roundoff; reduction-order
   differences must be reported, not silently absorbed.
4. Require both validity flags for every audited field and retain support/class
   alarms and maximum support extents.
5. Measure at least five alternating off/on repetitions with production output
   cadence. The current budget gate is median end-to-end ratio `<=1.15` and no
   paired ratio `>1.20`; exceeding it triggers re-costing, not a physics waiver.
6. One pair can diagnose the centroid definition difference but cannot support
   an independent-seed equivalence interval. Reuse at least four validation
   pairs if a method-insensitivity claim is contemplated.

## Local checks

The host/static contract tests require no GPU or simulator execution:

```bash
python -m pytest cpp/gh200_sim/tests/test_dual_centroid_contract.py -q
```

After an authorized validation run, enforce the aligned-frame gate and print
the observed maxima with:

```bash
python cpp/gh200_sim/scripts/compare_dual_centroids.py \
  legacy_trajectory.txt dual_centroids.txt
```

Observed maximum discrepancy: **not measured**. No GPU or simulator run was
made for this implementation, so the checker output must be recorded at the
first authorized `gputest` validation.

Runtime correctness and timing remain open until the gates above are completed.
The original dual-centroid candidate directly compiled with CUDA 12.8/sm_90:
its validation kernel used 48 registers, 384 bytes shared memory, and zero
stack/spill bytes. That evidence was not rerun for the composed overlay. The
combined source restores the formerly missing `tools/dump_phi.cu`, but this
CPU-only integration performed no CMake/CUDA compilation.
