# GH200 initializer/pairing, class-edge, and dual-centroid overlay

Date: 2026-08-07. Detached base:
`c130ce95de61ae7b1006255773f67cb094336464`. Worktree:
`C:\Users\Zirconix\source\repos\data_processing_gh200_combined_c130ce95`.
The dirty primary and all source-candidate worktrees were read-only. This is a
source overlay, not evidence of a production-capable binary.

## Composition and preserved contracts

- **Published-placement initialization.** The historical grid+jitter default is
  unchanged. Strict `--initial-centres` loading is fresh-run-only; the
  deterministic Palmieri CPU generator emits the accepted float table.
- **Matched-pair preparation.** The host tool atomically prepares and validates
  control/soft pairs, including nominal/realized density, uint32 campaign
  seeds, RNG and rounding provenance, source/binary hashes, byte-identical
  centre tables, and `PAIRING.txt` written last.
- **Terminal shape-class selector.** `PF_LARGE_CLASS_EDGE` is independently
  restricted by CMake and C++ to 192 or 208 (default 192). Classes 0--3 and all
  non-edge geometry are pinned; the selector changes only the terminal class.
- **Validation-only dual centroids.** `--dual-centroid-out` independently scans
  the current field for `phi`- and `phi^2`-weighted periodic centroids on the
  exact legacy trajectory frames. It requires `--out`, rejects benchmark mode
  and a shared output path, and adds no buffer, kernel launch, or file operation
  when absent. The hot solver kernels and legacy trajectory record/formatter
  remain unchanged.
- `dump_phi.cu` is restored byte-for-byte from the immutable source archive.
  The composed `CMakeLists.txt` carries the selector interface, strict host
  no-contraction policy, initializer/pairing tools and tests, both edge tests,
  the one-factor comparison, and the dual-centroid translation unit.

Exact dual-centroid semantics and the still-open scientific/runtime gates are
in `DUAL_CENTROID_VALIDATION.md`; its normalized source manifest is
`DUAL_CENTROID_VALIDATION_HASHES.md`.

## Local verification

No CMake/CUDA compilation, GPU or cluster access, simulator execution, or BU
consumption was performed.

MinGW-W64 GCC 13.1.0, C++17, `-Wall -Wextra -Werror -pedantic
-ffp-contract=off`, at `-O0` and `-O3`: the initializer, generator, strict CSV
validation, pairing integration, geometry suite, and one-factor comparator all
passed for `PF_LARGE_CLASS_EDGE=192` and 208. All four seed-1729 N=72/L=777
CSVs were byte-identical (SHA-256
`05d25e663a06ecdbfb31b2ae51a9b8ed395cb861e00d33a1a78beaa687d62cf5`),
and initializer stdout was identical across all four configurations.

```text
triple_case_e192_O0=PASS
triple_case_e208_O0=PASS
triple_case_e192_O3=PASS
triple_case_e208_O3=PASS
PALMIERI_INITIALIZER_TEST_PASS
PAIRING_PREPARATION_TEST_PASS
edge_192_contract=1
edge_208_contract=1
fixed_classes_equal=1
launch_smem_equal=1
sm90_legality=1
SHAPE_CLASS_EDGE_ONE_FACTOR_TEST_PASS
```

The dual-centroid host/static contract suite passed `14/14`. It checks CLI and
opt-in allocation/I/O boundaries, active-parity/frame alignment, read-only
sampling, output semantics, legacy-formatter preservation, normalized source
hashes, and complete solver-kernel invariants after projecting the selector's
comment-only delta back to the base spelling.

CMake 3.28.1 confirmed default 192, accepted 192/208, and rejected
191/200/224/non-numeric input. A full illegal-value configure stopped before
CUDA discovery. Direct C++ rejected edge 200 and a CMake-marked target missing
the propagated selector. GCC `-fanalyzer` passed both geometry configurations
and both Palmieri C++ programs; all Python sources passed AST parsing.

## Physical source hashes

SHA-256 below is over the exact working-tree bytes. This report intentionally
does not hash itself; its hash is supplied with the archive provenance.

```text
cfa5e7b42a14044bb4115067812c796aba464ce8cad88dbf74f2422533c607e8  .gitignore
a51104264729e7b362ea8c739de0ada29ba6e4f6047580ec3a3fd330ad20fd47  cpp/gh200_sim/.gitignore
23a298124780518c42b17aaad2b1df8b79751f0df4baa8f8007581063fccdcbf  cpp/gh200_sim/CMakeLists.txt
fedac19de8d509d481c8ec437eebfacc5fae6f3b1f76ed0ee68b7d3ce0e07d2f  cpp/gh200_sim/DUAL_CENTROID_VALIDATION.md
10e8dc84a73c8e08ebbbf11e548c5fc5503b47349747b3800b6bf3b783bd8cab  cpp/gh200_sim/DUAL_CENTROID_VALIDATION_HASHES.md
31d7640ce9a0a2332ad3378b233cd1858e1b2d7e93d8788d9df961617def2a0c  cpp/gh200_sim/README.md
7a773ae281d0c8438eac6029a7ae3e21f2afb9be0062405baf7f015828575b24  cpp/gh200_sim/cmake/PFLargeClassEdge.cmake
f1729037ae4815160ac2bb5351d3f87cf9e46c88ec7fecd5857283b125fb308c  cpp/gh200_sim/include/palmieri_initializer.hpp
27cb385bc5760843a17f1dcff8678a40cd0b22999d1ecf490e27624538b53136  cpp/gh200_sim/include/params.cuh
ba0086f6d1855a45c83ac600174cee784087b5f0c5ad6517fe27381c15a162b2  cpp/gh200_sim/include/sim.cuh
0b86bc8d39e8c6a5d9d7be70aa650480b529801b54731d7097984c29fe2eaa43  cpp/gh200_sim/include/validation_centroid.cuh
896cde9986dd855c26c9b156a66c8e8ab204c9567390ff8cf3f2cf78e3a00d57  cpp/gh200_sim/scripts/compare_dual_centroids.py
bacf2eb47c1406ec18165ed90122f2d34e9080d1d7e3d3694b361a831b09a6b8  cpp/gh200_sim/src/kernels.cu
e0c5efee251837d325d866017753d5b39276ddb8a431c71f5c3fd4300590c885  cpp/gh200_sim/src/main.cu
0a920b6d6d3bfbdcbcd0e8cfb95173c8fb37e44cf1885dbb7e722cde70b05a0d  cpp/gh200_sim/src/sim.cu
39dc0bcc202b2629cf2c729a6734ab30fcc70673c7498c43af7f1cb182e7f6ba  cpp/gh200_sim/src/validation_centroid.cu
0eaf7c773162ec6c517e7f04fcae7492dda3cb7870b3d307d090d6b4e1cdf9c1  cpp/gh200_sim/tests/host_cuda_stub/cuda_runtime.h
e99c02113343a7e17b244e03fb55debbb09585053499384b242482311f4e4558  cpp/gh200_sim/tests/shape_class_edge_compare.py
38253a7010a90f530c66b46cadd9ba2c2116c9842a52aefb9a1fe745ce10eb08  cpp/gh200_sim/tests/shape_class_edge_test.cpp
f9ebce4ae004edbdc6bab757ea2d093a5f94266a6bddd716ffbaf5d168bc50c8  cpp/gh200_sim/tests/test_dual_centroid_contract.py
80fcf78479d2c9597ddb2a1a6f773d30b3a32938105601e7b520f48b4f7b6a2a  cpp/gh200_sim/tools/dump_phi.cu
093f93471848ae07d8024fc865a6b10075b06e0af5f9067c732c104d67672f67  cpp/gh200_sim/tools/palmieri_centres.cpp
1a86ecbd988f19deaf29797e0b9389596234d777a5bd472135748348b09d2175  cpp/gh200_sim/tools/palmieri_initializer_test.cpp
6220cc29d631da98a96752735b08298620fd9c143470aa312f567a21270b1db4  cpp/gh200_sim/tools/palmieri_pairing.py
9cf1f1685c51d70dc93986f31ef6ebcd2d08e50bb0efdc33a1c74a091dab44b1  cpp/gh200_sim/tools/palmieri_pairing_test.py
```

## Remaining authorization gate

Compile both selector values with the intended Roihu CUDA toolchain, inspect
`ptxas -v` for registers/spills/local memory, then perform only the approved
zero-step/self-test and short dual-centroid validation on free `gputest`.
Require identical simulator state and legacy trajectory bytes with the sidecar
off/on, aligned frame keys, independent centroid agreement, support/class
alarms, and the frozen overhead threshold before scientific or production use.
This overlay does not authorize those runs.
