# GH200 / Roihu calibration — measured, not assumed

All numbers below were measured on Roihu `gputest` on 2026-07-30. They supersede
spec-sheet values. The roofline argument for the solver design must use these.

## Access

```bash
ssh roihu-gpu                      # roihu-gpu.csc.fi, user stevsilb
source /usr/share/lmod/lmod/init/bash
module use /appl/modulefiles/manual/general/aarch64
module load nvhpc                  # nvhpc 26.3 -> nvcc CUDA 13.0.88
# or: export MODULEPATH=/appl/modulefiles
#     module load spack/aarch64/v2026_03/gcc/15.2.0/cuda/13.0.2
```

`/appl/profile/zz-csc-env.sh` does **not** exist on Roihu — that path is Mahti's.

### Slurm

| | |
|---|---|
| Free partition | `gputest` — 15:00 walltime, 4 nodes, `gpu:gh200:4` per node, **unbilled** |
| Account | **`--account=project_2019216`** |
| Concurrency | **MaxJobs=1, MaxSubmitJobs=2** — GPU work is strictly serial |
| Billed partitions | `gpumedium`, `gpularge` — these consume the 400k GPU BU. Do not use. |

`project_2017848` has `MaxSubmitJobs=0` on `gputest` and cannot submit there, despite
being the project that owns the writable scratch. Use `project_2019216` for `gputest`
(its negative BU balance is irrelevant because `gputest` is unbilled).

Always pass `--mem`; Slurm otherwise assigns the full 217 GB node share.

## Device

| property | value |
|---|---|
| Name | NVIDIA GH200 120GB |
| Compute capability | 9.0 (`-arch=sm_90`) |
| SMs | 132 |
| Shared mem / SM | 233472 B |
| **Shared mem / block, opt-in max** | **232448 B** (default 49152; needs `cudaFuncSetAttribute`) |
| **L2** | **62914560 B (60 MB)**; `persistingL2CacheMaxSize` = 39321600 B (37.5 MB) |
| Global mem | 95.0 GiB |
| Bus | 6144 bit @ 2619 MHz → 4.02 TB/s theoretical |
| Cooperative launch | supported |
| `pageableMemoryAccess`, `hostNativeAtomicSupported` | both 1 — NVLink-C2C coherent, managed memory is cheap |
| Host | aarch64 Neoverse-V2, 288 logical CPUs, 868 GB / node |

## Achieved bandwidth

Theoretical is 4.02 TB/s. **Do not design against it.** Measured, on a 7.45 GiB buffer
well past L2:

| pattern | achieved |
|---|---|
| read only | 2.49 TB/s |
| write only | 3.35 TB/s |
| copy (read+write) | 2.73 TB/s |

9-point periodic stencil, one read + one write per pixel — this is the real access
pattern and therefore the practical speed limit:

| L | field | ms | effective | rate |
|---|---|---|---|---|
| 1024 | 4 MB | 0.0056 | 1.49 TB/s | 186 Gpix/s |
| 2048 | 16 MB | 0.0167 | 2.01 TB/s | 251 Gpix/s |
| 4096 | 64 MB | 0.0693 | 1.94 TB/s | 242 Gpix/s |
| 8192 | 256 MB | 0.2669 | 2.01 TB/s | 251 Gpix/s |

**Use 251 Gpix/s per read+write pass as the budget unit.**

`atomicAdd` to global (the S scatter): 385 Gatomic/s at L=1024 (L2-resident),
244 Gatomic/s at L=4096. Atomics are not the bottleneck.

## Baseline to beat

Existing `cell_sim` (post-coefficient-fix build), N=288, R=49, ρ=0.89, no I/O,
5000 steps on one GH200:

```
Domain 1563x1563, T=320, pool=235.9 MB, S=9.8 MB
[L2] persisting full S (9.8 MB) in carveout (max 39.3 MB)
wall=0.684s  ->  0.137 ms/step
```

For reference: the same case is 0.239 ms/step on an RTX 4090 and 0.866 ms/step on an
H100 1g.10gb MIG slice.

## Consequences for the design

1. **L2 persistence of S is already done** by the existing code. It is not an
   available win. At N=288 S is 9.8 MB against a 60 MB L2; S stays L2-resident up to
   roughly N=1600 (L=3663, S=53.6 MB), which covers the entire FSS production range.
   Above that (N=3200+) S spills and HBM traffic for S becomes real.

2. **The remaining win is kernel fusion.** The existing design launches three kernels
   per step (`scatter_S`, `reduce_mb`, `rhs_mb`) which between them read φ three times
   and S twice. Budget accounting at N=288 with an adaptive rect of W≈148
   (`hw = 2σ + K·R/4` at K=2, σ≈24.5), so 288 × 148² ≈ 6.3e6 active px/step:

   | design | passes | predicted |
   |---|---|---|
   | one read+write pass (floor) | 1 | 25 µs |
   | existing 3-kernel | ~5.5 equiv | 137 µs (measured) |
   | fused single kernel | ~2 HBM (S from L2) | **30–45 µs** |

   So the realistic target is **3–4× faster than 0.137 ms/step**, i.e. 35–45 µs/step.
   Anything claiming better than 25 µs at this size is claiming to beat the measured
   stencil roofline and is wrong.

3. **Shared memory is the enabling resource.** A 148² fp32 rect is 87.6 KB; φ and S
   both resident is 175 KB, inside the 232448 B opt-in limit, at 1 block/SM. (As
   implemented the worst case is the 144x176 class at 211,776 B — still inside the
   limit, but the headroom is 20,608 B, not 57 KB.) 132 SMs
   → 132 cells in flight, so N=288 is two waves. Occupancy is low, which is why the
   latency-hiding question is real and must be measured, not assumed.

4. **V needs no reduction pass.** The volume term uses V(φⁿ), and φⁿ was written at the
   end of step n−1, so V(φⁿ⁺¹) can be accumulated during the RHS write and carried.
   Ix/Iy cannot be carried — they depend on S.

5. **One replica per GPU.** 95 GiB holds even N=12800 (S=429 MB, pool≈O(N·W²)); the
   science needs seed ensembles, so 4 independent replicas per node beats any
   multi-GPU decomposition of a single replica.

## CUDA 13 API changes that bite

- `cudaDeviceProp::memoryClockRate` was **removed**. Use
  `cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, dev)`.

## Reproducing

`tools/gh200_probe.cu` in this directory. Build and run:

```bash
nvcc -O3 -arch=sm_90 -std=c++17 -o gh200_probe tools/gh200_probe.cu
srun --account=project_2019216 --partition=gputest --nodes=1 --ntasks=1 \
     --cpus-per-task=16 --mem=32G --gres=gpu:gh200:1 --time=00:10:00 ./gh200_probe
```
