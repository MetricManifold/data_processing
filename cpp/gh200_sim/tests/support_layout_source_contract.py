#!/usr/bin/env python3
"""Static contracts for the fixed-tile global geometry fallback."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(text, tokens, label):
    missing = [token for token in tokens if token not in text]
    if missing:
        raise RuntimeError("{} missing: {}".format(label, missing))


def main():
    params = (ROOT / "include/params.cuh").read_text(encoding="utf-8")
    require(params, (
        "constexpr int kClassFallback = 5;",
        "{kTilePitch - 2, kTilePitch - 2, 1, 1}",
        "class_containing_storage(",
        "class_smem_of(kClassFallback) == kScalarBytes",
    ), "fallback geometry")

    kernels = (ROOT / "src/kernels.cu").read_text(encoding="utf-8")
    require(kernels, (
        "void k_step_fallback(PF_GRID_CONSTANT const StepArgs A)",
        "A.cell[n].cls_written[input_parity]",
        "case kClassFallback: __syncthreads(); break;",
        "cudaLaunchKernelEx(&cfg, k_step, A);",
        "cudaLaunchKernelEx(&cfg, k_step_fallback, A);",
    ), "ordered fallback launch")
    if kernels.index("cudaLaunchKernelEx(&cfg, k_step, A);") > kernels.index(
            "cudaLaunchKernelEx(&cfg, k_step_fallback, A);"):
        raise RuntimeError("fallback launch precedes the shared-class update")

    checkpoint = (ROOT / "src/checkpoint.cu").read_text(encoding="utf-8")
    require(checkpoint, (
        "class_containing_storage(ext[0], ext[1], kPromoteSlack)",
        "const int32_t tile_t = kTilePitch;",
        "const int exact_cls = class_preserving_nonzero(",
    ), "checkpoint fallback")

    sim = (ROOT / "src/sim.cu").read_text(encoding="utf-8")
    require(sim, (
        'opt_.ckpt_dir + "/checkpoint_failed.bin"',
        "final_paths = checkpoint_paths(true, false);",
    ), "fatal checkpoint preservation")

    print("fallback_geometry_fixed_tile=1")
    print("fallback_ordered_after_shared_kernel=1")
    print("checkpoint_fallback_repack=1")
    print("failed_checkpoint_is_distinct=1")
    print("SUPPORT_LAYOUT_SOURCE_CONTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
