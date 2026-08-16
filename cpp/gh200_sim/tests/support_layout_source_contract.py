#!/usr/bin/env python3
"""Static source-authority gate for the representation-only support patch."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "common"

PINNED_UNCHANGED = {
    "include/kernels.cuh":
        "3b61cb5c248af38def480cf850866e8b9ae26e31c3c69dcd90773bd1a82814a1",
    "src/kernels.cu":
        "40f3ee0408e29602ba503be6d1a47e2548702b3f2c4eeec8d955d848c0c96566",
    "src/main.cu":
        "47753aa69ba20eba705c336c7b06567566cd444c9c241137f93ee2a7421a0192",
    "src/sim.cu":
        "16a66e8afe26f6e41cdbb2ccca132eb10605026e53a421055b223af031be6ae8",
    "include/sim.cuh":
        "45cde21c698a59f9d93545e957cc710d8ec4521d30709b9cb36a9a4a456af7ae",
    "src/validation_centroid.cu":
        "39dc0bcc202b2629cf2c729a6734ab30fcc70673c7498c43af7f1cb182e7f6ba",
}
CHECKPOINT_FORMAT_SHA256 = (
    "27d9b55abf591b606ad793318d19fdf83e427b7bae283032a374ff8d39e35da2"
)
TUMBLE_BLOCK = """
const Philox4 r = philox4x32_10(
    (uint32_t)(step & 0xFFFFFFFFull), (uint32_t)(step >> 32),
    (uint32_t)cs.global_id, 0xA5A5A5A5u,
    (uint32_t)(A.polarity_seed & 0xFFFFFFFFull),
    (uint32_t)(A.polarity_seed >> 32));
float theta = cs.theta;
int tumbled = 0;
if (philox_uniform53(r.v[0], r.v[1]) < A.p_tumble) {
    theta = (float)(2.0 * kPi * philox_uniform53(r.v[2], r.v[3]));
    tumbled = 1;
}
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for relative, expected in PINNED_UNCHANGED.items():
        observed = sha256(ROOT / relative)
        if observed != expected:
            raise RuntimeError("pinned non-layout source changed: {}".format(relative))
    if sha256(COMMON / "checkpoint_format.h") != CHECKPOINT_FORMAT_SHA256:
        raise RuntimeError("checkpoint ABI header changed")

    kernels = (ROOT / "src/kernels.cu").read_text(encoding="utf-8")
    if " ".join(kernels.split()).count(" ".join(TUMBLE_BLOCK.split())) != 2:
        raise RuntimeError("canonical tumble block is not present exactly twice")

    checkpoint = (ROOT / "src/checkpoint.cu").read_text(encoding="utf-8")
    required = (
        "const int exact_cls = class_preserving_nonzero(",
        "const int32_t tile_t = kTilePitch;",
        "out->file_tile_t = tile_t;",
    )
    if any(token not in checkpoint for token in required):
        raise RuntimeError("dynamic-tile checkpoint contract is incomplete")
    if "if (T == kTilePitch)" in checkpoint:
        raise RuntimeError("foreign-tile exact path is still pitch-gated")

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    module = (ROOT / "cmake/PFLargeClassEdge.cmake").read_text(encoding="utf-8")
    if "PF_EXTENDED_SUPPORT_LAYOUT=${PF_EXTENDED_SUPPORT_LAYOUT_VALUE}" not in cmake:
        raise RuntimeError("CMake targets do not receive the atomic layout selector")
    if "PF_EXTENDED_SUPPORT_LAYOUT" not in module or " OFF)" not in module:
        raise RuntimeError("compact layout is not the explicit CMake default")

    print("pinned_nonlayout_hashes=1")
    print("tumble_blocks_unchanged=1")
    print("checkpoint_v8_dynamic_tile=1")
    print("compact_default=1")
    print("SUPPORT_LAYOUT_SOURCE_CONTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
