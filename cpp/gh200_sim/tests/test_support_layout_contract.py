"""CPU/static gates for the opt-in GH200 extended support layout.

No test in this file invokes CUDA or a simulator executable.  The two host
compilations prove the selected geometry and compare the physics/RNG/checkpoint
observables emitted from the same headers.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[3]
GH200 = REPO / "cpp" / "gh200_sim"

# Checked directly against Roihu's pinned source.tar.gz on 2026-08-15.  The
# active kernels.cu differs from the pinned copy only in comments around the
# already-selected geometry. The exact canonical tumble block below must occur
# twice (fused and split), so layout-specific comment updates remain possible
# without weakening the RNG/dynamics guard.
PINNED_KERNELS_HEADER_SHA256 = (
    "3b61cb5c248af38def480cf850866e8b9ae26e31c3c69dcd90773bd1a82814a1"
)
CHECKPOINT_FORMAT_SHA256 = (
    "27d9b55abf591b606ad793318d19fdf83e427b7bae283032a374ff8d39e35da2"
)
PHYSICS_RNG_CHECKPOINT_FIELDS = {
    "p_tumble_bits": "3eb0c6f713f92497",
    "polarity_stream": "80090001",
    "philox": "78939ad7,b9b91b5e,7dbedec4,a173f38b",
    "uniform_bits": "3fde24e6b5ee6e46",
    "theta_bits": "3db33c28",
    "kappa_xi_lambda_mu": "10,1500,7,1",
    "checkpoint_version": "8",
    "checkpoint_sizes": "44,12,32,144",
}
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiler() -> str:
    requested = os.environ.get("CXX")
    candidates = ([requested] if requested else []) + ["c++", "g++", "clang++"]
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError("no GCC/Clang-compatible C++17 host compiler found")


def parse(stdout: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in fields:
            raise AssertionError(f"duplicate contract field {key!r}")
        fields[key] = value
    assert "SUPPORT_LAYOUT_CONTRACT_PASS" in stdout
    return fields


def compile_and_run(tmp: Path, extended: bool) -> dict[str, str]:
    tile, edge = (288, 224) if extended else (256, 208)
    suffix = ".exe" if os.name == "nt" else ""
    executable = tmp / (("extended" if extended else "compact") + suffix)
    command = [
        compiler(),
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-DPF_EXTENDED_SUPPORT_LAYOUT={int(extended)}",
        f"-DPF_EXPECTED_TILE_PITCH={tile}",
        f"-DPF_EXPECTED_LARGE_CLASS_EDGE={edge}",
        "-I",
        str(GH200 / "tests" / "host_cuda_stub"),
        "-I",
        str(GH200 / "include"),
        "-I",
        str(REPO / "cpp" / "common"),
        str(GH200 / "tests" / "support_layout_contract.cpp"),
        "-o",
        str(executable),
    ]
    built = subprocess.run(command, text=True, capture_output=True, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run(
        [str(executable)], text=True, capture_output=True, check=False
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    return parse(ran.stdout)


def assert_fields(actual: dict[str, str], expected: dict[str, str]) -> None:
    for key, value in expected.items():
        assert actual.get(key) == value, key


def test_compact_and_extended_geometry_with_identical_rng_tumble_and_abi() -> None:
    with tempfile.TemporaryDirectory(prefix="gh200_support_layout_") as raw:
        tmp = Path(raw)
        compact = compile_and_run(tmp, extended=False)
        extended = compile_and_run(tmp, extended=True)

    assert_fields(compact, {
        "layout": "compact",
        "tile_pitch": "256",
        "tile_area": "65536",
        "large_edge": "208",
        "max_support": "200",
        "large_smem": "183616",
        "launch_smem": "213504",
        "right_margin": "16",
        "extent201_class": "-1",
        "extent208_class": "-1",
        "old_tile_exact_200": "4",
        "old_tile_exact_208": "-1",
    })
    assert_fields(extended, {
        "layout": "extended",
        "tile_pitch": "288",
        "tile_area": "82944",
        "large_edge": "224",
        "max_support": "216",
        "large_smem": "211904",
        "launch_smem": "213504",
        "right_margin": "32",
        "extent201_class": "4",
        "extent208_class": "4",
        "old_tile_exact_200": "4",
        "old_tile_exact_208": "4",
    })
    assert_fields(compact, PHYSICS_RNG_CHECKPOINT_FIELDS)
    assert_fields(extended, PHYSICS_RNG_CHECKPOINT_FIELDS)

    geometry = {
        "layout",
        "tile_pitch",
        "tile_area",
        "large_edge",
        "max_support",
        "large_smem",
        "right_margin",
        "extent201_class",
        "extent208_class",
        "old_tile_exact_208",
    }
    assert {k: v for k, v in compact.items() if k not in geometry} == {
        k: v for k, v in extended.items() if k not in geometry
    }


def test_rng_tumble_and_checkpoint_sources_are_outside_the_patch() -> None:
    assert sha256(GH200 / "include" / "kernels.cuh") == (
        PINNED_KERNELS_HEADER_SHA256
    )
    kernel_source = (GH200 / "src" / "kernels.cu").read_text(encoding="utf-8")
    normalized_source = " ".join(kernel_source.split())
    normalized_tumble = " ".join(TUMBLE_BLOCK.split())
    assert normalized_source.count(normalized_tumble) == 2
    assert sha256(REPO / "cpp" / "common" / "checkpoint_format.h") == (
        CHECKPOINT_FORMAT_SHA256
    )


def test_checkpoint_exact_path_accepts_a_different_file_tile_edge() -> None:
    source = (GH200 / "src" / "checkpoint.cu").read_text(encoding="utf-8")
    assert "const int exact_cls = class_preserving_nonzero(" in source
    assert "if (T == kTilePitch)" not in source
    assert "const int32_t tile_t = kTilePitch;" in source
    assert "out->file_tile_t = tile_t;" in source


def test_rust_checkpoint_consumers_size_records_from_file_tile_t() -> None:
    analyzer = (
        REPO / "rust" / "cell_analyze" / "src" / "analysis" / "checkpoint.rs"
    ).read_text(encoding="utf-8")
    merger = (
        REPO / "rust" / "cell_analyze" / "src" / "analysis" /
        "merge_checkpoint.rs"
    ).read_text(encoding="utf-8")
    cpu_ref = (
        REPO / "rust" / "cpu_ref" / "src" / "checkpoint.rs"
    ).read_text(encoding="utf-8")

    assert "let tile_t: i32 = if version >= 7" in analyzer
    assert "bbox = BBox { x0: ox, y0: oy, x1: ox + tile_t" in analyzer
    assert "field_size > 500_000" in analyzer  # 288^2 = 82,944
    assert "let record_size = 32 + (tile_t as usize) * (tile_t as usize) * 4;" in merger
    assert "let tile_t = c.read_i32::<LittleEndian>()? as usize;" in cpu_ref
    assert "let tt = tile_t * tile_t;" in cpu_ref
