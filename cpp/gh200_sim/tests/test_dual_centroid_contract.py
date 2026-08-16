from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_dual_centroids as agreement  # noqa: E402


BASELINE_LEGACY_WRITER_SHA256 = (
    "d2ea89782cf8d01d32584c62abb6bcd292f23bc8d9d8db85525155c73b3f8fd0"
)
BASELINE_KERNELS_HEADER_SHA256 = (
    "740e73b0e1240a295047f2ca010b0d5d5b0c765b916b49ffcb3d26cb4146c796"
)
BASELINE_KERNELS_SOURCE_SHA256 = (
    "575f556982bd6b839367ff60a28fe86ca953253d460b0d3d84991cd202fc5d50"
)


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def periodic(value: float, period: float) -> float:
    return value - math.floor(value / period) * period


def valid_moments(total: float, first_x: float, first_y: float) -> bool:
    return (
        total > 0.0
        and math.isfinite(total)
        and math.isfinite(first_x)
        and math.isfinite(first_y)
    )


def dual_centroids(
    samples: list[tuple[int, int, float]], origin: tuple[int, int], period: int
) -> tuple[float, float, float, float, float, float]:
    sum_phi = sum(value for _, _, value in samples)
    sum_phi2 = sum(value * value for _, _, value in samples)
    x_phi = sum(value * x for x, _, value in samples) / sum_phi
    y_phi = sum(value * y for _, y, value in samples) / sum_phi
    x_phi2 = sum(value * value * x for x, _, value in samples) / sum_phi2
    y_phi2 = sum(value * value * y for _, y, value in samples) / sum_phi2
    return (
        periodic(origin[0] + x_phi, period),
        periodic(origin[1] + y_phi, period),
        periodic(origin[0] + x_phi2, period),
        periodic(origin[1] + y_phi2, period),
        sum_phi,
        sum_phi2,
    )


def test_legacy_trajectory_formatter_is_byte_unchanged_from_c130ce95() -> None:
    text = source("src/sim.cu")
    start = text.index("bool Sim::open_trajectory")
    end = text.index("void Sim::close_trajectory", start)
    observed = hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()
    assert observed == BASELINE_LEGACY_WRITER_SHA256


def test_solver_kernel_and_trajectory_abi_files_are_unchanged() -> None:
    header_hash = hashlib.sha256(source("include/kernels.cuh").encode()).hexdigest()
    assert header_hash == BASELINE_KERNELS_HEADER_SHA256
    # The composed large-edge selector changes one explanatory comment in the
    # hot kernel source, but no token. Project that exact comment back to the
    # c130ce95 spelling and require the complete file hash to become baseline.
    kernels = source("src/kernels.cu")
    selector_comment = (
        "        //     selected large edge - kPromoteSlack (184 or 200 px), not the\n"
        "        //     far tighter\n"
    )
    baseline_comment = (
        "        //     192 - kPromoteSlack = 184 px on an axis, not the far tighter\n"
    )
    assert kernels.count(selector_comment) == 1
    projected = kernels.replace(selector_comment, baseline_comment)
    source_hash = hashlib.sha256(projected.encode()).hexdigest()
    assert source_hash == BASELINE_KERNELS_SOURCE_SHA256


def test_normalized_hash_manifest_matches_sources() -> None:
    manifest = ROOT / "DUAL_CENTROID_VALIDATION_HASHES.md"
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        entries[relative] = digest
    assert "tests/test_dual_centroid_contract.py" in entries
    assert "src/validation_centroid.cu" in entries
    for relative, expected in entries.items():
        observed = hashlib.sha256(source(relative).encode("utf-8")).hexdigest()
        assert observed == expected, relative


def test_dual_path_is_explicitly_opt_in_and_separate() -> None:
    main = source("src/main.cu")
    sim = source("src/sim.cu")
    cmake = source("CMakeLists.txt")
    assert '"--dual-centroid-out"' in main
    assert "--dual-centroid-out requires --out" in main
    assert "--dual-centroid-out cannot be combined with --bench" in main
    assert "--dual-centroid-out must differ from --out" in main
    assert "if (!opt_.dual_centroid_path.empty())" in sim
    assert "if (d_dual_centroid_)" in sim
    assert "src/validation_centroid.cu" in cmake
    launch = sim.index("launch_validation_centroids")
    guarded = sim.rfind("if (d_dual_centroid_)", 0, launch)
    assert guarded >= 0
    assert "CU_WARN(cudaGetLastError())" in sim[launch : sim.index("}", launch)]


def test_validation_kernel_has_const_simulation_inputs_and_current_parity() -> None:
    header = source("include/validation_centroid.cuh")
    implementation = source("src/validation_centroid.cu")
    sim = source("src/sim.cu")
    signature = re.compile(
        r"launch_validation_centroids\(const float\* phi,\s*"
        r"const CellState\* cell,\s*const uint8_t\* cls,\s*"
        r"ValidationCentroidCell\* out",
        re.MULTILINE,
    )
    assert signature.search(header)
    assert "const float* __restrict__ phi" in implementation
    assert "const CellState* __restrict__ cell" in implementation
    assert "const uint8_t* __restrict__ clsv" in implementation
    assert "const int pin = (int)(steps_done_ & 1LL);" in sim
    assert "launch_validation_centroids(d_phi_[pin]" in sim
    assert "static_assert(sizeof(ValidationCentroidCell) == 64" in header


def test_periodic_local_lift_and_weight_definitions() -> None:
    # The active rect starts at global (9, 8) in an L=10 domain. Its support
    # crosses x=0, but local coordinates remain one unambiguous lift.
    result = dual_centroids(
        [(1, 1, 1.0), (3, 3, 0.5)], origin=(9, 8), period=10
    )
    x_phi, y_phi, x_phi2, y_phi2, sum_phi, sum_phi2 = result
    assert math.isclose(sum_phi, 1.5)
    assert math.isclose(sum_phi2, 1.25)
    assert math.isclose(x_phi, 2.0 / 3.0)
    assert math.isclose(y_phi, 29.0 / 3.0)
    assert math.isclose(x_phi2, 0.4)
    assert math.isclose(y_phi2, 9.4)
    assert not math.isclose(x_phi, x_phi2)


def test_local_lift_is_invariant_under_integer_recentering() -> None:
    period = 100
    mass = 4.0
    origin = 98
    moment = 13.0
    shift = 7
    before = periodic(origin + moment / mass, period)
    after = periodic((origin + shift) + (moment - shift * mass) / mass, period)
    assert math.isclose(before, 1.25)
    assert after == before


@pytest.mark.parametrize(
    ("total", "first_x", "first_y", "expected"),
    [
        (1.0, 0.0, 0.0, True),
        (0.0, 0.0, 0.0, False),
        (-1.0, 0.0, 0.0, False),
        (math.nan, 0.0, 0.0, False),
        (1.0, math.inf, 0.0, False),
    ],
)
def test_validity_contract(
    total: float, first_x: float, first_y: float, expected: bool
) -> None:
    assert valid_moments(total, first_x, first_y) is expected


def test_sidecar_column_contract_is_locked() -> None:
    sim = source("src/sim.cu")
    expected = (
        "# Format: time cell_id x_phi y_phi x_phi2_scan y_phi2_scan "
        "sum_phi sum_phi2_scan valid_phi valid_phi2_scan"
    )
    compact = " ".join(sim.replace('"\n            "', "").split())
    assert expected in compact
    assert "Invalid centroids are nan." in sim


def test_aligned_phi2_agreement_gate_reports_maxima(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.txt"
    sidecar = tmp_path / "dual.txt"
    legacy.write_text(
        "# v_A=1 N=1 Lx=100 Ly=100\n"
        "1.000000 7 99.999999 0.250000 0 0 1 0 0 1 1 12.500000\n",
        encoding="utf-8",
    )
    sidecar.write_text(
        "# Format: time cell_id x_phi y_phi x_phi2_scan y_phi2_scan "
        "sum_phi sum_phi2_scan valid_phi valid_phi2_scan\n"
        "1.000000 7 1 2 0.000001 0.2500001 10 12.500001 1 1\n",
        encoding="utf-8",
    )
    result = agreement.compare(legacy, sidecar)
    assert result["rows"] == 1
    assert result["x_phi2_periodic"].value == pytest.approx(2.0e-6)
    assert result["sum_phi2"].value == pytest.approx(1.0e-6)

    sidecar.write_text(
        "1.000000 7 1 2 0.01 0.25 10 12.5 1 1\n", encoding="utf-8"
    )
    with pytest.raises(agreement.GateFailure, match="x_phi2_periodic"):
        agreement.compare(legacy, sidecar)
