#!/usr/bin/env python3
"""Gate independently rescanned phi^2 centroids against legacy output.

This is a host-only post-processing check.  It never launches the simulator.
The tolerance accounts for the legacy float staging and six-decimal text
format while keeping the independent fp64 reduction comparison tight.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path


TEXT_ROUNDING_ABS = 5.1e-7
FLOAT_ULPS = 2.0


class GateFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class LegacyRow:
    x: float
    y: float
    volume: float


@dataclass(frozen=True)
class DualRow:
    x_phi: float
    y_phi: float
    x_phi2: float
    y_phi2: float
    sum_phi: float
    sum_phi2: float
    valid_phi: bool
    valid_phi2: bool


@dataclass(frozen=True)
class Maximum:
    value: float = -1.0
    key: tuple[str, int] | None = None
    limit: float = 0.0


def _insert_unique(rows: dict, key: tuple[str, int], value: object,
                   path: Path, line_number: int) -> None:
    if key in rows:
        raise GateFailure(f"{path}:{line_number}: duplicate key {key}")
    rows[key] = value


def read_legacy(path: Path) -> tuple[int, dict[tuple[str, int], LegacyRow]]:
    domain_side: int | None = None
    rows: dict[tuple[str, int], LegacyRow] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = re.search(r"\bLx=(\d+)\b", line)
            if match:
                domain_side = int(match.group(1))
            continue
        fields = line.split()
        if len(fields) != 12:
            raise GateFailure(
                f"{path}:{line_number}: expected 12 legacy columns, got {len(fields)}"
            )
        key = (fields[0], int(fields[1]))
        _insert_unique(
            rows, key,
            LegacyRow(float(fields[2]), float(fields[3]), float(fields[11])),
            path, line_number,
        )
    if domain_side is None or domain_side <= 0:
        raise GateFailure(f"{path}: missing positive Lx in legacy metadata")
    if not rows:
        raise GateFailure(f"{path}: no legacy data rows")
    return domain_side, rows


def read_dual(path: Path) -> dict[tuple[str, int], DualRow]:
    rows: dict[tuple[str, int], DualRow] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10:
            raise GateFailure(
                f"{path}:{line_number}: expected 10 sidecar columns, got {len(fields)}"
            )
        key = (fields[0], int(fields[1]))
        if fields[8] not in {"0", "1"} or fields[9] not in {"0", "1"}:
            raise GateFailure(
                f"{path}:{line_number}: validity fields must be 0 or 1"
            )
        _insert_unique(
            rows, key,
            DualRow(
                float(fields[2]), float(fields[3]),
                float(fields[4]), float(fields[5]),
                float(fields[6]), float(fields[7]),
                fields[8] == "1", fields[9] == "1",
            ),
            path, line_number,
        )
    if not rows:
        raise GateFailure(f"{path}: no sidecar data rows")
    return rows


def float32_ulp(value: float) -> float:
    """Spacing of binary32 around a finite, non-negative legacy observable."""
    rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    if not math.isfinite(rounded) or rounded < 0.0:
        raise GateFailure(f"cannot form binary32 tolerance around {value!r}")
    bits = struct.unpack("<I", struct.pack("<f", rounded))[0]
    following = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
    return following - rounded


def legacy_tolerance(value: float) -> float:
    return FLOAT_ULPS * float32_ulp(value) + TEXT_ROUNDING_ABS


def periodic_distance(lhs: float, rhs: float, period: int) -> float:
    delta = abs(lhs - rhs) % float(period)
    return min(delta, float(period) - delta)


def _update_maximum(current: Maximum, value: float,
                    key: tuple[str, int], limit: float) -> Maximum:
    return Maximum(value, key, limit) if value > current.value else current


def compare(legacy_path: Path, dual_path: Path) -> dict[str, Maximum | int]:
    period, legacy = read_legacy(legacy_path)
    dual = read_dual(dual_path)
    if legacy.keys() != dual.keys():
        missing = sorted(legacy.keys() - dual.keys())[:3]
        extra = sorted(dual.keys() - legacy.keys())[:3]
        raise GateFailure(
            "legacy/sidecar keys differ: "
            f"legacy={len(legacy)} sidecar={len(dual)} "
            f"missing_from_sidecar={missing} extra_in_sidecar={extra}"
        )

    maxima = {
        "x_phi2_periodic": Maximum(),
        "y_phi2_periodic": Maximum(),
        "sum_phi2": Maximum(),
    }
    failures: list[str] = []
    for key, old in legacy.items():
        new = dual[key]
        if not new.valid_phi:
            failures.append(f"{key}: valid_phi=0")
        elif (
            not all(math.isfinite(v) for v in (new.x_phi, new.y_phi, new.sum_phi))
            or new.sum_phi <= 0.0
            or not (0.0 <= new.x_phi < period)
            or not (0.0 <= new.y_phi < period)
        ):
            failures.append(f"{key}: invalid finite/range phi values")
        if not new.valid_phi2:
            failures.append(f"{key}: valid_phi2=0")
            continue
        if (
            not all(
                math.isfinite(v)
                for v in (new.x_phi2, new.y_phi2, new.sum_phi2)
            )
            or new.sum_phi2 <= 0.0
            or not (0.0 <= new.x_phi2 < period)
            or not (0.0 <= new.y_phi2 < period)
        ):
            failures.append(f"{key}: invalid finite/range phi2 values")
            continue
        comparisons = {
            "x_phi2_periodic": (
                periodic_distance(new.x_phi2, old.x, period),
                legacy_tolerance(old.x),
            ),
            "y_phi2_periodic": (
                periodic_distance(new.y_phi2, old.y, period),
                legacy_tolerance(old.y),
            ),
            "sum_phi2": (
                abs(new.sum_phi2 - old.volume),
                legacy_tolerance(old.volume),
            ),
        }
        for name, (difference, limit) in comparisons.items():
            if not math.isfinite(difference):
                failures.append(f"{key} {name}: non-finite difference")
                continue
            maxima[name] = _update_maximum(maxima[name], difference, key, limit)
            if difference > limit:
                failures.append(
                    f"{key} {name}: difference={difference:.17g} "
                    f"> limit={limit:.17g}"
                )
    if failures:
        preview = "\n".join(failures[:20])
        suffix = "" if len(failures) <= 20 else f"\n... {len(failures)-20} more"
        raise GateFailure(preview + suffix)
    maxima["rows"] = len(legacy)
    return maxima


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare independent sidecar phi^2 moments with legacy output"
    )
    parser.add_argument("legacy", type=Path)
    parser.add_argument("sidecar", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.legacy, args.sidecar)
    except (GateFailure, OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS rows={result['rows']}")
    for name in ("x_phi2_periodic", "y_phi2_periodic", "sum_phi2"):
        maximum = result[name]
        assert isinstance(maximum, Maximum)
        print(
            f"max_{name}_abs={maximum.value:.17g} key={maximum.key} "
            f"limit_at_max={maximum.limit:.17g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
