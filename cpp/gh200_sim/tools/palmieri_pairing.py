#!/usr/bin/env python3
"""Prepare and verify a provenance-locked Palmieri matched-pair directory."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple


SCHEMA = "palmieri_matched_pair_v1"
PAIRING_MAGIC = "PALMIERI_MATCHED_PAIR_V1"
CENTRES_NAME = "initial_centres.csv"
METADATA_NAME = "pairing.json"
MARKER_NAME = "PAIRING.txt"
CTRL_REL = Path("ctrl") / CENTRES_NAME
SOFT_REL = Path("soft") / CENTRES_NAME
UINT64_MAX = (1 << 64) - 1
UINT32_MAX = (1 << 32) - 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RNG_ENGINE = "mt19937_64"
UNIFORM_MAP = "(raw>>11)*2^-53"
COORDINATE_STORAGE = "binary32_before_later_acceptance_checks"


class PairingError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            aa = a.read(1024 * 1024)
            bb = b.read(1024 * 1024)
            if aa != bb:
                return False
            if not aa:
                return True


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(str(source), str(temporary))
        os.replace(str(temporary), str(destination))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def run_generator(generator: Path, arguments: Iterable[str]) -> Dict[str, str]:
    command = [str(generator)] + list(arguments)
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", errors="replace", check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PairingError(
            "centre generator/validator failed with exit {}: {}".format(
                completed.returncode, detail
            )
        )
    fields: Dict[str, str] = {}
    for line in completed.stdout.splitlines():
        for token in line.split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
    for required in ("method", "N", "L", "R", "accepted",
                     "minimum_periodic_distance", "table_fnv1a64"):
        if required not in fields:
            raise PairingError(
                "generator output omitted required field {!r}".format(required)
            )
    return fields


def numeric(value: float) -> str:
    return format(value, ".17g")


def validate_csv_with_generator(
    generator: Path, csv_path: Path, n_cells: int, side: float, radius: float
) -> Dict[str, str]:
    fields = run_generator(
        generator,
        ["--N", str(n_cells), "--side", numeric(side), "--radius",
         numeric(radius), "--validate", str(csv_path)],
    )
    if fields.get("mode") != "validate":
        raise PairingError("centre validator did not report mode=validate")
    try:
        reported_n = int(fields["N"], 10)
        accepted = int(fields["accepted"], 10)
        reported_side = float(fields["L"])
        reported_radius = float(fields["R"])
        minimum = float(fields["minimum_periodic_distance"])
    except ValueError as error:
        raise PairingError("centre validator emitted malformed diagnostics") from error
    if reported_n != n_cells or accepted != n_cells:
        raise PairingError("centre validator reported the wrong row count")
    if reported_side != side or reported_radius != radius:
        raise PairingError("centre validator reported different geometry")
    if n_cells > 1 and (not math.isfinite(minimum) or minimum < radius):
        raise PairingError("centre validator reported an invalid minimum distance")
    if not re.fullmatch(r"[0-9a-f]{16}", fields["table_fnv1a64"]):
        raise PairingError("centre validator emitted a malformed FNV fingerprint")
    return fields


def source_label(path: Path, source_root: Path) -> str:
    try:
        return path.relative_to(source_root).as_posix()
    except ValueError:
        return str(path)


def resolve_source(label: str, source_root: Path) -> Path:
    path = Path(label)
    return path if path.is_absolute() else source_root / path


def default_sources(source_root: Path) -> List[Path]:
    return [
        source_root / "tools" / "palmieri_centres.cpp",
        source_root / "include" / "palmieri_initializer.hpp",
        Path(__file__).resolve(),
    ]


def checked_uint64(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a base-10 integer") from error
    if value < 0 or value > UINT64_MAX:
        raise argparse.ArgumentTypeError("must be in [0, 2^64-1]")
    return value


def checked_uint32(text: str) -> int:
    value = checked_uint64(text)
    if value > UINT32_MAX:
        raise argparse.ArgumentTypeError(
            "must be in [0, 2^32-1] for checkpoint-v8 campaign safety"
        )
    return value


def checked_positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be numeric") from error
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def require_dict(value: object, name: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise PairingError("metadata field {!r} must be an object".format(name))
    return value


def require_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PairingError("metadata field {!r} is not a SHA-256".format(name))
    return value


def require_uint64(value: object, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PairingError("metadata field {!r} must be an integer".format(name))
    lower = 1 if positive else 0
    if value < lower or value > UINT64_MAX:
        raise PairingError("metadata field {!r} is out of range".format(name))
    return value


def require_uint32(value: object, name: str) -> int:
    value = require_uint64(value, name)
    if value > UINT32_MAX:
        raise PairingError(
            "metadata field {!r} exceeds checkpoint-v8 uint32 range".format(name)
        )
    return value


def require_geometry(metadata: Dict[str, object]) -> Tuple[int, float, float]:
    n_cells = require_uint64(metadata.get("N"), "N", positive=True)
    if n_cells > 2_147_483_647:
        raise PairingError("metadata N exceeds the generator's int range")
    side_value = metadata.get("L")
    radius_value = metadata.get("R")
    if isinstance(side_value, bool) or not isinstance(side_value, (int, float)):
        raise PairingError("metadata L must be numeric")
    if isinstance(radius_value, bool) or not isinstance(radius_value, (int, float)):
        raise PairingError("metadata R must be numeric")
    side = float(side_value)
    radius = float(radius_value)
    if not math.isfinite(side) or side <= 0.0:
        raise PairingError("metadata L must be finite and positive")
    if not math.isfinite(radius) or radius <= 0.0:
        raise PairingError("metadata R must be finite and positive")
    return int(n_cells), side, radius


def expected_marker(metadata_sha: str, centres_sha: str) -> bytes:
    return (
        PAIRING_MAGIC + "\n"
        + "schema=" + SCHEMA + "\n"
        + "metadata_sha256=" + metadata_sha + "\n"
        + "centres_sha256=" + centres_sha + "\n"
        + "ctrl=" + CTRL_REL.as_posix() + "\n"
        + "soft=" + SOFT_REL.as_posix() + "\n"
        + "validated=true\n"
    ).encode("ascii")


def validate_artifacts(
    output_root: Path,
    generator: Path,
    source_root: Path,
    metadata: Dict[str, object],
    check_marker: bool,
) -> Tuple[str, str]:
    if metadata.get("schema") != SCHEMA:
        raise PairingError("unsupported or missing pairing metadata schema")
    n_cells, side, radius = require_geometry(metadata)
    placement_seed = require_uint32(metadata.get("placement_seed"), "placement_seed")
    polarity_seed = require_uint32(metadata.get("polarity_seed"), "polarity_seed")
    del placement_seed, polarity_seed
    allowed = metadata.get("allowed_branch_difference")
    if not isinstance(allowed, str) or not allowed.strip():
        raise PairingError("allowed_branch_difference must be a non-empty string")
    density_definition = "N*pi*R^2/L^2"
    if metadata.get("realized_density_definition") != density_definition:
        raise PairingError("unexpected realized-density definition")
    nominal = metadata.get("nominal_density")
    if isinstance(nominal, bool) or not isinstance(nominal, (int, float)):
        raise PairingError("nominal_density must be numeric")
    nominal_density = float(nominal)
    if not math.isfinite(nominal_density) or nominal_density <= 0.0:
        raise PairingError("nominal_density must be finite and positive")
    nominal_side = math.ceil(
        math.sqrt(n_cells * math.pi * radius * radius / nominal_density)
    )
    if side != float(nominal_side):
        raise PairingError(
            "L is inconsistent with ceil(sqrt(N*pi*R^2/nominal_density))"
        )
    realized = metadata.get("realized_density")
    expected_density = n_cells * math.pi * radius * radius / (side * side)
    if isinstance(realized, bool) or not isinstance(realized, (int, float)):
        raise PairingError("realized_density must be numeric")
    if float(realized) != expected_density:
        raise PairingError("realized_density is inconsistent with N, L, and R")
    if metadata.get("rng_engine") != RNG_ENGINE:
        raise PairingError("rng_engine does not match the placement contract")
    if metadata.get("uniform_map") != UNIFORM_MAP:
        raise PairingError("uniform_map does not match the placement contract")
    if metadata.get("coordinate_storage") != COORDINATE_STORAGE:
        raise PairingError("coordinate_storage does not match the placement contract")

    generator_record = require_dict(metadata.get("generator"), "generator")
    recorded_generator_sha = require_sha(
        generator_record.get("sha256"), "generator.sha256"
    )
    if sha256_file(generator) != recorded_generator_sha:
        raise PairingError("generator binary SHA-256 does not match metadata")

    sources = metadata.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise PairingError("source_files must be a non-empty array")
    seen_sources = set()
    for index, entry_value in enumerate(sources):
        entry = require_dict(entry_value, "source_files[{}]".format(index))
        label = entry.get("path")
        if not isinstance(label, str) or not label:
            raise PairingError("source file path is missing")
        if label in seen_sources:
            raise PairingError("source file paths are duplicated")
        seen_sources.add(label)
        recorded = require_sha(entry.get("sha256"), "source_files.sha256")
        source = resolve_source(label, source_root)
        if not source.is_file() or sha256_file(source) != recorded:
            raise PairingError("source SHA-256 mismatch: {}".format(label))

    branches = require_dict(metadata.get("branches"), "branches")
    ctrl_record = require_dict(branches.get("ctrl"), "branches.ctrl")
    soft_record = require_dict(branches.get("soft"), "branches.soft")
    if ctrl_record.get("centres") != CTRL_REL.as_posix():
        raise PairingError("control centre-table path is not canonical")
    if soft_record.get("centres") != SOFT_REL.as_posix():
        raise PairingError("soft centre-table path is not canonical")
    ctrl_recorded_sha = require_sha(
        ctrl_record.get("sha256"), "branches.ctrl.sha256"
    )
    soft_recorded_sha = require_sha(
        soft_record.get("sha256"), "branches.soft.sha256"
    )
    if ctrl_recorded_sha != soft_recorded_sha:
        raise PairingError("branch metadata records different centre hashes")

    ctrl_path = output_root / CTRL_REL
    soft_path = output_root / SOFT_REL
    if not ctrl_path.is_file() or not soft_path.is_file():
        raise PairingError("one or both branch centre tables are missing")
    ctrl_sha = sha256_file(ctrl_path)
    soft_sha = sha256_file(soft_path)
    if ctrl_sha != soft_sha or not files_equal(ctrl_path, soft_path):
        raise PairingError("control and soft centre tables are not byte-identical")
    if ctrl_sha != ctrl_recorded_sha:
        raise PairingError("centre-table SHA-256 does not match metadata")

    ctrl_diag = validate_csv_with_generator(
        generator, ctrl_path, n_cells, side, radius
    )
    soft_diag = validate_csv_with_generator(
        generator, soft_path, n_cells, side, radius
    )
    method = metadata.get("method")
    if not isinstance(method, str) or not method:
        raise PairingError("metadata method is missing")
    if ctrl_diag["method"] != method or soft_diag["method"] != method:
        raise PairingError("initializer method does not match metadata")
    fnv = metadata.get("table_fnv1a64")
    if not isinstance(fnv, str) or re.fullmatch(r"[0-9a-f]{16}", fnv) is None:
        raise PairingError("metadata table_fnv1a64 is malformed")
    if ctrl_diag["table_fnv1a64"] != fnv or soft_diag["table_fnv1a64"] != fnv:
        raise PairingError("centre-table FNV fingerprint does not match metadata")

    metadata_path = output_root / METADATA_NAME
    if not metadata_path.is_file():
        raise PairingError("pairing metadata file is missing")
    metadata_sha = sha256_file(metadata_path)
    if check_marker:
        marker_path = output_root / MARKER_NAME
        if not marker_path.is_file():
            raise PairingError("PAIRING.txt is missing")
        if marker_path.read_bytes() != expected_marker(metadata_sha, ctrl_sha):
            raise PairingError("PAIRING.txt does not match the validated artifacts")
    return metadata_sha, ctrl_sha


def prepare_pair(args: argparse.Namespace) -> None:
    generator = args.generator.resolve()
    output_root = args.out.resolve()
    source_root = args.source_root.resolve()
    if not generator.is_file():
        raise PairingError("generator does not exist: {}".format(generator))
    if not args.allowed_branch_difference.strip():
        raise PairingError("--allowed-branch-difference must not be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    controlled = [
        output_root / CTRL_REL,
        output_root / SOFT_REL,
        output_root / METADATA_NAME,
        output_root / MARKER_NAME,
    ]
    existing = [path for path in controlled if path.exists()]
    if existing and not args.force:
        raise PairingError(
            "refusing to replace existing pairing artifact: {}".format(existing[0])
        )
    marker_path = output_root / MARKER_NAME
    metadata_path = output_root / METADATA_NAME
    if args.force:
        # Invalidate old claims before replacing either branch artifact.
        for obsolete in (marker_path, metadata_path):
            try:
                obsolete.unlink()
            except FileNotFoundError:
                pass
    (output_root / "ctrl").mkdir(parents=True, exist_ok=True)
    (output_root / "soft").mkdir(parents=True, exist_ok=True)

    sources = [path.resolve() for path in args.source]
    if not sources:
        sources = [path.resolve() for path in default_sources(source_root)]
    for source in sources:
        if not source.is_file():
            raise PairingError("source file does not exist: {}".format(source))
    generator_sha = sha256_file(generator)

    with tempfile.TemporaryDirectory(prefix=".pairing-stage-", dir=str(output_root)) as stage:
        staged_csv = Path(stage) / CENTRES_NAME
        generated = run_generator(
            generator,
            ["--N", str(args.N), "--side", numeric(args.side), "--radius",
             numeric(args.radius), "--seed", str(args.seed), "--out",
             str(staged_csv)],
        )
        if generated.get("mode") == "validate":
            raise PairingError("generator unexpectedly entered validation mode")
        atomic_copy(staged_csv, output_root / CTRL_REL)
        atomic_copy(staged_csv, output_root / SOFT_REL)

    ctrl_path = output_root / CTRL_REL
    soft_path = output_root / SOFT_REL
    if not files_equal(ctrl_path, soft_path):
        raise PairingError("branch copies differ before validation")
    ctrl_sha = sha256_file(ctrl_path)
    if ctrl_sha != sha256_file(soft_path):
        raise PairingError("branch SHA-256 values differ before validation")
    ctrl_diag = validate_csv_with_generator(
        generator, ctrl_path, args.N, args.side, args.radius
    )
    soft_diag = validate_csv_with_generator(
        generator, soft_path, args.N, args.side, args.radius
    )
    method = generated["method"]
    fnv = generated["table_fnv1a64"]
    for diagnostic in (ctrl_diag, soft_diag):
        if diagnostic["method"] != method or diagnostic["table_fnv1a64"] != fnv:
            raise PairingError("copied table diagnostics differ from generated table")
    if sha256_file(generator) != generator_sha:
        raise PairingError("generator binary changed during pairing preparation")

    source_records = [
        {"path": source_label(path, source_root), "sha256": sha256_file(path)}
        for path in sources
    ]
    metadata: Dict[str, object] = {
        "schema": SCHEMA,
        "N": args.N,
        "L": args.side,
        "R": args.radius,
        "nominal_density": args.nominal_density,
        "realized_density": args.N * math.pi * args.radius * args.radius
                            / (args.side * args.side),
        "realized_density_definition": "N*pi*R^2/L^2",
        "method": method,
        "rng_engine": RNG_ENGINE,
        "uniform_map": UNIFORM_MAP,
        "coordinate_storage": COORDINATE_STORAGE,
        "placement_seed": args.seed,
        "polarity_seed": args.polarity_seed,
        "table_fnv1a64": fnv,
        "generator": {"path": str(generator), "sha256": generator_sha},
        "source_files": source_records,
        "allowed_branch_difference": args.allowed_branch_difference.strip(),
        "branches": {
            "ctrl": {"centres": CTRL_REL.as_posix(), "sha256": ctrl_sha},
            "soft": {"centres": SOFT_REL.as_posix(), "sha256": ctrl_sha},
        },
        "generation_diagnostics": {
            key: generated[key]
            for key in ("accepted", "candidates", "rejected",
                        "minimum_periodic_distance")
            if key in generated
        },
    }
    payload = (json.dumps(metadata, indent=2, sort_keys=True,
                          ensure_ascii=True) + "\n").encode("ascii")
    atomic_write(metadata_path, payload)
    reread = json.loads(metadata_path.read_text(encoding="ascii"))
    metadata_sha, checked_sha = validate_artifacts(
        output_root, generator, source_root, reread, check_marker=False
    )
    if checked_sha != ctrl_sha:
        raise PairingError("centre hash changed before marker creation")
    try:
        atomic_write(marker_path, expected_marker(metadata_sha, ctrl_sha))
        validate_artifacts(
            output_root, generator, source_root, reread, check_marker=True
        )
    except BaseException:
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass
        raise
    print("pairing={}".format(output_root))
    print("centres_sha256={}".format(ctrl_sha))
    print("metadata_sha256={}".format(metadata_sha))
    print("method={}".format(method))
    print("validated=true")


def validate_pair(args: argparse.Namespace) -> None:
    generator = args.generator.resolve()
    output_root = args.out.resolve()
    source_root = args.source_root.resolve()
    if not generator.is_file():
        raise PairingError("generator does not exist: {}".format(generator))
    metadata_path = output_root / METADATA_NAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairingError("cannot read canonical pairing metadata") from error
    if not isinstance(metadata, dict):
        raise PairingError("pairing metadata root must be an object")
    metadata_sha, centres_sha = validate_artifacts(
        output_root, generator, source_root, metadata, check_marker=True
    )
    print("pairing={}".format(output_root))
    print("centres_sha256={}".format(centres_sha))
    print("metadata_sha256={}".format(metadata_sha))
    print("validated=true")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Prepare or validate one byte-identical Palmieri run pair"
    )
    subparsers = result.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--generator", required=True, type=Path)
    common.add_argument("--out", required=True, type=Path)
    common.add_argument(
        "--source-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="root used to resolve recorded source paths",
    )

    prepare = subparsers.add_parser("prepare", parents=[common])
    prepare.add_argument("--N", required=True, type=int)
    prepare.add_argument("--side", required=True, type=checked_positive_float)
    prepare.add_argument("--radius", required=True, type=checked_positive_float)
    prepare.add_argument(
        "--nominal-density", required=True, type=checked_positive_float
    )
    prepare.add_argument("--seed", required=True, type=checked_uint32)
    prepare.add_argument("--polarity-seed", required=True, type=checked_uint32)
    prepare.add_argument("--allowed-branch-difference", required=True)
    prepare.add_argument(
        "--source", action="append", type=Path, default=[],
        help="source file to hash (repeatable; defaults to generator sources)",
    )
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(function=prepare_pair)

    validate = subparsers.add_parser("validate", parents=[common])
    validate.set_defaults(function=validate_pair)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare" and (args.N < 1 or args.N > 2_147_483_647):
        print("[fatal] --N must be in [1, INT_MAX]", file=sys.stderr)
        return 2
    try:
        if args.command == "prepare":
            expected_side = math.ceil(
                math.sqrt(args.N * math.pi * args.radius * args.radius
                          / args.nominal_density)
            )
            if args.side != float(expected_side):
                raise PairingError(
                    "--side must equal ceil(sqrt(N*pi*R^2/--nominal-density)) "
                    "({})".format(expected_side)
                )
        args.function(args)
    except (OSError, PairingError, ValueError) as error:
        print("[fatal] {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
