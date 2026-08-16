#!/usr/bin/env python3
"""CPU-only integration tests for Palmieri matched-pair preparation."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def invoke(arguments, expected_success=True):
    completed = subprocess.run(
        arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", errors="replace", check=False
    )
    succeeded = completed.returncode == 0
    if succeeded != expected_success:
        raise RuntimeError(
            "unexpected exit {} for {!r}\nstdout:\n{}\nstderr:\n{}".format(
                completed.returncode, arguments, completed.stdout,
                completed.stderr
            )
        )
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    args = parser.parse_args()
    tool = args.tool.resolve()
    generator = args.generator.resolve()
    source_root = tool.parent.parent
    python = sys.executable

    with tempfile.TemporaryDirectory(prefix="palmieri-pairing-test-") as temporary:
        root = Path(temporary)
        pair = root / "pair"
        prepare = [
            python, str(tool), "prepare", "--generator", str(generator),
            "--source-root", str(source_root), "--out", str(pair),
            "--N", "8", "--side", "259", "--radius", "49",
            "--nominal-density", "0.9",
            "--seed", "7", "--polarity-seed", "90007",
            "--allowed-branch-difference",
            "cell-0 stiffness only: ctrl gamma=1; soft gamma=0.35",
        ]
        invoke(prepare)
        ctrl = pair / "ctrl" / "initial_centres.csv"
        soft = pair / "soft" / "initial_centres.csv"
        marker = pair / "PAIRING.txt"
        metadata_path = pair / "pairing.json"
        if not marker.is_file() or ctrl.read_bytes() != soft.read_bytes():
            raise RuntimeError("successful preparation did not lock equal tables")
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        expected_density = 8 * math.pi * 49 * 49 / (259 * 259)
        if metadata.get("realized_density") != expected_density:
            raise RuntimeError("realized density was not recorded exactly")
        for required in (
            "N", "L", "R", "method", "placement_seed", "polarity_seed",
            "generator", "source_files", "allowed_branch_difference",
            "branches", "nominal_density", "realized_density", "rng_engine",
            "uniform_map", "coordinate_storage",
        ):
            if required not in metadata:
                raise RuntimeError("metadata omitted {}".format(required))
        invoke([python, str(tool), "validate", "--generator", str(generator),
                "--source-root", str(source_root), "--out", str(pair)])

        original_soft = soft.read_bytes()
        soft.write_bytes(original_soft + b"tamper\n")
        invoke([python, str(tool), "validate", "--generator", str(generator),
                "--source-root", str(source_root), "--out", str(pair)],
               expected_success=False)
        soft.write_bytes(original_soft)

        original_metadata = metadata_path.read_bytes()
        original_marker = marker.read_bytes()
        metadata["nominal_density"] = 0.8
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        invoke([python, str(tool), "validate", "--generator", str(generator),
                "--source-root", str(source_root), "--out", str(pair)],
               expected_success=False)
        metadata_path.write_bytes(original_metadata)
        marker.write_bytes(original_marker)

        alternate = root / "alternate.csv"
        invoke([
            str(generator), "--N", "8", "--side", "259", "--radius", "49",
            "--seed", "8", "--out", str(alternate),
        ])
        if alternate.read_bytes() == ctrl.read_bytes():
            raise RuntimeError("test precondition failed: alternate table is equal")
        soft.write_bytes(alternate.read_bytes())
        invoke([python, str(tool), "validate", "--generator", str(generator),
                "--source-root", str(source_root), "--out", str(pair)],
               expected_success=False)
        soft.write_bytes(original_soft)

        marker.write_text("PALMIERI_MATCHED_PAIR_V1\nvalidated=true\n",
                          encoding="ascii")
        invoke([python, str(tool), "validate", "--generator", str(generator),
                "--source-root", str(source_root), "--out", str(pair)],
               expected_success=False)

        mismatch = root / "unpaired"
        (mismatch / "ctrl").mkdir(parents=True)
        (mismatch / "soft").mkdir(parents=True)
        (mismatch / "ctrl" / "initial_centres.csv").write_bytes(ctrl.read_bytes())
        (mismatch / "soft" / "initial_centres.csv").write_bytes(
            alternate.read_bytes()
        )
        invoke([python, str(tool), "validate", "--generator", str(generator),
                "--source-root", str(source_root), "--out", str(mismatch)],
               expected_success=False)
        if (mismatch / "PAIRING.txt").exists():
            raise RuntimeError("validation failure created PAIRING.txt")

        nominal_mismatch = root / "bad-nominal"
        invoke([
            python, str(tool), "prepare", "--generator", str(generator),
            "--source-root", str(source_root), "--out", str(nominal_mismatch),
            "--N", "8", "--side", "259", "--radius", "49",
            "--nominal-density", "0.8", "--seed", "7",
            "--polarity-seed", "90007", "--allowed-branch-difference",
            "cell-0 stiffness only",
        ], expected_success=False)
        if (nominal_mismatch / "PAIRING.txt").exists():
            raise RuntimeError("nominal-density failure created PAIRING.txt")

        seed_overflow = root / "bad-seed"
        invoke([
            python, str(tool), "prepare", "--generator", str(generator),
            "--source-root", str(source_root), "--out", str(seed_overflow),
            "--N", "8", "--side", "259", "--radius", "49",
            "--nominal-density", "0.9", "--seed", "4294967296",
            "--polarity-seed", "90007", "--allowed-branch-difference",
            "cell-0 stiffness only",
        ], expected_success=False)
        if (seed_overflow / "PAIRING.txt").exists():
            raise RuntimeError("seed-range failure created PAIRING.txt")

        polarity_overflow = root / "bad-polarity-seed"
        invoke([
            python, str(tool), "prepare", "--generator", str(generator),
            "--source-root", str(source_root), "--out", str(polarity_overflow),
            "--N", "8", "--side", "259", "--radius", "49",
            "--nominal-density", "0.9", "--seed", "7",
            "--polarity-seed", "4294967296", "--allowed-branch-difference",
            "cell-0 stiffness only",
        ], expected_success=False)
        if (polarity_overflow / "PAIRING.txt").exists():
            raise RuntimeError("polarity-seed-range failure created PAIRING.txt")

    print("pairing_success=1")
    print("pairing_tamper_rejection=1")
    print("pairing_valid_table_mismatch_rejection=1")
    print("pairing_nominal_density_validation=1")
    print("pairing_uint32_seed_enforcement=1")
    print("pairing_metadata_tamper_rejection=1")
    print("pairing_marker_tamper_rejection=1")
    print("PAIRING_PREPARATION_TEST_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
