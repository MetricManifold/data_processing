#!/usr/bin/env python3
"""Compare compact/extended CPU contracts and prove a representation-only delta."""

import argparse
import subprocess
import sys


def read(executable):
    completed = subprocess.run(
        [executable], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="ascii", errors="strict", check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "{} failed with {}: {}".format(
                executable, completed.returncode, completed.stderr.strip()
            )
        )
    fields = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in fields:
                raise RuntimeError("duplicate test field: {}".format(key))
            fields[key] = value
    if "SUPPORT_LAYOUT_TEST_PASS" not in completed.stdout:
        raise RuntimeError("{} omitted its pass sentinel".format(executable))
    return fields


COMMON = {
    "class0": "144,144,64,64",
    "class1": "176,144,32,64",
    "class2": "144,176,64,32",
    "class3": "160,160,32,32",
    "staged_raw": "213440",
    "launch_aligned": "213504",
    "old_tile_exact_200": "4",
    "p_tumble_bits": "3eb0c6f713f92497",
    "polarity_stream": "80090001",
    "philox": "78939ad7,b9b91b5e,7dbedec4,a173f38b",
    "uniform_bits": "3fde24e6b5ee6e46",
    "theta_bits": "3db33c28",
    "checkpoint_version": "8",
    "checkpoint_sizes": "44,12,32,144",
}


def expected(extended):
    if extended:
        specific = {
            "layout": "extended",
            "tile_pitch": "288",
            "tile_area": "82944",
            "large_edge": "224",
            "class4": "224,224,32,32",
            "large_raw": "211904",
            "large_aligned": "211968",
            "large_to_staged_margin": "1536",
            "tile_right_margin": "32",
            "max_support_extent": "216",
            "extent201_class": "4",
            "extent208_class": "4",
            "old_tile_exact_208": "4",
        }
    else:
        specific = {
            "layout": "compact",
            "tile_pitch": "256",
            "tile_area": "65536",
            "large_edge": "208",
            "class4": "208,208,32,32",
            "large_raw": "183616",
            "large_aligned": "183680",
            "large_to_staged_margin": "29824",
            "tile_right_margin": "16",
            "max_support_extent": "200",
            "extent201_class": "-1",
            "extent208_class": "-1",
            "old_tile_exact_208": "-1",
        }
    result = dict(COMMON)
    result.update(specific)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", required=True)
    parser.add_argument("--extended", required=True)
    args = parser.parse_args()
    compact = read(args.compact)
    extended = read(args.extended)
    if compact != expected(False):
        raise RuntimeError("compact audit differs from the locked contract")
    if extended != expected(True):
        raise RuntimeError("extended audit differs from the locked contract")

    geometry = {
        "layout", "tile_pitch", "tile_area", "large_edge", "class4",
        "large_raw", "large_aligned", "large_to_staged_margin",
        "tile_right_margin", "max_support_extent", "extent201_class",
        "extent208_class", "old_tile_exact_208",
    }
    if {k: v for k, v in compact.items() if k not in geometry} != {
        k: v for k, v in extended.items() if k not in geometry
    }:
        raise RuntimeError("a non-representation contract field changed")
    print("compact_contract=1")
    print("extended_contract=1")
    print("physics_rng_checkpoint_equal=1")
    print("SUPPORT_LAYOUT_ONE_FACTOR_TEST_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
