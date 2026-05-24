import subprocess

import pytest

from conftest import CELL_SIM


@pytest.mark.parametrize("bad_args,bad_flag", [
    (["-n", "abc"], "-n"),
    (["--v-A", "NaN"], "--v-A"),
    (["--tau", "-1"], "--tau"),
    (["--gamma", "badselector"], "--gamma"),
])
def test_malformed_numeric_cli_values_fail_loudly(tmp_path, bad_args, bad_flag):
    outdir = tmp_path / "out"
    cmd = [
        CELL_SIM,
        "-n", "1",
        "-N", "320",
        "-r", "20",
        "-t", "0.01",
        "--dt", "0.01",
        "--seed", "42",
        "--trajectory-samples", "0",
        "--save-interval", "0",
        *bad_args,
        "-o", str(outdir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    combined = result.stderr + result.stdout
    assert result.returncode != 0
    assert "[cli] bad value" in combined
    assert bad_flag in combined