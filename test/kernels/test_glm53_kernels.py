"""TP4 MXFP4 arithmetic, multi-token state and page-table regression."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

CHECKS = Path(__file__).with_name("glm53")


@pytest.mark.parametrize("requests,query", [(1, 16), (2, 8), (4, 4)])
def test_mxfp4_tp4(tmp_path, requests, query):
    """Check independent MoE numerics and causal serial/batched equivalence."""
    if not (
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 4
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    ):
        pytest.skip("needs 4 gfx950 GPUs")

    launch = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
    ]

    def run(script, args):
        subprocess.run(
            launch + [str(CHECKS / script), *args],
            check=True,
            timeout=1800,
        )

    # The independent golden checks FP4 codes/scales, gate/up, activation,
    # down/reduction, untouched cache regions, eager replay and graph replay.
    run(
        "check_mxfp4.py",
        [
            "--rows",
            str(requests * query),
            "--request-width",
            str(query),
            "--capacity",
            "131072" if requests == 4 else "64",
        ],
    )
    indexed = str(tmp_path / "indexed")
    serial = str(tmp_path / "serial")
    args = [
        "--activation",
        "fp4",
        "--requests",
        str(requests),
        "--query-tokens",
        str(query),
        "--capacity",
        "64",
        "--expanded-page-table",
        "--route-ties",
        "--check-routing",
        "--check-causality",
    ]
    try:
        # Freeze the indexed serial selection so the comparison isolates
        # cross-cohort KV visibility and physical page-table row addressing.
        run(
            "check_multitoken.py",
            args + ["--schedule", "serial", "--save-control", indexed],
        )
        args += ["--external-control", indexed]
        run(
            "check_multitoken.py",
            args + ["--schedule", "serial", "--save-control", serial],
        )
        run("check_multitoken.py", args + ["--check-control", serial])
    finally:
        for artifact in tmp_path.glob("*.rank*.pt"):
            artifact.unlink()
