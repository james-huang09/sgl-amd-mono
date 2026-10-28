"""TP4/TP8 MXFP4 arithmetic, multi-token state and page-table regression."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CHECKS = Path(__file__).with_name("glm53")


def check_a8w8(run, tmp_path, requests, query, capacity, perf_args):
    # Match the A4W4 arithmetic/perf case's full indexer, live lengths,
    # seed and cache capacity; gate A8W8 with its own serial control.
    a8_args = [
        "--activation",
        "fp8",
        "--requests",
        str(requests),
        "--query-tokens",
        str(query),
        "--seed",
        "821",
        "--capacity",
        capacity,
        "--check-routing",
        "--check-causality",
    ]
    a8_indexed, a8_serial = str(tmp_path / "a8_indexed"), str(tmp_path / "a8_serial")
    run(
        "check_multitoken.py",
        a8_args
        + [
            "--schedule",
            "serial",
            "--save-control",
            a8_indexed,
        ],
    )
    fixed = a8_args + ["--external-control", a8_indexed]
    run(
        "check_multitoken.py",
        fixed
        + [
            "--schedule",
            "serial",
            "--save-control",
            a8_serial,
        ],
    )
    run("check_multitoken.py", fixed + ["--check-control", a8_serial])
    run("check_multitoken.py", a8_args + perf_args)


def check_mxfp4_tp(tmp_path, requests, query, tp):
    """Check independent MoE numerics and causal serial/batched equivalence."""
    if not (
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= tp
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    ):
        pytest.skip(f"needs {tp} gfx950 GPUs")

    launch = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={tp}",
    ]

    def run(script, args):
        subprocess.run(
            launch + [str(CHECKS / script), *args],
            check=True,
            timeout=1800,
        )

    perf = os.environ.get("GLM53_PERF", "0")
    if perf not in ("0", "1"):
        raise ValueError("GLM53_PERF must be 0 or 1")
    a8w8 = os.environ.get("GLM53_PERF_A8W8", "0")
    if a8w8 not in ("0", "1") or (a8w8 == "1" and perf != "1"):
        raise ValueError("GLM53_PERF_A8W8 must be 0 or 1 and requires GLM53_PERF=1")
    perf_args = (
        [
            "--perf",
            "--perf-samples",
            os.environ.get("GLM53_PERF_SAMPLES", "100"),
            "--perf-warmup",
            os.environ.get("GLM53_PERF_WARMUP", "20"),
            "--perf-kernel-replays",
            os.environ.get("GLM53_PERF_KERNEL_REPLAYS", "20"),
        ]
        if perf == "1"
        else []
    )
    capacity = "131072" if query == 4 else "64"

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
            capacity,
        ]
        + perf_args,
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
        if a8w8 == "1":
            check_a8w8(run, tmp_path, requests, query, capacity, perf_args)
    finally:
        for artifact in tmp_path.glob("*.rank*.pt"):
            artifact.unlink()


@pytest.mark.parametrize("requests,query", [(1, 16), (2, 8), (4, 4), (1, 4), (2, 4)])
def test_mxfp4_tp4(tmp_path, requests, query):
    check_mxfp4_tp(tmp_path, requests, query, 4)


@pytest.mark.parametrize("requests", [1, 2])
def test_mxfp4_tp8(tmp_path, requests):
    check_mxfp4_tp(tmp_path, requests, 4, 8)


@pytest.mark.parametrize("requests", [1, 2])
def test_a8w8_tp8(tmp_path, requests):
    """Original TP8 MTP3 widths with synthetic serial controls and replay."""
    if not (
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 8
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    ):
        pytest.skip("needs 8 gfx950 GPUs")
    launch = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
    ]

    def run(script, args):
        subprocess.run(launch + [str(CHECKS / script), *args], check=True, timeout=1800)

    perf = os.environ.get("GLM53_PERF", "0")
    if perf not in ("0", "1"):
        raise ValueError("GLM53_PERF must be 0 or 1")
    perf_args = (
        [
            "--perf",
            "--perf-samples",
            os.environ.get("GLM53_PERF_SAMPLES", "100"),
            "--perf-warmup",
            os.environ.get("GLM53_PERF_WARMUP", "20"),
            "--perf-kernel-replays",
            os.environ.get("GLM53_PERF_KERNEL_REPLAYS", "20"),
        ]
        if perf == "1"
        else []
    )
    try:
        check_a8w8(run, tmp_path, requests, 4, "131072", perf_args)
    finally:
        for artifact in tmp_path.glob("*.rank*.pt"):
            artifact.unlink()
