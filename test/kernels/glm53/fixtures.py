"""Nonzero shuffled-FP8 fixtures shared by cross-version GLM checks."""

import hashlib
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist

from sgl_amd_mono.glm53.gfx950.packing import pack_bf16, pack_fp8, pack_ptpc_fp8
from sgl_amd_mono.glm53.gfx950.runner import (
    INDEX_PAGE_BYTES,
    Glm53Caches,
    Glm53Step,
    Glm53Weights,
)

HIDDEN, INDEX_DIM, TOPK = 6144, 128, 2048


def read_pairs(rt, name, columns, dtype=torch.bfloat16):
    rows = rt.sc.get("_tile_rows", rt.width)
    result = []
    for tile in range(rt.width // rows):
        offset = rt.sc[name] + tile * rt.sc.get("_tile_stride", 0)
        count = rows * columns // (2 if dtype == torch.bfloat16 else 1)
        words = (
            rt.scratch[offset : offset + count * 8].view(torch.int32).reshape(count, 2)
        )
        result.append(words[:, 0].contiguous().view(dtype).reshape(rows, columns))
    return torch.cat(result)


def latency_stats(samples):
    """Nearest-rank P95 of individual replay or kernel invocation samples."""
    ordered = sorted(samples)
    return dict(
        samples=len(ordered),
        median_us=statistics.median(ordered),
        p95_us=ordered[math.ceil(0.95 * len(ordered)) - 1],
        min_us=ordered[0],
        max_us=ordered[-1],
        mean_us=statistics.mean(ordered),
    )


def add_perf_args(parser):
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--perf-samples", type=int, default=100)
    parser.add_argument("--perf-warmup", type=int, default=20)
    parser.add_argument("--perf-kernel-replays", type=int, default=20)


def validate_perf_args(parser, args):
    if args.perf and (
        args.perf_samples < 20 or args.perf_warmup < 1 or args.perf_kernel_replays < 20
    ):
        parser.error(
            "--perf requires at least 20 samples/kernel replays and one warmup"
        )
    if args.perf and args.profile_launches:
        parser.error("--perf already profiles kernels; omit --profile-launches")


def measure_replay(args, graph, restore, verify, mono_calls_per_replay=1):
    """Shared A4/A8 full replay timing and separate GPU-kernel profiling."""
    rank, tp = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        print("PERF_PROGRESS warmup", flush=True)
    for _ in range(args.perf_warmup):
        restore()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    timing_started = time.time()
    for _ in range(args.perf_samples):
        restore()
        dist.barrier()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed_us = start.elapsed_time(end) * 1000
        assert math.isfinite(elapsed_us) and elapsed_us > 0
        samples.append(elapsed_us)
    timing_ended = time.time()
    verify()
    if rank == 0:
        print("PERF_PROGRESS replay timing complete; kernel profile", flush=True)
    restore()
    dist.barrier()
    profile_started = time.time()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        # Prime ROCm graph tracing inside the recorder, outside the measured
        # range. The first replay can be absent from the device event stream.
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        # Keep every rank's recorder active for the entire TP replay sequence.
        # Recorder setup/teardown skew can otherwise drop a boundary launch.
        with torch.profiler.record_function("glm53_measured_kernel_replays"):
            dist.barrier()
            for _ in range(args.perf_kernel_replays):
                graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
    profile_ended = time.time()
    verify()
    windows = [
        event.time_range
        for event in profile.events()
        if event.name == "glm53_measured_kernel_replays"
        and event.device_type == torch.autograd.DeviceType.CPU
    ]
    assert len(windows) == 1, "profiler lost the measured replay range"
    window = windows[0]
    kernels = defaultdict(list)
    for event in sorted(profile.events(), key=lambda e: e.time_range.start):
        if (
            event.device_type == torch.autograd.DeviceType.CUDA
            and event.name != "glm53_measured_kernel_replays"
            and window.start <= event.time_range.start
            and event.time_range.end <= window.end
        ):
            kernels[event.name].append(event.time_range.elapsed_us())
    samples_all_ranks, kernels_all_ranks = [None] * tp, [None] * tp
    dist.all_gather_object(samples_all_ranks, samples)
    dist.all_gather_object(kernels_all_ranks, dict(kernels))
    # Validate collectively so a local capture failure cannot strand peers
    # gathering results while the failed rank enters collective runtime close.
    for owner, owned in enumerate(kernels_all_ranks):
        counts = {name: len(values) for name, values in owned.items()}
        assert owned, f"rank {owner}: profiler captured no GPU events"
        assert sum(n for name, n in counts.items() if "glm5_monokernel" in name) == (
            args.perf_kernel_replays * mono_calls_per_replay
        ), f"rank {owner}: unexpected profiler monokernel calls: {counts}"
    if rank != 0:
        return None
    print("PERF_PROGRESS kernel profile complete", flush=True)
    slowest = [max(values) for values in zip(*samples_all_ranks)]
    breakdown = []
    for name in sorted(set().union(*(k.keys() for k in kernels_all_ranks))):
        per_rank = []
        for owner, owned in enumerate(kernels_all_ranks):
            values = owned.get(name, [])
            if values:
                per_rank.append(
                    dict(
                        rank=owner,
                        calls_per_replay=len(values) / args.perf_kernel_replays,
                        mean_total_us_per_replay=sum(values) / args.perf_kernel_replays,
                        **latency_stats(values),
                    )
                )
        breakdown.append(
            dict(
                name=name,
                max_rank_mean_total_us_per_replay=max(
                    r["mean_total_us_per_replay"] for r in per_rank
                ),
                per_rank=per_rank,
                samples_all_ranks_us=[k.get(name, []) for k in kernels_all_ranks],
            )
        )
    source_root = Path(__file__).resolve().parents[3]
    sources = [
        "sgl_amd_mono/glm53/gfx950/kernel.py",
        "sgl_amd_mono/glm53/gfx950/runner.py",
        "sgl_amd_mono/glm53/gfx950/layout.py",
        "test/kernels/glm53/fixtures.py",
        "test/kernels/glm53/check_mxfp4.py",
        "test/kernels/glm53/check_multitoken.py",
        "test/kernels/test_glm53_kernels.py",
    ]
    return dict(
        warmup_replays=args.perf_warmup,
        graph_steps=1,
        weight_allocations=1,
        completion_boundary="max-rank HIP-event full-layer replay",
        percentile_method="nearest-rank",
        cache_restore_timed=False,
        timing_window_unix=[timing_started, timing_ended],
        kernel_profile_window_unix=[profile_started, profile_ended],
        correctness="passed",
        **latency_stats(slowest),
        samples_max_rank_us=slowest,
        per_rank=[latency_stats(values) for values in samples_all_ranks],
        samples_all_ranks_us=samples_all_ranks,
        kernel_profile_replays=args.perf_kernel_replays,
        kernel_profile_priming_replays=1,
        kernel_breakdown=breakdown,
        source_sha256={
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in sources
        },
    )


def make_weights(tp, device, ties=False, seed=719, expert_factory=None):
    g = torch.Generator(device=device).manual_seed(seed)
    heads, inter = 64 // tp, 2048 // tp

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device) * scale

    def fp8(rows, k, *, packed=False, scale_k=128, experts=None):
        shape = (rows, k) if experts is None else (experts, rows, k)
        q = rnd(*shape, scale=0.2).to(torch.float8_e4m3fn)
        scales = torch.full(
            (*shape[:-2], math.ceil(rows / 128), math.ceil(k / scale_k)),
            k**-0.5,
            dtype=torch.float32,
            device=device,
        )
        return (pack_fp8(q) if packed else pack_ptpc_fp8(q)), scales

    qa, sqa = fp8(2624, HIDDEN)
    qb, sqb = fp8(heads * 256, 2048)
    uk, suk = fp8(heads * 512, 192, packed=True, scale_k=64)
    uv, suv = fp8(heads * 256, 512, packed=True)
    wo, so = fp8(HIDDEN, heads * 256)
    if expert_factory is None:
        ug, sug = fp8(2 * inter, HIDDEN, experts=257)
        dn, sdn = fp8(HIDDEN, inter, experts=257)
    else:
        ug, sug, dn, sdn = expert_factory(inter, HIDDEN, g, device)
    ik, sik = fp8(INDEX_DIM, HIDDEN)
    iq, siq = fp8(32 * INDEX_DIM, 2048)
    iw = pack_bf16(rnd(32, HIDDEN, scale=0.0 if ties else HIDDEN**-0.5).bfloat16())
    keep = (
        ik,
        sik,
        iw,
        iq,
        siq,
        torch.ones(128, device=device),
        torch.zeros(128, device=device),
    )
    index = torch.tensor(
        [t.data_ptr() for t in keep] + [0], dtype=torch.int64, device=device
    )
    return Glm53Weights(
        g_in=torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
        g_q=torch.ones(2048, dtype=torch.bfloat16, device=device),
        g_kv=torch.ones(512, dtype=torch.bfloat16, device=device),
        g_post=torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
        w_qkv_a=qa,
        s_qkv_a=sqa,
        w_q_b=qb,
        s_q_b=sqb,
        w_uk=uk,
        s_uk=suk,
        w_uv=uv,
        s_uv=suv,
        w_o=wo,
        s_o=so,
        w_r=pack_bf16(rnd(256, HIDDEN, scale=0.02).bfloat16()),
        bias=rnd(256, scale=0.05),
        w_ug=ug,
        s_ug=sug,
        w_dn=dn,
        s_dn=sdn,
        shuffled=True,
        index=index,
        keep=keep,
    )


def fixture(
    rows,
    capacity,
    device,
    pad_rows,
    full_context=False,
    seed=720,
    slot_offset=0,
    request_width=1,
):
    g = torch.Generator(device=device).manual_seed(seed)
    pages_per_request = capacity // 64
    assert rows % request_width == 0
    requests = rows // request_width
    pages = requests * pages_per_request + 1
    table = torch.zeros(
        requests, pages_per_request + 3, dtype=torch.int32, device=device
    )
    table[:, :pages_per_request] = (
        torch.randperm(pages - 1, generator=g, device=device).reshape(requests, -1)
        + 1
        + slot_offset // 64
    )
    req = torch.arange(requests - 1, -1, -1, dtype=torch.int32, device=device)
    req = req.repeat_interleave(request_width)
    lengths = ([max(1, capacity - 3), 65, 2049, 1, 4097, 127, 2048, 129] * 2)[:rows]
    if full_context:
        lengths = [capacity - 3 - row for row in range(rows)]
    if request_width > 1:
        lengths = [
            capacity - request_width - 2 - row // request_width + row % request_width
            for row in range(rows)
        ]
    lengths = [min(n, capacity) for n in lengths]
    if pad_rows:
        lengths[-pad_rows:] = [0] * pad_rows
    pos = torch.tensor(
        [max(0, n - 1) for n in lengths], dtype=torch.int64, device=device
    )
    slots = table[req.long(), pos // 64].long() * 64 + pos % 64
    for row, n in enumerate(lengths):
        if not n:
            slots[row] = -1
    counts = torch.tensor(
        [min(n, TOPK) for n in lengths], dtype=torch.int32, device=device
    )
    indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), counts.cumsum(0).int()]
    )
    h = torch.randn(rows, HIDDEN, generator=g, device=device).bfloat16()
    angles = (
        torch.arange(capacity, device=device).float()[:, None]
        * (8000000 ** (-torch.arange(0, 64, 2, device=device).float() / 64))[None, :]
    )
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    kv = (torch.randn(pages * 64, 576, generator=g, device=device) * 0.1).bfloat16()
    values = (torch.randn(pages, 64, 128, generator=g, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    shuffled = (
        values.view(torch.uint8)
        .reshape(pages, 4, 16, 8, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    scales = torch.ones(pages, 64, dtype=torch.float32, device=device)
    idx = torch.cat([shuffled.reshape(pages, -1), scales.view(torch.uint8)], dim=1)
    idx = idx.reshape(pages, 64, 132)
    if slot_offset:
        # Translate identical logical data past the 32-bit buffer byte limit.
        # The unused prefix remains covered by check_untouched().
        kv_storage = torch.zeros(
            slot_offset + kv.shape[0], 576, dtype=kv.dtype, device=device
        )
        kv_storage[slot_offset:].copy_(kv)
        kv = kv_storage
        idx_storage = torch.zeros(
            slot_offset // 64 + pages, 64, 132, dtype=idx.dtype, device=device
        )
        idx_storage[slot_offset // 64 :].copy_(idx)
        idx = idx_storage
    step = Glm53Step(
        pos,
        slots,
        indptr,
        req,
        table,
        torch.full((rows * TOPK + 1,), -7, dtype=torch.int32, device=device),
    )
    return h, cos, sin, Glm53Caches(kv, idx), step, lengths


def check_untouched(before, after, written_slots):
    valid = written_slots[written_slots >= 0].long().unique()
    mask = torch.ones(before.kv.shape[0], dtype=torch.bool, device=valid.device)
    mask[valid] = False
    assert torch.equal(
        before.kv.view(torch.uint8)[mask], after.kv.view(torch.uint8)[mask]
    ), "untouched KV rows changed"
    changed = (before.idx != after.idx).reshape(-1)
    d = torch.arange(128, device=valid.device)
    for slot in valid.tolist():
        page, row = slot // 64, slot % 64
        offset = (
            page * INDEX_PAGE_BYTES
            + (row // 16) * 2048
            + (d // 16) * 256
            + (row % 16) * 16
            + d % 16
        )
        changed[offset] = False
        scale = page * INDEX_PAGE_BYTES + 8192 + row * 4
        changed[scale : scale + 4] = False
    assert not changed.any(), "untouched index-cache bytes changed"


def tensor_digest(tensor):
    """Hash full state on the test server without saving large cache tensors."""
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for chunk in raw.split(16 * 1024 * 1024):
        digest.update(chunk.cpu().numpy().tobytes())
    return digest.hexdigest()
