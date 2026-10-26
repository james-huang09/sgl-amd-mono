"""GLM-5.3 paged FP8 scores/top-k and replica replay, with nonzero weights and replay checks.

torchrun --standalone --nproc-per-node=4 test/kernels/glm53/check_indexer.py

Uses the plugin's real runner and cache ABI, without importing SGLang. This is a
synthetic layer/protocol check, not a checkpoint or serving accuracy result.
The independent score oracle consumes the unchanged upstream stage's queries
and head weights; it does not establish a full-layer independent golden.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from fixtures import (
    HIDDEN,
    TOPK,
    check_untouched,
    fixture,
    make_weights,
    tensor_digest,
)

from sgl_amd_mono.glm53.gfx950.layout import INDEX_SEL_CHUNK, INDEX_SEL_PAR_CHUNKS

REPLAY_BOUNDARY = INDEX_SEL_PAR_CHUNKS * INDEX_SEL_CHUNK

from sgl_amd_mono.glm53.gfx950.runner import (
    INDEX_PAGE_BYTES,
    Glm53Build,
    Glm53Caches,
    Glm53LayerOp,
    Glm53Runtime,
    Glm53Step,
)

LAYERS = 3  # odd count exercises slot parity across step boundaries


def pairs(rt, name, shape, bf16=False):
    tile_rows = rt.sc.get("_tile_rows", rt.width)
    if tile_rows != rt.width:
        from types import SimpleNamespace

        parts = []
        for tile in range(rt.width // tile_rows):
            sc = {k: v for k, v in rt.sc.items() if k != "_tile_rows"}
            view = SimpleNamespace(
                width=tile_rows,
                sc=sc,
                sy=rt.sy,
                epoch=rt.epoch,
                scratch=rt.scratch[tile * rt.sc["_tile_stride"] :],
                peer=SimpleNamespace(
                    bytes=rt.peer.bytes[tile * rt.sy["_tile_stride"] :]
                ),
            )
            parts.append(pairs(view, name, (tile_rows, *shape[1:]), bf16))
        return torch.cat(parts)
    n = math.prod(shape)
    off = rt.sc[name]
    raw = rt.scratch
    count = n // 2 if bf16 else n
    words = (
        raw[off : off + count * 8]
        .view(torch.int32)
        .reshape(count, 2)[:, 0]
        .contiguous()
    )
    return words.view(torch.bfloat16 if bf16 else torch.float32).reshape(shape)


def scores(rt, lengths):
    from sgl_amd_mono.glm53.check import launch_scores

    return launch_scores(rt, rt.width).contiguous().view(torch.float32)


def score_oracle(rt, cache, step, cos, sin, lengths):
    """Independent torch score arithmetic from the score stage's inputs."""
    rows = rt.width
    q = pairs(rt, "index_q", (rows, 32, 128), bf16=True).float()
    w = pairs(rt, "index_w", (rows, 32))
    new_k = pairs(rt, "index_k_new", (rows, 128), bf16=True).float()
    c, s = cos[step.positions].float()[:, None], sin[step.positions].float()[:, None]
    q0, q1 = q[:, :, :64:2].clone(), q[:, :, 1:64:2].clone()
    q[:, :, :64:2], q[:, :, 1:64:2] = q0 * c - q1 * s, q0 * s + q1 * c
    for span in (1, 2, 4, 8, 16, 32, 64):
        z = q.reshape(rows, 32, -1, 2, span)
        a, b = z[..., 0, :].clone(), z[..., 1, :].clone()
        z[..., 0, :], z[..., 1, :] = a + b, a - b
    q = (q * (128**-0.5)).bfloat16().float()
    pages = cache.idx.shape[0]
    raw = cache.idx.reshape(pages, -1)
    kvals = (
        raw[:, :8192]
        .contiguous()
        .reshape(pages, 4, 8, 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    kvals = kvals.view(torch.float8_e4m3fn).float().reshape(pages, 64, 128)
    scales = raw[:, 8192:].contiguous().view(torch.float32)
    actual = scores(rt, lengths)
    worst = 0.0
    for row, length in enumerate(lengths):
        if not length:
            continue
        logical = torch.arange(length, device=q.device)
        physical = step.page_table[step.req_ids[row].long(), logical // 64].long()
        keys = kvals[physical, logical % 64] * scales[physical, logical % 64, None]
        keys[-1] = new_k[row]
        ref = ((q[row] @ keys.T).relu() * w[row, :, None]).sum(dim=0)
        got = actual[row, :length]
        # The independent RoPE/butterfly path can cross BF16 rounding boundaries.
        rel = (got - ref).norm() / ref.norm().clamp_min(1e-20)
        assert torch.isfinite(got).all() and float(rel) < 0.01, (row, float(rel))
        worst = max(worst, float(rel))
    return worst


def run(args):
    rank, tp = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    assert tp in (4, 8) and args.rows in (2, 4, 6, 8, 12, 16)
    assert 0 <= args.pad_rows < args.rows
    assert args.capacity >= 64 and args.capacity % 64 == 0
    assert args.slot_offset >= 0 and args.slot_offset % 64 == 0
    assert args.request_width >= 1 and args.rows % args.request_width == 0
    assert "gfx950" in torch.cuda.get_device_properties(device).gcnArchName
    weights = make_weights(tp, device, args.ties, args.seed)
    h, cos, sin, initial, step, lengths = fixture(
        args.rows,
        args.capacity,
        device,
        args.pad_rows,
        args.full_context,
        args.seed + 1,
        args.slot_offset,
        args.request_width,
    )
    runtimes, caches, steps = [], [], []
    if getattr(args, "kv_fp8", False):
        from msgspec.structs import replace

        initial = replace(initial, kv=initial.kv.to(torch.float8_e4m3fn))
    try:
        for replica in (0, 1):
            rt = Glm53Runtime(
                Glm53Build(
                    rows=args.rows,
                    tp=tp,
                    launches_per_step=LAYERS,
                    with_indexer=True,
                    index_max_seq=args.capacity,
                    fp8_shuffled=True,
                    kv_fp8=getattr(args, "kv_fp8", False),
                ),
                rank,
                dist.group.WORLD,
                device,
            )
            runtimes.append(rt)
            rt.set_deadline(30)
            # Poison payloads, retaining generation zero and clear guard words.
            rt.scratch.view(torch.int32)[::2] = 0x7FC00000
            rt.peer.bytes[: rt.sy["flags"]].view(torch.int32)[::2] = 0x7FC00000
            caches.append(Glm53Caches(initial.kv.clone(), initial.idx.clone()))
            steps.append(
                Glm53Step(
                    step.positions,
                    step.slots,
                    step.indptr,
                    step.req_ids,
                    step.page_table,
                    step.topk.clone(),
                )
            )
            print(f"rank={rank} compile replica={replica}", flush=True)
            rt.precompile()
        ops = [
            [Glm53LayerOp(rt, weights, layer) for layer in range(LAYERS)]
            for rt in runtimes
        ]
        dist.barrier()

        def run_step(which):
            x = h
            for op in ops[which]:
                x = op.launch(x, caches[which], cos, sin, steps[which])
            runtimes[which].advance_step()
            return x

        outputs = [None, None]

        def compare():
            torch.cuda.synchronize()
            for rt in runtimes:
                assert not rt.epoch[3:6].any(), "guard error"
            assert torch.isfinite(outputs[0]).all()
            assert torch.equal(
                *[
                    pairs(rt, "index_q", (args.rows, 32, 128), bf16=True)
                    for rt in runtimes
                ]
            ), "replica changed projected queries"
            assert torch.equal(
                caches[0].kv.view(torch.uint8), caches[1].kv.view(torch.uint8)
            ), "replica changed KV bytes"
            assert torch.equal(caches[0].idx, caches[1].idx), (
                "replica changed index-cache bytes"
            )
            score_vectors = [scores(rt, lengths) for rt in runtimes]
            for row, length in enumerate(lengths):
                assert torch.equal(
                    score_vectors[0][row, :length], score_vectors[1][row, :length]
                ), "replica changed scores"
                if length:
                    # Independent exact membership/tie oracle from complete scores.
                    logical = torch.argsort(
                        score_vectors[0][row, :length], descending=True, stable=True
                    )[:TOPK]
                    physical = (
                        step.page_table[step.req_ids[row].long(), logical // 64].long()
                        * 64
                        + logical % 64
                    )
                    start = int(step.indptr[row])
                    selected = steps[1].topk[start : start + len(logical)].long()
                    assert torch.equal(
                        selected.sort().values, physical.sort().values
                    ), "top-k differs from stable torch sort"
                if args.ties and length:
                    logical = torch.arange(min(length, TOPK), device=device)
                    slots = (
                        step.page_table[step.req_ids[row].long(), logical // 64].long()
                        * 64
                        + logical % 64
                    )
                    start = int(step.indptr[row])
                    selected = steps[1].topk[start : start + len(logical)].long()
                    assert torch.equal(selected, slots), (
                        "ties did not select earliest logical positions"
                    )
            assert torch.equal(steps[0].topk, steps[1].topk), (
                "replica changed ordered physical top-k"
            )
            assert torch.equal(outputs[0], outputs[1]), "replica changed layer output"
            # Reconstruct TP ownership: every rank must produce the same hidden stream.
            local_output = outputs[1].cpu()
            root_output = local_output.clone()
            dist.broadcast(root_output, src=0)
            assert torch.equal(root_output, local_output), "TP ranks disagree"

        for repeat in range(2):
            for which in (0, 1):
                if repeat == 1 and which == 1:
                    dist.barrier()
                    if rank == 0:
                        time.sleep(0.05)
                outputs[which] = run_step(which)
            compare()
        oracle_rel = score_oracle(runtimes[0], caches[0], step, cos, sin, lengths)
        graphs = []
        for which in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[which] = run_step(which)
            graphs.append(graph)
        compare()
        generator = torch.Generator(device=device).manual_seed(args.seed + 4)
        initial_lengths = list(lengths)
        written_slots = [step.slots.clone()]

        def set_lengths(new_lengths):
            lengths[:] = new_lengths
            step.positions.copy_(
                torch.tensor([max(0, n - 1) for n in lengths], device=device)
            )
            slots = (
                step.page_table[step.req_ids.long(), step.positions // 64].long() * 64
                + step.positions % 64
            )
            slots[torch.tensor(lengths, device=device) == 0] = -1
            step.slots.copy_(slots)
            counts = torch.tensor([min(n, TOPK) for n in lengths], device=device)
            step.indptr[1:].copy_(counts.cumsum(0))
            written_slots.append(slots.clone())

        replay_lengths = []
        for replay in range(args.replays):
            h.copy_(torch.randn(h.shape, generator=generator, device=device))
            # Cross the long/short boundary through the same captured pointers,
            # including mixed rows and empty work; no host graph reselection.
            new_lengths = [max(1, n - replay % 6) if n else 0 for n in initial_lengths]
            if args.capacity > REPLAY_BOUNDARY:
                boundaries = [
                    args.capacity - 3,
                    REPLAY_BOUNDARY,
                    REPLAY_BOUNDARY + 1,
                    4095,
                    args.capacity // 2,
                    args.capacity - 2,
                    1,
                    REPLAY_BOUNDARY - 1,
                ]
                for row, n in enumerate(initial_lengths):
                    if n:
                        new_lengths[row] = boundaries[(replay + row) % len(boundaries)]
            if args.request_width > 1:
                for begin in range(0, args.rows, args.request_width):
                    first = max(
                        1, min(new_lengths[begin], args.capacity - args.request_width)
                    )
                    for offset in range(args.request_width):
                        if initial_lengths[begin + offset]:
                            new_lengths[begin + offset] = first + offset
            set_lengths(new_lengths)
            replay_lengths.append(list(lengths))
            for graph in graphs:
                graph.replay()
            compare()
        set_lengths(initial_lengths)
        for graph in graphs:
            graph.replay()
        compare()
        # Separate captured replicas can be replayed in a different order/count.
        # Only the replayed runtime may consume an epoch.
        before = [int(rt.epoch[0]) for rt in runtimes]
        for which in (1, 1, 0):
            graphs[which].replay()
        compare()
        assert [int(rt.epoch[0]) for rt in runtimes] == [before[0] + 1, before[1] + 2]
        for cache in caches:
            check_untouched(initial, cache, torch.cat(written_slots))
        result = dict(
            tp=tp,
            rows=args.rows,
            request_width=args.request_width,
            capacity=args.capacity,
            initial_live_keys=initial_lengths,
            final_live_keys=lengths,
            ties=args.ties,
            seed=args.seed,
            slot_offset=args.slot_offset,
            graph_replays=args.replays,
            replay_live_keys=replay_lengths,
            dispatch="layer_launch",
            extra_replica_replays=[1, 2],
            delayed_peer_ms=50,
            correctness="passed",
            score_oracle_rel=oracle_rel,
        )
        if args.save_control or args.check_control:
            snapshot = {
                "input": h.cpu(),
                "output": outputs[0].cpu(),
                "indices": torch.where(
                    steps[0].topk >= 0,
                    steps[0].topk - args.slot_offset,
                    steps[0].topk,
                ).cpu(),
            }
            for name, shape in (
                ("a", (args.rows, HIDDEN)),
                ("o", (args.rows, 64 // tp, 256)),
                ("q_lat", (args.rows, 64 // tp, 512)),
            ):
                snapshot[name] = pairs(runtimes[0], name, shape, bf16=True).cpu()
            if args.exact_control:
                full_scores = scores(runtimes[0], lengths)
                snapshot["state_sha256"] = {
                    "kv": tensor_digest(caches[0].kv[args.slot_offset :]),
                    "index_cache": tensor_digest(
                        caches[0].idx.reshape(-1)[
                            args.slot_offset // 64 * INDEX_PAGE_BYTES :
                        ]
                    ),
                    "index_q": tensor_digest(
                        pairs(runtimes[0], "index_q", (args.rows, 32, 128), bf16=True)
                    ),
                    "scores": [
                        tensor_digest(full_scores[row, :length])
                        for row, length in enumerate(lengths)
                    ],
                }
            if args.save_control:
                target = Path(f"{args.save_control}.rank{rank}.pt")
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(snapshot, target)
            if args.check_control:
                control = torch.load(
                    f"{args.check_control}.rank{rank}.pt", weights_only=True
                )
                metrics = {}
                for name in snapshot:
                    if name == "state_sha256":
                        assert snapshot[name] == control[name], (
                            "source control changed cache, query or score bytes"
                        )
                        continue
                    if name in ("input", "indices"):
                        assert torch.equal(snapshot[name], control[name]), (
                            f"source control changed {name}"
                        )
                    else:
                        if args.exact_control:
                            assert torch.equal(snapshot[name], control[name]), (
                                f"source control changed {name} bytes"
                            )
                        actual, expected = snapshot[name].float(), control[name].float()
                        relative = (
                            actual - expected
                        ).norm() / expected.norm().clamp_min(1e-30)
                        max_abs = (actual - expected).abs().max()
                        # A separate source-control gate for changed attention
                        # reduction tiles, stricter than the serving cosine gate.
                        # Independent runtime replicas remain bitwise exact in every run.
                        assert (
                            torch.isfinite(actual).all() and float(relative) <= 1e-3
                        ), (name, float(relative))
                        metrics[name] = [float(relative), float(max_abs)]
                gathered = [None for _ in range(tp)]
                dist.all_gather_object(gathered, metrics)
                result["source_control_max_rank"] = {
                    name: [max(m[name][i] for m in gathered) for i in range(2)]
                    for name in metrics
                }
                result["source_control_exact"] = args.exact_control
        if rank == 0:
            print("RESULT " + json.dumps(result), flush=True)
    finally:
        for rt in runtimes:
            rt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--request-width", type=int, default=1)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--pad-rows", type=int, default=1)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--ties", action="store_true")
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--kv-fp8", action="store_true")
    parser.add_argument(
        "--slot-offset",
        type=int,
        default=0,
        help="Translate the physical cache by this many slots (a multiple of 64).",
    )
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Start every active request near the compiled context limit.",
    )
    parser.add_argument(
        "--save-control",
        help="Remote prefix for small control tensors, one file per rank.",
    )
    parser.add_argument(
        "--check-control",
        help="Compare output and selected intermediates to a saved source control.",
    )
    parser.add_argument(
        "--exact-control",
        action="store_true",
        help="Require exact source-control tensors and full cache/query/score hashes.",
    )
    args = parser.parse_args()
    if not 1 <= args.layers <= 128:
        parser.error("--layers must be in [1, 128]")
    LAYERS = args.layers
    if args.exact_control and not (args.save_control or args.check_control):
        parser.error("--exact-control requires --save-control or --check-control")
    dist.init_process_group("gloo")
    try:
        run(args)
    finally:
        dist.destroy_process_group()
