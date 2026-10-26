# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from ROCm/aiter#6173 at 56d6390def (Apache-2.0):
# aiter/ops/flydsl/kernels/glm5_mono/glm/layout.py
# Multi-token planning adapted from ROCm/FlyDSL#1256 at 032072c94064.

"""Compile-time storage layout and CTA schedule for the GLM-5 MonoKernel."""

from sgl_amd_mono.common.plan import cdiv

from .config import (
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    QKV_A_ROWS,
    TOP_K,
    V_DIM,
)
from .tiles import (
    BLOCKS,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
)

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
INDEX_TILE = 16
INDEX_KEYS_PER_TASK = 64
# a long row's paged top-k runs on up to this many slices (CTAs) per row, each with
# staging room for this many threshold-bin candidates
INDEX_SEL_SLICES = 16
INDEX_CAND_STAGE_CAP = 16384
# keys a select chunk (a CTA's 512 threads x 8), and the chunks past which a row selects
# on up to INDEX_SEL_SLICES CTAs: up to 8 chunks the hand-offs cost what the slices save
INDEX_SEL_CHUNK = 4096
INDEX_SEL_PAR_CHUNKS = 8


def select_slices(bound, slices=INDEX_SEL_SLICES):
    """(slices, chunks a slice) of a row of ``bound`` > 0 keys whose top-k runs on
    several CTAs: runs of whole chunks in position order, at most INDEX_SEL_SLICES, none
    empty. The kernel calls it on traced Int32 and the CPU test on ints."""
    chunks = (bound + INDEX_SEL_CHUNK - 1) // INDEX_SEL_CHUNK
    per = (chunks + slices - 1) // slices
    return (chunks + per - 1) // per, per


def index_select_slices(samples: int, cohort_rows: bool = False) -> int:
    """Keep every row's selectors in the upper half of its resident grid."""
    blocks = BLOCKS // (2 if cohort_rows else 1)
    return min(INDEX_SEL_SLICES, blocks // 2 // samples)


DCP_SUMMARY_PAIRS = KV_LORA // 2 + 2


def split_acc_head(head_group, lane_group: int, element: int):
    """Map one split-PV MFMA row to its local attention head."""

    return head_group * WAVES + lane_group * 4 + element


def split_score_column(wave: int, lane: int):
    """Locate this wave's head column in a 16x16 MFMA score tile."""

    return wave + 16 * ((lane % 16) // 4)


def fp8_kv_upper_pair_lane(lane):
    return (lane & -4) + 2


def fp8_pe_upper_pair_lane(lane):
    return (lane & -2) + 1


N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_GROUPS = HIDDEN // 32
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
assert XQ_WAVES * 4 <= WAVES


def dn_tile(samples: int, expert_mxfp4: bool = False) -> int:
    """Return rows per expert-down/FFN-reduce task for the tuned schedule."""

    return 32 if samples == 1 or (samples > 4 and expert_mxfp4) else HIDDEN // BLOCKS


def row_tile_size(samples: int, native_fp4_mfma: bool = False) -> int:
    """Bound resident query state as in FlyDSL #1256's sixteen-token path."""
    if not 1 <= samples <= 16:
        raise ValueError("GLM mono-kernel supports 1 to 16 total query tokens")
    if native_fp4_mfma and samples > 12:
        return next(tile for tile in range(12, 0, -1) if samples % tile == 0)
    return samples if samples <= 12 else 8


def cache_samples_per_task(samples: int, npes: int) -> int:
    """Distribute the wide TP4 cache/normalization work over query rows."""
    return 1 if npes == 4 and samples >= 8 else samples


def uv_reuse(samples: int, npes: int, dcp_size: int = 1) -> int:
    """Number of value tiles consuming one locally merged latent."""
    return 4 if npes == 4 and samples >= 8 and dcp_size == 1 else 1


def sparse_keys_per_task(samples: int, heads: int = WAVES) -> int:
    """Use narrower sparse tiles when samples or attention heads fill LDS."""

    return 32 if heads > 2 * WAVES else 64


def sample_wave_batches(samples: int) -> int:
    return (samples + WAVES - 1) // WAVES


def ug_task_rounds(inter: int, blocks: int = BLOCKS) -> int:
    return (inter + blocks - 1) // blocks


def down_prefetch_batch(
    samples: int, native_fp4_mfma: bool = False, inter: int = INTER
) -> int:
    if samples <= 4:
        return 9
    if native_fp4_mfma:
        return (
            4 if samples <= 6 else 2 if inter > INTER and samples in (8, 10, 12) else 3
        )
    return 4


def down_x_words(
    samples: int, inter: int, expert_mxfp4: bool, native_fp4_mfma: bool = False
) -> int:
    return (
        samples * MOE_SLOTS * inter // (4 if native_fp4_mfma or not expert_mxfp4 else 2)
    )


def dcp_summary_index(source, sample, tile, item, samples, n_uv):
    return ((source * samples + sample) * n_uv + tile) * DCP_SUMMARY_PAIRS + item


def dcp_uv_owner(tile, output_heads, rank):
    return tile // (V_DIM // UV_TILE) // output_heads == rank


def dcp_local_uv_tile(tile, output_heads):
    return tile % (output_heads * V_DIM // UV_TILE)


def ug_split(samples: int, inter: int = INTER):
    """Return the balanced up/gate leftover split for batches two and four."""

    full_tiles, remainder = divmod((samples * TOP_K + 1) * (inter // UG_TILE), BLOCKS)
    if samples not in (2, 4) or remainder == 0 or BLOCKS % remainder:
        return None
    segments = BLOCKS // remainder
    if (HIDDEN // 128) % segments or (HIDDEN // 128) // segments > WAVES // 2:
        return None
    return full_tiles, segments


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    heads: int,
    npes: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    native_fp4_mfma: bool = False,
):
    """Return byte offsets for per-rank scratch and symmetric peer buffers."""
    from sgl_amd_mono.common.guard import FLAG_BYTES

    tile = row_tile_size(samples, native_fp4_mfma)
    if tile != samples:
        if (samples != 16 and not native_fp4_mfma) or npes != 4:
            raise ValueError("tiled GLM launches require 16 rows at TP4")
        scratch, symmetric = layout(
            tile,
            heads,
            npes,
            sparse_attention_topk,
            with_indexer,
            index_max_seq,
            inter,
            output_heads,
            dcp_size,
            native_fp4_mfma,
        )
        count = samples // tile
        scratch = dict(scratch, _tile_rows=tile, _tile_count=count)
        scratch["_tile_barriers"] = count * scratch["_tile_stride"]
        scratch["_bytes"] = scratch["_tile_barriers"] + _align(
            max(2, count - 1) * (BLOCKS + 1) * 8
        )
        symmetric = dict(symmetric)
        symmetric["flags"] = count * symmetric["_tile_stride"]
        symmetric["_bytes"] = symmetric["flags"] + FLAG_BYTES
        return scratch, symmetric
    split_count = sparse_attention_topk // sparse_keys_per_task(samples, heads)
    output_heads = heads if output_heads is None else output_heads
    pair_bytes = 8
    items = [
        ("q_a", samples * Q_LORA * pair_bytes),
        ("q_an", samples * Q_LORA // 2 * pair_bytes),
        ("kv_a", samples * (KV_LORA + PE_DIM) * pair_bytes),
        ("kvnew", samples * KV_LORA * pair_bytes),
        ("penew", samples * PE_DIM * pair_bytes),
        ("q_nope", samples * heads * NOPE_DIM * pair_bytes),
        ("q_pe", samples * heads * PE_DIM * pair_bytes),
        ("q_lat", samples * heads * KV_LORA * pair_bytes),
        ("sp_acc", samples * split_count * heads * KV_LORA * pair_bytes),
        ("sp_m", samples * split_count * heads * pair_bytes),
        ("sp_l", samples * split_count * heads * pair_bytes),
        ("o", samples * output_heads * V_DIM * pair_bytes),
        ("a", samples * HIDDEN * pair_bytes),
        ("scores", samples * N_EXPERTS * pair_bytes),
        ("xq", samples * HIDDEN // 4 * pair_bytes),
        (
            "xqs",
            samples * (XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS) * pair_bytes,
        ),
        ("sel", samples * MOE_SLOTS * pair_bytes),
        ("prob", samples * MOE_SLOTS * pair_bytes),
        ("mid", samples * MOE_SLOTS * inter * pair_bytes),
        ("ugp", BLOCKS * samples * 2 * UG_TILE * pair_bytes),
        ("xqd", samples * HIDDEN * 4),
    ]
    if with_indexer:
        items += [
            ("index_k", samples * INDEX_DIM * pair_bytes),
            ("index_k_new", samples * INDEX_DIM // 2 * pair_bytes),
            ("index_ready", samples * pair_bytes),
            ("index_w", samples * INDEX_HEADS * pair_bytes),
            ("index_q", samples * INDEX_Q_ROWS // 2 * pair_bytes),
        ]
        if npes == 4 and (samples >= 8 or (native_fp4_mfma and samples == 7)):
            # Separate tags from normalized input and later MoE scratch reuse.
            # Cohorts (including native 14 = 2 x 7 rows) publish SGL's
            # RoPE+Hadamard query here once per row.
            items += [("index_q_rot", samples * INDEX_Q_ROWS // 2 * pair_bytes)]
        items += [("index_scores", samples * index_max_seq * pair_bytes)]
        items += [
            ("indices", samples * sparse_attention_topk * 4),
            ("indices_ready", samples * pair_bytes),
            # parallel top-k: per slice a 4096-bin histogram and its ready pair, per
            # row the threshold (bin prefix, keys still needed, keys in the bin), per
            # slice its winners' positions, its (key, position) candidates and their
            # counts
            ("index_part_hist", samples * INDEX_SEL_SLICES * 4096 * 4),
            ("index_part_ready", samples * INDEX_SEL_SLICES * pair_bytes),
            ("index_thresh", samples * 4 * pair_bytes),
            ("index_win_stage", samples * INDEX_SEL_SLICES * sparse_attention_topk * 4),
            ("index_cand_stage", samples * INDEX_SEL_SLICES * INDEX_CAND_STAGE_CAP * 8),
            ("index_gather_count", samples * INDEX_SEL_SLICES * pair_bytes),
        ]

    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset
    scratch.update(_tile_rows=samples, _tile_count=1, _tile_stride=offset)
    if samples >= 8 and npes == 4:
        scratch["_tile_barriers"] = offset
        scratch["_bytes"] += _align((BLOCKS + 1) * 8)

    part = npes * samples * HIDDEN * pair_bytes
    region = 2 * part
    dcp_part = (
        dcp_size * samples * (heads * V_DIM // UV_TILE) * DCP_SUMMARY_PAIRS * pair_bytes
        if dcp_size > 1
        else 0
    )
    offset = 2 * region + 2 * dcp_part
    symmetric = {
        "attn": 0,
        "ffn": region,
        "dcp": 2 * region,
        "_part_stride": part,
        "_dcp_part_stride": dcp_part,
    }
    # Keep the guard outside every payload arena.
    symmetric["flags"] = offset
    symmetric["_tile_stride"] = offset
    symmetric["_bytes"] = offset + FLAG_BYTES
    return scratch, symmetric


def stage_tasks(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    inter: int = INTER,
    npes: int = 1,
    dcp_size: int = 1,
    cohort_rows: bool = False,
):
    """Return ``(stage name, task count)`` pairs in execution order."""

    tile = row_tile_size(samples, expert_mxfp4)
    if tile != samples:
        if (samples != 16 and not expert_mxfp4) or npes != 4:
            raise ValueError("tiled GLM launches require 16 rows at TP4")
        stages = stage_tasks(
            tile,
            heads,
            sparse_attention_topk,
            with_indexer,
            index_max_seq,
            expert_mxfp4,
            inter,
            npes,
            dcp_size,
            samples // tile == 2,
        )
        return [
            (f"tile{part}/{name}", count)
            for part in range(samples // tile)
            for name, count in stages
        ]
    tasks = [
        ("qkv_a", N_QKV_A),
        ("q_norm", samples),
        ("cache", samples // cache_samples_per_task(samples, npes)),
        ("q_b", heads * (NOPE_DIM + PE_DIM) // Q_B_TILE),
    ]
    if with_indexer:
        tasks += [("index_q", INDEX_Q_ROWS // INDEX_TILE)]
    tasks += [("uk", heads * KV_LORA // UK_TILE)]
    if with_indexer:
        tasks += [
            (
                "index_score",
                samples
                * cdiv(
                    index_max_seq,
                    INDEX_KEYS_PER_TASK,
                ),
            ),
            ("index_hist", samples * index_select_slices(samples, cohort_rows)),
            ("index_thresh", samples),
            ("index_gather", samples * index_select_slices(samples, cohort_rows)),
        ]
        tasks += [("index_select", samples)]
    tasks += [
        (
            "split",
            samples
            * (
                heads
                // (
                    16
                    if expert_mxfp4
                    and npes == 4
                    and heads == 16
                    and (samples == 12 or cohort_rows)
                    else WAVES
                )
            )
            * (sparse_attention_topk // sparse_keys_per_task(samples, heads)),
        ),
        (
            "uv",
            samples * (heads * V_DIM // UV_TILE) // uv_reuse(samples, npes, dcp_size),
        ),
        ("o", N_ROW_TILES),
        ("router", samples * N_ROUTER),
        (
            "ug",
            (
                max(BLOCKS, inter)
                if samples == 1
                else samples
                * max(
                    BLOCKS,
                    ((MOE_SLOTS * inter // UG_TILE + BLOCKS - 1) // BLOCKS) * BLOCKS,
                )
            ),
        ),
        ("down", HIDDEN // dn_tile(samples, expert_mxfp4)),
    ]
    if npes == 4 and samples >= 8:
        union_tasks = (samples * TOP_K + 1) * (inter // 8)
        capacity = cdiv(union_tasks, BLOCKS) * BLOCKS
        if expert_mxfp4 and cohort_rows:
            width = 64 if with_indexer else 128
            chunks = 2 * samples * TOP_K + 2
            rounds = cdiv(chunks * (inter // width), BLOCKS)
            capacity = max(capacity, rounds * (BLOCKS // 2) * (width // 8))
        tasks = [
            (name, max(count, capacity) if name == "ug" else count)
            for name, count in tasks
        ]
    return tasks
