"""Check-mode scores must come from the selected CP epoch and every row tile."""

from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.glm53.check import launch_scores
from sgl_amd_mono.glm53.gfx950.layout import INDEX_LOCAL_CAPACITY, layout


def runtime(rows, capacity, cp):
    sc, sy = layout(rows, 16, 4, 2048, True, capacity, indexer_cp=cp)
    return SimpleNamespace(
        key=SimpleNamespace(
            index_max_seq=capacity, indexer_cp=cp, launches_per_step=3, tp=4
        ),
        width=rows,
        rank=0,
        sc=sc,
        sy=sy,
        epoch=torch.tensor([2]),
        scratch=torch.zeros(sc["_bytes"], dtype=torch.uint8),
        peer=SimpleNamespace(
            bytes=torch.zeros(sy["_bytes"], dtype=torch.uint8), group=object()
        ),
    )


@pytest.mark.parametrize("cp", [False, True])
def test_check_reads_all_cohorts_and_only_the_current_epoch(cp):
    rt = runtime(16, 64, cp)
    expected = torch.arange(16 * 64, dtype=torch.int32).reshape(16, 64)
    tile = rt.sc["_tile_rows"]
    for start in range(0, 16, tile):
        if cp:
            at = rt.sy["index_scores"] + (start // tile) * rt.sy["_tile_stride"]
            current = at + rt.sy["_index_scores_stride"]
            rt.peer.bytes[at : at + tile * 64 * 8].view(torch.int32).reshape(
                tile, 64, 2
            )[..., 0] = -123
            raw = rt.peer.bytes
        else:
            current = rt.sc["index_scores"] + (start // tile) * rt.sc["_tile_stride"]
            raw = rt.scratch
        raw[current : current + tile * 64 * 8].view(torch.int32).reshape(tile, 64, 2)[
            ..., 0
        ] = expected[start : start + tile]
    assert torch.equal(launch_scores(rt, 14, layer=1), expected[:14])


def test_long_cp_check_reconstructs_live_shards_and_keeps_short_rows(monkeypatch):
    from sgl_amd_mono.glm53.gfx950.layout import index_shard_tiles

    capacity = INDEX_LOCAL_CAPACITY + 64
    rt = runtime(2, capacity, True)
    stride = rt.sy["_index_scores_stride"]
    at = rt.sy["index_scores"] + stride
    rt.peer.bytes[at : at + 2 * INDEX_LOCAL_CAPACITY * 8].view(torch.int32).reshape(
        2, INDEX_LOCAL_CAPACITY, 2
    )[1, :, 0] = 77

    def gather(out, local, group):
        assert group is rt.peer.group
        for rank, tensor in enumerate(out):
            tensor.fill_(rank + 1)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    step = SimpleNamespace(
        positions=torch.tensor([capacity - 1, 5]), slots=torch.tensor([100, 101])
    )
    result = launch_scores(rt, 2, step, layer=1)
    for rank in range(4):
        begin, end = index_shard_tiles(capacity, rank, 4)
        assert (result[0, begin * 64 : min(end * 64, capacity)] == rank + 1).all()
    assert (result[1, :6] == 77).all()
