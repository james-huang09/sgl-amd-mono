"""Check-mode scores must include every row tile, including partial output rows."""

from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.glm53.check import launch_scores
from sgl_amd_mono.glm53.gfx950.layout import layout


@pytest.mark.parametrize("rows", [8, 14, 16])
def test_check_reads_all_cohorts_and_partial_rows(rows):
    sc, sy = layout(16, 16, 4, 2048, True, 64)
    rt = SimpleNamespace(
        key=SimpleNamespace(index_max_seq=64),
        width=16,
        sc=sc,
        scratch=torch.zeros(sc["_bytes"], dtype=torch.uint8),
    )
    expected = torch.arange(16 * 64, dtype=torch.int32).reshape(16, 64)
    tile = sc["_tile_rows"]
    for start in range(0, 16, tile):
        at = sc["index_scores"] + (start // tile) * sc["_tile_stride"]
        rt.scratch[at : at + tile * 64 * 8].view(torch.int32).reshape(tile, 64, 2)[
            ..., 0
        ] = expected[start : start + tile]
    assert torch.equal(launch_scores(rt, rows), expected[:rows])
