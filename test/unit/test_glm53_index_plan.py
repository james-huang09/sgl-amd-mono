"""Wide row arenas, query coverage and local task planning."""

import pytest

from sgl_amd_mono.glm53.gfx950.layout import (
    layout,
    stage_tasks,
    uv_reuse,
)


@pytest.mark.parametrize("capacity", [4096, 131072])
def test_sixteen_row_arenas_and_guards_are_disjoint(capacity):
    pytest.importorskip("flydsl")
    sc, sy = layout(16, 16, 4, 2048, True, capacity, inter=512)
    tile_sc, tile_sy = layout(8, 16, 4, 2048, True, capacity, inter=512)
    assert sc["_tile_rows"] == 8 and sc["_tile_count"] == 2
    assert sc["_tile_barriers"] == 2 * sc["_tile_stride"]
    assert sc["_bytes"] > sc["_tile_barriers"]
    assert sy["flags"] == 2 * sy["_tile_stride"]
    assert sy["_bytes"] - sy["flags"] == 256
    assert sc["_tile_stride"] == tile_sc["_tile_stride"]
    assert sy["_tile_stride"] == tile_sy["flags"]
    stages = dict(stage_tasks(16, 16, 2048, True, capacity, inter=512, npes=4))
    for tile in range(2):
        assert stages[f"tile{tile}/cache"] == 8
        assert stages[f"tile{tile}/uv"] == 8 * 16


def test_dcp_retains_per_projection_summary_tasks():
    assert uv_reuse(8, 4, 1) == 4
    assert uv_reuse(8, 4, 4) == 1
    tasks = dict(stage_tasks(8, 16, 2048, npes=4, dcp_size=4))
    assert tasks["uv"] == 8 * 16 * 4


@pytest.mark.parametrize("rows,tp", [(0, 4), (13, 4), (15, 4), (17, 4), (16, 8)])
def test_unsupported_physical_width_fails_before_allocation(rows, tp):
    from sgl_amd_mono.glm53.gfx950.runner import Glm53Build

    with pytest.raises(ValueError):
        Glm53Build(rows, tp, 1, True, 4096, True)


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("capacity", [64, 4096, 16384, 16448, 1048576])
def test_query_tiles_cover_all_heads(tp, capacity):
    tasks = dict(stage_tasks(4, 64 // tp, 2048, True, capacity, npes=tp))
    assert tasks["index_q"] == 256
