"""The GLM-5.3 gfx950 parallel top-k plan on CPU (RFC #43949): a row past
INDEX_SEL_PAR_CHUNKS chunks selects on INDEX_SEL_SLICES CTAs, each a run of whole
chunks, and the kernel calls the same ``select_slices`` on traced values. On Python
ints: the slices tile a row's chunks once in position order, and each width's slice
tasks fit the half of the grid they run on, after the score stage and before the
stages that read the selection."""

import pytest

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.glm53.gfx950 import layout as L
from sgl_amd_mono.glm53.spec import WIDTHS

PAR = L.INDEX_SEL_PAR_CHUNKS * L.INDEX_SEL_CHUNK
BOUNDS = [PAR + 1, 2 * PAR, 2 * PAR + 1, 65536, 65537, 131072, 262144 + 48,
          (1 << 20) - 1, 1 << 20]  # fmt: skip


@pytest.mark.parametrize("bound", BOUNDS + list(range(PAR + 1, 1 << 20, 7919)))
def test_slices_tile_a_rows_chunks_once_in_position_order(bound):
    n, per = L.select_slices(bound)
    chunks = -(-bound // L.INDEX_SEL_CHUNK)
    assert 1 <= n <= L.INDEX_SEL_SLICES and per >= 1
    # slice k takes chunks [k * per, min((k + 1) * per, chunks)): each nonempty, all taken
    assert (n - 1) * per < chunks <= n * per


@pytest.mark.parametrize("rows", [w for w in WIDTHS if w <= 12])
def test_slice_tasks_fit_half_the_grid_between_score_and_split(rows):
    stages = L.stage_tasks(rows, 8, 2048, with_indexer=True, index_max_seq=1 << 20)
    tasks = dict(stages)
    slices = L.index_select_slices(rows)
    assert tasks["index_hist"] == tasks["index_gather"] == rows * slices
    assert rows * slices <= BLOCKS // 2
    assert tasks["index_thresh"] == tasks["index_select"] == rows
    order = [name for name, _ in stages]
    chain = [
        "index_score",
        "index_hist",
        "index_thresh",
        "index_gather",
        "index_select",
        "split",
    ]
    assert [order.index(name) for name in chain] == sorted(
        order.index(name) for name in chain
    )


@pytest.mark.parametrize("rows", range(1, 17))
def test_native_parallel_selection_has_disjoint_tile_storage(rows):
    scratch, symmetric = L.layout(rows, 16, 4, 2048, True, 131072, native_fp4_mfma=True)
    tile, count = scratch["_tile_rows"], scratch["_tile_count"]
    slices = L.index_select_slices(tile, count == 2)
    grid = BLOCKS // (2 if count == 2 else 1)
    assert tile * slices <= grid // 2
    for bound in BOUNDS:
        n, per = L.select_slices(bound, slices)
        chunks = -(-bound // L.INDEX_SEL_CHUNK)
        assert 1 <= n <= slices
        assert (n - 1) * per < chunks <= n * per
    assert scratch["index_part_hist"] < scratch["index_part_ready"]
    assert scratch["index_win_stage"] < scratch["index_cand_stage"]
    assert scratch["index_cand_stage"] < scratch["index_gather_count"]
    assert scratch["index_gather_count"] < scratch["_tile_stride"]
    assert symmetric["flags"] >= count * symmetric["_tile_stride"]
