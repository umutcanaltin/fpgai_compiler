from pathlib import Path
import re


RUN_ROOT = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs")


def _source(design: str) -> str:
    p = RUN_ROOT / design / "hls/src/deeplearn.cpp"
    if not p.exists():
        return ""
    return p.read_text()


def _types(design: str) -> str:
    p = RUN_ROOT / design / "hls/include/fpgai_types.h"
    if not p.exists():
        return ""
    return p.read_text()


def test_paper_tiling_designs_emit_distinct_dense_tiles_if_generated() -> None:
    designs = {
        "kv260_tiling_small": ("1", "1"),
        "kv260_tiling_medium": ("2", "2"),
        "kv260_tiling_large": ("4", "4"),
    }

    if not all((RUN_ROOT / d).exists() for d in designs):
        return

    sources = {}
    for design, (tile_in, tile_out) in designs.items():
        src = _source(design)
        types = _types(design)
        assert "dense_out_in_tiled" in src
        assert "planner_tile" in types

        pattern = rf"dense_out_in_tiled<8,\s*4,\s*{tile_in},\s*{tile_out},"
        assert re.search(pattern, src), f"{design} did not emit expected first-layer tile"

        assert f"planner_tile: {{'in': {tile_in}, 'out': {tile_out}}}" in types

        sources[design] = src

    assert sources["kv260_tiling_small"] != sources["kv260_tiling_medium"]
    assert sources["kv260_tiling_medium"] != sources["kv260_tiling_large"]


def test_conv_tiling_accepts_generated_tm_tn_tk_aliases() -> None:
    from fpgai.engine import planner as planner_module

    tile = planner_module._conv_tile_from_cfg(
        {"optimization": {"tiling": {"conv": {"tm": 4, "tn": 2, "tk": 3}}}},
        "conv0",
        {"ic": 8, "oc": 16, "oh": 7, "ow": 5},
    )

    assert tile == {
        "ic": 2,
        "oc": 4,
        "oh": 3,
        "ow": 3,
    }


def test_conv_tiling_accepts_layer_specific_generated_tm_tn_tk_aliases() -> None:
    from fpgai.engine import planner as planner_module

    tile = planner_module._conv_tile_from_cfg(
        {
            "optimization": {
                "tiling": {
                    "conv": {"tm": 4, "tn": 2, "tk": 3},
                    "layers": {
                        "conv0": {"tm": 5, "tn": 3, "tk": 2},
                    },
                }
            }
        },
        "conv0",
        {"ic": 8, "oc": 16, "oh": 7, "ow": 5},
    )

    assert tile == {
        "ic": 3,
        "oc": 5,
        "oh": 2,
        "ow": 2,
    }

