from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpgai.backends.vivado.boards import board_names, get_board
from fpgai.backends.vivado.vivado_bridge import generate_vivado_bridge_for_artifact


def _minimal_hls_artifact(tmp_path: Path) -> Path:
    build = tmp_path / "artifact" / "build"
    hls = build / "hls"
    hls.mkdir(parents=True)
    (hls / "run_hls.tcl").write_text("# hls script\n", encoding="utf-8")
    (build / "manifest.json").write_text(
        json.dumps({"top_name": "deeplearn"}),
        encoding="utf-8",
    )
    return tmp_path / "artifact"


def test_board_registry_contains_supported_vivado_bridge_boards() -> None:
    assert set(board_names()) >= {"pynq_z2", "kv260", "kr260"}

    pynq = get_board("pynq-z2")
    kv260 = get_board("kv260")
    kr260 = get_board("kr260")

    assert pynq.ps_type == "processing_system7"
    assert kv260.ps_type == "zynq_ultra_ps_e"
    assert kr260.ps_type == "zynq_ultra_ps_e"

    assert pynq.supports_bridge_generation is True
    assert kv260.supports_bridge_generation is True
    assert kr260.supports_bridge_generation is True
    assert kv260.supports_vivado_impl is True
    assert kr260.supports_vivado_impl is True


@pytest.mark.parametrize(
    ("board", "expected_ps", "forbidden_ps", "expected_part"),
    [
        ("pynq_z2", "processing_system7", "zynq_ultra_ps_e", "xc7z020clg400-1"),
        ("kv260", "zynq_ultra_ps_e", "processing_system7", "xck26-sfvc784-2LV-c"),
        ("kr260", "zynq_ultra_ps_e", "processing_system7", "xck26-sfvc784-2LV-c"),
    ],
)
def test_generate_vivado_bridge_uses_board_specific_ps(
    tmp_path: Path,
    board: str,
    expected_ps: str,
    forbidden_ps: str,
    expected_part: str,
) -> None:
    artifact_dir = _minimal_hls_artifact(tmp_path)

    result = generate_vivado_bridge_for_artifact(
        artifact_dir,
        board_name=board,
        run_impl_default=True,
    )

    bridge = Path(result["vivado_bridge_dir"])
    create_bd = (bridge / "scripts" / "create_bd.tcl").read_text(encoding="utf-8")
    run_vivado = (bridge / "scripts" / "run_vivado.tcl").read_text(encoding="utf-8")
    manifest = json.loads((bridge / "vivado_bridge_manifest.json").read_text(encoding="utf-8"))

    assert expected_ps in create_bd
    assert forbidden_ps not in create_bd
    assert f'set part "{expected_part}"' in run_vivado

    assert manifest["board"] == board
    assert manifest["part"] == expected_part
    assert manifest["ps_type"] == expected_ps
    assert manifest["supports_bridge_generation"] is True
    assert manifest["supports_vivado_synth"] is True
    assert manifest["supports_vivado_impl"] is True
    assert manifest["vivado_impl_requested"] is True
    assert manifest["bitstream_requested"] is True


def test_unknown_vivado_board_fails_cleanly(tmp_path: Path) -> None:
    artifact_dir = _minimal_hls_artifact(tmp_path)

    with pytest.raises(KeyError, match="Unknown Vivado board"):
        generate_vivado_bridge_for_artifact(artifact_dir, board_name="unknown_board")
