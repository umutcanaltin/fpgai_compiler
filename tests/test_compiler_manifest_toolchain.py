from __future__ import annotations

from types import SimpleNamespace

from fpgai.engine.compiler import Compiler


def test_resolved_toolchain_summary_preserves_vivado_and_vitis_settings64() -> None:
    from fpgai.engine.compiler import _resolved_toolchain_summary

    raw = {
        "toolchain": {
            "vivado": {
                "enabled": True,
                "settings64": "/tools/Xilinx/Vivado/2023.2/settings64.sh",
                "executable": "vivado",
                "extra_private_key": "should_not_be_copied",
            },
            "vitis_hls": {
                "enabled": True,
                "settings64": "/tools/Xilinx/Vitis_HLS/2023.2/settings64.sh",
                "executable": "vitis_hls",
            },
        }
    }

    summary = _resolved_toolchain_summary(raw)

    assert summary["vivado"]["settings64"] == "/tools/Xilinx/Vivado/2023.2/settings64.sh"
    assert summary["vivado"]["executable"] == "vivado"
    assert "extra_private_key" not in summary["vivado"]
    assert summary["vitis_hls"]["settings64"] == "/tools/Xilinx/Vitis_HLS/2023.2/settings64.sh"


def test_manifest_source_records_toolchain_summary() -> None:
    import pathlib

    source = pathlib.Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")
    assert '"toolchain": _resolved_toolchain_summary(self.cfg.raw)' in source
    assert '"toolchain": _resolved_toolchain_summary(raw)' in source
