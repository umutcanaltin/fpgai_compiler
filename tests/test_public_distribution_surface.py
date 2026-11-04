"""Guard public-facing source distribution against internal research leakage.

These checks cover declared packaging and preview policy; they do not scan
Git history, establish third-party rights or authorize a public release.
"""
from __future__ import annotations

from pathlib import Path

from scripts import prepare_public_preview as preview


ROOT=Path(__file__).resolve().parents[1]


def test_curated_sdist_deliberately_omits_historical_research_tree():
    text=(ROOT/"MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include examples *.py *.json *.md" in text
    for line in ("prune paper","prune rl_environment","prune tests",
                 "prune docs/maintainers","prune build","prune dist"):
        assert line in text
    assert "recursive-include docs *.md" not in text
    assert "recursive-include rl_environment" not in text
    assert "recursive-include tests" not in text


def test_gallery_and_community_guides_are_in_clean_room_preview():
    expected={
        "examples/make_toy_onnx.py",
        "examples/make_neural_network_onnx.py",
        "examples/compile_neural_network.py",
        "examples/make_mlp_onnx.py",
        "examples/NETWORK_GALLERY.md",
        "examples/README.md",
        "tests/test_public_neural_network_examples.py",
        "docs/USING_FPGAI.md","docs/SUPPORTED_FEATURES.md",
        "docs/NEUROSCIENTIST_QUICKSTART.md",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
    }
    assert expected <= preview.PUBLIC_FILES | preview.PUBLIC_TESTS
    assert all(preview.allowed(p,profile="compiler") for p in expected)


def test_history_vendor_workflows_native_outputs_remain_excluded():
    denied=(
        "paper/compiler_ir/R3I_ISOLATED_ALLO_CAPABILITY_RUNBOOK_20261104.md",
        "paper/compiler_ir/REPRODUCIBILITY_LEDGER.md",
        "docs/maintainers/COMPILER_IR_README_HISTORY_20261104.md",
        "rl_environment/HANDOFF.md",
        ".github/workflows/compiler-ir-native-vitis.yml",
        ".github/workflows/cpu-regression.yml",
        "build/gallery/private_sample.onnx",
        "examples/generated_private_weights.npz",
        "src/fpgai_research/experiments/private_tensor.pt",
    )
    for path in denied:
        assert not preview.allowed(path,profile="compiler")
        assert not preview.allowed(path,profile="full-research")


def test_public_readme_and_documentation_offer_actionable_network_walkthrough():
    readme=(ROOT/"README.md").read_text(encoding="utf-8")
    examples=(ROOT/"examples/NETWORK_GALLERY.md").read_text(encoding="utf-8")
    guide=(ROOT/"docs/USING_FPGAI.md").read_text(encoding="utf-8")
    for name in ("cnn_pool_dense","cnn_strided_dense",
                 "cnn_two_conv_softmax"):
        assert name in examples
    for text in ("examples/make_neural_network_onnx.py",
                 "examples/compile_neural_network.py"):
        assert text in readme and text in examples and text in guide
    assert "--verify-host" in examples
    assert "make_mlp_onnx.py" in examples
    assert "PRIMEarxiv" not in readme
    assert "RESEARCH_SPRINT_MASTER_PLAN" not in readme
