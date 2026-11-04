"""Original external-compiler checkouts are not benchmark executions."""
import json
from pathlib import Path

import pytest

from fpgai_research.experiments import cst_prior_art_environment as peer


def test_no_streamtensor_runnable_pin_fails_closed_without_checkout(tmp_path):
    result=peer.inspect_checkout(peer="StreamTensor",checkout=tmp_path/"missing")
    assert result["exact_checkout"] is False
    assert result["source_checkout_status"]=="NO_VERIFIED_RUNNABLE_REVISION"
    assert result["peer_compiler_executed"] is False


def test_unknown_peer_and_missing_checkout_rejected(tmp_path):
    with pytest.raises(ValueError,match="Unsupported"):
        peer.inspect_checkout(peer="unverified_tool",checkout=tmp_path)
    with pytest.raises(ValueError,match="real directory"):
        peer.inspect_checkout(peer="ST-Flow",checkout=tmp_path/"missing")


def test_exact_source_hash_and_remote_are_not_functionality_proof(
    monkeypatch,tmp_path
):
    checkout=tmp_path/"checkout"
    checkout.mkdir()
    def git(root,*args):
        assert root==checkout
        if args==("rev-parse","--show-toplevel"):
            return str(checkout)
        if args==("rev-parse","HEAD"):
            return peer.PINS["ST-Flow"]["revision"]
        if args==("status","--porcelain"):
            return ""
        if args==("remote","get-url","origin"):
            return "git@github.com:UCLA-VAST/ST-Flow.git"
        pytest.fail("unexpected git call")
    monkeypatch.setattr(peer,"_git",git)
    result=peer.inspect_checkout(peer="ST-Flow",checkout=checkout)
    assert result["exact_checkout"] is True
    assert result["pin_type"]=="PACT26_EVALUATION_TAG"
    assert result["peer_compiler_executed"] is False
    assert result["certificate_parity_established"] is False
    assert result["matched_hardware_qor_established"] is False
    output=tmp_path/"receipt"
    peer.create_report(peer="ST-Flow",checkout=checkout,output=output)
    assert json.loads((output/"peer_preflight.json").read_text())==result
    with pytest.raises(FileExistsError,match="create-only"):
        peer.create_report(peer="ST-Flow",checkout=checkout,output=output)


@pytest.mark.parametrize("remote",[
    "https://attacker.invalid/github.com/UCLA-VAST/ST-Flow",
    "https://github.com/UCLA-VAST/ST-Flow-fork.git",
    "file:///tmp/ST-Flow",
    "https://example.net/fake",
])
def test_no_fake_remote_can_qualify_pin(monkeypatch,tmp_path,remote):
    checkout=tmp_path/"repo"
    checkout.mkdir()
    def git(_, *args):
        return ({
            ("rev-parse","--show-toplevel"):str(checkout),
            ("rev-parse","HEAD"):peer.PINS["ST-Flow"]["revision"],
            ("status","--porcelain"):"",
            ("remote","get-url","origin"):remote,
        })[args]
    monkeypatch.setattr(peer,"_git",git)
    assert peer.inspect_checkout(peer="ST-Flow",checkout=checkout)[
        "exact_checkout"] is False


def test_dirty_peer_checkout_not_treated_as_original_artifact(monkeypatch,tmp_path):
    checkout=tmp_path/"repo"
    checkout.mkdir()
    def git(_, *args):
        return ({
            ("rev-parse","--show-toplevel"):str(checkout),
            ("rev-parse","HEAD"):peer.PINS["Allo"]["revision"],
            ("status","--porcelain"):"M source.cc",
            ("remote","get-url","origin"):"https://github.com/cornell-zhang/allo.git",
        })[args]
    monkeypatch.setattr(peer,"_git",git)
    record=peer.inspect_checkout(peer="Allo",checkout=checkout)
    assert record["clean_checkout"] is False
    assert record["exact_checkout"] is False
    assert record["pin_type"]=="MAIN_SNAPSHOT_NOT_PLDI_EVALUATION_REVISION"
