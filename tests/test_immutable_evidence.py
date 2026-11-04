"""Atomic reference evidence must never silently change or disappear."""
import pytest
from fpgai_research.experiments.immutable_evidence import immutable_evidence_directory


def _save(stage, data=b"correct"):
    (stage/"reference_outputs.npy").write_bytes(data)
    (stage/"reference_evaluation.json").write_text('{"evidence":"E0"}')


def test_successful_atomic_publication(tmp_path):
    dest=tmp_path/"reference"
    with immutable_evidence_directory(dest) as stage:
        _save(stage)
        assert not dest.exists()
    assert (dest/"reference_outputs.npy").read_bytes()==b"correct"


def test_identical_rerun_does_not_change_previous_files(tmp_path):
    dest=tmp_path/"reference"
    with immutable_evidence_directory(dest) as stage: _save(stage)
    before=(dest/"reference_outputs.npy").stat().st_mtime_ns
    with immutable_evidence_directory(dest) as stage: _save(stage)
    assert (dest/"reference_outputs.npy").stat().st_mtime_ns==before


def test_changed_evidence_rejected_without_overwrite(tmp_path):
    dest=tmp_path/"reference"
    with immutable_evidence_directory(dest) as stage: _save(stage)
    with pytest.raises(FileExistsError,match="different content"):
        with immutable_evidence_directory(dest) as stage: _save(stage,b"wrong")
    assert (dest/"reference_outputs.npy").read_bytes()==b"correct"


def test_rollback_after_writer_failure(tmp_path):
    dest=tmp_path/"reference"
    with pytest.raises(RuntimeError):
        with immutable_evidence_directory(dest) as stage:
            _save(stage)
            raise RuntimeError("source failed")
    assert not dest.exists()
    assert not list(tmp_path.glob(".reference.stage-*"))


def test_symlink_destination_is_rejected(tmp_path):
    dest=tmp_path/"reference"
    dest.symlink_to(tmp_path/"missing",target_is_directory=True)
    with pytest.raises(FileExistsError):
        with immutable_evidence_directory(dest): pass
    assert dest.is_symlink()


def test_reject_nested_symlink_candidate(tmp_path):
    dest=tmp_path/"reference"
    (tmp_path/"other").write_text("nothing")
    with pytest.raises(ValueError,match="linked"):
        with immutable_evidence_directory(dest) as stage:
            _save(stage)
            (stage/"linked").symlink_to(tmp_path/"other")
    assert not dest.exists()


def test_forged_extra_file_cannot_be_accepted_as_identical(tmp_path):
    dest=tmp_path/"reference"
    with immutable_evidence_directory(dest) as stage: _save(stage)
    (dest/"extra.json").write_text("{}")
    with pytest.raises(FileExistsError,match="different content"):
        with immutable_evidence_directory(dest) as stage: _save(stage)
    assert (dest/"extra.json").exists()


def test_no_empty_receipts(tmp_path):
    with pytest.raises(ValueError,match="empty"):
        with immutable_evidence_directory(tmp_path/"empty"): pass
    assert not (tmp_path/"empty").exists()
