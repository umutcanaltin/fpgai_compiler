"""Vendor receipts cannot bind source bytes mutated by the vendor subprocess."""
from hashlib import sha256
from pathlib import Path
import json
import stat
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis


def case(tmp_path):
    c=tmp_path/'case'
    c.mkdir()
    source=c/'model.cpp'
    source.write_text('int original = 1;\n')
    (c/'tb.cpp').write_text('// unchanged testbench\n')
    (c/'manifest.json').write_text(json.dumps({'cpp_sha256':sha256(source.read_bytes()).hexdigest(),
                           'testbench_sha256':sha256((c/'tb.cpp').read_bytes()).hexdigest()}))
    (c/'run_hls.tcl').write_text('exit\n')
    return c


def script(tmp_path,text):
    runner=tmp_path/'fake_tool'
    runner.write_text('#!/bin/sh\n'+text+'\n')
    runner.chmod(runner.stat().st_mode|stat.S_IXUSR)
    return str(runner)


def test_mutation_during_run_is_reported_and_not_attested(tmp_path):
    c=case(tmp_path)
    executable=script(tmp_path,'echo "int modified = 2;" > model.cpp; exit 0')
    receipt=run_vitis(c,executable=executable)
    assert receipt['returncode']==0
    assert not receipt['completed_successfully']
    assert not receipt['inputs_stable_during_execution']
    assert receipt['input_hashes_before_execution']['model_cpp']!=receipt['input_hashes_after_execution']['model_cpp']
    # Restoring original bytes cannot turn this run into E2/E3 evidence.
    (c/'model.cpp').write_text('int original = 1;\n')
    assert summarize_case(str(c))['evidence']=='E1-HLS-source'


def test_unchanged_input_manifest_is_recorded(tmp_path):
    c=case(tmp_path)
    receipt=run_vitis(c,executable=script(tmp_path,'exit 0'))
    assert receipt['completed_successfully']
    assert receipt['inputs_stable_during_execution']
    assert receipt['input_hashes_before_execution']==receipt['input_hashes_after_execution']
