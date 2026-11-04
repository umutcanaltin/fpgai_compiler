"""C2 original finite mutation adequacy: preserve every mutant and survivor."""
from __future__ import annotations

import csv
import json

import pytest

from fpgai_research.experiments.cst_certificate_mutation_adequacy import (
    run,FAULTS,SCHEMA,
)


def test_C2_original_complete_E0_mutation_matrix_and_no_E1_inflation(tmp_path):
    result=run()
    assert result["schema"]==SCHEMA
    assert result["prespecified_graphs"]==3
    assert result["prespecified_Phi_policies"]==4
    assert result["prespecified_fault_classes"]==len(FAULTS)==8
    assert result["prespecified_mutants"]==3*4*8
    assert result["original_gcc_calls"]==0
    assert result["original_vitis_calls"]==0
    assert result["E1_observed_fault_detection_established"] is False
    classes=result["class_specific_adequacy"]
    assert set(classes)==set(FAULTS)
    assert all(entry["prespecified_mutants"]==12
               for entry in classes.values())
    # There is no legitimate basis for dropping surviving E0 mutants.
    assert (result["combined_E0_mutants_killed"]+
            result["combined_E0_mutants_survived"])==96
    for fault in ("premature_A_reduction_commit",
                  "premature_B_reduction_commit","premature_Add_commit"):
        assert classes[fault]["generic_safety_killed"]==12
        assert classes[fault]["combined_E0_survivors"]==0
    conform=classes["nonconforming_safe_Phi_issue_order"]
    assert conform["independent_phi_killed"]==12
    # General alias/readiness legality and this particular fixed Phi
    # issue-rule conformity are distinct proof properties.
    assert conform["generic_safety_killed"]==0

    target=tmp_path/"C2_original_adequacy"
    persisted=run(output=target)
    assert persisted["prespecified_mutants"]==96
    assert json.loads((target/"summary.json").read_text())==persisted
    with (target/"cases.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==96
    assert len({(r["shape"],r["policy"],r["fault_class"]) for r in rows})==96
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=target)
