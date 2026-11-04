"""Prepare a separate fixed-point integer/fraction split investigation.

Creates new, byte-identical input capsules and single-case model-level
ArchitectureRequests. This is numerical diagnostics, not an eighth
architecture axis and not vendor execution. No original evidence is edited.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from fpgai_research.experiments.full_pipeline import SCHEMA as PLAN_SCHEMA, parse_plan
from fpgai_research.experiments.full_suite import (
    SCHEMA as SUITE_SCHEMA, _source, validate_suite,
)


def _digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def prepare_precision_split_spec(*, suite_path, prepared_suite, directory,
                                 width=8, integer_splits=(3,4,5)) -> dict:
    """Repackage identical pinned study inputs for new immutable 8-bit trials.

    The resulting suite definition can be passed to prepare-suite in a
    different fresh output directory. No local Vitis or Vivado is invoked.
    """
    if type(width) is not int or width < 2:
        raise ValueError("Fixed-point width must be an integer >= 2")
    if (not isinstance(integer_splits, (tuple,list)) or not integer_splits
        or len(set(integer_splits))!=len(integer_splits)
        or any(type(i) is not int or not 1 <= i < width for i in integer_splits)):
        raise ValueError("Integer splits must be unique integers in 1..width-1")
    definition=Path(suite_path)
    if definition.is_symlink() or not definition.is_file():
        raise ValueError("Original suite definition must be a regular file")
    parsed=json.loads(definition.read_text())
    original=validate_suite(parsed)
    source_root=definition.parent.resolve()
    prepared=Path(prepared_suite)
    if prepared.is_symlink() or not prepared.is_dir():
        raise ValueError("Original prepared suite must be a regular directory")
    declared=json.loads((prepared/"suite_results.json").read_text())
    if declared.get("schema") != SUITE_SCHEMA:
        raise ValueError("Prepared suite schema does not match original suite")
    indexed={row.get("id"):row for row in declared.get("studies",[])}
    if set(indexed)!={row["id"] for row in original}:
        raise ValueError("Prepared study inventory differs from source manifest")
    for row in original:
        if indexed[row["id"]].get("status")!="E1-PREPARED":
            raise ValueError("All original study assets must be source-prepared")
    if len(original)*len(integer_splits)>16:
        raise ValueError("Diagnostic sweep exceeds maximum suite model count")
    target=Path(directory)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Refusing to replace precision sweep specification")

    with TemporaryDirectory(prefix=".fpgai-precision-",dir=target.parent) as temp:
        root=Path(temp)
        assets=root/"assets"
        plans=root/"plans"
        assets.mkdir()
        plans.mkdir()
        copied={}
        experiments=[]
        for item in original:
            name=item["id"]
            study=prepared/name
            if study.is_symlink() or not study.is_dir():
                raise ValueError("Expected a regular prepared study: "+name)
            prepared_record=json.loads((study/"full_experiment.json").read_text())
            planfile=study/"experiment_plan.json"
            if planfile.is_symlink() or not planfile.is_file():
                raise ValueError("Missing pinned original experiment plan")
            original_plan=json.loads(planfile.read_text())
            plan=parse_plan(original_plan)
            original_assets={}
            for field, receipt_key in (
                ("onnx","model_sha256"),
                ("input","inputs_sha256"),
                ("labels","labels_sha256")
            ):
                if field not in item:
                    continue
                source=_source(source_root,item[field])
                actual=_digest(source)
                if prepared_record.get(receipt_key)!=actual:
                    raise ValueError(
                        f"Original {name}/{field} bytes disagree with prepared study")
                relative=f"assets/{name}_{field}{source.suffix}"
                copiedfile=root/relative
                copyfile(source,copiedfile)
                if _digest(copiedfile)!=actual:
                    raise ValueError("Source asset changed while copying")
                copied[relative]=actual
                original_assets[field]=relative
            if plan["operator"]!=prepared_record["operator"]:
                raise ValueError("Pinned operator differs from prepared study")
            for integer_bits in integer_splits:
                identity=f"{name}_w{width}i{integer_bits}"
                if len(identity)>40:
                    raise ValueError("Generated suite study ID too long")
                base=original_plan["default"].copy()
                base["precision_bits"]=width
                base["integer_bits"]=integer_bits
                proposed={
                    "schema":PLAN_SCHEMA,
                    "operator":plan["operator"],
                    "default":base,
                    "variants":{},
                    "interactions":[],
                    "target_part":plan["target_part"],
                    "clock_ns":plan["clock_ns"],
                    "tolerance":plan["tolerance"],
                }
                parsed_plan=parse_plan(proposed)
                if len(parsed_plan["points"])!=1:
                    raise AssertionError("Numerical split study must have one baseline case")
                nameplan=f"plans/{identity}.json"
                (root/nameplan).write_text(
                    json.dumps(proposed,sort_keys=True,indent=2)+"\n")
                trial={"id":identity,"family":item["family"],
                       **original_assets,"plan":nameplan}
                experiments.append(trial)
        spec={"schema":SUITE_SCHEMA,"experiments":experiments,
              "notes":"Separate 8-bit signed AP_RND/AP_SAT integer-width diagnostic; not 7-axis study"}
        validate_suite(spec)
        (root/"suite.json").write_text(json.dumps(spec,sort_keys=True,indent=2)+"\n")
        result={
            "schema":"fpgai-research/precision-split-specification/v1",
            "source_suite_sha256":_digest(definition),
            "source_prepared_suite":str(prepared),
            "precision_bits":width,
            "integer_splits":list(integer_splits),
            "case_count":len(experiments),
            "models":sorted({row["id"] for row in original}),
            "copied_asset_sha256":copied,
            "evidence":"E0-configuration-only",
            "vendor_executed":False,
            "warnings":[
                "8-bit range may saturate intermediate activations; CSim required",
                "Standalone split experiments do not prove the 7-axis paper campaign",
                "New Vitis cases must preserve the original output-error tolerance"
            ],
        }
        (root/"precision_study.json").write_text(
            json.dumps(result,sort_keys=True,indent=2)+"\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Precision sweep destination appeared during preparation")
        root.rename(target)
    return result
