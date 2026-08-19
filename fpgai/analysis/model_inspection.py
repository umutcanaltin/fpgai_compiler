from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from fpgai.capabilities.capabilities import capability_for


def _tensor_dict(
    graph: Any,
    name: str,
) -> Dict[str, Any]:
    spec = graph.get_tensor(name)

    return {
        "name": name,
        "shape": list(getattr(spec, "shape", ()) or ()),
        "dtype": str(getattr(spec, "dtype", "unknown")),
    }


def _constant_summary(
    constants: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], int, int]:
    rows: List[Dict[str, Any]] = []
    total_values = 0
    total_bytes = 0

    for name in sorted(constants):
        value = np.asarray(constants[name])
        count = int(value.size)
        size_bytes = int(value.nbytes)

        total_values += count
        total_bytes += size_bytes

        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "values": count,
                "bytes": size_bytes,
            }
        )

    return rows, total_values, total_bytes


@dataclass(frozen=True)
class ModelInspection:
    model_path: str
    pipeline_mode: str
    graph_name: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    operator_counts: Dict[str, int]
    operators: List[Dict[str, Any]]
    constants: List[Dict[str, Any]]
    parameter_values: int
    parameter_bytes: int
    disallowed_operators: List[str]
    unsupported_operators: List[str]
    limited_operators: List[str]
    gap_audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def compilation_ready(self) -> bool:
        return (
            not self.disallowed_operators
            and not self.unsupported_operators
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "pipeline_mode": self.pipeline_mode,
            "graph_name": self.graph_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "operator_counts": self.operator_counts,
            "operators": self.operators,
            "constants": self.constants,
            "parameter_values": self.parameter_values,
            "parameter_bytes": self.parameter_bytes,
            "disallowed_operators": self.disallowed_operators,
            "unsupported_operators": self.unsupported_operators,
            "limited_operators": self.limited_operators,
            "compilation_ready": self.compilation_ready,
            "gap_audit": self.gap_audit,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
        )

    def summary(self) -> str:
        lines = [
            "=============== FPGAI Model Inspection ===============",
            f"Model                 : {self.model_path}",
            f"Graph                 : {self.graph_name}",
            f"Pipeline mode         : {self.pipeline_mode}",
            f"Inputs                : {self.inputs}",
            f"Outputs               : {self.outputs}",
            f"Operators             : {sum(self.operator_counts.values())}",
            f"Operator counts       : {self.operator_counts}",
            f"Constants             : {len(self.constants)}",
            f"Parameter values      : {self.parameter_values}",
            f"Parameter bytes       : {self.parameter_bytes}",
            "-------------------------------------------------------",
            f"Disallowed operators  : {self.disallowed_operators}",
            f"Unsupported operators : {self.unsupported_operators}",
            f"Limited operators     : {self.limited_operators}",
            f"Compilation ready     : {self.compilation_ready}",
            "=======================================================",
        ]

        return "\n".join(lines)

    def write_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_text(
            self.to_json() + "\n",
            encoding="utf-8",
        )
        return output


def inspect_graph(
    graph: Any,
    *,
    model_path: str,
    pipeline_mode: str,
    allowed_operators: Sequence[str],
) -> ModelInspection:
    allowed = set(allowed_operators)

    operator_counts = dict(
        sorted(
            Counter(
                op.op_type
                for op in graph.ops
            ).items()
        )
    )

    operators: List[Dict[str, Any]] = []
    disallowed: set[str] = set()
    unsupported: set[str] = set()
    limited: set[str] = set()

    for index, op in enumerate(graph.ops):
        capability = capability_for(
            op.op_type,
            pipeline_mode,
        )
        is_allowed = op.op_type in allowed

        if not is_allowed:
            disallowed.add(op.op_type)

        if capability.status == "unsupported":
            unsupported.add(op.op_type)
        elif capability.status == "limited":
            limited.add(op.op_type)

        operators.append(
            {
                "index": index,
                "name": op.name,
                "op_type": op.op_type,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "allowed": is_allowed,
                "capability": capability.to_dict(),
            }
        )

    constants, parameter_values, parameter_bytes = (
        _constant_summary(
            getattr(
                graph,
                "constants",
                {},
            )
            or {}
        )
    )

    from fpgai.analysis.model_gap import audit_model_gaps
    gap_audit = audit_model_gaps(graph, pipeline_mode=pipeline_mode)

    return ModelInspection(
        model_path=str(model_path),
        pipeline_mode=str(pipeline_mode),
        graph_name=str(
            getattr(
                graph,
                "name",
                "main",
            )
        ),
        inputs=[
            _tensor_dict(graph, name)
            for name in graph.inputs
        ],
        outputs=[
            _tensor_dict(graph, name)
            for name in graph.outputs
        ],
        operator_counts=operator_counts,
        operators=operators,
        constants=constants,
        parameter_values=parameter_values,
        parameter_bytes=parameter_bytes,
        disallowed_operators=sorted(disallowed),
        unsupported_operators=sorted(unsupported),
        limited_operators=sorted(limited),
        gap_audit=gap_audit,
    )


def inspect_config(
    cfg: Any,
) -> ModelInspection:
    from fpgai.frontend import import_model_source
    from fpgai.layers.composites import expand_composite_layers
    from fpgai.ir.passes import (
        assign_stable_names,
        insert_activations,
    )

    raw = getattr(
        cfg,
        "raw",
        {},
    ) or {}

    operators = raw.get(
        "operators",
        {},
    ) or {}

    defaults = operators.get(
        "defaults",
        {},
    ) or {}

    activation = defaults.get(
        "activation_insert",
        {},
    ) or {}

    graph = import_model_source(
        cfg.model.path,
        format_hint=getattr(cfg.model, "format", None),
        source_framework=getattr(cfg.model, "framework", None),
        pipeline_mode=getattr(cfg.pipeline, "mode", "inference"),
        target_board=((raw.get("targets", {}).get("platform", {}) or {}).get("board") or (raw.get("targets", {}) or {}).get("board")),
    )
    graph = expand_composite_layers(graph)

    activation_kind = str(
        activation.get(
            "kind",
            "none",
        )
    ).lower()

    if activation_kind != "none":
        graph = insert_activations(
            graph,
            kind=activation_kind,
            alpha=float(
                activation.get(
                    "alpha",
                    0.1,
                )
            ),
            except_last=bool(
                activation.get(
                    "except_last",
                    True,
                )
            ),
        )

    graph = assign_stable_names(graph)

    return inspect_graph(
        graph,
        model_path=cfg.model.path,
        pipeline_mode=cfg.pipeline.mode,
        allowed_operators=cfg.operators.supported,
    )

def write_model_inspection_report(
    inspection: ModelInspection,
    out_dir: str | Path,
    *,
    resource_prediction: Mapping[str, Any] | None = None,
    timing_prediction: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write model-inspection artifacts for CLI/report workflows."""
    output = Path(out_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_json = output / "model_profile.json"
    resource_json = output / "resource_prediction.json"
    timing_json = output / "timing_prediction.json"
    summary_md = output / "prediction_summary.md"
    gap_json = output / "model_gap_audit.json"
    gap_md = output / "model_gap_audit.md"

    inspection.write_json(profile_json)
    gap_json.write_text(json.dumps(inspection.gap_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gap = dict(inspection.gap_audit or {})
    gap_lines = [
        "# FPGAI Real-Model Compilation Gap Audit",
        "",
        f"- Graph: `{gap.get('graph')}`",
        f"- Pipeline mode: `{gap.get('pipeline_mode')}`",
        f"- Operators: `{gap.get('operator_count', 0)}`",
        f"- Operator-contract ready: `{gap.get('compilation_ready_at_operator_contract_level', False)}`",
        "",
        "## Blockers",
        "",
        f"- Unsupported operator types: `{gap.get('unsupported_operator_types', [])}`",
        f"- Limited operator types: `{gap.get('limited_operator_types', [])}`",
        f"- Dynamic-shape operator types: `{gap.get('dynamic_shape_operator_types', [])}`",
        f"- Typed tensor blockers: `{gap.get('typed_tensor_blockers', [])}`",
        f"- Persistent runtime-state blockers: `{[item.get('name') if isinstance(item, dict) else item for item in gap.get('runtime_state_blockers', [])]}`",
        f"- Explicit post-processing partition candidates: `{[item.get('name') for item in gap.get('postprocess_partition_candidates', [])]}`",
        "",
        "## Typed tensors",
        "",
        f"- Integer/boolean tensors: `{[item.get('name') for item in gap.get('integer_tensors', [])]}`",
        f"- Index tensors: `{[item.get('name') for item in gap.get('index_tensors', [])]}`",
        "",
        "## Persistent state",
        "",
        f"- State tensors: `{[item.get('name') for item in gap.get('state_tensors', [])]}`",
        f"- State requirements: `{gap.get('runtime_state_requirements', [])}`",
        "",
        "## Detection output contract",
        "",
        f"- Contract: `{gap.get('detection_output_contract')}`",
        "",
        "## Explicit post-processing boundaries",
        "",
        *([f"- `{item.get('name')}` ({item.get('op_type')}): `{item.get('recommended_partition')}` — {item.get('reason')}" for item in gap.get('postprocess_partition_candidates', [])] or ["- None"]),
        "",
        "The audit is model-agnostic: it reports graph/operator/runtime requirements and does not select a model-specific compiler path.",
        "",
    ]
    gap_md.write_text("\n".join(gap_lines), encoding="utf-8")

    if resource_prediction is not None:
        resource_json.write_text(
            json.dumps(
                dict(resource_prediction),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if timing_prediction is not None:
        timing_json.write_text(
            json.dumps(
                dict(timing_prediction),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    lines = [
        "# FPGAI Model Inspection and Prediction Summary",
        "",
        "This report is generated before HLS/Vivado execution.",
        "Resource and timing prediction artifacts are added in the next prediction step when estimator inputs are available.",
        "",
        "## Model profile",
        "",
        f"- Model: `{inspection.model_path}`",
        f"- Pipeline mode: `{inspection.pipeline_mode}`",
        f"- Compilation ready: `{inspection.compilation_ready}`",
        f"- Operators: `{sum(inspection.operator_counts.values())}`",
        f"- Operator counts: `{inspection.operator_counts}`",
        f"- Parameter values: `{inspection.parameter_values}`",
        f"- Parameter bytes: `{inspection.parameter_bytes}`",
        "",
        "## Operator support",
        "",
        f"- Disallowed operators: `{inspection.disallowed_operators}`",
        f"- Unsupported operators: `{inspection.unsupported_operators}`",
        f"- Limited operators: `{inspection.limited_operators}`",
        f"- Dynamic-shape operator types: `{inspection.gap_audit.get('dynamic_shape_operator_types', [])}`",
        f"- Typed tensor blockers: `{inspection.gap_audit.get('typed_tensor_blockers', [])}`",
        f"- Persistent-state blockers: `{inspection.gap_audit.get('runtime_state_blockers', [])}`",
        "",
        "## Prediction status",
        "",
        "- `model_profile.json`: generated",
        (
            "- `resource_prediction.json`: generated"
            if resource_prediction is not None
            else "- `resource_prediction.json`: not generated"
        ),
        (
            "- `timing_prediction.json`: generated"
            if timing_prediction is not None
            else "- `timing_prediction.json`: not generated"
        ),
        "",
    ]

    summary_md.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    paths = {
        "model_profile_json": str(profile_json),
        "prediction_summary_md": str(summary_md),
        "model_gap_audit_json": str(gap_json),
        "model_gap_audit_md": str(gap_md),
    }

    if resource_prediction is not None:
        paths["resource_prediction_json"] = str(resource_json)

    if timing_prediction is not None:
        paths["timing_prediction_json"] = str(timing_json)

    return paths

