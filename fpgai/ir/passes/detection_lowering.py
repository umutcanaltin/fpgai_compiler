from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from fpgai.ir import Graph


@dataclass(frozen=True)
class DetectionOutputPlan:
    schema: str
    output_tensor: str
    layout: str
    box_format: str
    class_count: int
    score_activation: str
    coordinate_space: str
    pyramid_strides: tuple[int, ...]
    postprocess_partition: str
    nms_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "output_tensor": self.output_tensor,
            "layout": self.layout,
            "box_format": self.box_format,
            "class_count": self.class_count,
            "score_activation": self.score_activation,
            "coordinate_space": self.coordinate_space,
            "pyramid_strides": list(self.pyramid_strides),
            "postprocess_partition": self.postprocess_partition,
            "nms_required": self.nms_required,
        }


def plan_detection_output(
    graph: Graph,
    *,
    output_tensor: str,
    class_count: int,
    layout: str = "batch_channels_candidates",
    box_format: str = "xywh",
    score_activation: str = "sigmoid",
    coordinate_space: str = "input_pixels",
    pyramid_strides: Sequence[int] = (),
    postprocess_partition: str = "ps_or_host",
    nms_required: bool = True,
) -> DetectionOutputPlan:
    """Attach a generic raw-detection output/runtime contract.

    The contract describes the tensor leaving the accelerator and the explicit
    post-processing boundary. It does not identify or special-case a detector
    family; any detection graph may use the same semantics.
    """
    tensor = graph.get_tensor(str(output_tensor))
    if tensor is None:
        raise KeyError(f"IRDET001: unknown detection output tensor {output_tensor!r}")
    if int(class_count) <= 0:
        raise ValueError("IRDET002: class_count must be positive")
    layout_value = str(layout).strip().lower()
    if layout_value not in {"batch_channels_candidates", "batch_candidates_channels"}:
        raise ValueError("IRDET003: unsupported detection output layout")
    box_value = str(box_format).strip().lower()
    if box_value not in {"xywh", "xyxy", "ltrb"}:
        raise ValueError("IRDET004: box_format must be xywh, xyxy, or ltrb")
    score_value = str(score_activation).strip().lower()
    if score_value not in {"sigmoid", "softmax", "none"}:
        raise ValueError("IRDET005: score_activation must be sigmoid, softmax, or none")
    partition = str(postprocess_partition).strip().lower().replace("-", "_")
    if partition not in {"ps", "host", "ps_or_host", "pl", "none"}:
        raise ValueError("IRDET006: unsupported detection postprocess partition")
    strides = tuple(int(x) for x in pyramid_strides)
    if any(x <= 0 for x in strides):
        raise ValueError("IRDET007: pyramid strides must be positive")

    tensor.semantics.tags = tuple(tensor.semantics.tags) + (
        "detection_output",
        f"box_format:{box_value}",
        f"layout:{layout_value}",
    )
    plan = DetectionOutputPlan(
        schema="fpgai.detection-output-plan/v1",
        output_tensor=str(output_tensor),
        layout=layout_value,
        box_format=box_value,
        class_count=int(class_count),
        score_activation=score_value,
        coordinate_space=str(coordinate_space),
        pyramid_strides=strides,
        postprocess_partition=partition,
        nms_required=bool(nms_required),
    )
    payload = plan.to_dict()
    graph.semantics.runtime_contract["detection_output"] = payload
    graph.metadata["detection_output_plan"] = payload
    return plan


@dataclass(frozen=True)
class DetectionDecodePlan:
    schema: str
    distance_tensor: str
    decoded_box_tensor: str
    dfl_bins: int
    grid_origin: float
    pyramid_strides: tuple[int, ...]
    input_box_format: str
    output_box_format: str
    coordinate_space: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "distance_tensor": self.distance_tensor,
            "decoded_box_tensor": self.decoded_box_tensor,
            "dfl_bins": self.dfl_bins,
            "grid_origin": self.grid_origin,
            "pyramid_strides": list(self.pyramid_strides),
            "input_box_format": self.input_box_format,
            "output_box_format": self.output_box_format,
            "coordinate_space": self.coordinate_space,
        }


def plan_detection_decode(
    graph: Graph,
    *,
    distance_tensor: str,
    decoded_box_tensor: str,
    dfl_bins: int,
    pyramid_strides: Sequence[int],
    grid_origin: float = 0.5,
    input_box_format: str = "ltrb",
    output_box_format: str = "xywh",
    coordinate_space: str = "input_pixels",
) -> DetectionDecodePlan:
    """Attach a generic distribution/grid/stride box-decode contract.

    Arithmetic remains ordinary FPGAI ops (Softmax/Mul/ReduceSum/Add/Sub/etc.);
    this plan only records how those tensors are interpreted by the deployment runtime.
    """
    for tensor_name in (distance_tensor, decoded_box_tensor):
        if graph.get_tensor(str(tensor_name)) is None:
            raise KeyError(f"IRDET008: unknown detection decode tensor {tensor_name!r}")
    if int(dfl_bins) <= 1:
        raise ValueError("IRDET009: dfl_bins must be greater than one")
    strides = tuple(int(x) for x in pyramid_strides)
    if not strides or any(x <= 0 for x in strides):
        raise ValueError("IRDET010: detection decode requires positive pyramid strides")
    in_fmt = str(input_box_format).strip().lower()
    out_fmt = str(output_box_format).strip().lower()
    if in_fmt not in {"ltrb", "xywh", "xyxy"} or out_fmt not in {"ltrb", "xywh", "xyxy"}:
        raise ValueError("IRDET011: unsupported box format in detection decode plan")
    plan = DetectionDecodePlan(
        schema="fpgai.detection-decode-plan/v1",
        distance_tensor=str(distance_tensor),
        decoded_box_tensor=str(decoded_box_tensor),
        dfl_bins=int(dfl_bins),
        grid_origin=float(grid_origin),
        pyramid_strides=strides,
        input_box_format=in_fmt,
        output_box_format=out_fmt,
        coordinate_space=str(coordinate_space),
    )
    payload = plan.to_dict()
    graph.semantics.runtime_contract["detection_decode"] = payload
    graph.metadata["detection_decode_plan"] = payload
    graph.get_tensor(str(decoded_box_tensor)).semantics.tags = tuple(graph.get_tensor(str(decoded_box_tensor)).semantics.tags) + ("decoded_boxes", f"box_format:{out_fmt}")
    return plan


__all__ = ["DetectionOutputPlan", "DetectionDecodePlan", "plan_detection_output", "plan_detection_decode"]
