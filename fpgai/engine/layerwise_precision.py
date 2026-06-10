from __future__ import annotations

from typing import Any, Dict

from fpgai.numerics.precision_policy import (
    resolve_precision_for_op,
)


def resolve_layerwise_precision(
    graph,
    raw_cfg: Dict[str, Any],
) -> None:
    for index, op in enumerate(graph.ops):
        resolved = resolve_precision_for_op(
            raw_cfg,
            op,
            index,
        )

        op.attrs["precision"] = resolved.specs
        op.attrs["precision_tag"] = f"op{index}"
        op.attrs["precision_rule_indices"] = list(
            resolved.matched_rule_indices
        )