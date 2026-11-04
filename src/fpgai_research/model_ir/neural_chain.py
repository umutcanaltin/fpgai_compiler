"""R2z: typed, bounded end-to-end CNN -> Dense Model IR composition.

Reuses the canonical FunctionalModel/ArchitectureModel and existing
Conv2D/MatMul/Gemm/activation/flatten HLS emission. No duplicate
convolution kernel, no hidden graph rewrite, no tensor-lifetime alias
or physical-BRAM claim. This is a complete-tensor sequential baseline;
branching graphs and arbitrary network topologies remain a different
supported research family.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import prod

from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint, functional_fingerprint, verify_refinement,
)

SCHEMA = "fpgai-research/r2z-cnn-dense-typed-sequential-network-E0/v1"
CONV_KINDS = frozenset(("Conv",))
DENSE_KINDS = frozenset(("Gemm","MatMul"))
PRE_DENSE_KINDS = frozenset(("Conv","Relu","MaxPool","GlobalAveragePool","Flatten"))
POST_DENSE_KINDS = frozenset(("Gemm","MatMul","Relu","Sigmoid","Softmax"))
ALLOWED_KINDS = PRE_DENSE_KINDS | POST_DENSE_KINDS
MAX_MODEL_ELEMENTS = 250_000
MAX_CONSTANT_ELEMENTS = 150_000


@dataclass(frozen=True)
class NeuralStage:
    operator: str
    kind: str
    activation_tensor: str
    output_tensor: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    architecture_iteration_shape: tuple[int, ...]
    requested_pe: int
    requested_simd: int
    requested_replica: int
    requested_fold: int
    requested_tile: tuple[int, ...]
    requested_pipeline_ii: int | None
    exchange: str = "complete-producer-tensor-before-next-stage"


@dataclass(frozen=True)
class NeuralBoundary:
    tensor: str
    producer: str
    consumer: str
    shape: tuple[int, ...]
    logical_elements: int
    source_memory: str = "full-contiguous-materialized-tensor"
    live_region: str = "producer-stage-complete-through-consumer-stage-complete"


@dataclass(frozen=True)
class NeuralChainIR:
    architectural: ArchitecturalModel
    stages: tuple[NeuralStage, ...]
    boundaries: tuple[NeuralBoundary, ...]
    total_declared_intermediate_elements: int
    total_constant_elements: int
    has_conv: bool
    has_dense: bool
    transition_operator: str
    schema: str = SCHEMA


def resolve_neural_chain(
    arch: ArchitecturalModel, *, max_operators: int=24,
    max_model_elements: int=MAX_MODEL_ELEMENTS,
    max_constant_elements: int=MAX_CONSTANT_ELEMENTS,
) -> NeuralChainIR:
    if not isinstance(arch, ArchitecturalModel):
        raise TypeError("R2z requires resolved ArchitecturalModel")
    if (type(max_operators) is not int or not 3 <= max_operators <= 64
            or type(max_model_elements) is not int or max_model_elements < 1
            or type(max_constant_elements) is not int
            or max_constant_elements < 1):
        raise ValueError("R2z requires finite positive source graph limits")
    model = arch.functional
    model.validate()
    if (len(model.inputs)!=1 or len(model.outputs)!=1
            or not 3 <= len(model.operators) <= max_operators
            or len(model.operators) != len(arch.points)):
        raise ValueError("R2z requires bounded single-input/output neural network")
    if model.tensor(model.inputs[0]).data is not None:
        raise ValueError("R2z runtime input must not be an initializer")
    if len(model.tensor(model.inputs[0]).shape) != 4:
        raise ValueError("R2z expects a rank-four NCHW runtime input")
    if not all(op.kind in ALLOWED_KINDS for op in model.operators):
        raise ValueError("R2z unsupported neural layer; no implicit fallback")
    if len({(point.effective.precision_bits, point.effective.integer_bits)
            for point in arch.points}) != 1:
        raise ValueError("R2z mixed fixed-point formats require explicit casts")
    # This recomputes the supported source geometry from FunctionalModel
    # and forbids silent changes to any resolved stage-level mechanism.
    verify_refinement(lower(arch))

    previous=model.inputs[0]
    stages=[]
    boundaries=[]
    const_used=set()
    dense=False
    conv=False
    flat=False
    transition=None
    previous_op=None
    for index, (op,point) in enumerate(zip(model.operators,arch.points)):
        if point.operator != op.name or point.kind != op.kind:
            raise ValueError("R2z operator/architecture identity mismatch")
        if (op.inputs[0]!=previous
                or (index==len(model.operators)-1
                    and op.outputs[0]!=model.outputs[0])):
            raise ValueError("R2z requires one sequential activation chain")
        if index < len(model.operators)-1 and op.outputs[0] in model.outputs:
            raise ValueError("R2z external output must be final stage")
        shape=model.tensor(previous).shape
        produced=model.tensor(op.outputs[0]).shape
        if shape[0] != produced[0]:
            raise ValueError("R2z batch dimension must be preserved")
        if op.kind=="Conv":
            if dense or flat or len(shape)!=4 or len(produced)!=4:
                raise ValueError("R2z Conv2D only before flatten/dense transition")
            if len(op.inputs) not in (2,3):
                raise ValueError("R2z Conv requires static weights and optional bias")
            conv=True
        elif op.kind in ("MaxPool","GlobalAveragePool"):
            if dense or flat or len(shape)!=4 or len(produced)!=4:
                raise ValueError("R2z NCHW pool must precede dense stages")
        elif op.kind=="Flatten":
            if flat or dense or len(shape)!=4 or len(produced)!=2:
                raise ValueError("R2z one NCHW->rank2 Flatten required")
            if int(op.attr("axis",1)) != 1:
                raise ValueError("R2z supports Flatten axis 1 only")
            flat=True
            transition=op.name
        elif op.kind in DENSE_KINDS:
            if not flat or len(shape)!=2 or len(produced)!=2:
                raise ValueError("R2z Dense needs preceding Flatten and rank2 operands")
            if len(op.inputs) not in ((2,3) if op.kind=="Gemm" else (2,)):
                raise ValueError("R2z Dense expects constant RHS and optional Gemm bias")
            dense=True
        elif op.kind in ("Relu","Sigmoid","Softmax"):
            if op.kind!="Relu" and not dense:
                raise ValueError("R2z sigmoid/softmax only following Dense")
            if len(shape) not in (2,4) or produced != shape:
                raise ValueError("R2z activation must preserve shape")
            if not dense and flat:
                raise ValueError("R2z rank2 activation before Dense is not admitted")
        if dense and op.kind not in POST_DENSE_KINDS:
            raise ValueError("R2z no NCHW, Conv or flatten after Dense")
        if flat and op.kind not in (DENSE_KINDS|{"Flatten"}) and not dense:
            raise ValueError("R2z requires Dense directly after Flatten")
        # The existing canonical HLS emitter supports static compile-time
        # kernel/weight/bias arrays. This boundary explicitly excludes
        # dynamic weights and hidden additional input tensors.
        for const_name in op.inputs[1:]:
            tensor=model.tensor(const_name)
            if tensor.data is None:
                raise ValueError("R2z weight or bias must be a static initializer")
            const_used.add(const_name)
        if op.kind not in (CONV_KINDS | DENSE_KINDS) and len(op.inputs)!=1:
            raise ValueError("R2z activation/pool/flatten must be unary")
        if op.kind not in (CONV_KINDS | DENSE_KINDS):
            a=point.effective
            if ((a.parallelism,a.pes,a.simd,a.serialization)!=(1,1,1,1)
                    or a.tile):
                raise ValueError("R2z scalar neural stages do not implement spatial Phi")
        if previous_op is not None:
            boundaries.append(NeuralBoundary(
                previous, previous_op, op.name, shape, prod(shape)
            ))
        a=point.effective
        stages.append(NeuralStage(
            operator=op.name,kind=op.kind,
            activation_tensor=previous,output_tensor=op.outputs[0],
            input_shape=shape,output_shape=produced,
            architecture_iteration_shape=point.iteration_shape,
            requested_pe=a.pes,requested_simd=a.simd,
            requested_replica=a.parallelism,requested_fold=a.serialization,
            requested_tile=a.tile,requested_pipeline_ii=a.pipeline_ii,
        ))
        previous=op.outputs[0]
        previous_op=op.name
    if not (conv and dense and flat) or transition is None:
        raise ValueError("R2z mixed network must contain Conv, Flatten and Dense")
    if previous != model.outputs[0]:
        raise ValueError("R2z final operator must produce network output")
    constant_names={t.name for t in model.tensors if t.data is not None}
    if const_used!=constant_names:
        raise ValueError("R2z unused or undeclared source initializer")
    total=sum(prod(t.shape) for t in model.tensors)
    total_const=sum(prod(model.tensor(n).shape) for n in const_used)
    if total>max_model_elements or total_const>max_constant_elements:
        raise ValueError("R2z finite network tensor/weight budget exceeded")
    return NeuralChainIR(
        architectural=arch,stages=tuple(stages),
        boundaries=tuple(boundaries),
        total_declared_intermediate_elements=sum(b.logical_elements
                                                 for b in boundaries),
        total_constant_elements=total_const,
        has_conv=True,has_dense=True,transition_operator=transition,
    )


def neural_chain_receipt(ir:NeuralChainIR)->dict:
    if not isinstance(ir,NeuralChainIR) or ir.schema!=SCHEMA:
        raise ValueError("R2z typed NeuralChainIR required")
    model=ir.architectural.functional
    contract={
        "schema":SCHEMA,
        "functional_sha256":functional_fingerprint(model),
        "effective_architecture_sha256":
            architecture_fingerprint(ir.architectural),
        "operators":[asdict(x) for x in ir.stages],
        "boundaries":[asdict(x) for x in ir.boundaries],
        "operators_preserved":len(ir.stages),
        "intermediate_boundaries":len(ir.boundaries),
        "intermediate_full_tensor_elements":
            ir.total_declared_intermediate_elements,
        "constant_elements":ir.total_constant_elements,
        "nchw_to_dense_transition":ir.transition_operator,
        "source_memory_policy":"full-materialized-sequential-no-inplace-alias",
        "native_vitis_calls":0,
        "physical_BRAM":None,
        "limitations":[
            "Bounded sequential mixed Conv/Pool/Flatten/Dense/activation only",
            "No branching, dynamic weights, mixed numeric formats or arbitrary NN operators",
            "Full source tensors; no scratch reuse, streaming or in-place implementation",
            "E0 source semantics and requested Phi, not achieved HLS timing/area",
        ],
    }
    contract["contract_sha256"]=sha256(json.dumps(
        contract,sort_keys=True,separators=(",",":"),allow_nan=False
    ).encode()).hexdigest()
    return contract
