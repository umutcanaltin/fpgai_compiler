"""Bounded first-class Phi-conditioned tensor-value relations for static CST row chains.

This refines previously hash-only TensorExchange boundaries with explicit
logical elements, all producer reduction owners, all consumer read owners,
and a source scratch address. The existing row-atomic checker certifies
temporal non-aliasing; the separately executed GCC value tracer checks
actual write/read events.

It does *not* infer HLS cycle-level readiness, bank-port conflicts or
general DAG fan-out. It is an E0 **spatial ownership/read cardinality** IR.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.row_chain import RowChainIR, resolve_row_chain
from fpgai_research.model_ir.temporal_schedule import blocked_schedule, check_schedule


def _hash(value) -> str:
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MappedTensorValue:
    tensor: str
    row: int
    column: int
    producer: str
    consumer: str
    producer_required_contributions: int
    consumer_required_reads: int
    producer_phi_owner_sha256: str
    consumer_phi_read_sha256: str
    source_slot_address: int
    ready_rule: str = "all_logical_producer_contributions_completed"
    release_rule: str = "all_logical_consumer_reads_completed"


@dataclass(frozen=True)
class MappedValueIR:
    row_chain: RowChainIR
    row_block: int
    values: tuple[MappedTensorValue,...]
    schedule_sha256: str
    projection_sha256: str
    evidence: str = "E0-explicit-bounded-Phi-spatial-value-contracts"


def resolve_mapped_values(architecture: ArchitecturalModel, *,
                          row_block: int = 1,
                          max_values: int = 20_000,
                          max_contributions: int = 100_000) -> MappedValueIR:
    if type(max_values) is not int or max_values<1:
        raise ValueError("Finite positive mapped-value bound required")
    if type(max_contributions) is not int or max_contributions<1:
        raise ValueError("Finite positive reduction/read event bound required")
    ir=resolve_row_chain(architecture)
    if type(row_block) is not int or not 1<=row_block<=ir.row_count:
        raise ValueError("Invalid row_block for mapped values")
    schedule=blocked_schedule(ir.row_count,len(ir.chain),row_block)
    proof=check_schedule(ir,schedule,row_slots=row_block,
                         max_element_events=max_contributions)
    if proof["status"]!="VERIFIED":
        raise ValueError("Mapped tensor values require independently verified whole-row plan")
    f=architecture.functional
    ops=f.operators
    points=architecture.points
    count=sum(prod(b.shape) for b in ir.composed.boundaries)
    needed=0
    for stage,b in enumerate(ir.composed.boundaries):
        prod_op=ops[stage]
        con_op=ops[stage+1]
        producer_contributions=(points[stage].iteration_shape[2]
            if prod_op.kind in ("MatMul","Gemm") else 1)
        consumer_reads=(points[stage+1].iteration_shape[1]
            if con_op.kind in ("MatMul","Gemm") else 1)
        needed += prod(b.shape)*(producer_contributions+consumer_reads)
    if count>max_values or needed>max_contributions:
        raise ValueError("Exact mapped tensor relation exceeds finite event bound")
    values=[]
    for stage,b in enumerate(ir.composed.boundaries):
        upstream,downstream=ops[stage:stage+2]
        producer_point,consumer_point=points[stage:stage+2]
        m,n=b.shape
        pk=(producer_point.iteration_shape[2]
            if upstream.kind in ("MatMul","Gemm") else 1)
        cn=(consumer_point.iteration_shape[1]
            if downstream.kind in ("MatMul","Gemm") else 1)
        for row in range(m):
            for col in range(n):
                producer_sites = [
                    ((row,col,k),coordinate(producer_point,(row,col,k)))
                    for k in range(pk)
                ] if upstream.kind in ("MatMul","Gemm") else [
                    ((row,col),coordinate(producer_point,(row,col)))
                ]
                consumer_sites = [
                    ((row,nc,col),coordinate(consumer_point,(row,nc,col)))
                    for nc in range(cn)
                ] if downstream.kind in ("MatMul","Gemm") else [
                    ((row,col),coordinate(consumer_point,(row,col)))
                ]
                values.append(MappedTensorValue(
                    tensor=b.tensor,row=row,column=col,
                    producer=upstream.name,consumer=downstream.name,
                    producer_required_contributions=len(producer_sites),
                    consumer_required_reads=len(consumer_sites),
                    producer_phi_owner_sha256=_hash(producer_sites),
                    consumer_phi_read_sha256=_hash(consumer_sites),
                    source_slot_address=(row%row_block)*n+col))
    payload=[{
        "tensor":v.tensor,"element":[v.row,v.column],
        "producer":v.producer,"consumer":v.consumer,
        "contributions":v.producer_required_contributions,
        "reads":v.consumer_required_reads,
        "producer_phi":v.producer_phi_owner_sha256,
        "consumer_phi":v.consumer_phi_read_sha256,
        "slot":v.source_slot_address,
    } for v in values]
    return MappedValueIR(ir,row_block,tuple(values),
                         proof["schedule_sha256"],_hash(payload))


def mapped_value_receipt(ir: MappedValueIR) -> dict:
    if not isinstance(ir,MappedValueIR):
        raise TypeError("Expected typed MappedValueIR")
    return {
        "schema":"fpgai-research/cst-mapped-value-ir/v1",
        "row_block":ir.row_block,"value_elements":len(ir.values),
        "logical_producer_contributions":sum(
            x.producer_required_contributions for x in ir.values),
        "logical_consumer_reads":sum(x.consumer_required_reads for x in ir.values),
        "phi_conditioned_projection_sha256":ir.projection_sha256,
        "verified_row_stage_schedule_sha256":ir.schedule_sha256,
        "first_value_projections":[{
            "tensor":v.tensor,"logical_element":[v.row,v.column],
            "producer_phi":v.producer_phi_owner_sha256,
            "consumer_phi":v.consumer_phi_read_sha256,
            "source_slot":v.source_slot_address,
        } for v in ir.values[:3]],
        "evidence":"E0-spatial-Phi-ownership-and-logical-read-cardinality",
        "native_vitis_calls":0,
        "limits":[
            "Whole-row atomic schedule, not an inferred element/cycle event ordering",
            "Single-consumer sequential static chain; no arbitrary DAG fan-out",
            "Source value readiness requires separate actual-GCC E1 evidence",
            "No physical bank allocation, port modeling or original Vitis result",
        ],
    }
