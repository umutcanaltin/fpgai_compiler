"""C2: source-pinned, compiled-once finite decision catalog for any proposer.

Reuses exact original unified_transition from the source compiler, retaining
the *typed accepted next state* for a later decision. This avoids an
unnecessary SECOND compiler transition when consuming an already certified
catalog entry. Constructing the catalog still compiles EVERY candidate,
which may cost more than attempting one blind proposal. E0 is NOT E1/E3/E4.

No Gym, agents, reward models, vendor toolchains or new source compilers.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from fpgai_research.model_ir.decision_interface import DecisionAction, DecisionState
from fpgai_research.model_ir.fanout_decision_interface import FanoutDecisionState
from fpgai_research.model_ir.unified_decision_interface import (
    MAX_ACTIONS, UnifiedDecision, unified_state_hash, unified_transition,
)

SCHEMA = "fpgai-research/C2-source-attested-compiled-once-action-catalog/v1"
CERTIFIED = "BOUNDED_CERTIFIED_E0"
STATUSES = (CERTIFIED, "UNKNOWN_E0", "REJECTED")
State = DecisionState | FanoutDecisionState


def _sha64(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class CompiledActionCatalog:
    parent: State
    parent_sha256: str
    actions: tuple[DecisionAction, ...]
    decisions: tuple[UnifiedDecision, ...]
    compiler_transitions: int
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != SCHEMA
                or type(self.actions) is not tuple
                or type(self.decisions) is not tuple
                or len(self.actions) != len(self.decisions)
                or not 0 <= len(self.actions) <= MAX_ACTIONS
                or self.compiler_transitions != len(self.actions)
                or not _sha64(self.parent_sha256)):
            raise ValueError("C2 compiled-once catalog population invalid")

    def mask(self) -> tuple[bool, ...]:
        """Strict certified source-only mask, never promotes UNKNOWN."""
        return tuple(dec.status == CERTIFIED and dec.e0_certified
                     for dec in self.decisions)

    def statuses(self) -> tuple[str, ...]:
        return tuple(dec.status for dec in self.decisions)

    def as_receipt(self) -> dict:
        return {
            "schema": self.schema,
            "parent_source_state_sha256": self.parent_sha256,
            "candidate_count": len(self.actions),
            "statuses": self.statuses(),
            "e0_certified_mask": self.mask(),
            "rows": [d.as_receipt() for d in self.decisions],
            "compiler_transitions_to_construct": self.compiler_transitions,
            "additional_compiler_transitions_to_select_cached": 0,
            "original_gcc_calls": 0, "native_vitis_calls": 0,
            "original_vivado_calls": 0, "trained_agents": 0,
            "hardware_feasibility_checked": False,
            "limitations": [
                "Catalog construction invokes the ordinary compiler once for every action",
                "One blind proposal may be less expensive than full catalog precomputation",
                "Only finite complete-design actions, not partial assignments or hardware legality",
                "A cached E0 source certificate does not prove CSim/CSynth/physical feasibility",
            ],
        }


def build_catalog(state: State,
                  actions: tuple[DecisionAction, ...]) -> CompiledActionCatalog:
    """One *real* source compiler transition per action, exactly once."""
    if (type(actions) is not tuple or len(actions) > MAX_ACTIONS
            or any(not isinstance(a, DecisionAction) for a in actions)):
        raise TypeError("C2 compiled-once catalog requires <=8 typed actions")
    before = unified_state_hash(state)
    decisions = tuple(unified_transition(state, a) for a in actions)
    if unified_state_hash(state) != before:
        raise ValueError("C2 catalog build mutated the original compiler state")
    for a, d in zip(actions, decisions):
        if (not isinstance(d, UnifiedDecision)
                or d.original_state is not state
                or d.parent_state_sha256 != before
                or d.action != a
                or d.status not in STATUSES
                or d.e0_certified is not (d.status == CERTIFIED)
                or d.source_admissible is not (d.status != "REJECTED")):
            raise ValueError("C2 source compiler action decision disagrees with catalog")
        if d.status == "REJECTED":
            if d.next_state is not state or d.state_sha256 is not None:
                raise ValueError("C2 rejected action changed state or source identity")
        else:
            if (d.next_state is state or not _sha64(d.state_sha256)
                    or not _sha64(d.cpp_sha256)
                    or not _sha64(d.certificate_sha256)
                    or unified_state_hash(d.next_state) != d.state_sha256
                    or d.state_sha256 == before):
                raise ValueError("C2 accepted action lacks source-attested next state")
    return CompiledActionCatalog(state, before, actions, decisions, len(actions))


def select_certified(catalog: CompiledActionCatalog, index: int, *,
                     current_state: State) -> UnifiedDecision:
    """Return original certified typed transition without a second compiler.

    Only an in-process, immutable, source-verified catalog is allowed.
    Saved JSON cannot be used to forge a certified next state.
    """
    if not isinstance(catalog, CompiledActionCatalog):
        raise TypeError("C2 selection requires an in-process compiled catalog")
    if type(index) is not int or not 0 <= index < len(catalog.decisions):
        raise IndexError("C2 action index outside enumerated source catalog")
    if current_state is not catalog.parent:
        raise ValueError("C2 cannot apply cached action to a different state object")
    before = unified_state_hash(current_state)
    if before != catalog.parent_sha256:
        raise ValueError("C2 cached action original source identity changed")
    result = catalog.decisions[index]
    if (result.action != catalog.actions[index]
            or result.original_state is not current_state
            or result.parent_state_sha256 != before):
        raise ValueError("C2 cached action binding invalid")
    if result.status != CERTIFIED or result.e0_certified is not True:
        raise ValueError("C2 refuses REJECTED or UNKNOWN_E0 cached action")
    if (result.source_admissible is not True
            or result.next_state is current_state
            or not _sha64(result.cpp_sha256)
            or not _sha64(result.certificate_sha256)
            or not _sha64(result.state_sha256)
            or result.state_sha256 != unified_state_hash(result.next_state)
            or result.state_sha256 == before):
        raise ValueError("C2 cached certified compiler identity invalid")
    return result
