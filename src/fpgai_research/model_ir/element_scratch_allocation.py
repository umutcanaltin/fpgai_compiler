"""CST R2c: exact finite source-event value lifetimes and optimal unit-slot packing.

Research-only SOURCE EVENT model, not FPGA RAM banking, address ports,
physical BRAM requirements, clock cycles or externally novel optimization.
All tensors have scalar unit slots in a single scratch pool. A value
lives from COMMIT until its final actual READ (inclusive); later COMMITs
may reclaim a slot only after that final read.

A deterministic interval-coloring allocator is paired with an
independently implemented bounded exhaustive slot oracle and a separate
source-order address-replay verifier. The optimality statement holds
ONLY for the fixed known total event order and equal-size scalar
scratch values with no bank/port or extra buffering constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from heapq import heappush, heappop
import json


SCHEMA="fpgai-research/cst-r2c-finite-unit-slot-lifetimes-e0/v1"


@dataclass(frozen=True)
class ValueInterval:
    tensor: str
    element: int
    commit_index: int
    last_read_index: int

    @property
    def key(self):
        return (self.tensor,self.element)


@dataclass(frozen=True)
class ScratchAllocation:
    intervals: tuple[ValueInterval,...]
    addresses: tuple[tuple[str,int,int],...]
    unit_slots: int
    peak_simultaneous_live_values: int
    schedule_sha256: str
    schema: str = SCHEMA

    def slot_map(self) -> dict[tuple[str,int],int]:
        return {(name,idx):slot for name,idx,slot in self.addresses}


def _sha(events) -> str:
    from dataclasses import asdict as _asdict
    return sha256(json.dumps([_asdict(e) for e in events],
                             sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


def extract_intervals(events, *, max_events:int=100_000)->tuple[ValueInterval,...]:
    """Fail closed on non-closed lifetimes, duplicates and reads before COMMIT."""
    if (type(max_events) is not int or max_events<1 or
            not isinstance(events,tuple) or len(events)>max_events):
        raise ValueError("Finite immutable original event tuple required")
    commits={}
    final_reads={}
    for index,event in enumerate(events):
        key=(event.tensor,event.element)
        if (not isinstance(event.tensor,str) or
                not event.tensor or type(event.element) is not int or
                event.element<0 or event.kind not in ("COMMIT","READ")):
            raise ValueError("Malformed typed scratch event")
        if event.kind=="COMMIT":
            if key in commits:
                raise ValueError("DUPLICATE_VALUE_COMMIT")
            commits[key]=index
        else:
            if key not in commits:
                raise ValueError("VALUE_READ_BEFORE_COMMIT")
            final_reads[key]=index
    if commits.keys()!=final_reads.keys():
        raise ValueError("VALUE_WITHOUT_FINAL_READ")
    return tuple(sorted((
        ValueInterval(k[0],k[1],start,final_reads[k])
        for k,start in commits.items()
    ),key=lambda x:(x.commit_index,x.tensor,x.element)))


def _peak(intervals)->int:
    events=[]
    for iv in intervals:
        events.append((iv.commit_index,1))
        events.append((iv.last_read_index,-1))
    # All event indices are unique for an original one-event-at-a-time trace.
    live=peak=0
    for _,delta in sorted(events):
        live+=delta
        if live<0:
            raise AssertionError("Negative original value liveness")
        peak=max(peak,live)
    if live:
        raise AssertionError("Incomplete original source value release")
    return peak


def verify_allocation(events,addresses:dict[tuple[str,int],int],*,unit_slots:int):
    """Independent event-by-event alias check; concrete failure reason/address."""
    if (type(unit_slots) is not int or unit_slots<0 or
            not isinstance(addresses,dict) or
            any(type(slot) is not int or not 0<=slot<unit_slots
                for slot in addresses.values())):
        raise ValueError("Invalid finite scratch capacity/address type")
    intervals=extract_intervals(events)
    expected={iv.key for iv in intervals}
    if set(addresses)!=expected:
        raise ValueError("MISSING_OR_EXTRA_VALUE_ADDRESS")
    ends={iv.key:iv.last_read_index for iv in intervals}
    live_by_slot={}
    for i,event in enumerate(events):
        key=(event.tensor,event.element)
        slot=addresses[key]
        if event.kind=="COMMIT":
            if slot in live_by_slot:
                raise ValueError(
                    "OVERWRITE_LIVE_VALUE index="+str(i)+
                    " slot="+str(slot)+" old="+str(live_by_slot[slot])+
                    " new="+str(key))
            live_by_slot[slot]=key
        else:
            if live_by_slot.get(slot)!=key:
                raise ValueError(
                    "READ_STALE_OR_UNREADY_VALUE index="+str(i)+
                    " slot="+str(slot)+" expected="+str(key))
            if ends[key]==i:
                del live_by_slot[slot]
    if live_by_slot:
        raise ValueError("UNRELEASED_SCRATCH_ADDRESSES")
    return {
        "schema":SCHEMA,"status":"EXACT_SOURCE_SLOT_REPLAY_VERIFIED",
        "unit_slots":unit_slots,
        "values":len(expected),
        "peak_simultaneous_live_values":_peak(intervals),
        "original_vendor_calls":0,
        "evidence":"E0-fixed-source-trace-memory-safety-only",
    }


def allocate_minimum_slots(events,*,max_events:int=100_000)->ScratchAllocation:
    """Deterministic optimal interval packing for *one fixed source order*.

    Uses an earliest-release min-heap and reuses the smallest free slot;
    all unit intervals share a total event order, so the chromatic
    number equals maximum simultaneous live values. Independent E0
    replay plus optional exhaustive oracle test this assertion.
    """
    intervals=extract_intervals(events,max_events=max_events)
    occupied=[]  # min-heap of (last_read_event, physical_slot)
    free=[]
    addresses={}
    used=0
    for value in intervals:
        while occupied and occupied[0][0]<value.commit_index:
            _,slot=heappop(occupied)
            heappush(free,slot)
        if free:
            slot=heappop(free)
        else:
            slot=used
            used+=1
        addresses[value.key]=slot
        heappush(occupied,(value.last_read_index,slot))
    peak=_peak(intervals)
    if used!=peak:
        raise AssertionError("Interval packing failed independent peak bound")
    verify_allocation(events,addresses,unit_slots=used)
    ordered=tuple(sorted((name,idx,slot) for (name,idx),slot in addresses.items()))
    return ScratchAllocation(intervals,ordered,used,peak,_sha(events))


def exhaustive_minimum_slots(events,*,max_values:int=12)->int:
    """Independent small-instance reference oracle: k-color overlapping intervals.

    Searches k=1..N with a separate DFS, NOT the heap allocator or its
    computed max-live bound. Factorially expensive; strict finite limit.
    """
    intervals=extract_intervals(events)
    if (type(max_values) is not int or not 1<=max_values<=16 or
            len(intervals)>max_values):
        raise ValueError("Exact oracle supported only on bounded tiny value populations")
    def conflict(a,b):
        return a.commit_index<=b.last_read_index and b.commit_index<=a.last_read_index
    n=len(intervals)
    if n==0:
        return 0
    adjacency=tuple(tuple(j for j in range(i) if conflict(iv,intervals[j]))
                    for i,iv in enumerate(intervals))
    def exists(colors):
        assigned=[-1]*n
        def search(i):
            if i==n:
                return True
            forbidden={assigned[j] for j in adjacency[i] if assigned[j]>=0}
            for slot in range(colors):
                if slot not in forbidden:
                    assigned[i]=slot
                    if search(i+1):
                        return True
                    assigned[i]=-1
            return False
        return search(0)
    for k in range(1,n+1):
        if exists(k):
            return k
    raise AssertionError("Finite interval coloring enumeration did not terminate")
