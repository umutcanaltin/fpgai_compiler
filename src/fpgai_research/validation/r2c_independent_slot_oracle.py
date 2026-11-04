"""Independent bounded exact source lifetime/slot oracle for R2c experiments.

Intentionally DOES NOT import or call the R2c lifetime extractor,
greedy allocator, peak checker, source-address verifier or their
internal data structures. It independently reconstructs first
COMMIT and final READ for each value directly from raw events,
then enumerates all finite k-slot assignments and accepts the
smallest non-overlapping one. Bounded small-domain research oracle;
not a practical general FPGA memory scheduler.
"""
from __future__ import annotations

SCHEMA="fpgai-research/r2c-independent-source-value-bruteforce-oracle/v1"


def oracle_minimum_unit_slots(events,*,max_values:int=11)->dict:
    if (not isinstance(events,tuple) or
            type(max_values) is not int or not 1<=max_values<=12):
        raise ValueError("Independent oracle requires finite immutable events")
    born={}
    last={}
    for i,e in enumerate(events):
        if (type(e.tensor) is not str or not e.tensor or
                type(e.element) is not int or e.element<0 or
                e.kind not in ("COMMIT","READ")):
            raise ValueError("Independent oracle invalid source event")
        key=(e.tensor,e.element)
        if e.kind=="COMMIT":
            if key in born:
                raise ValueError("INDEPENDENT_ORACLE_DUPLICATE_COMMIT")
            born[key]=i
        else:
            if key not in born:
                raise ValueError("INDEPENDENT_ORACLE_PREMATURE_READ")
            last[key]=i
    if born.keys()!=last.keys():
        raise ValueError("INDEPENDENT_ORACLE_INCOMPLETE_VALUE_LIFETIME")
    keys=tuple(sorted(born,key=lambda x:(born[x],x)))
    if len(keys)>max_values:
        raise ValueError("Independent brute-force reference exceeded tiny domain")
    spans=[(born[key],last[key]) for key in keys]
    n=len(spans)
    if not n:
        return {"schema":SCHEMA,"status":"EXACT_ORACLE_VERIFIED",
                "minimum_unit_slots":0,"values":0,
                "source_events":len(events),"original_vendor_calls":0}
    def overlaps(i,j):
        return spans[i][0]<=spans[j][1] and spans[j][0]<=spans[i][1]
    for slots in range(1,n+1):
        assigned=[-1]*n
        def dfs(i):
            if i==n:
                return True
            forbidden={assigned[j] for j in range(i)
                       if overlaps(i,j) and assigned[j]>=0}
            for physical in range(slots):
                if physical not in forbidden:
                    assigned[i]=physical
                    if dfs(i+1):
                        return True
            assigned[i]=-1
            return False
        if dfs(0):
            return {
                "schema":SCHEMA,"status":"EXACT_ORACLE_VERIFIED",
                "minimum_unit_slots":slots,
                "values":n,"source_events":len(events),
                "method":"independent-event-scan-and-exhaustive-coloring",
                "original_vendor_calls":0,
                "limitations":[
                    "Finite scalar-value intervals only, no port/bank/arbitration",
                    "Fixed total source event order; no dynamic hardware schedules",
                ],
            }
    raise AssertionError("Independent bounded slot enumeration exhausted")
