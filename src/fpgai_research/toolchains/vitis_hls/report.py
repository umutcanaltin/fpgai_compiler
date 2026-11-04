"""Vitis HLS C-simulation outcome and numerical metric parser."""
from pathlib import Path
import math
import re

_METRICS = re.compile(
    r"FPGAI_METRICS\s+samples=(\d+)\s+outputs=(\d+)\s+"
    r"cosine=([^\s]+)\s+rmse=([^\s]+)\s+maxabs=([^\s]+)\s+failures=(\d+)"
)

_CLASSIFICATION = re.compile(
    r"FPGAI_CLASSIFICATION\s+samples=(\d+)\s+compiled_correct=(\d+)\s+"
    r"reference_correct=(\d+)\s+class_disagreements=(\d+)"
)

def discover_csim_success(case_directory: str | Path) -> dict:
    root = Path(case_directory)
    candidates = [root/"logs"/"vitis.stdout.log",
       root/"fpgai_hls_project"/"solution1"/"csim"/"report"/"fpgai_model_csim.log",
       root/"fpgai_hls_project"/"solution1"/"csim"/"build"/"csim_results.log"]
    for file in candidates:
        if not file.is_file():
            continue
        value = file.read_text(errors="replace")
        successful_tool=bool(re.search(r"CSim done with 0 errors|FPGAI_CSIM_STATUS=completed",value,re.I))
        failed_tool="FPGAI_CSIM_STATUS=failed" in value
        # A concatenated log with conflicting or repeated numerical metrics
        # is ambiguous even when its first record looks successful.
        metric_lines=[line for line in value.splitlines() if "FPGAI_METRICS" in line]
        class_lines=[line for line in value.splitlines() if "FPGAI_CLASSIFICATION" in line]
        if len(metric_lines)!=1 or len(class_lines)>1:
            continue
        metrics=_METRICS.search(metric_lines[0])
        if class_lines and not _CLASSIFICATION.search(class_lines[0]):
            continue
        if not (successful_tool or failed_tool) or not metrics:
            continue
        samples, outputs, cosine, rmse, maximum, failures = metrics.groups()
        try:
            values = [float(cosine),float(rmse),float(maximum)]
            if (not all(math.isfinite(x) for x in values)
                or not (-1.000001 <= values[0] <= 1.000001)
                or values[1] < 0 or values[2] < 0):
                continue
        except ValueError:
            continue
        classification=_CLASSIFICATION.search(class_lines[0]) if class_lines else None
        if classification:
            observed_n,compiled,reference,disagreed=map(int,classification.groups())
            if observed_n!=int(samples) or not (0<=compiled<=observed_n and 0<=reference<=observed_n and 0<=disagreed<=observed_n):
                continue
        return {"passed":successful_tool and not failed_tool and int(failures)==0 and int(samples)>0 and int(outputs)>0,
            "marker":"FPGAI_CSIM_STATUS=failed" if failed_tool else "CSim completed","source":str(file.relative_to(root)),
            "samples":int(samples),"outputs_per_sample":int(outputs),
            "cosine":values[0],"rmse":values[1],"max_absolute_error":values[2],
            "failures":int(failures),
            "classification":({
                "samples":observed_n,
                "compiled_correct":compiled,
                "reference_correct":reference,
                "class_disagreements":disagreed,
                "compiled_top1_accuracy":compiled/observed_n,
                "reference_top1_accuracy":reference/observed_n,
                "class_disagreement_rate":disagreed/observed_n
            } if classification else None)}
    return {"passed":False,"marker":None,"source":None,"metrics_missing":True}
