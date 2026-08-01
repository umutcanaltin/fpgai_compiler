from __future__ import annotations
import subprocess, sys
from pathlib import Path
from .loading_errors import OperatorLoadIssue

def validate_operator_module_in_subprocess(package_root: str|Path, module_path: str, symbol: str, timeout_sec: float=10.0):
    root=Path(package_root).resolve(); target=(root/module_path).resolve()
    if root not in target.parents: return False, OperatorLoadIssue("OPLOAD004","module_path","Unsafe module path")
    code=("import importlib.util,sys\n"
          "p,s=sys.argv[1],sys.argv[2]\n"
          "spec=importlib.util.spec_from_file_location('_fpgai_check',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
          "assert hasattr(m,s)\nprint('ok')\n")
    try:
        cp=subprocess.run([sys.executable,"-c",code,str(target),symbol],cwd=root,env={"PATH":""},capture_output=True,text=True,timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return False, OperatorLoadIssue("OPLOAD012","subprocess","Operator validation timed out")
    if cp.returncode!=0: return False, OperatorLoadIssue("OPLOAD013","subprocess",cp.stderr.strip() or "Subprocess validation failed")
    return True,None
