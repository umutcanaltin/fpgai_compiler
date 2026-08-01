from pathlib import Path
from fpgai.operators.external import validate_operator_module_in_subprocess

def test_subprocess_loader_validates_symbol(tmp_path: Path):
    module=tmp_path/"plugin.py"; module.write_text("def define_operator():\n    return None\n")
    ok,issue=validate_operator_module_in_subprocess(tmp_path,"plugin.py","define_operator")
    assert ok and issue is None
