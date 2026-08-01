from pathlib import Path
from fpgai.registries import RegistryCatalogue, RegistrySource
from fpgai.operators.external import OperatorLoadRequest, load_operator_packages, write_external_operator_loading_report

def test_loading_report_is_machine_readable(tmp_path: Path):
    c=RegistryCatalogue(); c.register_package("examples/packages/scale_bias_operator",RegistrySource.PROJECT_LOCAL)
    result=load_operator_packages(OperatorLoadRequest(c,("community.scale_bias_operator",),"approved_for_reference"))
    path=write_external_operator_loading_report(result,tmp_path/"report.json")
    assert '"status": "passed"' in path.read_text()
