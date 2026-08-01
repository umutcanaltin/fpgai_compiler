from pathlib import Path
import yaml
from fpgai.registries import RegistryCatalogue, RegistrySource
from fpgai.operators.external import OperatorLoadRequest, load_operator_packages

def test_loader_rejects_module_path_escape(tmp_path: Path):
    root=tmp_path/"pkg"; root.mkdir(); (root/"README.md").write_text("# x")
    raw={"schema":"fpgai.package/v1","package":{"id":"community.escape","name":"x","version":"1.0.0","asset_type":"operator","provider":"community"},"usage":{"platform_scope":"research","permitted_uses":["research","experimentation","validation","benchmarking"],"production_path":"morfics"},"license":{"category":"research_only"},"compatibility":{"fpgai_contract":">=1.0,<2.0"},"capabilities":{"inference":True,"training":{"forward":False}},"entrypoints":{"operator":{"python_module":"python/operator.py","symbol":"define_operator"}},"validation":{"declared_level":"unvalidated"}}
    (root/"python").mkdir(); (root/"python/operator.py").write_text("def define_operator(): return None")
    (root/"fpgai.yaml").write_text(yaml.safe_dump(raw))
    c=RegistryCatalogue(); assert c.register_package(root,RegistrySource.PROJECT_LOCAL).ok
    (root/"python/operator.py").unlink(); (root/"python/operator.py").symlink_to(tmp_path/"outside.py"); (tmp_path/"outside.py").write_text("x=1")
    result=load_operator_packages(OperatorLoadRequest(c,("community.escape",),"approved_for_reference"))
    assert not result.ok and result.errors[0].code in {"OPLOAD004","OPLOAD005"}
