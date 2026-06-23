from fpgai.experiments.design_matrix import expand_design_matrix, render_template


def test_expand_design_matrix_cartesian_product():
    cfg = {
        "name": "demo",
        "command_template": "python main.py benchmark --config {config_path} --policy {policy}",
        "defaults": {"config_path": "configs/examples/default_compile.yml", "board": "kv260"},
        "parameters": {"policy": ["balanced", "latency_first"], "bits": [8, 16]},
    }
    points = expand_design_matrix(cfg)
    assert len(points) == 4
    assert points[0].board == "kv260"
    assert "--policy balanced" in points[0].command
    assert points[-1].parameters["bits"] == 16


def test_expand_design_matrix_limit():
    cfg = {"name": "demo", "parameters": {"a": [1, 2, 3], "b": [4, 5]}}
    points = expand_design_matrix(cfg, limit=3)
    assert len(points) == 3


def test_render_template_leaves_unknown_tokens():
    assert render_template("x {known} {missing}", {"known": 1}) == "x 1 {missing}"
