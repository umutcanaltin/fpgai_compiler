from fpgai.implementations.mixed_backend.physical import (
    _required_hls_ports,
    _wrapper_source,
)


def test_clockless_hls_ports_are_accepted():
    ports = {
        "input_data": ("input", 16),
        "input_valid": ("input", 1),
        "output_data": ("output", 16),
        "output_valid": ("output", 1),
    }
    mapping = _required_hls_ports(ports)
    assert "clock" not in mapping
    assert "reset" not in mapping
    assert mapping["input_data"] == "input_data"


def test_clockless_hls_wrapper_does_not_emit_clock_binding():
    ports = {
        "input_data": ("input", 16),
        "input_valid": ("input", 1),
        "output_data": ("output", 16),
        "output_valid": ("output", 1),
    }
    mapping = _required_hls_ports(ports)
    source = _wrapper_source(
        top_name="mixed_top",
        hls_top="scale2_hls",
        hls_ports=mapping,
        hls_port_info=ports,
        vhdl_top="scale_bias_vhdl",
        data_width=16,
    )
    assert ".ap_clk(" not in source
    assert ".input_data(input_data)" in source
    assert ".output_valid(hls_valid)" in source
