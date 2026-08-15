from fpgai.backends.vivado.vivado_bridge import _create_bd_tcl


def test_kv260_bd_uses_requested_pl_clock():
    tcl = _create_bd_tcl('kv260', 'deeplearn', 'zynq_ultra_ps_e', 250.0)
    assert 'CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {250}' in tcl


def test_zynq7000_bd_uses_requested_pl_clock():
    tcl = _create_bd_tcl('pynq_z2', 'deeplearn', 'processing_system7', 125.0)
    assert 'CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {125}' in tcl
