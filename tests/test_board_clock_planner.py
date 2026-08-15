from fpgai.analysis.board_clock_planner import ClockProbePoint, choose_realizable_clock, realizable_targets_from_probes, summarize_clock_probes


def _p(req, actual, status='probed'):
    return ClockProbePoint(req,actual,(1000.0/actual if actual else None),status,'/tmp/probe',0)


def test_clock_policy_is_explicit_and_never_silent():
    values=[300.0,333.333333]
    assert choose_realizable_clock(340,values,policy='exact_only') is None
    assert choose_realizable_clock(340,values,policy='nearest') == 333.333333
    assert choose_realizable_clock(340,values,policy='nearest_below') == 333.333333


def test_probe_summary_and_adaptive_targets_deduplicate_aliases():
    points=[_p(310,300.03),_p(320,300.03),_p(340,333.333333),_p(350,333.333333)]
    assert realizable_targets_from_probes(points) == (300.03,333.333333)
    report=summarize_clock_probes(points)
    assert report['status']=='passed'
    assert len(report['realizable_mhz'])==2
    assert report['request_aliases']['300.03']==[310,320]


def test_probe_tcl_materializes_board_interfaces_and_validates_before_target_generation():
    from fpgai.analysis.board_clock_planner import _probe_tcl
    tcl = _probe_tcl(board_name='kv260', requested_mhz=333.333)
    assert 'validate_bd_design' in tcl
    assert 'make_bd_intf_pins_external [get_bd_intf_pins zynq_ultra_ps_e_0/DDR]' in tcl
    assert 'make_bd_intf_pins_external [get_bd_intf_pins zynq_ultra_ps_e_0/FIXED_IO]' in tcl
    assert 'generate_target all $bd_file' in tcl
    assert 'FPGAI-CLOCK-PROBE ERROR:' in tcl
    assert 'CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {333.333}' in tcl
    assert 'CONFIG.PSU__USE__M_AXI_GP0 {0}' in tcl
    assert 'CONFIG.PSU__USE__S_AXI_GP0 {0}' in tcl
    assert 'connect_bd_net $probe_clk $pin' in tcl


def test_probe_point_serializes_failure_reason():
    point = ClockProbePoint(300.0, None, None, 'tool_failed', '/tmp/probe', 23, 'generate_target failed')
    assert point.to_dict()['failure_reason'] == 'generate_target failed'
