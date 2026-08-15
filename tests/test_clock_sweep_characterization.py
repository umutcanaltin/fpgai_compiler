from fpgai.analysis.clock_sweep import ClockSweepPoint, summarize_clock_sweep


def test_clock_sweep_reports_implementation_fmax_bracket():
    report=summarize_clock_sweep([
        ClockSweepPoint(200,'passed',True,1.0),
        ClockSweepPoint(250,'passed',True,0.2),
        ClockSweepPoint(275,'timing_failed',False,-0.1),
    ])
    assert report['highest_passing_mhz']==250
    assert report['first_failing_mhz_above_pass']==275
    assert report['fmax_bracket_mhz']=={'lower_bound':250,'upper_bound':275}
