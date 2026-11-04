"""Malformed or incomplete HLS XML must never become E3 paper measurements."""
import pytest
from fpgai_research.toolchains.vitis_hls.package import parse_csynth_xml


def report(tmp_path, *, lut='10', latency='100', worst='110', ii='1', ii_max='2'):
    f=tmp_path/'report.xml'
    f.write_text('<Profile><AreaEstimates><Resources>'
                 f'<LUT>{lut}</LUT></Resources></AreaEstimates><PerformanceEstimates>'
                 f'<SummaryOfOverallLatency><Best-caseLatency>{latency}</Best-caseLatency>'
                 f'<Worst-caseLatency>{worst}</Worst-caseLatency>'
                 f'<Interval-min>{ii}</Interval-min><Interval-max>{ii_max}</Interval-max>'
                 '</SummaryOfOverallLatency></PerformanceEstimates></Profile>')
    return f


def test_valid_report_keeps_parsed_numbers_and_fingerprint(tmp_path):
    r=parse_csynth_xml(str(report(tmp_path)))
    assert r['measurements']['lut']=='10'
    assert r['measurements']['latency_best']=='100'
    assert len(r['report_sha256'])==64


@pytest.mark.parametrize('kwargs',[
    {'lut':'not-a-number'}, {'latency':'-'}, {'worst':'40'},
    {'ii':'0'}, {'ii_max':'0'}, {'ii':'4','ii_max':'2'},
    {'lut':'-1'}, {'latency':'nan'}
])
def test_bad_report_cannot_become_hls_synthesis_evidence(tmp_path,kwargs):
    with pytest.raises(ValueError,match='HLS'):
        parse_csynth_xml(str(report(tmp_path,**kwargs)))


def test_resource_only_xml_is_not_enough(tmp_path):
    path=tmp_path/'report.xml'
    path.write_text('<Profile><AreaEstimates><Resources><LUT>123</LUT></Resources>'
                    '</AreaEstimates><PerformanceEstimates><SummaryOfOverallLatency>'
                    '</SummaryOfOverallLatency></PerformanceEstimates></Profile>')
    with pytest.raises(ValueError,match='latency'):
        parse_csynth_xml(str(path))
