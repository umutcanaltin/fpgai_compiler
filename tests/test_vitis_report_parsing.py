"""Vitis XML resource estimator must never confuse capacity with utilization."""
import pytest
from fpgai_research.toolchains.vitis_hls.package import parse_csynth_xml

def test_used_resources_are_not_available_capacity(tmp_path):
    file=tmp_path/"fpgai_model_csynth.xml"
    file.write_text("""<Profile><AreaEstimates>
      <AvailableResources><LUT>100000</LUT></AvailableResources>
      <Resources><LUT>107</LUT><FF>33</FF><DSP>2</DSP></Resources>
      </AreaEstimates><PerformanceEstimates><SummaryOfOverallLatency>
      <Best-caseLatency>123</Best-caseLatency><Interval-min>1</Interval-min>
      </SummaryOfOverallLatency></PerformanceEstimates></Profile>""")
    result=parse_csynth_xml(str(file))
    assert result["measurements"]["lut"]=="107"
    assert result["measurements"]["ii_min"]=="1"

def test_capacity_only_report_is_not_accepted(tmp_path):
    file=tmp_path/"fpgai_model_csynth.xml"
    file.write_text("<Profile><AreaEstimates><AvailableResources><LUT>100000</LUT></AvailableResources></AreaEstimates></Profile>")
    with pytest.raises(ValueError,match="lacks area estimates"):
        parse_csynth_xml(str(file))


def test_undefined_native_latency_is_not_numerical(tmp_path):
    """Vitis 'undef' must not become a cycle count or HLS E3 qualification."""
    file=tmp_path/"fpgai_model_csynth.xml"
    file.write_text("""<Profile><AreaEstimates><Resources>
      <LUT>107</LUT><FF>33</FF><DSP>2</DSP>
      </Resources></AreaEstimates><PerformanceEstimates>
      <SummaryOfOverallLatency><Best-caseLatency>undef</Best-caseLatency>
      <Worst-caseLatency>undef</Worst-caseLatency>
      </SummaryOfOverallLatency></PerformanceEstimates></Profile>""")
    with pytest.raises(ValueError,match="latency_best.*undef"):
        parse_csynth_xml(str(file))
