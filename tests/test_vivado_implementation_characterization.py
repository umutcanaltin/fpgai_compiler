from fpgai.analysis.vivado_implementation_characterization import characterize_vivado_implementation


def _write_common_reports(tmp_path, *, timing_text: str):
    reports = tmp_path / "vivado_bridge" / "reports"
    reports.mkdir(parents=True)
    (reports / "utilization_impl.rpt").write_text("""
| CLB LUTs | 1234 |
| CLB Registers | 2345 |
| Block RAM Tile | 6 |
| DSPs | 12 |
| URAM | 2 |
""", encoding="utf-8")
    (reports / "timing_impl.rpt").write_text(timing_text, encoding="utf-8")
    (reports / "power_impl.rpt").write_text(
        "Total On-Chip Power (W) | 2.50\nDynamic (W) | 2.00\nDevice Static (W) | 0.50\n",
        encoding="utf-8",
    )


def test_vivado_characterization_parses_implementation_reports(tmp_path):
    _write_common_reports(
        tmp_path,
        timing_text="WNS=0.500 | TNS=0.000 | WHS=0.100 | THS=0.000\n",
    )

    payload = characterize_vivado_implementation(
        tmp_path,
        target_clock_mhz=200.0,
        external_provenance={"package_lock": "package-lock.yml"},
    )

    assert payload["status"] == "passed"
    assert payload["validation_level"] == "vivado_implemented"
    assert payload["clock"]["timing_met"] is True
    assert payload["clock"]["derived_fmax_mhz"] > 200.0
    assert payload["clock"]["fmax_derivation_status"] == "slack_based_estimate_only"
    assert payload["resources"] == {"lut": 1234, "ff": 2345, "bram": 6, "dsp": 12, "uram": 2}
    assert payload["power"]["total_on_chip_power_w"] == 2.5


def test_vivado_characterization_does_not_publish_nonphysical_fmax(tmp_path):
    _write_common_reports(
        tmp_path,
        timing_text="WNS=5.216 | TNS=0.000 | WHS=0.010 | THS=0.000\n",
    )

    payload = characterize_vivado_implementation(tmp_path, target_clock_mhz=200.0)

    assert payload["status"] == "passed"
    assert payload["clock"]["timing_met"] is True
    assert payload["clock"]["derived_achieved_period_ns"] is None
    assert payload["clock"]["derived_fmax_mhz"] is None
    assert payload["clock"]["fmax_derivation_status"] == "not_computable_wns_exceeds_or_equals_target_period"
