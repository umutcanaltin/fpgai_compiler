"""Parse routed Vivado implementation reports; incomplete routing is not E4."""
from hashlib import sha256
from pathlib import Path
import re

_ALIASES = {
  "lut":("CLB LUTs","Slice LUTs","LUT as Logic"),
  "ff":("CLB Registers","Slice Registers","Register as Flip Flop"),
  "dsp":("DSPs","DSP48 Blocks","DSP48E2"),
  "bram":("Block RAM Tile","RAMB36","Block RAM"),
  "uram":("URAM","UltraRAM")
}

def _resource(text: str, names):
    for name in names:
        match = re.search(r"^\s*\|\s*"+re.escape(name)+
          r"\s*\|\s*([0-9][0-9,]*(?:\.\d+)?)\s*\|",text,re.M|re.I)
        if match:
            number = float(match.group(1).replace(",",""))
            return int(number) if number.is_integer() else number
    return None

def _route_counter(route: str, *labels: str) -> int | None:
    """Read colon-delimited Vivado net counters, including dotted labels.

    Vivado UG906 omits the 'unrouted nets' row entirely when none exist.
    A count of zero cannot be inferred from that omission alone.
    """
    readings = []
    for label in labels:
        pattern = (r"^\s*" + re.escape(label) +
                   r"\.*\s*:\s*([0-9][0-9,]*)\s*:?\s*$")
        readings.extend(int(value.replace(",", "")) for value in
                        re.findall(pattern, route, re.I | re.M))
    if not readings:
        return None
    if len(set(readings)) != 1:
        raise ValueError("Conflicting Vivado route status counters")
    return readings[0]


def _complete_route(route: str) -> bool:
    if "Design Route Status" not in route:
        raise ValueError("Cannot confirm complete routing from the route status report")
    routable = _route_counter(route, "# of routable nets", "Number of routable nets")
    routed = _route_counter(route, "# of fully routed nets", "Number of fully routed nets")
    errors = _route_counter(route, "# of nets with routing errors",
                            "Number of nets with routing errors")
    unrouted = _route_counter(route, "# of unrouted nets",
                              "Number of unrouted nets", "Unrouted Nets")
    if routable is None or routed is None or errors is None:
        raise ValueError("Cannot confirm complete routing from the route status report")
    if errors != 0 or (unrouted is not None and unrouted != 0):
        raise ValueError("Vivado reports unrouted nets or routing errors")
    if routable < 1 or routed != routable:
        raise ValueError("Vivado fully routed net count does not match routable nets")
    return True


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def _worst_negative_slack(timing: str) -> float:
    """Extract WNS from the *Design Timing Summary*, not a path group table."""
    location = timing.find("Design Timing Summary")
    if location == -1:
        raise ValueError("Vivado timing summary does not contain WNS")
    section = timing[location:]
    section = section.split("| Clock Summary", 1)[0]
    section = section.split("Clock Summary", 1)[0]
    # Some Vivado report formats print one label/value per line.
    direct = re.search(r"^\s*WNS\s*\(ns\)\s*[:|]\s*(" + _NUMBER + r")\s*$",
                       section, re.M | re.I)
    if direct:
        return float(direct.group(1))
    # Standard Vivado 2023.2 'Design Timing Summary' header has WNS(ns)
    # as its FIRST column; the next line is a dashed ruler, then numeric
    # columns beginning with WNS. Do not read a later intra-clock table.
    lines = section.splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"^\s*WNS\s*\(ns\)(?:\s|$)", line, re.I):
            continue
        if not ("TNS(ns)" in line or "TNS (ns)" in line):
            continue
        candidates = lines[index + 1:index + 5]
        for candidate in candidates:
            stripped = candidate.strip()
            if not stripped or re.fullmatch(r"[-\s|:]+", stripped):
                continue
            values = re.findall(r"(?<![A-Za-z0-9_.])" + _NUMBER +
                                r"(?![A-Za-z0-9_.])", stripped)
            if len(values) >= 2 and re.match(r"^\s*" + _NUMBER + r"\s+", candidate):
                return float(values[0])
            break
    # Earlier exporter versions sometimes write an explicit Slack label.
    direct = re.search(r"^\s*Slack\s*\(\s*(?:MET|VIOLATED)\s*\)\s*:\s*(" +
                       _NUMBER + r")\s*ns\s*$", section, re.M | re.I)
    if direct:
        return float(direct.group(1))
    raise ValueError("Vivado timing summary does not contain WNS")


def _drc_summary(path: Path) -> dict:
    """Describe implementation-rule violations without implying board readiness.

    E4 refers to routed/timed RTL on a target part, not a pin-constrained
    bitstream. DRC status is a separate, source-attested observation.
    """
    missing={"available":False,"summary_verified":False,
             "violations":None,"critical_warnings":None,
             "unconstrained_io":None,"rules":{},"clean":None,
             "board_deployment_qualified":False}
    if not path.is_file():
        return missing
    contents=path.read_text(errors="replace")
    count=re.search(r"^\s*Violations found:\s*([0-9,]+)\s*$",contents,re.I|re.M)
    # Vivado reports may omit the rule table when there are zero violations.
    if not count:
        return {**missing,"available":True}
    total=int(count.group(1).replace(",",""))
    rows={}
    pattern=(r"^\|\s*([A-Z][A-Z0-9_-]*)\s*\|\s*"
             r"(Critical Warning|Warning|Error|Advisory|Info)\s*\|"
             r"[^|\n]*\|\s*([0-9,]+)\s*\|\s*$")
    for match in re.finditer(pattern,contents,re.I|re.M):
        rule,severity,number=match.groups()
        rule=rule.upper()
        if rule in rows:
            return {**missing,"available":True}
        rows[rule]={"severity":severity,"violations":int(number.replace(",",""))}
    verified=sum(x["violations"] for x in rows.values())==total and (total==0 or bool(rows))
    if not verified:
        return {**missing,"available":True}
    critical=sum(row["violations"] for row in rows.values()
                 if row["severity"].lower() in ("critical warning","error"))
    unconstrained=any(rows.get(rule,{}).get("violations",0)>0
                      for rule in ("NSTD-1","UCIO-1"))
    return {"available":True,"summary_verified":True,
            "violations":total,"critical_warnings":critical,
            "unconstrained_io":unconstrained,"rules":rows,
            "clean":total==0,
            "board_deployment_qualified":False}


def parse_vivado_reports(directory: str | Path) -> dict:
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("Symlinked original Vivado report root")
    report_dir = root/"reports" if (root/"reports").is_dir() else root/"vivado"/"reports"
    if report_dir.is_symlink():
        raise ValueError("Symlinked original Vivado report directory")
    paths = {n:report_dir/n for n in ("route_status.rpt","timing_summary.rpt","utilization.rpt")}
    drc_path=report_dir/"drc.rpt"
    if any(p.is_symlink() for p in (*paths.values(),drc_path)):
        raise ValueError("Symlinked original Vivado physical report rejected")
    absent = [n for n,p in paths.items() if not p.is_file() or not p.read_text().strip()]
    if absent:
        raise FileNotFoundError("Missing Vivado report: "+", ".join(absent))
    route = paths["route_status.rpt"].read_text(errors="replace")
    timing = paths["timing_summary.rpt"].read_text(errors="replace")
    usage = paths["utilization.rpt"].read_text(errors="replace")
    _complete_route(route)
    wns = _worst_negative_slack(timing)
    resources = {k:_resource(usage,v) for k,v in _ALIASES.items()}
    if all(v is None for v in resources.values()):
        raise ValueError("Vivado utilization rows not recognized")
    raw_hashes={n:sha256(p.read_bytes()).hexdigest() for n,p in paths.items()}
    if drc_path.is_file():
        raw_hashes["drc.rpt"]=sha256(drc_path.read_bytes()).hexdigest()
    return {"evidence":"E4-route-and-timing-reports",
            "timing":{"wns_ns":wns,"timing_met":wns>=0},
            "utilization":resources,
            "drc":_drc_summary(drc_path),
            "raw_report_sha256":raw_hashes,
            "routed_nets_complete":True}
