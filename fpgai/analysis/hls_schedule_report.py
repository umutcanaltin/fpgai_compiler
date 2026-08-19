from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class HLSLoopSchedule:
    """Requested and achieved schedule information for one HLS loop."""

    name: str
    requested_ii: int | None = None
    achieved_ii: int | None = None
    latency_min: int | None = None
    latency_max: int | None = None
    tripcount_min: int | None = None
    tripcount_max: int | None = None

    @property
    def ii_met(self) -> bool | None:
        if self.requested_ii is None or self.achieved_ii is None:
            return None
        return self.achieved_ii <= self.requested_ii

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requested_ii": self.requested_ii,
            "achieved_ii": self.achieved_ii,
            "ii_met": self.ii_met,
            "latency_min": self.latency_min,
            "latency_max": self.latency_max,
            "tripcount_min": self.tripcount_min,
            "tripcount_max": self.tripcount_max,
        }


@dataclass(frozen=True)
class HLSScheduleReport:
    """Normalized HLS scheduling report parsed from text/XML reports."""

    source: str
    loops: tuple[HLSLoopSchedule, ...]

    @property
    def failed_loops(self) -> tuple[HLSLoopSchedule, ...]:
        return tuple(loop for loop in self.loops if loop.ii_met is False)

    @property
    def summary(self) -> dict[str, Any]:
        with_requested = [
            loop for loop in self.loops if loop.requested_ii is not None
        ]
        with_achieved = [
            loop for loop in self.loops if loop.achieved_ii is not None
        ]
        return {
            "source": self.source,
            "loop_count": len(self.loops),
            "loops_with_requested_ii": len(with_requested),
            "loops_with_achieved_ii": len(with_achieved),
            "failed_ii_count": len(self.failed_loops),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "loops": [loop.to_dict() for loop in self.loops],
        }


_INT_RE = re.compile(r"-?\d+")


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "?", "NA", "N/A"}:
        return None
    match = _INT_RE.search(text)
    return int(match.group(0)) if match else None


def _first_text(node: ET.Element, names: Iterable[str]) -> str | None:
    wanted = {name.lower() for name in names}
    for child in node.iter():
        tag = child.tag.split("}")[-1].lower()
        if tag in wanted and child.text is not None:
            text = child.text.strip()
            if text:
                return text
    return None


def _loop_name(node: ET.Element, fallback: str) -> str:
    for key in ("name", "Name", "loop_name", "LoopName"):
        value = node.attrib.get(key)
        if value:
            return str(value)
    text = _first_text(node, ("Name", "LoopName", "loop_name"))
    return text or fallback


def _from_loop_node(node: ET.Element, fallback: str) -> HLSLoopSchedule:
    return HLSLoopSchedule(
        name=_loop_name(node, fallback),
        requested_ii=_parse_int(
            _first_text(
                node,
                (
                    "PipelineII",
                    "TargetII",
                    "RequestedII",
                    "requested_ii",
                    "target_ii",
                ),
            )
        ),
        achieved_ii=_parse_int(
            _first_text(
                node,
                (
                    "AchievedII",
                    "FinalII",
                    "Interval",
                    "achieved_ii",
                    "final_ii",
                ),
            )
        ),
        latency_min=_parse_int(
            _first_text(node, ("LatencyMin", "min_latency", "MinLatency"))
        ),
        latency_max=_parse_int(
            _first_text(node, ("LatencyMax", "max_latency", "MaxLatency"))
        ),
        tripcount_min=_parse_int(
            _first_text(node, ("TripCountMin", "min_tripcount", "MinTripCount"))
        ),
        tripcount_max=_parse_int(
            _first_text(node, ("TripCountMax", "max_tripcount", "MaxTripCount"))
        ),
    )


def parse_hls_schedule_xml(path: str | Path) -> HLSScheduleReport:
    """Parse loop II information from a Vivado/Vitis HLS XML report.

    The parser is intentionally tolerant because report tag names differ
    between HLS versions and between csynth/syn reports.
    """

    report_path = Path(path)
    root = ET.parse(report_path).getroot()
    candidates = []
    for node in root.iter():
        tag = node.tag.split("}")[-1].lower()

        # HLS XML reports usually store loop rows in <Loop> nodes, but
        # parent sections are often named SummaryOfLoopLatency.  Those
        # parents also contain PipelineII/AchievedII as descendants, so a
        # descendant search alone would count the summary section as an
        # extra loop.  Keep only row-like loop records.
        is_loop_record = (
            tag == "loop"
            or tag.endswith("loop")
            or tag in {"looprecord", "loopentry", "loopinfo"}
        ) and not tag.startswith("summary")

        if not is_loop_record:
            continue

        if (
            _first_text(node, ("AchievedII", "FinalII", "Interval"))
            is not None
            or _first_text(node, ("PipelineII", "TargetII", "RequestedII"))
            is not None
        ):
            candidates.append(node)

    loops = tuple(
        _from_loop_node(node, f"loop_{index}")
        for index, node in enumerate(candidates)
    )
    return HLSScheduleReport(source=str(report_path), loops=loops)


_TEXT_LOOP_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.$:-]*)"
    r"[^\n]*?(?:target|requested|pipeline)\s*II\s*[:=]\s*(?P<req>\d+)"
    r"[^\n]*?(?:achieved|final|interval)\s*II\s*[:=]\s*(?P<ach>\d+)",
    re.IGNORECASE,
)

# Vitis HLS 2023.2 emits the scheduling decision in stdout/logs as:
#   Pipelining result : Target II = 1, Final II = 1, Depth = 5, loop 'NAME'
_VITIS_PIPELINING_RESULT_RE = re.compile(
    r"Pipelining\s+result\s*:\s*Target\s+II\s*=\s*(?P<req>\d+)\s*,"
    r"\s*Final\s+II\s*=\s*(?P<ach>\d+)\s*,"
    r"\s*Depth\s*=\s*(?P<depth>\d+)\s*,\s*loop\s+'(?P<name>[^']+)'",
    re.IGNORECASE,
)


def _parse_vitis_csynth_table(text: str) -> tuple[HLSLoopSchedule, ...]:
    """Parse loop rows from the Vitis 2023.2 csynth hierarchy table.

    The table does not preserve the pragma's requested II, but it does expose
    the achieved loop interval, iteration latency and trip count.  Requested
    II therefore remains ``None`` here; when a Vitis log is also available the
    explicit Target/Final II parser provides both values.
    """

    loops: list[HLSLoopSchedule] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 10:
            continue
        columns = [item.strip() for item in stripped.strip("|").split("|")]
        if len(columns) < 9:
            continue
        marker = columns[0]
        if not marker.startswith("o "):
            continue
        name = marker[2:].strip()
        # csynth hierarchy columns: name, issue, slack, latency cycles,
        # latency ns, iteration latency, interval, trip count, pipelined, ...
        achieved = _parse_int(columns[6]) if len(columns) > 6 else None
        iteration_latency = _parse_int(columns[5]) if len(columns) > 5 else None
        trip_count = _parse_int(columns[7]) if len(columns) > 7 else None
        pipelined = columns[8].strip().lower() if len(columns) > 8 else ""
        if achieved is None or pipelined not in {"yes", "true"}:
            continue
        loops.append(
            HLSLoopSchedule(
                name=name,
                requested_ii=None,
                achieved_ii=achieved,
                latency_min=iteration_latency,
                latency_max=iteration_latency,
                tripcount_min=trip_count,
                tripcount_max=trip_count,
            )
        )
    return tuple(loops)


def parse_hls_schedule_text(path: str | Path) -> HLSScheduleReport:
    """Parse requested/achieved II from a plain-text HLS report or log.

    Supports the historic compact ``requested II / achieved II`` form, the
    Vitis HLS 2023.2 scheduling log, and the loop hierarchy table in
    ``csynth.rpt``.
    """

    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8", errors="replace")
    parsed: list[HLSLoopSchedule] = []

    # Parse the historic compact form line-by-line, but leave native Vitis
    # "Pipelining result" records to the dedicated parser below.  The old
    # regex is intentionally permissive and would otherwise misidentify the
    # log prefix (for example ``INFO:``) as a loop name.
    for line in text.splitlines():
        if "pipelining result" in line.lower():
            continue
        for match in _TEXT_LOOP_RE.finditer(line):
            parsed.append(
                HLSLoopSchedule(
                    name=match.group("name"),
                    requested_ii=int(match.group("req")),
                    achieved_ii=int(match.group("ach")),
                )
            )

    for match in _VITIS_PIPELINING_RESULT_RE.finditer(text):
        parsed.append(
            HLSLoopSchedule(
                name=match.group("name"),
                requested_ii=int(match.group("req")),
                achieved_ii=int(match.group("ach")),
                latency_min=int(match.group("depth")),
                latency_max=int(match.group("depth")),
            )
        )

    parsed.extend(_parse_vitis_csynth_table(text))

    # Deduplicate identical observations because some reports embed the same
    # loop row in more than one section.  Preserve distinct requested/achieved
    # observations when they genuinely differ.
    unique: dict[tuple[Any, ...], HLSLoopSchedule] = {}
    for loop in parsed:
        key = (
            loop.name,
            loop.requested_ii,
            loop.achieved_ii,
            loop.latency_min,
            loop.latency_max,
            loop.tripcount_min,
            loop.tripcount_max,
        )
        unique.setdefault(key, loop)

    return HLSScheduleReport(source=str(report_path), loops=tuple(unique.values()))


def parse_hls_schedule_report(path: str | Path) -> HLSScheduleReport:
    """Parse an HLS schedule report from XML or text based on suffix/content."""

    report_path = Path(path)
    if report_path.suffix.lower() == ".xml":
        return parse_hls_schedule_xml(report_path)

    head = report_path.read_text(
        encoding="utf-8",
        errors="replace",
    )[:256].lstrip()
    if head.startswith("<"):
        return parse_hls_schedule_xml(report_path)
    return parse_hls_schedule_text(report_path)


def compare_requested_achieved_ii(
    requested_by_layer: Mapping[str, int],
    report: HLSScheduleReport,
) -> dict[str, Any]:
    """Match requested layer II values against loop names in an HLS report."""

    comparisons: list[dict[str, Any]] = []
    for layer_name, requested_ii in requested_by_layer.items():
        matched = [
            loop
            for loop in report.loops
            if layer_name in loop.name and loop.achieved_ii is not None
        ]
        achieved = min(
            (loop.achieved_ii for loop in matched if loop.achieved_ii is not None),
            default=None,
        )
        met = None if achieved is None else achieved <= int(requested_ii)
        comparisons.append(
            {
                "layer_name": layer_name,
                "requested_ii": int(requested_ii),
                "achieved_ii": achieved,
                "matched_loops": [loop.name for loop in matched],
                "ii_met": met,
            }
        )

    return {
        "source": report.source,
        "layer_count": len(requested_by_layer),
        "matched_layer_count": sum(
            1 for item in comparisons if item["achieved_ii"] is not None
        ),
        "failed_layer_count": sum(
            1 for item in comparisons if item["ii_met"] is False
        ),
        "layers": comparisons,
    }


def discover_hls_schedule_reports(root: str | Path) -> tuple[Path, ...]:
    """Discover likely HLS schedule reports below a project directory.

    The function intentionally accepts common Vivado/Vitis HLS report layouts:
    ``*_csynth.xml``, ``csynth.xml``, ``*_csynth.rpt``, and schedule/log text
    files containing requested/achieved II information.
    """

    root_path = Path(root)
    if not root_path.exists():
        return tuple()

    patterns = (
        "**/*csynth*.xml",
        "**/*csynth*.rpt",
        "**/*schedule*.xml",
        "**/*schedule*.rpt",
        "**/*schedule*.log",
        "**/vitis_hls.log",
        "**/logs/vitis_hls_stdout.log",
        "**/logs/*vitis*hls*.log",
    )
    found: set[Path] = set()
    for pattern in patterns:
        for path in root_path.glob(pattern):
            if path.is_file():
                found.add(path)

    return tuple(sorted(found, key=lambda item: str(item)))


def summarize_hls_schedule_reports(
    root: str | Path,
    requested_by_layer: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Parse all discovered HLS schedule reports into one normalized summary."""

    reports = []
    comparisons = []
    for path in discover_hls_schedule_reports(root):
        try:
            report = parse_hls_schedule_report(path)
        except Exception as exc:  # pragma: no cover - defensive for tool logs
            reports.append(
                {
                    "source": str(path),
                    "parse_error": str(exc),
                    "summary": {
                        "source": str(path),
                        "loop_count": 0,
                        "loops_with_requested_ii": 0,
                        "loops_with_achieved_ii": 0,
                        "failed_ii_count": 0,
                    },
                    "loops": [],
                }
            )
            continue

        report_dict = report.to_dict()
        reports.append(report_dict)
        if requested_by_layer:
            comparisons.append(
                compare_requested_achieved_ii(requested_by_layer, report)
            )

    total_loops = sum(
        int(item.get("summary", {}).get("loop_count", 0)) for item in reports
    )
    failed_loops = sum(
        int(item.get("summary", {}).get("failed_ii_count", 0)) for item in reports
    )
    parse_errors = sum(1 for item in reports if "parse_error" in item)

    matched_layers = max(
        (int(item.get("matched_layer_count", 0)) for item in comparisons),
        default=0,
    )
    failed_layers = max(
        (int(item.get("failed_layer_count", 0)) for item in comparisons),
        default=0,
    )

    return {
        "root": str(Path(root)),
        "report_count": len(reports),
        "parse_error_count": parse_errors,
        "loop_count": total_loops,
        "failed_loop_count": failed_loops,
        "requested_layer_count": len(requested_by_layer or {}),
        "matched_layer_count": matched_layers,
        "failed_layer_count": failed_layers,
        "reports": reports,
        "comparisons": comparisons,
    }


def write_hls_schedule_summary(
    root: str | Path,
    output_path: str | Path,
    requested_by_layer: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Write a normalized HLS schedule summary JSON file."""

    import json

    summary = summarize_hls_schedule_reports(
        root,
        requested_by_layer=requested_by_layer,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
