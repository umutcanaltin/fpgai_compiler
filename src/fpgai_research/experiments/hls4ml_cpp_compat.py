"""Bounded lexical compatibility fix for hls4ml 1.3.0's invalid C++ typedef names.

Some ONNX-derived types remain /path/Name even when the ONNX layer
and tensor identities were normalized. The official writer emitted
these identifiers and Vitis 2023.2 GCC rejected them. Repair only
the exact ill-formed type identifiers and all their C++ references
within the disposable generated E1 project; record before/after
digests. Never change arithmetic tokens or generated trained weights.

This compatibility adapter is part of the comparator methodology;
neither unmodified upstream success nor native E2/E3/E4 is claimed.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re

SCHEMA = "fpgai-research/hls4ml-cpp-identifier-compatibility/v1"
SOURCE_EXTENSIONS = frozenset({".h", ".hpp", ".cpp", ".cc", ".cxx"})
# Detect ONLY an illegal /identifier as a typedef's FINAL declarator.
# C++ support headers legitimately contain division within template args:
#   typedef typename retval<AP_MAX((_AP_W + 7) / 8, 8), _AP_S>::Type RetType;
# The earlier broad detector matched the arithmetic '/' and generated a
# false-positive at ap_int_base.h:123. Do not scan bundled arithmetic libs.
UNSAFE_TYPE = re.compile(
    r"(?m)^[ \t]*typedef[ \t]+[^;\r\n]{1,240}?[ \t]+"
    r"(?P<symbol>/[A-Za-z_][A-Za-z0-9_./]*)"
    r"(?P<invalid_suffix>[ \t]*(?:<[^;\r\n]*)?)[ \t]*;"
)
VALID_TYPE_ALIAS = re.compile(r"/[A-Za-z_][A-Za-z0-9_./]*")
WORD_BOUNDARY = r"[A-Za-z0-9_./]"
MODEL_TYPE_HEADER_NAMES = frozenset({"defines.h", "parameters.h"})


def _is_project_owned_source(project: Path, file: Path) -> bool:
    """Never mutate Vitis/ap_types/nnet_utils/weights vendor or data files."""
    parts = file.relative_to(project).parts
    return len(parts) == 1 or (len(parts) == 2 and parts[0] == "firmware")


def is_model_type_header(file: Path) -> bool:
    """Header is a hls4ml-written model type/config file, not support code."""
    return file.name in MODEL_TYPE_HEADER_NAMES and not any(
        part in {"ap_types", "nnet_utils", "weights"} for part in file.parts
    )


def _cpp_files(project: Path) -> list[Path]:
    found = []
    for file in sorted(project.rglob("*")):
        if file.is_symlink():
            raise ValueError("Generated hls4ml project contains linked source")
        if not file.is_file() or file.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if (file.stat().st_size > 16 * 1024 * 1024
                or len(found) >= 3000):
            raise ValueError("Oversized generated source inventory")
        found.append(file)
    return found


def repair_unsafe_hls4ml_typedefs(project_dir) -> dict:
    """Source-only, bounded, deterministic generated-identifier adaptation."""
    project = Path(project_dir)
    if project.is_symlink() or not project.is_dir():
        raise ValueError("Missing regular generated HLS project")
    # Only project-owned model files can be changed. Scan the entire
    # project for links/bounds, but never parse Vitis bundled arithmetic
    # headers as generated model declarations.
    files = [file for file in _cpp_files(project)
             if _is_project_owned_source(project, file)]
    texts = {file: file.read_text(encoding="utf-8") for file in files}
    offenders = set()
    for file, content in texts.items():
        if not is_model_type_header(file):
            continue
        for match in UNSAFE_TYPE.finditer(content):
            symbol = match.group("symbol")
            if (not VALID_TYPE_ALIAS.fullmatch(symbol)
                    or match.group("invalid_suffix").strip()):
                line = content.count("\n", 0, match.start()) + 1
                raise ValueError(
                    "Unsupported invalid C++ typedef alias syntax in generated header "
                    f"(line {line}); refusing speculative source edits")
            offenders.add(symbol)
    # Sorted identities, not private names, drive the deterministic fix.
    aliases = {}
    for i, symbol in enumerate(sorted(offenders)):
        proposed = f"hls4ml_compat_type_{i}"
        if any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(proposed)}(?![A-Za-z0-9_])",
                         text) for text in texts.values()):
            raise ValueError("Generated C++ type compatibility alias collides")
        aliases[symbol] = proposed
    revised = dict(texts)
    for symbol, target in aliases.items():
        pattern = re.compile(
            rf"(?<!{WORD_BOUNDARY}){re.escape(symbol)}(?!{WORD_BOUNDARY})"
        )
        replacements = 0
        for file, content in revised.items():
            if pattern.search(content):
                # A single token in a string, include path or filename could
                # be numerical/configuration data; refuse uncontrolled edits.
                for line in content.splitlines():
                    if pattern.search(line) and (line.lstrip().startswith("#include")
                                                 or '"' in line or "'" in line):
                        raise ValueError("Unsafe type is embedded in quoted C++ data")
                if "firmware/weights/" in file.relative_to(project).as_posix():
                    raise ValueError("Refusing any changes to generated weight files")
                changed, n = pattern.subn(target, content)
                revised[file] = changed
                replacements += n
        if replacements < 1:
            raise ValueError("Inconsistent generated type repair inventory")
    for file, content in revised.items():
        if is_model_type_header(file):
            match = UNSAFE_TYPE.search(content)
            if match:
                line = content.count("\n", 0, match.start()) + 1
                raise ValueError(
                    f"Unrepaired invalid hls4ml typedef at generated header line {line}")
    modified = []
    for file in files:
        before = texts[file]
        after = revised[file]
        if before == after:
            continue
        rel = file.relative_to(project).as_posix()
        modified.append({
            "relative_path": rel,
            "original_sha256": sha256(before.encode("utf-8")).hexdigest(),
            "adapted_sha256": sha256(after.encode("utf-8")).hexdigest(),
        })
    # Fully validate before writing, so rejected repairs have zero mutation.
    for file in files:
        if revised[file] != texts[file]:
            file.write_text(revised[file], encoding="utf-8")
    return {
        "schema": SCHEMA,
        "renamed_invalid_typedef_count": len(aliases),
        "modified_source_file_count": len(modified),
        "modified_source_inventory": modified,
        "trained_weight_files_modified": False,
        "only_generated_cpp_identifier_references_modified": True,
        "original_onnx_source_modified": False,
        "native_compilation_verified": False,
        "fixed_point_numerics_verified": False,
        "native_synthesis_or_implementation_executed": False,
    }



def export_failed_hls4ml_header_diagnostic(project_dir, output_dir) -> dict:
    """Persist only offending typedef LINES for a failed E1 attempt.

    This private diagnostic excludes complete headers, trained weights,
    implementation code and E1-qualification receipts. No raw identifiers
    appear on stdout or in the machine-readable metadata.
    """
    import json
    from tempfile import TemporaryDirectory

    project = Path(project_dir)
    dest = Path(output_dir)
    if (dest.exists() or dest.is_symlink()
            or project.is_symlink() or not project.is_dir()):
        raise ValueError("Unsafe or existing failed-source diagnostic destination")
    candidates = []
    for file in _cpp_files(project):
        if not _is_project_owned_source(project, file):
            continue
        if not is_model_type_header(file):
            continue
        rel = file.relative_to(project)
        raw = file.read_bytes()
        content = raw.decode("utf-8")
        matches = list(UNSAFE_TYPE.finditer(content))
        if matches:
            candidates.append((rel.as_posix(), file, raw, content, matches))
    if not candidates:
        raise ValueError("No offending generated non-weight header available for diagnosis")
    rel_name, source, raw, content, matches = candidates[0]
    line_numbers = [
        content.count("\n", 0, match.end() - 1) + 1 for match in matches
    ]
    # Persist ONLY the exact malformed typedef statements, not the entire
    # header (which might contain unrelated private generated material).
    offending_lines = "".join(
        f"line {line}: {match.group(0).strip().splitlines()[-1]}\n"
        for line, match in zip(line_numbers, matches)
    )
    result = {
        "schema": "fpgai-research/hls4ml-failed-header-diagnostic/v1",
        "evidence": "E1-FAILED-PRIVATE-TYPEDEF-LINES-ONLY",
        "original_header_sha256": sha256(raw).hexdigest(),
        "private_original_generated_header": rel_name,
        "invalid_typedef_line_numbers": line_numbers,
        "invalid_typedef_count": len(matches),
        "full_generated_header_preserved": False,
        "trained_weight_header_copied": False,
        "e1_source_qualified": False,
        "csim_executed": False,
        "native_e3_e4_executed": False,
        "external_qor_comparison_eligible": False,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls4ml-diag-", dir=dest.parent) as temp:
        stage = Path(temp) / "diagnostic"
        stage.mkdir(mode=0o700)
        (stage / "offending_typedefs_private.txt").write_text(
            offending_lines, encoding="utf-8")
        (stage / "diagnostic.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Failed-source diagnostic appeared concurrently")
        stage.rename(dest)
    return result
