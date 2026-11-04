"""Build an arXiv-compatible *source* ZIP from the canonical article and evidence.

Never fetches private build artifacts or invokes vendor tools. Refuses partial
source sets: conditional LaTeX evidence inputs must all be present and pinned.
The resulting archive still requires independent scientific/IP review.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import zipfile

_TEX_INPUT = re.compile(r"\\(?:input|IfFileExists)\{([^{}]+\.tex)\}")
_RESTRICTED = re.compile(r"(^/|(^|/)\.\.(/|$)|[^A-Za-z0-9_+.,=/\-])")


def _ordinary(source, *, limit=1024 * 1024):
    if (source.is_symlink() or not source.is_file()
            or not 0 < source.stat().st_size <= limit):
        raise ValueError("Missing, unsafe, or too-large LaTeX source input")
    return source.read_bytes()


def collect_article_sources(root):
    root = Path(root)
    master = root / "paper" / "main.tex"
    text = _ordinary(master).decode("utf-8")
    if (r"\documentclass{article}" not in text
            or text.count(r"\begin{document}") != 1
            or text.count(r"\end{document}") != 1
            or r"\today" in text):
        raise ValueError("Canonical standard-article source is not arXiv-ready")
    referenced = sorted(set(_TEX_INPUT.findall(text)))
    if not referenced or "paper/semantic_table.tex" not in referenced:
        raise ValueError("Canonical measured-evidence input inventory is missing")
    if not any(path.startswith("build/mnist_e4_paper") for path in referenced):
        raise ValueError("Routed native observations missing from manuscript source")
    if not any(path.startswith("build/mnist_paper_e3") for path in referenced):
        raise ValueError("Native E3 observations missing from manuscript source")
    bundle = {"main.tex": text.encode("utf-8")}
    for rel in referenced:
        if _RESTRICTED.search(rel) or "//" in rel or not rel.endswith(".tex"):
            raise ValueError("Unsafe TeX input relative path")
        source = root / rel
        bundle[rel] = _ordinary(source)
    # Avoid uploading stale private build logs, raw model weights, model
    # predictions, generated C++, or documents not needed for compilation.
    # Only the exact dependencies extracted from the paper are included.
    if len(bundle) > 32:
        raise ValueError("Unbounded arXiv source inventory")
    return bundle


def _scientific_latex_presentation(source: bytes) -> bytes:
    """Normalize only internal stage labels in a submission *copy* of LaTeX.

    Keep original qualified exported tables intact for byte-for-byte
    source re-audit; numeric values, commands, filenames and labels are
    not rewritten. The source inventory records both original and
    presentation digests. This is not a source-evidence upgrade.
    """
    text = source.decode("utf-8")
    # Treat compound references before individual stage aliases.
    patterns = (
        (r"\bE0/E1\b", "software-model validation and HLS source generation"),
        (r"\bE2/E3/E4\b", "C simulation, HLS synthesis and physical implementation"),
        (r"\bE3/E4\b", "HLS synthesis and physical implementation"),
        (r"\bE3-estimate\b", "HLS synthesis-estimate"),
        (r"\bE0\b", "software-model"),
        (r"\bE1\b", "HLS-source"),
        (r"\bE2\b", "C-simulation"),
        (r"\bE3\b", "HLS-synthesis"),
        (r"\bE4\b", "post-route"),
    )
    for old, new in patterns:
        text = re.sub(old, new, text)
    for code, label in (("E_0", "software"), ("E_1", "source"),
                        ("E_2", "simulation"), ("E_3", "synthesis"),
                        ("E_4", "implementation")):
        text = text.replace("$" + code + "$", label)
    if re.search(r"\bE[0-4]\b|\$E_[0-4]", text):
        raise ValueError("Internal stage codes remain in arXiv LaTeX source")
    return text.encode("utf-8")


def package_arxiv_article(repository, output_file):
    target = Path(output_file)
    if target.exists() or target.is_symlink():
        raise FileExistsError("ArXiv source ZIP must use a new path")
    if target.suffix != ".zip":
        raise ValueError("Expected an arXiv source ZIP")
    raw_bundle = collect_article_sources(repository)
    bundle = {
        name: _scientific_latex_presentation(data)
        for name, data in raw_bundle.items()
    }
    manifest = {
        "schema": "fpgai-research/arxiv-article-source-inventory/v1",
        "top_level_tex": "main.tex",
        "article_class": "article",
        "file_sha256": {name: sha256(data).hexdigest()
                        for name, data in sorted(bundle.items())},
        "source_original_sha256": {
            name: sha256(data).hexdigest()
            for name, data in sorted(raw_bundle.items())
        },
        "internal_stage_label_normalization_on_submission_copy_only": True,
        "all_conditional_evidence_inputs_present": True,
        "vendor_tools_executed": False,
        "source_bundle_is_submission_or_ip_approval": False,
        "manual_pdf_review_and_rights_approval_required": True,
    }
    bundle["source_inventory.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".arxiv-source-", dir=target.parent) as temp:
        staged = Path(temp) / "article.zip"
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, data in sorted(bundle.items()):
                member = zipfile.ZipInfo(name)
                member.date_time = (2026, 10, 27, 0, 0, 0)
                member.external_attr = (0o100644 << 16)
                member.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(member, data)
        if target.exists() or target.is_symlink():
            raise FileExistsError("ArXiv destination appeared during staging")
        staged.rename(target)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(package_arxiv_article(args.repository, args.output),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
