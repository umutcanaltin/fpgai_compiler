"""Audit markdown relative links within a curated source-release preview.

Run locally; uses only tracked files, does not change visibility, execute
native vendors, or read private experiment folders. A successful result is
not a full IP, security, history or scientific release clearance.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
from pathlib import PurePosixPath, Path
from urllib.parse import unquote, urlsplit

from scripts.prepare_public_preview import plan, allowed, SCHEMA as PREVIEW_SCHEMA

SCHEMA="fpgai-research/public-docs-link-audit/v1"
LINK_RE=re.compile(r"\[[^\]]+\]\(([^\s)]+)(?:\s+[^)]*)?\)")


def target_of(owner:str,href:str)->str|None:
    """Resolve a Markdown link to a path in the release-preview root."""
    if href.startswith(("#","mailto:","http://","https://")):
        return None
    if href.startswith("//"):
        return None
    uri=urlsplit(href)
    if uri.scheme or href.startswith("/"):
        return None
    rel=unquote(uri.path)
    if not rel:
        return None
    if "\\" in rel or "\x00" in rel:
        return "!unsafe-link"
    dest=posixpath.normpath(posixpath.join(
        posixpath.dirname(owner),rel))
    if dest in (".","..") or dest.startswith("../"):
        return "!outside-release"
    return PurePosixPath(dest).as_posix()


def _inventory(root:Path,profile:str)->set[str]:
    """Use the manifest when independently auditing a staged preview.

    The source checkout still uses tracked Git files. A staged preview is
    intentionally Git-history-free and must never silently fall back to a
    parent checkout's Git index or untrusted arbitrary directory contents.
    """
    manifest=root/"PREVIEW_NOT_FOR_PUBLICATION.json"
    if not manifest.exists() and not manifest.is_symlink():
        return set(plan(root,profile)["selected_paths"])
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Unsafe staged preview manifest")
    doc=json.loads(manifest.read_text(encoding="utf-8"))
    paths=doc.get("selected_paths")
    if (doc.get("schema")!=PREVIEW_SCHEMA
            or doc.get("profile")!=profile
            or doc.get("ready_to_publish") is not False
            or not isinstance(paths,list)
            or not all(isinstance(x,str) and allowed(x,profile=profile)
                       for x in paths)
            or len(paths)!=len(set(paths))
            or "README.md" not in paths):
        raise ValueError("Invalid staged preview population or publication status")
    actual=set()
    for file in root.rglob("*"):
        if file.is_symlink():
            raise ValueError("Staged preview contains a symbolic link")
        if file.is_file():
            actual.add(file.relative_to(root).as_posix())
    expected=set(paths)|{"PREVIEW_NOT_FOR_PUBLICATION.json"}
    if actual!=expected:
        missing=sorted(expected-actual)
        extra=sorted(actual-expected)
        raise ValueError(
            f"Staged preview file inventory mismatch: missing={missing}; extra={extra}")
    return set(paths)


def audit_source_links(root:Path,profile:str)->dict:
    root=root.resolve()
    included=_inventory(root,profile)
    markdown=sorted(f for f in included if f.endswith(".md"))
    unresolved=[]
    checked=0
    for src in markdown:
        file=root/src
        if not file.is_file() or file.is_symlink():
            raise ValueError("Public source file missing or unsafe: "+src)
        for match in LINK_RE.finditer(file.read_text(encoding="utf-8")):
            link=match.group(1)
            dest=target_of(src,link)
            if dest is None:
                continue
            checked+=1
            if dest not in included:
                unresolved.append({
                    "source":src,"target":dest,
                    "line":file.read_text(encoding="utf-8").count(
                        "\n",0,match.start())+1,
                })
    return {
        "schema":SCHEMA,"profile":profile,
        "source_markdown_files":len(markdown),
        "checked_local_markdown_links":checked,
        "unresolved_public_links":unresolved,
        "all_local_public_links_resolve":not unresolved,
        "public_release_authorized":False,
    }


def main(argv:list[str]|None=None)->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile",choices=("compiler","full-research"),default="compiler")
    ap.add_argument("--root",default=None,
                    help="Optional checkout or staged-preview root directory")
    args=ap.parse_args(argv)
    root=(Path(args.root) if args.root else Path(__file__).resolve().parents[1])
    out=audit_source_links(root,args.profile)
    print(json.dumps(out,indent=2,sort_keys=True))
    if out["unresolved_public_links"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
