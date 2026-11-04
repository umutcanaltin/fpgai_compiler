"""Read-only fail-closed preflight; NEVER changes GitHub visibility.

A passing local checklist is necessary, not sufficient: an owner must approve
third-party/IP rights, scan full history, choose the license, and explicitly
make the repository public. This script never prints secret values.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


def check(root):
    root=Path(root).resolve()
    tracked=subprocess.run(["git","ls-files","-z"],cwd=root,
                            check=True,capture_output=True).stdout
    paths=[p.decode("utf-8") for p in tracked.split(b"\0") if p]
    suspect_ext=(".pem",".key",".pfx",".p12",".pt",".pth",
                 ".npz",".h5",".hdf5",".bit",".onnx")
    suspicious=[p for p in paths if p.lower().endswith(suspect_ext)]
    # Existence of a license does not imply owner actually owns the rights;
    # institutional approval must be separately confirmed.
    license_present=any((root/name).is_file()
                       for name in ("LICENSE","LICENSE.md","LICENSE.txt"))
    approval=(root/"release-approval.json")
    approved=False
    if approval.exists() and not approval.is_symlink():
        try:
            doc=json.loads(approval.read_text())
            approved=bool(doc.get("institutional_ip_release_approved") is True
                          and doc.get("license_selected_and_approved") is True
                          and doc.get("history_scan_reviewed") is True
                          and doc.get("third_party_rights_reviewed") is True)
        except (ValueError,OSError):
            approved=False
    scan_available=shutil.which("gitleaks") is not None
    blockers=[]
    if not license_present:
        blockers.append("No approved repository LICENSE committed")
    if not approved:
        blockers.append("Missing local-only written IP/license/history/rightsholder approvals")
    if suspicious:
        blockers.append("Tracked potentially private model/credential artifact paths")
    if not scan_available:
        blockers.append("Full Git-history secret scan not verified: gitleaks unavailable")
    # Even when scanner is present, it MUST be run with --all history and
    # separately reviewed by a person; existence alone never means audited.
    blockers.append("Human verification of full-history secret/license scan is required")
    blockers.append("Clean-room release and full dependency license audit not verified")
    return {
        "schema":"fpgai-research/opensource-readiness/v1",
        "ready_to_switch_visibility":False,
        "repository_visibility_changed":False,
        "tracked_files_checked":len(paths),
        "potentially_private_tracked_paths":suspicious,
        "approved_license_present":license_present,
        "local_release_approvals_present":approved,
        "history_scanner_available":scan_available,
        "blockers":blockers,
    }


def main():
    root=Path(__file__).resolve().parents[1]
    print(json.dumps(check(root),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
