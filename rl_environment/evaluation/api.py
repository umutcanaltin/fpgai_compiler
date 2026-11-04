"""Stable public read-only audit/report API, preserving original source modules."""
def audit_native_study(path):
    from ..gym_audit import audit
    return audit(path)


def report_completed_stages(studies, output, *, require_all_seven=False):
    from ..gym_curriculum_report import export
    return export(studies, output, require_all_seven=require_all_seven)
