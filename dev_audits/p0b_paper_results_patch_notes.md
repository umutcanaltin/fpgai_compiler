# P0B Paper Results Aggregator Patch Notes

This patch adds the canonical paper master-result aggregator.

New canonical owner:

- `fpgai/reporting/paper_results.py`

Purpose:

- Read existing FPGAI compile reports.
- Produce `paper_results/master_results.json`.
- Produce `paper_results/master_results.csv`.
- Produce `paper_results/master_results.md`.
- Produce `paper_results/schema/master_result_schema.json`.
- Produce `paper_results/schema/master_result_schema.md`.

This replaces the previous habit of many standalone paper scripts with a single master result schema.

Professional terminology rule:

- New fields use `validation`, `support_status`, `required_validation`, and `status` terminology.
- The aggregator may read legacy keys from existing reports, but it does not introduce new product fields named `truth` or `evidence`.
