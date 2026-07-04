# R1 Deletion Policy

No source file is deleted in R1.

Reason: static usage detection is not enough for this project because many modules are invoked by CLI entry points, dynamic report generation, examples, or paper pipelines. Deletion requires:

1. import/reference audit,
2. replacement path if any,
3. focused tests,
4. full selected test suite,
5. example compile validation,
6. artifact smoke / validation trace pass.

Deletion starts in a dedicated cleanup sprint after R0/R1 classify candidates.
