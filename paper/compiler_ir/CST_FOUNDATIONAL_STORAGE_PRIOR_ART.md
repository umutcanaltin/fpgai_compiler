# CST source-storage prior art: accepted critical novelty challenge

As of November 3, 2026, the independent Claude review correctly
identified a missing class of prior art in the repository's novelty
gate. The generic relationship between a legal schedule, value
lifetimes and storage cannot be claimed as an invention of CST.

- Thies, Vivien, Sheldon and Amarasinghe, **A Unified Framework for
  Schedule and Storage Optimization**, PLDI 2001,
  DOI 10.1145/378795.378852.
  Joint storage/schedule optimisation via occupancy vectors.
- Strout, Carter, Ferrante and Simon, **Schedule-Independent Storage
  Mapping for Loops**, ASPLOS 1998,
  DOI 10.1145/291006.291015.
  Universal occupancy vectors for storage across legal schedules.
- Bhaskaracharya, Bondhugula and Cohen, **SMO: An Integrated
  Approach to Intra-Array and Inter-Array Storage Optimization**,
  POPL 2016. Fixed-schedule cross-array and within-array reuse.
- Lefebvre and Feautrier, **Automatic Storage Management for Parallel
  Programs**, Parallel Computing 1998: full-text check outstanding.
- Darte, Schreiber and Villard, **Lattice-Based Memory Allocation**,
  IEEE Transactions on Computers 2005: full-text check outstanding.

These are *prior-work comparison requirements*, not completed
published peer reproductions. The user's independently audited R2c
E0/E1 source cohort at e6db1ec has 12/12 source scratch
saving pairs and 9/9 unique Phi-induced minimum scalar-slot
changes, but still no physical native BRAM outcome, hardware
schedule, general fan-out graph or external firstness result.

The defensible prospective paper question is narrower:
can compositional operator mapping semantics give independent
per-value readiness/lifetime certificates whose predictive
power and mapped hardware consequences are empirically
validated beyond existing schedule/storage frameworks?

See CLAUDE_REVIEW_ACTION_REGISTER_20261103.md.
