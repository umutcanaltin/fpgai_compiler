# Who may use FPGAI Research?

This page distinguishes **current repository access** from the intended
**future academic research-only source license**. It is important not to
confuse an intention to publish research with a present right to copy,
redistribute, modify, or commercially deploy this code.

## Current status (before institutional approval)

The repository remains **private** with **no effective LICENSE** and no
verified grant of public redistribution rights. Existing collaborators
may use materials **only within their individually authorized access
and any applicable institutional or contractual terms**. Repository
access does not by itself grant rights to republish, sublicense,
commercialize, or upload derived source or generated FPGA IP.

A draft [academic research-only license proposal](ACADEMIC_RESEARCH_LICENSE_PROPOSAL.md)
exists for rights-holder review. It is **not an operative license**.

## Proposed permissions after an approved release

The currently proposed policy is *research-only, non-commercial,
source-available*; it is **not OSI-approved open-source software**.
The final terms may change after legal review.

| Intended user / activity | Proposed outcome | Important boundary |
|---|---|---|
| University student studying FPGA compilation | Academic research/teaching use | Attribution and applicable student/institution permissions |
| Faculty or researcher reproducing a paper | Non-commercial research and verification | Do not republish private model weights, vendor assets or unpublished study logs |
| Non-profit academic laboratory extending operators | Internal research changes | Public redistribution of modified source would require separate approval |
| Independent researcher | Non-commercial research, once terms expressly include them | No consultancy/client deliverable under the research permission |
| University lab working on a commercial customer contract | **Not automatically authorized** | Research institution identity alone does not make a commercial deliverable non-commercial |
| Company evaluating for a future product | **Outside the proposed permission** | Requires separate written commercial evaluation/development terms |
| Commercial product, hosted API, customer deployment or shipped FPGA image | **Not permitted under the proposal** | Negotiate separate rights from the authorized licensor(s) |
| Academic paper citing results | Scholarly description and citation | Citation is not permission to disclose restricted source/data |
| Copy to a public fork or package registry | **Not permitted under the proposal** | Separate written redistribution permission required |

No table entry above grants rights **today**. Only a finalized,
properly authorized license governs released code.

## How can authorized researchers get started?

For source-only study, install Python and use the
[neural-network gallery](../examples/NETWORK_GALLERY.md); no proprietary
FPGA tools are needed. To experiment with HLS C simulation and
synthesis, consult [AMD environment setup](AMD_TOOLCHAIN_SETUP.md).
Running a model on actual hardware requires the further integration
documented in [FPGA deployment](FPGA_DEPLOYMENT.md).

Researchers must obtain their own AMD tools, licenses, board platforms,
and applicable licenses for datasets and models. Sharing a source
compiler does not grant AMD's tool licenses or cover third-party IP.

## Publication and citations

Use [CITATION.md](../CITATION.md) for source revisions, study
population and measured evidence attribution. Do not fabricate an
archive DOI or publication acceptance. Independent study data and the
original private repository history are not included in the proposed
public source snapshot.

## Decision required before issuing any real license

The actual rights holders must approve the exact files and terms,
including the academic/non-commercial definitions, redistribution,
source modifications, use of generated RTL, commercial exceptions,
and third-party legal obligations. A full Git-history security review
and contributor rights audit remain mandatory.

See [license decision](LICENSE_DECISION.md) and
[release checklist](OPEN_SOURCE_RELEASE_CHECKLIST.md). No public
publication, redistribution grant or licensing approval has occurred
as a result of writing these guides.
