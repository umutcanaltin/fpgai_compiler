# External HLS integration security

External source packages are untrusted by default. E4A performs metadata and
path validation before copying files. Absolute paths, traversal, missing files,
and symlinked package files are rejected. The package root is not modified.

Generating a project does not execute HLS or C++ code. C simulation and vendor
tool execution are separate trust stages. Morfics managed workers will later
provide container isolation, restricted networking, resource limits, secrets
isolation, and organization approval.
