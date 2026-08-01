# Package discovery security boundary

Discovery is metadata-only. It may read manifests, validate declared files, calculate hashes, and construct registry metadata. It must not import package modules, invoke subprocesses, access the network, install packages, run vendor tools, or execute testbenches.

Security rules:

- recursive scanning is bounded to a maximum depth of five;
- symbolic-link directories and symbolic-link manifests are not followed;
- declared package files may not escape the package root through symbolic links;
- hidden, generated, cache, build, and virtual-environment directories are skipped;
- invalid packages are quarantined rather than registered;
- same package ID and version with different manifest content is an explicit conflict;
- no source-priority rule silently replaces conflicting package content.
