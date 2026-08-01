# External operator security

Discovery, manifest validation, registry inventory, and dependency resolution never execute package code. In-process loading is only for explicitly trusted research packages. Subprocess validation provides process separation and timeouts but is not a complete sandbox. Morfics production workers must add container isolation, network restrictions, resource limits, secret isolation, and organization approval.
