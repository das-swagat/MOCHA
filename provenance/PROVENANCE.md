# Release provenance

The research Pointcept source directory used for the final MOCHA experiments
was preserved as a source snapshot and does not contain its original `.git`
metadata.

For that reason, MOCHA release provenance is based on:

1. SHA256 hashes of the exact implementation files;
2. canonical run/config snapshots;
3. canonical checkpoint manifests and SHA256 hashes;
4. validated edge package hashes and benchmark evidence.

Canonical implementations:

- `MOCHAPretrainerV3NOMAELite`
- `TargetMOCHASegmentorV2`

Both are implemented in the preserved Pointcept `models/default.py` snapshot.

MultiMOCHA data loading is implemented in
`multimocha_unlabeled.py`, with manifest construction in
`build_multimocha_manifest.py`.
