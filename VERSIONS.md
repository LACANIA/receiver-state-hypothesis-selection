# Source and manuscript versions

This source-only update builds on public main commit `934353c23d21f61f01412ea823deca78adc5d37e`, read and verified on 8 November 2026. The repository's commit history identifies the exact publication commit for this update. Source/configuration inventories are versioned with the code; saved data and publication documents are distributed separately.

The associated manuscript is distributed separately and has SHA256 `3638fdc219610764dcbb8ad75d0e98624613071bf8fe6c0a49b58d7ea5b561a9`. It uses six main figures, six main tables and a two-page Online Resource 1 containing two tables and one figure. No eighteen-table Excel supplement is implied by this code release.

M0–M14, the native frozen EGSHS and the original D2 execution identity remain unchanged. Forty of 41 primary src/leo_positioning modules are byte-identical to the frozen source; model_candidates.py retains the existing publication interface, while the historical full version remains in the separate snapshots. D2 uses its publication-spec interpreter and report-epoch reconstruction; do not assume all-input equivalence to the native selector.

The new D120 adapter is a packaging interface around unchanged historical functions. Only its input-check mode was exercised in this update. Numerical reproduction, fitting and selector execution were not run. SOURCE_MANIFEST.json records current file hashes and retained source identities; UPSTREAM_SOURCE_MANIFEST.json preserves the original upstream source mapping.
