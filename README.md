# Receiver State Hypothesis Selection

Code for Receiver-State Hypothesis Selection for Doppler-Only LEO Positioning under Residual Blindness.

This source-only repository provides M0–M14 estimator and EGSHS implementations, frozen configurations, third-party data links, and manual reproduction instructions. Manuscripts, supplementary publication files, correspondence, audit reports, datasets, saved experimental outputs, and artwork are excluded. CSV files included here contain rule configurations, input identities, or code inventories rather than result tables.

Start with [REPRODUCTION.md](REPRODUCTION.md), [DATA_SOURCES.md](DATA_SOURCES.md), and [REPRODUCIBILITY_SCOPE.md](REPRODUCIBILITY_SCOPE.md). [CODE_FILE_CATALOG.md](CODE_FILE_CATALOG.md) describes file purposes. [VERSIONS.md](VERSIONS.md) identifies the frozen sources and associated manuscript. The repository does not provide a single-command reproduction of the entire study.

## Code organization

| Location | Purpose |
|---|---|
| `src/leo_positioning/` | Frozen measurement models, candidate estimators and selection functions |
| `src/flagship/` | Historical Python namespace for information analysis, D2 reconstruction and controlled-data interfaces |
| `manual_workflows/tree/` | Historical stage sources and their separate frozen dependencies |
| `code_provenance/development_baselines/` | Development comparison worker, serialization correction and provenance sources |
| `code_provenance/d120/` | Archived detection-scale calculation, input index and an explicit optional rerun adapter |
| `code_provenance/motion_clock/` | Archived geometry-generation provenance; not an alternative selector |
| `configs/` | Selection rules, protocols, input identity indexes and message checks |
| `environment/` | Recorded software dependencies |
| `third_party/` | Included message definitions and original notices |

Repeated modules in separate frozen snapshots preserve experiment-specific identities. They have not been merged into a new solver. Historical source identifiers are retained for traceability.

## Local inspection

Run `python -B check_code.py` for hashes and syntax without fitting or selection. The optional `--imports` mode imports scientific modules and is not required for static inspection. Neither mode establishes numerical end-to-end reproducibility. D120 additionally supports a hashes-only input check documented in REPRODUCIBILITY_SCOPE.md.

Author-code licensing and third-party obligations are described in [LICENSE_NOTICE.md](LICENSE_NOTICE.md); no new licence is granted by this update. See VERSIONS.md and the commit history for the source identities and update status.
