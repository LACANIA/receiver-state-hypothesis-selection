# Reproduction scope and required inputs

This update adds source and input identities for results already reported in the manuscript. It contains no new experiment, candidate, threshold or saved result. Obtain external and author-held inputs before executing historical stages. This is a source-only handoff; inspection, candidate fitting, selection and post-selection evaluation are different operations.

## Detection scale D120

The paper uses 48 archived 120-second windows: 12 Development and 36 receiver-position holdouts in the same orbit/time domain. `code_provenance/d120/D120_48_WINDOW_INPUT_INDEX.csv` identifies each geometry by collection, sequence, filename and SHA256. Obtain these registered files from the corresponding author; their presence in this index does not mean their data are public. Use a geometry root containing the listed geometry_development/ and independent_validation/ collections.

`compute_project_theory_original.py` is the unchanged historical source. Its original main routine also handles other historical windows, checks and plots and contains machine-specific paths. Do not execute or import that file merely to inspect it. Its D120 science functions use the fixed receiver anchor, zero reference velocity, full-window centered times, Gaussian sigma 0.30 m/s, alpha 0.05 and power 0.80. Half-windows select negative and nonnegative centered times; independent nuisance Schur matrices are added for the joint-half comparison. The full-window fit instead shares nuisance parameters.

`reproduce_d120.py` is a newly packaged adapter, not the historical execution source. It verifies the archive SHA and extracts six unchanged function definitions without executing historical top-level statements. It processes only the 48 indexed inputs. It neither imports EGSHS nor performs candidate fitting. The hashes-only mode performs no numerical calculation:

```text
python -B code_provenance/d120/reproduce_d120.py --geometry-root YOUR_GEOMETRY_ROOT --check-inputs
```

An explicit later rerun can use `--run --output NEW_EMPTY_OUTPUT_PATH` instead of `--check-inputs`. No such numerical rerun was performed while assembling this update. The adapter has static and input-identity checks only; numerical parity with historical outputs still requires an authorized rerun. Historical saved window and half-window results remain obtainable from the corresponding author.

The historical manuscript displays the upper percentile as 1405.6 m. The saved unrounded value is 1405.5496107297806 m; direct one-decimal rounding differs. The adapter reports unrounded values and does not overwrite this documented historical display convention. The median paired improvement is calculated from individual paired ratios, not from a ratio of median radii.

## Development comparisons

`code_provenance/development_baselines/selection_worker.py` contains the locked comparison rules. `selection_worker_interface_correction.py` is the recorded group-ID serialization correction; it does not change formulas, candidate subsets or tie rules. The earlier paper_exp03a source documents preparation and is not the final comparison worker.

The code imports `baseline_common.py`, whose historical locations must be mapped to local working copies. Supply the protocol and lock under BASELINE_ROOT/protocol and the group registry under BASELINE_ROOT/partition; byte-identical configuration copies are in configs/development_baselines/. Supply the truth-free candidate input, reference selections, scenario inventory and corresponding frozen SELECTOR_SOURCE. The worker checks 900 candidates and 60 groups and filters PAPER_EXP01_HOLDOUT. B2 uses eligible M0–M14 held-out RMSE; AIC, the article's FSC (historical AICc), and BIC use compatible M0–M4 with the registered noise scale. Post-selection truth is separate. The full historical observation preparation remains in manual_workflows/; candidate inputs and historical inventories must be regenerated in that order or requested from the corresponding author.

## Motion–clock G1–G3

`configs/MOTION_CLOCK_G1_G3_INPUT_INDEX.csv` identifies the archived arrays and their hashes, coordinates, start/report times, visible counts and pre-noise ranks. The idealized orbit functions use 550 km altitude and 53 degrees inclination. Two interleaved sets of 20 positions per plane in 12 planes give 480 catalogue objects; all-epoch visibility selection gives 11/10/6 satellites for G1/G2/G3. These are not the actual OMM satellite population used in D2.

The corresponding trajectory/common-term functions are in src/flagship/controlled_motion_clock.py. Their orbit_functions loader reads the preserved functions under manual_workflows/tree/flagship/nextgen_egshs/17_clock_dev_classical_baselines/ and 20_matched_observation_support_experiment/. `code_provenance/motion_clock/geometry28_original.py` documents the exact catalogue and visibility construction; it imports the historical common28 environment and is a provenance source, not a standalone controller. Use the indexed archived arrays to retain exact geometry. Source adjacency does not authorize using other window/probe selectors for the manuscript's native EGSHS result.

## D2 and measured data

Use DATA_SOURCES.md for original-provider access. D2 requires the archived historical orbital snapshot, its frozen generation inventory, and the 110-second reporting reconstruction; a current online orbital feed cannot substitute for that snapshot. Historical native EGSHS and the D2 publication-spec interpreter are separate execution paths. The existing identity checks do not establish equality on every possible exceptional input.

## Figures and numerical results

Conceptual diagrams were drawn separately. For numerical result plots, obtain the saved arrays or regenerate results with the corresponding inputs, then follow the paper's metric, grouping and reporting definitions. Final publication layout sources and saved plot data are not bundled in this code-only update. No complete figure-reconstruction claim is made. Saved experimental outputs remain available through the manuscript's stated access route.
