"""Input check or explicit D120 rerun using unchanged archived functions.

This adapter was added for packaging. It was not used to produce the paper.
Scientific functions are loaded as AST definitions to avoid the archive's
top-level directory creation and unrelated historical analyses.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json

ARCHIVE_SHA256 = '8974bef2530dc737d9fe3aaa1e4254ce2be6d1e99023282956d32da1c6058850'
FUNCTIONS = {'required_noncentrality', 'residualize', 'geometry_arrays',
             'schur_from_arrays', 'visible_plane_angle_deg', 'half_window_metrics'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-root', type=Path, required=True,
                        help='Root containing geometry_development/ and independent_validation/.')
    parser.add_argument('--index', type=Path, default=Path(__file__).with_name('D120_48_WINDOW_INPUT_INDEX.csv'))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-inputs', action='store_true', help='Hashes only; no numerical analysis.')
    mode.add_argument('--run', action='store_true', help='Explicitly recompute only the registered D120 analysis.')
    parser.add_argument('--output', type=Path, help='New output directory; must not already exist.')
    args = parser.parse_args()
    with args.index.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 48
    assert sum(r['dataset'] == 'DATA04' for r in records) == 12
    assert sum(r['dataset'] == 'DATA07' for r in records) == 36
    inputs = [(r, args.geometry_root / r['source_collection'] / r['input_filename']) for r in records]
    failures = [{'sequence_id': r['sequence_id'], 'path': str(p)} for r,p in inputs
                if not p.is_file() or digest(p) != r['input_sha256']]
    if failures:
        raise SystemExit(json.dumps({'input_check': 'FAIL', 'failures': failures}, indent=2))
    archive = Path(__file__).with_name('compute_project_theory_original.py')
    assert digest(archive) == ARCHIVE_SHA256, 'Archived scientific source identity mismatch'
    if args.check_inputs:
        print(json.dumps({'input_check': 'PASS', 'geometry_files': 48,
                          'scientific_source_hash': ARCHIVE_SHA256, 'numerical_analysis_run': False}))
        return
    if args.output is None or args.output.exists():
        raise SystemExit('--run requires a new --output directory; existing outputs are never overwritten')
    # Imports and all numerical operations occur only in the explicitly requested run mode.
    import math
    import numpy as np
    import pandas as pd
    from scipy.linalg import subspace_angles
    from scipy.optimize import brentq
    from scipy.stats import chi2, ncx2
    tree = ast.parse(archive.read_text(encoding='utf-8-sig'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    assert {n.name for n in selected} == FUNCTIONS
    scope = {'np': np, 'pd': pd, 'math': math, 'json': json,
             'Path': Path, 'subspace_angles': subspace_angles, 'brentq': brentq,
             'chi2': chi2, 'ncx2': ncx2, 'ALPHA': 0.05, 'SIGMA': 0.30}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(archive), 'exec'), scope)
    scope['LAMBDA_REQ_LRT_80'] = scope['required_noncentrality'](3, 0.80)
    rows = []
    for record, path in inputs:
        metadata = {**record, 'satellite_tier': '', 'information_stratum': ''}
        result = scope['half_window_metrics'](metadata, path)
        assert result is not None
        rows.append(result)
    frame = pd.DataFrame(rows)
    full = frame['required_radius_lrt80_shared_full_m'].to_numpy()
    better_half = np.minimum(frame['required_radius_lrt80_left_m'], frame['required_radius_lrt80_right_m'])
    summary = {'windows': 48, 'sigma_mps': 0.30, 'alpha': 0.05, 'power': 0.80,
               'required_noncentrality': scope['LAMBDA_REQ_LRT_80'],
               'median_full_radius_m': float(np.median(full)),
               'full_radius_p05_m': float(np.quantile(full, 0.05)),
               'full_radius_p95_m': float(np.quantile(full, 0.95)),
               'median_better_half_m': float(np.median(better_half)),
               'median_separate_joint_m': float(frame['required_radius_lrt80_separate_joint_m'].median()),
               'median_paired_gain': float(frame['joint_radius_gain_vs_better_half'].median()),
               'note': 'Unrounded recomputed values; historical displayed rounding is not overwritten.'}
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / 'd120_per_window.csv', index=False)
    (args.output / 'd120_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
