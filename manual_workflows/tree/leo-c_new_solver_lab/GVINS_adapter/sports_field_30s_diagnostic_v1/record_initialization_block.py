"""Record the user-required initialization stop; never import scientific runtime."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import sys

OUT = Path(__file__).resolve().parent
A = OUT.parent
LAB = A.parent
N = LAB / 'paper_draft/manuscript_portfolio/workspaces/PAPER_F_SATELLITE_NAVIGATION_FLAGSHIP/nextgen_egshs'
UTC = datetime.now(timezone.utc)
RUN = 'GVINS_CLOCK_BYPASS_DIAGNOSTIC_V1_' + UTC.strftime('%Y%m%dT%H%M%SZ')
TASK = 'GVINS_SPORTS_FIELD_30S_CLOCK_GATE_BYPASS_DIAGNOSTIC_V1'
DECISION = 'GVINS_DIAGNOSTIC_INITIALIZATION_BLOCKED'
NEXT = 'GVINS_GPS_L1_NONTRUTH_INITIALIZATION_CONTRACT_V1'
assert not (OUT / 'RUN_SEAL.json').exists(), 'Existing run must not be overwritten.'

def sha(p):
    return hashlib.sha256(io_path(p).read_bytes()).hexdigest()

def io_path(p):
    absolute = str(Path(p).absolute())
    return Path('\\\\?\\' + absolute) if sys.platform == 'win32' and not absolute.startswith('\\\\?\\') else Path(absolute)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(name, data):
    p = OUT / name
    assert not p.exists(), str(p)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

native_manifest = read(A / 'source_manifest.json')
checks = []
for entry in native_manifest['native_source_checks']:
    p = Path(entry['resolved_path'])
    actual = sha(p)
    checks.append({'path': str(p), 'expected_sha256': entry['expected_sha256'],
                   'sha256': actual, 'matches': actual == entry['expected_sha256']})
assert checks and all(x['matches'] for x in checks), 'Native source identity conflict.'

wrapper_manifest = read(A / 'gps_l1_wrapper/output_manifest.json')
assert wrapper_manifest['run_id'] == 'GVINS_GPS_L1_WRAPPER_V1_20260910T112243Z'
wrapper = A / 'gps_l1_wrapper/wrapper.py'
bound_wrapper = next(x for x in wrapper_manifest['files'] if Path(x['path']) == wrapper)
assert sha(wrapper) == bound_wrapper['sha256']
lock_path = A / 'sports_field_pilot/time_alignment_v2/pilot_interval_lock_v2.json'
v2_seal = read(lock_path.parent / 'final_output_seal.json')
bound_lock = next(x for x in v2_seal['files'] if Path(x['path']) == lock_path)
assert sha(lock_path) == bound_lock['sha256']
lock = read(lock_path)
assert lock['start_gps_week'] == 2134
assert abs(lock['start_gps_tow_s'] - 83928.102) < 1e-8
assert abs(lock['end_gps_tow_s'] - 83958.102) < 1e-8

specs = [
    ('CURRENT_NATIVE_ENTRY', N / '21_egshs_reference_and_weak_support_analysis/runtime/scripts/release_runtime.py',
     196, 206, 'sealed_initialization_prior',
     'Consumes initial_ecef_m supplied by a caller; no observation-derived position initializer is supplied here.'),
    ('FROZEN_FIELD_INITIALIZER', Path('leo-c/_new_solver_lab/final_release/MA_BGTR_v7_2_freeze_r2/scripts/release_runtime.py'),
     196, 226, 'p_true + 10_000.0 * east',
     'east_10km/east_100km are offsets from p0_true_m; forbidden truth-associated initialization for this pilot.'),
    ('URBANNAV_RECONSTRUCTED_PRIOR', N / '04_urbannav_historical_initialization_trace_reconstruction_v1_1/URBANNAV_DIAGNOSTIC_RECONSTRUCTION_V1_1/instrumented_runtime/build_historical_initialization_priors.py',
     105, 137, "'truth_derived':True",
     'Source first-row receiver_true coordinates define the anchor; sanitized candidate inputs retain a truth-derived prior.'),
    ('GVINS_COARSE_LOCALIZATION', LAB / 'GVINS-main/GVINS-main/estimator/src/initial/gnss_vi_initializer.cpp',
     16, 40, 'psr_pos(accum_obs, accum_ephems, iono_params)',
     'Native coarse localization uses pseudorange positioning, outside the permitted candidate/initialization inputs.'),
    ('WRAPPER_FIXED_STATE_FIXTURE', A / 'gps_l1_wrapper/native_leaf_qa.py',
     24, 31, 'deterministic Earth-scale test state',
     'Deterministic QA state tests implementation identity; it is not a bound operational location prior for sports_field.'),
]
evidence = []
for identity, path, start, end, needle, finding in specs:
    content = io_path(path).read_text(encoding='utf-8-sig')
    assert needle in content, (identity, needle)
    lines = content.splitlines()
    evidence.append({'identity': identity, 'path': str(path), 'sha256': sha(path),
                     'line_start': start, 'line_end': end,
                     'excerpt': '\n'.join(f'{i+1}: {lines[i]}' for i in range(start-1, min(end, len(lines)))),
                     'finding': finding, 'accepted_for_current_pilot': False})

save('initialization_evidence.json', {'run_id': RUN, 'sources': evidence,
     'scope': 'Bound native entry, historical field/UrbanNav initialization, native GVINS coarse initializer and wrapper QA fixture. Not an exhaustive audit of all project assets.',
     'missing_input': 'An explicitly sourced non-RTK/PVT, non-pseudorange, non-truth-associated coarse receiver position prior.'})

seal = {'task': TASK, 'run_id': RUN, 'created_utc': UTC.isoformat(), 'status': 'PRE_EXECUTION_BLOCKED',
        'decision': DECISION, 'data_identity': 'REAL_GNSS_RF_DIAGNOSTIC_PILOT',
        'source_runs': ['GVINS_TIME_ALIGNMENT_V2_20260910T103857Z', wrapper_manifest['run_id']],
        'pilot': {'gps_week': 2134, 'tow_start': 83928.102, 'tow_end_exclusive': 83958.102,
                  'canonical_epochs': 30, 'rows': 180, 'prns': [2,5,6,13,29,30],
                  'count_identity': 'User-locked design and prior manifest; no observation rescan in this blocked task.'},
        'initialization': 'INITIALIZATION_CONTRACT_BLOCKED', 'candidate_runs': 0, 'selector_runs': 0,
        'pre_truth_candidate_output_seal': False,
        'note': 'This seals a prerequisite failure, not completed candidate outputs.',
        'native_source_checks': checks, 'wrapper_sha256': sha(wrapper), 'pilot_lock_sha256': sha(lock_path),
        'clock_bypass_authorized_by_current_task': True, 'clock_bypass_implemented': False}
save('RUN_SEAL.json', seal)
models = [{'model': f'M{i}', 'status': 'NOT_RUN_INITIALIZATION_CONTRACT_BLOCKED',
           'fit_attempted': False, 'fit_success': None, 'residual_rmse': None,
           'validation_metric': None, 'b0': None, 'bdot': None,
           'strict_eligible': None, 'bypass_eligible': None,
           'strict_reject_reason': None, 'position_error': None} for i in range(15)]
metrics = {'run_id': RUN, 'decision': DECISION, 'planned_models': 15, 'candidate_fits_attempted': 0,
           'candidate_fits_success_count': 0, 'count_interpretation': 'No attempts; not 15 failed fits.',
           'candidate_status': models, 'strict_clock_rejected': None, 'bypass_recovered': None,
           'clock_gate_primary_blocker': None, 'residual_position_disagreement': None,
           'validation_position_disagreement': None, 'scientific_comparison_status': 'NOT_EVALUATED',
           'new_threshold_selected': False, 'paper_updated': False, 'next_task': NEXT}
save('diagnostic_metrics.json', metrics)
for name, identity in [('strict_selector_trace.json', 'STRICT_FROZEN'),
                       ('bypass_selector_trace.json', 'CLOCK_GATE_BYPASS_DIAGNOSTIC_SELECTOR')]:
    save(name, {'run_id': RUN, 'identity': identity, 'status': 'NOT_RUN',
                'reason': 'INITIALIZATION_CONTRACT_BLOCKED', 'selected_model': None,
                'selected_position_error': None, 'trace_events': [], 'fallback_status': 'NOT_EVALUATED'})
save('POSITION_OUTCOME_FIREWALL_AUDIT.json', {'run_id': RUN,
     'scope': 'Task action/source review and recorder import inspection; no candidate process was launched.',
     'candidate_process_started': False, 'candidate_input_created': False,
     'rtk_pvt_values_loaded': False, 'position_outcomes_computed': False,
     'forward_qa_clock_used_as_input': False, 'pseudorange_used_as_input': False,
     'native_runtime_imported': False, 'raw_bag_scanned': False,
     'forbidden_initializers_rejected': [e['identity'] for e in evidence if e['identity'] in
         ['FROZEN_FIELD_INITIALIZER', 'URBANNAV_RECONSTRUCTED_PRIOR', 'GVINS_COARSE_LOCALIZATION']],
     'runtime_firewall_execution_test': 'NOT_APPLICABLE_NO_CANDIDATE_EXECUTION'})
save('decision.json', {'task': TASK, 'run_id': RUN, 'decision': DECISION,
                      'blocker': 'INITIALIZATION_CONTRACT_BLOCKED', 'next_task': NEXT})

report = f'''# Sports field 30 s clock-gate bypass diagnostic

任务：{TASK}  
run_id：{RUN}  
实际时间：{UTC.isoformat()}  
decision：**{DECISION}**

## 当前结果

任务在用户规定的初始化前置检查处停止。M0–M14执行0次，EGSHS执行0次，尚无定位误差、资格恢复数量或残差排序结果。严格版本与absolute-clock旁路版本均未运行；本轮没有据旧QA数据推断它们会选择哪个候选。

本任务已经明确授权诊断性绕过20/30 m/s absolute b0门，因此此前GNSS阈值尚未完成预注册并非本轮停止依据。实际缺口是合法的初始位置来源。原窗口保持GPS week2134、TOW[83928.102,83958.102)，计划30个1 Hz历元、6颗卫星、180行，没有重新选窗。

## 初始化源码证据

1. 当前task21绑定入口 `release_runtime.py:196–206` 读取调用者提供的 `sealed_initialization_prior.initial_ecef_m`，没有从当前Doppler观测构造位置初值的过程。封存一个初值文件不会改变其数据来源身份。
2. 原field冻结入口 `release_runtime.py:196–226` 先取 `p0_true_m`，再构造 `p_true + 10_000.0 * east` 或100 km偏移，违反本轮禁止truth-associated anchor的要求。
3. UrbanNav历史初始化重建脚本第105–137行从 `receiver_true_x_m/y_m/z_m` 取得anchor，并明确记录 `truth_derived=True`。候选进程只读取清理后的初值，并不能使初值变成独立测得的位置。
4. GVINS的 `GNSSVIInitializer::coarse_localization` 第16–40行调用 `psr_pos`；使用这一路径会引入本轮禁止的pseudorange定位。
5. wrapper固定状态QA中的Earth-scale测试向量用于验证预测和Jacobian，没有绑定为sports_field实际粗略位置先验，本轮未将它改作运行初值。

全部路径、对应行与SHA见 `initialization_evidence.json`。本次检查范围限定于上述入口和相关协议，未宣称穷尽整个项目中的所有可能初始化算法。

## 来源与执行边界

与 `source_manifest.json` 比较的{len(checks)}项冻结源码SHA全部一致，wrapper.py与前置wrapper output manifest一致，原pilot lock与V2封存记录一致。没有修改原科学源码、clock规则、权重、selector、论文或历史结果。

本轮读取源码和来源/协议记录，没有读取RTK/PVT状态值，没有执行position evaluation，也没有向candidate提供前置公共clock序列。`RUN_SEAL.json` 明确标为PRE_EXECUTION_BLOCKED，不能当作完成候选后的pre-truth seal。两个selector trace仅记录NOT_RUN，未制造门控事件。

## 未生成的实验结果

由于全部候选尚未执行，`candidate_fit_summary.csv`、`clock_gate_impact.csv`、`residual_position_ordering.csv`、`selector_comparison.csv` 暂不生成。逐模型NOT_RUN清单保存在 `diagnostic_metrics.json`；未取得的残差、b0、资格与位置误差全部为null，不能用零替代。M2能否拟合、clock门是否为主要阻断、Residual Blindness是否出现均为NOT_EVALUATED。

## 唯一后续方向

`{NEXT}`：先绑定一个独立于RTK/PVT、pseudorange及真实轨迹的粗略位置来源，并登记坐标、坐标系、来源和共同初始化方式。例如用户独立提供的场地粗坐标可以作为待批准来源；本轮没有自行查询或选取坐标。完成该输入合同后，才继续当前已锁定窗口的诊断。候选首次运行前仍需完成原计划中的wrapper子集调用和clock旁路调用点检查。

本次停止不构成候选优化失败，也没有产生GNSS泛化或clock门作用的实验证据。
'''
(OUT / 'report.md').write_text(report, encoding='utf-8')
assert all(sha(c['path']) == c['sha256'] for c in checks)
assert sha(wrapper) == seal['wrapper_sha256']
save('verification.json', {'run_id': RUN, 'status': 'BLOCKED_STATE_RECORD_VERIFIED',
     'native_sources_unchanged': len(checks), 'wrapper_source_unchanged': True,
     'candidate_count_not_run': len(models), 'candidate_fits_attempted': 0, 'EGSHS_runs': 0,
     'all_unobserved_scientific_metrics_null': all(m['b0'] is None and m['fit_success'] is None and m['position_error'] is None for m in models),
     'no_positioning_result_created': True, 'python_executable': sys.executable,
     'candidate_or_selector_imports': [], 'remaining_blocker': 'INITIALIZATION_CONTRACT_BLOCKED',
     'result_csv_disposition': 'NOT_CREATED_BECAUSE_EXECUTION_STOPPED_AT_REQUIRED_PREREQUISITE'})
save('output_manifest.json', {'run_id': RUN, 'created_utc': datetime.now(timezone.utc).isoformat(),
     'files': [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'output_manifest.json']})
terminal = {
    'TASK_STATUS': DECISION, 'TOTAL_REPORT': str(OUT / 'report.md'),
    'INITIALIZATION': 'INITIALIZATION_CONTRACT_BLOCKED',
    'CANDIDATE_FITS_SUCCESS': '0/15; attempted=0; all NOT_RUN',
    'STRICT_CLOCK_REJECTED': 'NOT_EVALUATED', 'BYPASS_RECOVERED': 'NOT_EVALUATED',
    'M2_FIT': 'NOT_RUN', 'M2_B0': 'NOT_AVAILABLE',
    'STRICT_SELECTED': 'NOT_RUN', 'BYPASS_SELECTED': 'NOT_RUN',
    'RESIDUAL_POSITION_DISAGREEMENT': 'NOT_EVALUATED',
    'VALIDATION_POSITION_DISAGREEMENT': 'NOT_EVALUATED',
    'CLOCK_GATE_PRIMARY_BLOCKER': 'NOT_ASSESSED',
    'NEW_THRESHOLD_SELECTED': 'false', 'PAPER_UPDATED': 'false', 'NEXT_TASK': NEXT}
for key, value in terminal.items():
    print(f'{key}={value}')
