from pathlib import Path
from datetime import datetime, timezone
import csv, json, pickle
import numpy as np
from bridge import Bridge, O, A, read, save, sha

m=read(O/'diagnostic_metrics.json'); protocol=read(O/'protocol.json'); seal=read(O/'PRE_TRUTH_SEAL.json'); b=Bridge()
with (O/'pretruth_results.pkl').open('rb') as f: results=pickle.load(f)
checks=[]
for n in ['strict','bypass']:
    for payload in results['payloads'][n]:
        for cap in payload['full_capture']:
            rmse=float(np.sqrt(np.mean(cap['residual']**2)))
            assert abs(rmse-cap['row']['residual_rmse_mps'])<1e-12
        row=payload['row'];short=payload['model'].split('_')[0]
        if n=='bypass' and int(short[1:])<10:
            strict=next(x for x in results['payloads']['strict'] if x['model']==payload['model'])
            assert pickle.dumps(strict)==pickle.dumps(payload),short
            checks.append(dict(model=short,shared_full_train_fit=True))
for name,rows in read(O/'tables.json').items():
    with (O/name).open(newline='',encoding='utf-8') as f: actual=list(csv.DictReader(f))
    assert len(actual)==len(rows)
    for a,r in zip(actual,rows):
        for key,value in r.items():
            got=a[key]
            if value is None: assert got==''
            elif isinstance(value,bool): assert got.lower()==str(value).lower()
            elif isinstance(value,(float,int)): assert float(got)==float(value)
            else: assert got==str(value)
for x in b.sources: assert sha(x['resolved_path'])==x['expected_sha256']
for x in read(O/'RUN_SEAL.json')['files']: assert sha(x['path'])==x['sha256'],x['path']
for x in read(O/'PRE_TRUTH_SEAL.json')['files']: assert sha(x['path'])==x['sha256']
assert len(checks)==10
epoch_rows=read(O/'epoch_position_errors.json')
pair_rows=read(O/'tables.json')['residual_position_ordering.csv']
for mode in ['strict','bypass']:
    assert len([x for x in pair_rows if x['run']==mode])==105
    for name,row in m['rows'][mode].items():
        errors=[x['position_error_m'] for x in epoch_rows if x['run']==mode and x['model']==name]
        assert len(errors)==30 and errors[-1]==row['position_error_m']
        assert abs(float(np.sqrt(np.mean(np.square(errors))))-row['trajectory_epoch_rmse_m'])<1e-8
summary=[read(O/'raw'/s/'execution_summary.json') for s in ['strict','bypass']]
assert all(not s['denied'] for s in summary)
calls=sum(m['actual_nonlinear_solver_calls'].values()); assert calls==52
decision='GVINS_DIAGNOSTIC_CANDIDATES_WORK_BUT_SELECTOR_OTHER_LIMITATIONS'
next_task='GVINS_GPS_L1_SHORT_ARC_IDENTIFIABILITY_AND_INITIALIZATION_REVIEW_V1'
save(O/'decision.json',dict(task=protocol['task'],run_id=m['run_id'],decision=decision,
   identity=m['identity'],status='COMPLETED_DIAGNOSTIC',clock_gate_primary_blocker=False,
   residual_position_disagreement_observed=True,validation_position_disagreement_observed=True,
   selector_bypass_non_degenerate=False,strict_eligible=0,bypass_eligible=1,
   new_threshold_selected=False,paper_updated=False,independent_validation=False,next_task=next_task))
save(O/'verification.json',dict(status='PASS_EXECUTION_AND_EVALUATION_IDENTITY',run_id=m['run_id'],
    utc=datetime.now(timezone.utc).isoformat(),native_source_sha_matches=len(b.sources),
    run_seal_source_hashes_unchanged=True,pretruth_output_hashes_unchanged=True,
    shared_fit_checks=checks,unique_top_level_runs=20,nonlinear_solver_calls=calls,
    nonlinear_solver_calls_by_function=m['actual_nonlinear_solver_calls'],
    selector_distinct_results=2,v71_selection_invocations=4,
    selector_call_identity='Two scientific diagnostic decisions plus two identical replay calls through frozen v72 shell.',
    finite_position_outputs_strict=15,finite_position_outputs_bypass=15,
    strict_native_numerical_success=13,bypass_native_numerical_success=15,
    csv_full_precision_roundtrip=True,csv_rows={'candidate_fit_summary':15,'clock_gate_impact':1,
        'residual_position_ordering':210,'selector_comparison':2},
    position_error_epoch_records=len(epoch_rows),pair_counts=m['pair_counts'],
    light_time_max_error_s=max(s['light_time_max_error_s'] for s in summary),
    light_time_max_iterations=max(s['light_time_max_iterations'] for s in summary),
    candidate_reference_file_reads=0,rtk_derived_common_initialization=True,
    native_weighting_changed=False,science_threshold_selected=False,paper_updated=False,
    evaluation_console_issue='Missing clean import affected only final console print after all evaluation artifacts had been saved; corrected without rerunning fits or selections.',
    independent_validation=False))
mapping='''# Measurement and diagnostic gate mapping

All existing scientific files and wrapper.py remain unchanged. The new bridge installs process-local aliases by original function identity, including full/train/held-out, static, CTD, projected, robust, GIR, cascade and covariance consumers. Complete alias targets are recorded in integration_qa.json.

The three measurement leaves are models.residuals_and_jacobian, trajectory_models.residuals_and_jacobian_ctd and be_gtr_solver.h_and_jacobian_x. Their lookup verifies exact satellite metadata arrays and time-row identity. Satellite arrays are receive-time broadcast metadata for lookup only; every prediction uses candidate-position-dependent retarded GNSS propagation. The original t0 stays at the locked interval start. No array is a substitute for physical prediction. Exact memoization uses the entire geometric state and row indices, with no rounding or approximate reuse.

quality.physical_plausibility always computes its original result. Diagnostic mode removes only the beta0_limit reason, retaining numerical, speed and bdot handling. This leaf also serves GIR proposal checks. risk_veto.severe_risk_veto always computes all original reasons and removes only abs(beta0_estimated_mps)>30 in diagnostic mode. hierarchical_gates._refine_gate uses an in-memory function with precisely the beta>30 disjunct removed; speed>300, bdot>1 and every other statement remain unchanged. Stored cascade training rules and their 0.02 drift check remain active.

M0–M9 reuse byte-identical full/train payloads. M10–M14 have explicitly separate strict and bypass fit paths because GIR tests physical qualification during optimization. Cascades retain their native base fits and original refinement budgets. There are 20 top-level executions and 52 actual nonlinear solver calls, including internal full/train/base/refinement work. No alternate seed, window, optimizer budget or starting point was tried.

The strict and bypass selector inputs were saved before RTK position-error evaluation. v71 produced two distinct decisions; the original v72 shell was checked by two deterministic v71 replay calls. A separate same-strict-fit, postfit-only qualification table was computed before reference loading; it does not represent a third selected method. Final eligibility is stage-specific: CSV fields strict_eligible/bypass_eligible mean native base-filter eligibility; family and secondary ranking paths are in the selector traces.

Native ephemeris covariance diagnostics retain their original zero_or_not_configured identity because no exogenous covariance contract was supplied. This is not a claim of zero broadcast error. No heteroscedastic raw Doppler weights were inserted. The native full, train and held-out roles remain 180,126 and54 rows.
'''
(O/'INTEGRATION_AND_BYPASS_MAPPING.md').write_text(mapping,encoding='utf-8')

s=m['rows']['strict'];t=m['rows']['bypass'];a,bsel=m['selectors']
main=['M0_static_position','M1_static_position_bias','M2_ctd_full','M3_ctd_no_drift','M4_ctd_no_bias','M12_ctd_full_plus_gir_refine']
table=['| Candidate | Full RMSE (m/s) | Trimmed validation (m/s) | b0 (m/s) | bdot (m/s²) | Endpoint error (m) | Strict / bypass base eligible |',
       '|---|---:|---:|---:|---:|---:|---|']
def f(x): return 'NA' if x is None else f'{x:.6f}'
for name in main:
    x=s[name];z=t[name]
    table.append(f"| {name.split('_')[0]} | {f(x['residual_rmse_mps'])} | {f(x['validation_metric_mps'])} | {f(x['b0_mps'])} | {f(x['bdot_mps2'])} | {x['position_error_m']:.3f} | {x['base_eligible']} / {z['base_eligible']} |")
report=f'''# Initialization-assisted sports_field clock-gate diagnostic

任务：{protocol['task']}  
run_id：{m['run_id']}  
正式decision：**{decision}**  
数据身份：**INITIALIZATION_ASSISTED_REAL_RF_DIAGNOSTIC**

## 结论

已完成同一30 s真实GPS L1片段的15个严格候选及5个GIR相关旁路候选，共20次顶层执行、52次实际非线性solver调用。M0–M9的full/train拟合在两种资格版本间逐值复用。严格版本13/15满足native numerical_success，旁路版本15/15；这不等于所有候选都完成了收敛，也不等于位置准确。BE分支M5/M6/M8/M9在严格版本中有有限输出但converged=false，全部记录保留。

absolute-clock限制确实拒绝了真实GNSS clock量级，不过移除该限制后只有M1恢复基础资格，其他14个候选仍被其他条件排除。因此本片段不能支持“只改clock门即可形成有效GNSS定位”的结论。两个选择输出都具有公里级误差，旁路后的实际选择更差。

## 初始化和数据身份

用户明确授权RTK仅构造预先固定扰动的共同初始化。读取位置值以前，protocol.json已经固定：首个canonical历元的RTK-fixed primary位置加局部East 10000 m，速度初值为零，联合clock初值为零，投影prior为原B0_none。偏移不根据任何候选结果调整，全体候选和训练拟合共享该初值。

anchor位于TOW83929.002，模型原点仍为锁定区间起点83928.102，二者相差0.9 s；这里使用近似共同位置先验，没有借PVT速度进行初值传播。因此10000 m表示相对anchor的构造偏移，不宣称相对模型原点真实位置恰好10000 m。原初始化阻断记录保留在上一级目录，没有覆盖。

区间仍为GPS week2134、TOW[83928.102,83958.102)，1 Hz共30个历元、6颗GPS L1卫星、180行，PRN为2/5/6/13/29/30。原生70/30规则产生126训练行、54留出行。最终共同报告历元为最后一条canonical观测TOW83958.002，未将其误写为区间右端点。30个轨迹历元不构成30次独立定位实验。

## 严格与旁路选择

| Version | Selected model | Base eligible | Fallback | Full residual RMSE (m/s) | Trimmed validation (m/s) | Endpoint error (m) |
|---|---|---|---|---:|---:|---:|
| STRICT_EGSHS | M8 | False | True | {a['residual_rmse_mps']:.6f} | {a['validation_metric_mps']:.6f} | {a['position_error_m']:.3f} |
| CLOCK_GATE_BYPASS_DIAGNOSTIC_SELECTOR | M1 | True | False | {bsel['residual_rmse_mps']:.6f} | {bsel['validation_metric_mps']:.6f} | {bsel['position_error_m']:.3f} |

严格版本没有基础合格候选，原fallback按validation返回M8。该输出不应描述为通过clock及风险资格的正式选择。旁路版本只有M1通过基础资格，实际路径为no_dynamic_candidate，再返回static family的M1。这个变化发生在同一份M1/M8拟合结果上；GIR重拟合没有产生进入最终候选竞争的新分支。旁路选择仍缺少跨族竞争，不能称为已展示非退化的GNSS状态选择能力。

严格输出中10个候选的最终b0触发20 m/s检查。对同一严格fit仅重算postfit资格，只有M1恢复基础资格；允许GIR旁路拟合后，基础合格集合仍仅有M1。这里的“10个clock检查失败”与“1个仅因absolute-clock恢复基础资格”是不同计数。另有GIR proposal的clock影子失败，详见raw分区中的shadow计数，不能以最终b0计数代替。

## 重点固定候选

{chr(10).join(table)}

上表为严格fit数值；M12旁路fit的RMSE为{t['M12_ctd_full_plus_gir_refine']['residual_rmse_mps']:.6f} m/s、位置误差{t['M12_ctd_full_plus_gir_refine']['position_error_m']:.3f} m，仍未通过基础资格。全部15模型、两种fit路径及拒绝理由均见candidate_fit_summary.csv与diagnostic_metrics.json。

M2正常产生收敛和有限输出，b0={s['M2_ctd_full']['b0_mps']:.6f} m/s、bdot={s['M2_ctd_full']['bdot_mps2']:.6f} m/s²，clock量级接近前置物理QA背景，未向其输入任何81.5 m/s修正。它的全记录RMSE降至0.308192 m/s，终点位置误差却达到37.272 km。旁路后，M2仍触发位置协方差proxy、条件数以及真实数据动态预测增益相关检查；其proxy约114.97 km、normal condition约7.52e15属于相关的局部信息诊断，不能作为彼此独立的多项证据累计。

M12/M13的严格GIR refinement未形成数值成功状态，保留其base位置作为已保存输出；旁路使GIR可以进行更新，仍触发原cascade的速度和abs(bdot)<0.02等条件。M3/M4分别估计约141.85/207.52 m/s速度，原motion资格继续排除这些输出。本轮没有整体关闭physical_plausibility。

## Residual Blindness与预测排序

完整枚举每个版本的105个候选对，严格版本观察到56对残差—位置排序分歧、50对trimmed-validation—位置排序分歧；旁路版本分别为46对和44对。两组共享大量拟合，候选对也相互依赖，这些计数只描述本pilot，不能解释为独立样本频率或总体概率。所有有限已保存输出参与完整枚举，包括未合格或未收敛候选，标志字段可用于明确分层。

一个同clock、只释放速度的例子是M1→M3：全记录RMSE从0.707325降至0.441475 m/s，终点误差从19.174增至907.725 km。无clock的M0→M4中，残差和trimmed validation均大幅下降，终点误差从1168.757增至1331.199 km，且M4速度不合格。这些是保存输出上的排序事实，不将某个门控认定为独立因果作用。

M1与M2之间，残差从0.707325降至0.308192 m/s，终点误差从19.174增至37.272 km，而trimmed validation从0.556261升至0.943598 m/s；在这一对上，留出证据确实揭示了dynamic输出的代价。相反，完整枚举中的其他候选对仍显示预测排序分歧。完整不利与有利对照都保留在residual_position_ordering.csv。

## 实现、隔离与来源

使用原wrapper的候选位置依赖light-time、广播星历直接解析速度、satellite clock及完整Earth-rotation表达。全记录/训练/留出5种基础状态的预测和Jacobian映射核验通过，共15项；单位、原时间原点和clock线性列不变。严格与旁路只在20/30 m/s absolute b0资格处形成差异，0.1、dynamic 0.02、cascade strict0.02和全部非clock规则保留。具体调用点见INTEGRATION_AND_BYPASS_MAPPING.md。

候选进程只获得授权的共同初值和原严格观测schema，没有读取RTK/PVT文件、PVT速度、pseudorange、前置逐历元clock或真实误差。RTK构造初值是明确的例外，不能据“候选进程直接读取为零”声称完全truth-free。候选/选择文件在{seal['utc']}形成PRE_TRUTH_SEAL，随后评价器才读取RTK位置计算误差，评价完成时间为{m['evaluation_utc']}。

55项冻结源码SHA、RUN_SEAL内执行代码与PRE_TRUTH输出SHA核对通过。实际v71调用4次，其中2次产生严格/旁路决定，另2次通过原v72壳作一致性复核，没有增加独立实验。评价脚本最终终端打印曾缺少clean导入，该问题发生在评价文件保存之后，已修正导入，没有重跑拟合或选择。

原native AIC/BIC为用户允许的条件性补充，本轮未列入：当前GNSS运行没有锁定用于该基线的likelihood/noise-scale合同，不能套用其他受控实验的sigma或补造准则。已有M0/M1/M2/M3/M4/M12固定输出均完整保留。

## 后续研究判断

优先建议 **{next_task}**。先利用本轮保存状态检查短弧位置可辨识性和共同初值依赖，再决定是否值得另行设计clock-domain规则。当前证据表明absolute-clock门是适配限制之一，解除后仍存在位置支撑、预测增益及原family语义的限制。不能从一次辅助初始化pilot判断整个方法不成立，也不能以更小残差替代真实位置评价。

本轮没有选择新threshold、生成noise、变更窗口或更新论文；没有完成独立真实定位验证。四张结果表分别为[candidate_fit_summary.csv](candidate_fit_summary.csv)、[clock_gate_impact.csv](clock_gate_impact.csv)、[residual_position_ordering.csv](residual_position_ordering.csv)、[selector_comparison.csv](selector_comparison.csv)。
'''
(O/'report.md').write_text(report,encoding='utf-8')
save(O/'output_manifest.json',dict(run_id=m['run_id'],utc=datetime.now(timezone.utc).isoformat(),
   files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*'))
          if p.is_file() and p.name!='output_manifest.json' and '__pycache__' not in p.parts]))
for key,val in dict(TASK_STATUS=decision,TOTAL_REPORT=str(O/'report.md'),
   INITIALIZATION='RTK first canonical position + fixed East 10 km; v=0; joint clock=0; initialization-assisted',
   CANDIDATE_FITS_SUCCESS='STRICT 13/15 native numerical_success; BYPASS 15/15; not all converged',
   STRICT_CLOCK_REJECTED=10,BYPASS_RECOVERED=1,
   M2_FIT='converged; full RMSE 0.308192 m/s; endpoint error 37271.914 m',M2_B0=s['M2_ctd_full']['b0_mps'],
   STRICT_SELECTED='M8 / 9960.844 m / low-quality fallback',BYPASS_SELECTED='M1 / 19173.775 m / only base-eligible candidate',
   RESIDUAL_POSITION_DISAGREEMENT='strict 56/105; bypass 46/105',
   VALIDATION_POSITION_DISAGREEMENT='strict 50/105; bypass 44/105',CLOCK_GATE_PRIMARY_BLOCKER='false',
   NEW_THRESHOLD_SELECTED='false',PAPER_UPDATED='false',NEXT_TASK=next_task).items(): print(f'{key}={val}')
