"""Post-selection RTK position evaluation; this file is denied to candidate processes."""
from pathlib import Path
from datetime import datetime, timezone
import json, pickle, itertools
import numpy as np
import pyarrow.parquet as pq
from bridge import Bridge, O, A, read, save, sha, clean

seal=read(O/'PRE_TRUTH_SEAL.json')
for item in seal['files']: assert sha(item['path'])==item['sha256']
b=Bridge() # Load only the public candidate observation schema and source types for pickle.
with (O/'pretruth_results.pkl').open('rb') as f: data=pickle.load(f)
# Pure postfit qualification on strict states is computed and saved before RTK evaluation.
import pandas as pd
b.bypass=True
same=[b.requalify(p['row']) for p in data['payloads']['strict']]
same,_=b.mods['robust_gates'].apply_robust_hard_gate(same)
same=b.rt._enrich_risk(pd.DataFrame(same),b.obs)
same=b.rt._add_observable_outlier_evidence(same,dict(dataset_type='real'))
best=b.mods['family_evidence'].best_candidate(same,b.mods['family_evidence'].STATIC_MODELS)
same_eligible,same_reasons=b.mods['hierarchical_gates']._base_filter(same,best,b.mods['hierarchical_gates'].HierarchicalGateConfig())
same_names=set(same_eligible.candidate_model)
save(O/'same_fit_qualification_pretruth.json',dict(utc=datetime.now(timezone.utc).isoformat(),
     base_eligible=sorted(same_names),reasons=same_reasons,position_reference_loaded=False,
     purpose='Pure postfit diagnostic qualification only; no extra selector invocation'))
inputs=read(O/'input_manifest.json'); epochs=np.asarray(inputs['row_times_ns'],dtype=np.int64)
refpath=A/'sports_field_pilot/time_alignment_v2/pilot_rtk_reference.parquet'
binding=next(x for x in inputs['bindings'] if Path(x['path'])==refpath)
assert sha(refpath)==binding['sha256']
reference=pq.read_table(refpath,columns=['gpst_calendar_ns','primary_px','primary_py','primary_pz',
                        'reference_eligible','valid_fix','carr_soln']).to_pandas().set_index('gpst_calendar_ns').loc[epochs]
assert reference.reference_eligible.all() and reference.valid_fix.all() and (reference.carr_soln==2).all()
truth=reference[['primary_px','primary_py','primary_pz']].to_numpy(float)
times=(epochs-b.origin)/1e9;end=times[-1]
comparisons=[];rows={};gates={};initial=read(O/'initialization_prior.json')
epochs_table=[]

for mode,frame in data['rows'].items():
    b.bypass=mode=='bypass'
    best=b.mods['family_evidence'].best_candidate(frame,b.mods['family_evidence'].STATIC_MODELS)
    eligible,reasons=b.mods['hierarchical_gates']._base_filter(frame,best,b.mods['hierarchical_gates'].HierarchicalGateConfig())
    eligible_names=set(eligible.candidate_model)
    gates[mode]=dict(base_eligible=sorted(eligible_names),reasons=reasons)
    result={}
    for ix,r in frame.iterrows():
        model=r.candidate_model
        payload=next(p for p in data['payloads'][mode] if p['model']==model)
        final=np.array([r.get('final_ecef_'+k+'_m',np.nan) for k in ['x','y','z']],float)
        velocity=np.array([r.get('estimated_v'+k+'_mps',np.nan) for k in ['x','y','z']],float)
        positions=final[None,:]+(times-end)[:,None]*velocity[None,:]
        error=np.linalg.norm(positions-truth,axis=1)
        capture=next((x for x in reversed(payload['full_capture']) if x['model']==model),None)
        if capture:
            assert np.array_equal(final,capture['positions'][-1])
            assert np.array_equal(velocity,capture['velocity'])
        out=dict(model=model,fit_source=r.fit_source,fit_success=bool(r.numerical_success),
            full_fit_numerical_success=bool(capture['numerical_success']) if capture else None,
            full_fit_converged=bool(r.converged),residual_rmse_mps=r.full_residual_rmse_mps,
            validation_metric_mps=r.trimmed_validation_rmse_mps,
            raw_validation_rmse_mps=r.raw_validation_rmse_mps,
            validation_score_mps=b.mods['family_evidence'].validation_score(r),
            b0_mps=r.beta0_estimated_mps,bdot_mps2=r.beta_dot_estimated_mps2,
            speed_mps=r.estimated_speed_mps,condition_number=r.condition_number,
            physical_qualified=bool(r.physical_plausible),quality_pass=bool(r.quality_pass),
            severe_risk_veto=bool(r.severe_risk_veto),severe_risk_reason=r.severe_risk_reason,
            base_eligible=model in eligible_names,base_reject_reason=reasons[ix],
            stored_refine_gate_pass=bool(r.get('refine_gate_pass',True)),stored_refine_reason=r.get('refine_gate_reason',''),
            fit_failure_reason=r.failure_reason,position_error_m=float(error[-1]),
            trajectory_epoch_rmse_m=float(np.sqrt(np.mean(error**2))),
            trajectory_epoch_max_error_m=float(error.max()),
            CLOCK_GATE_SHADOW_FAIL=bool(np.isfinite(r.beta0_estimated_mps) and abs(r.beta0_estimated_mps)>20),
            position_cov_proxy_m=r.get('position_cov_sqrt_trace_m',np.nan),
            fit_path_affected_by_clock_gate=int(model.split('_')[0][1:])>=10,
            objective_rss_m2ps2=float(np.sum(capture['residual']**2)) if capture else np.nan,
            native_robust_cost_per_dof=r.get('robust_cost_per_dof',np.nan))
        result[model]=out
        for i,t in enumerate(epochs):
            epochs_table.append(dict(run=mode,model=model,epoch_ns=int(t),position_error_m=float(error[i]),
                estimated_x_m=positions[i,0],estimated_y_m=positions[i,1],estimated_z_m=positions[i,2]))
    rows[mode]=result

summary=[]
for model in rows['strict']:
    s=rows['strict'][model];t=rows['bypass'][model]
    summary.append(dict(model=model,fit_success=s['fit_success'],residual_rmse=s['residual_rmse_mps'],
        validation_metric=s['validation_metric_mps'],b0=s['b0_mps'],bdot=s['bdot_mps2'],
        strict_eligible=s['base_eligible'],bypass_eligible=t['base_eligible'],
        strict_reject_reason=s['base_reject_reason'],bypass_reject_reason=t['base_reject_reason'],
        position_error=s['position_error_m'],bypass_position_error_m=t['position_error_m'],
        bypass_residual_rmse_mps=t['residual_rmse_mps'],bypass_validation_metric_mps=t['validation_metric_mps'],
        bypass_b0_mps=t['b0_mps'],bypass_bdot_mps2=t['bdot_mps2'],
        fit_path_affected_by_clock_gate=s['fit_path_affected_by_clock_gate'],
        full_fit_converged=s['full_fit_converged'],strict_full_fit_success=s['full_fit_numerical_success'],
        numerical_success_bypass=t['fit_success'],bypass_fit_source=t['fit_source'],
        CLOCK_GATE_SHADOW_FAIL=t['CLOCK_GATE_SHADOW_FAIL'],
        strict_severe_risk_reason=s['severe_risk_reason'],bypass_severe_risk_reason=t['severe_risk_reason'],
        strict_refine_reason=s['stored_refine_reason'],bypass_refine_reason=t['stored_refine_reason'],
        comparison_eligibility_stage='native_base_filter; family/near-tie trace stored separately',
        position_error_unit='m',b0_unit='m/s',bdot_unit='m/s^2'))

pair_rows=[]
nested={frozenset(x) for x in [('M0','M1'),('M0','M4'),('M1','M3'),('M3','M2'),('M4','M3'),('M4','M2')]}
for mode,rr in rows.items():
    for x,y in itertools.combinations(rr.values(),2):
        de=y['position_error_m']-x['position_error_m']
        dr=y['residual_rmse_mps']-x['residual_rmse_mps']
        dv=y['validation_metric_mps']-x['validation_metric_mps']
        comparable=bool(np.isfinite([de,dr,dv]).all())
        pair_rows.append(dict(run=mode,model_i=x['model'],model_j=y['model'],
           delta_residual_j_minus_i_mps=dr,delta_validation_j_minus_i_mps=dv,
           delta_position_error_j_minus_i_m=de,
           lower_residual_higher_position_error=bool(dr*de<0) if comparable else None,
           lower_validation_higher_position_error=bool(dv*de<0) if comparable else None,
           finite_comparable=comparable,
           clean_nested_pair=frozenset([x['model'].split('_')[0],y['model'].split('_')[0]]) in nested,
           mathematical_identity='Same 180 raw rows and endpoint; raw RMSE descriptive ordering, optimization objectives may differ'))

selectors=[]
for mode,selection in data['selections'].items():
    row=rows[mode][selection['model']]
    selectors.append(dict(run=mode,selector_identity='STRICT_EGSHS' if mode=='strict' else 'CLOCK_GATE_BYPASS_DIAGNOSTIC_SELECTOR',
        selected_model=row['model'],position_error_m=row['position_error_m'],residual_rmse_mps=row['residual_rmse_mps'],
        validation_metric_mps=row['validation_metric_mps'],b0_mps=row['b0_mps'],bdot_mps2=row['bdot_mps2'],
        fallback_status=bool(selection['low_quality']),base_eligible=row['base_eligible'],reason=selection['reason'],
        selected_fit_source=row['fit_source'],initialization_assisted=True))

strict_names=set(gates['strict']['base_eligible']);bypass_names=set(gates['bypass']['base_eligible'])
clock_rejected=sum(not x['physical_qualified'] and x['CLOCK_GATE_SHADOW_FAIL'] for x in rows['strict'].values())
only_clock=[m for m in same_names if m not in strict_names]
restored=sorted(bypass_names-strict_names)
impact=[dict(planned_models=15,strict_fit_success=sum(x['fit_success'] for x in rows['strict'].values()),
    strict_absolute_clock_shadow_fail=sum(x['CLOCK_GATE_SHADOW_FAIL'] for x in rows['strict'].values()),
    strict_physical_clock_rejected=clock_rejected,
    recovered_same_fit_base_qualification=len(only_clock),recovered_same_fit_models=';'.join(sorted(only_clock)),
    bypass_base_recovered=len(restored),bypass_base_recovered_models=';'.join(restored),
    bypass_still_base_ineligible=15-len(bypass_names),strict_base_eligible_count=len(strict_names),
    bypass_base_eligible_count=len(bypass_names),fit_path_additional_models=5,
    count_identity='One pilot, 15 hypotheses; full/train success and base eligibility are distinct')]

internal={}
for fitmode in ['strict','bypass']:
    for p in (O/'raw'/fitmode).glob('M*.seal.json'):
        for key,value in read(p)['solver_counts'].items(): internal[key]=internal.get(key,0)+value
pair_counts={mode:dict(pairs=sum(x['run']==mode and x['finite_comparable'] for x in pair_rows),
       residual_disagreements=sum(x['run']==mode and bool(x['lower_residual_higher_position_error']) for x in pair_rows),
       validation_disagreements=sum(x['run']==mode and bool(x['lower_validation_higher_position_error']) for x in pair_rows))
       for mode in ['strict','bypass']}
metrics=dict(run_id=read(O/'protocol.json')['run_id'],identity='INITIALIZATION_ASSISTED_REAL_RF_DIAGNOSTIC',
    evaluation_utc=datetime.now(timezone.utc).isoformat(),rows=rows,selectors=selectors,impact=impact,
    pair_counts=pair_counts,actual_top_level_candidate_runs=20,actual_nonlinear_solver_calls=internal,
    original_initialization_block_preserved=True,threshold_selected=False,paper_updated=False,
    native_aic_bic='NOT_REPORTED: no GNSS native likelihood noise-scale contract locked; no baseline rule invented',
    independent_real_positioning_validation=False,
    reference_identity='RTK fixed primary position; same device/capture, initialization-assisted',
    report_epoch_ns=int(epochs[-1]),epochs_are_independent_replicates=False)
save(O/'diagnostic_metrics.json',metrics)
save(O/'epoch_position_errors.json',epochs_table)
save(O/'tables.json',{'candidate_fit_summary.csv':summary,'clock_gate_impact.csv':impact,
                     'residual_position_ordering.csv':pair_rows,'selector_comparison.csv':selectors})
save(O/'evaluation_complete.json',dict(utc=metrics['evaluation_utc'],pretruth_seal_sha256=sha(O/'PRE_TRUTH_SEAL.json'),
        reference_sha256=sha(refpath),position_results_created=True,selectors_not_rerun_after_truth=True))
print(json.dumps(dict(selectors=clean(selectors),impact=clean(impact),pair_counts=pair_counts,solver_calls=internal),indent=2))
