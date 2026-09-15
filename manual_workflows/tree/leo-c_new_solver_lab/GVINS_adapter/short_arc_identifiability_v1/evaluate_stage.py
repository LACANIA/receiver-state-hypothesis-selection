from common import *
import pickle
from dataclasses import asdict
from itertools import combinations
from collections import Counter
import pandas as pd
from scipy.cluster.hierarchy import linkage

def main():
    assert not (O/'POSTFIT_EVALUATION.json').exists()
    schedule=read(O/'fit_schedule.json');paths=[O/'raw'/f"{s['fit_id']}.pkl" for s in schedule]
    assert len(paths)==54 and all(p.exists() for p in paths)
    for p in paths:assert sha(p)==read(p.with_suffix('.seal.json'))['sha256']
    # All outputs completed before the evaluator loads position reference.
    save(O/'FIT_OUTPUT_SEAL.json',dict(utc=now(),files=[dict(path=str(p),sha256=sha(p)) for p in paths],configurations=54,reference_role='Already authorized for prefit information and initialization, never fit-objective input',selector_calls=0))
    src=check_sources();rtroot=Path(next(x['resolved_path'] for x in src if Path(x['resolved_path']).name=='models.py')).parent.parent
    sys.path.insert(0,str(rtroot))
    from leo_positioning import uncertainty_diagnostics as ud,risk_veto as rv
    refs=read(O/'reference_qa_only.json');inf=read(O/'information_full.json');cfg=read(O/'aggregation_contract.json')
    # Recompare velocity paths on identical available rows. Raw sensitivity
    # matrices have fewer rows where the fixed position-derivative stencil gaps.
    matrices=read(O/'local_matrices.json');obsrows=read(O/'candidate_observations.json')
    refmap={r['gpst_calendar_ns']:r for r in refs['reference']};sensitivity=[]
    for T in WINDOWS:
        kept=[r for r in obsrows if r['receive_gpst_ns']<ORIGIN+T*10**9 and refmap[r['receive_gpst_ns']]['reference_eligible']]
        commonmask=np.array([np.isfinite(vec(refmap[r['receive_gpst_ns']],'position_derived_v')).all() for r in kept])
        frame=enu(vec(refs['endpoints'][str(T)],'primary_p'))
        for m in MODELS:
            key=f'{T}_NATURAL_SUPPORT_'
            p=matrices[key+'PVT_PRIMARY_'+m];s=matrices[key+'POSITION_DERIVED_SENSITIVITY_'+m]
            a=np.array(p['Jp'])[commonmask];n=np.array(p['Jn']).reshape(len(kept),-1)[commonmask]
            dp,_,_,_=information(a,n,frame)
            ds=next(r for r in inf if r['window']==T and r['model']==m and r['velocity']=='POSITION_DERIVED_SENSITIVITY')
            assert len(a)==ds['rows']
            sensitivity.append(dict(window=T,model=m,common_rows=len(a),primary_full_rows=len(kept),primary_common_lambda_min=dp['lambda_min'],position_derived_lambda_min=ds['lambda_min'],lambda_ratio_same_rows=ds['lambda_min']/dp['lambda_min'],weak_proxy_ratio_same_rows=ds['weak_position_std_proxy_m']/dp['weak_position_std_proxy_m']))
    save(O/'velocity_sensitivity_common_rows.json',sensitivity)
    models={};rows=[];points={};calls=Counter();enriched={};residual_checks=[]
    for spec,path in zip(schedule,paths):
        with path.open('rb') as f:payload=pickle.load(f)
        assert payload['spec']==spec
        calls.update(payload['solver_counts'])
        T=spec['window'];m=spec['model'];r=dict(payload['row']);full=payload['full_result'];tr=payload['train_result'];x=np.array(full['state'])
        v=np.zeros(3) if m in ['M0','M1'] else x[3:6]
        q=x[:3]+T*v;truth=vec(refs['endpoints'][str(T)],'primary_p');err=q-truth
        fr=enu(truth);J=np.array(payload['full_J']);a=J[:,:3];n=J[:,3:].copy()
        if m not in ['M0','M1']:n[:,:3]-=T*a
        d,S,C,dirs=information(a,n,fr)
        residual=np.array(payload['full_residual']);sigma2=float(residual@residual/max(len(residual)-len(x),1))
        native=ud.covariance_diagnostics_from_jacobian(residual,J,np.ones(len(residual)),'static' if m in ['M0','M1'] else 'trajectory',max((o['receive_gpst_ns']-ORIGIN)/1e9 for o in read(O/'candidate_observations.json') if o['receive_gpst_ns']<ORIGIN+T*10**9))
        r.update(native);enriched[spec['fit_id']]=r
        refi=next(z for z in inf if z['window']==T and z['model']==m and z['velocity']=='PVT_PRIMARY' and z['support']=='NATURAL_SUPPORT')
        weak=np.array(refi['weak_ecef']);strong=np.array(refi['strong_ecef'])
        row=dict(fit_id=spec['fit_id'],model=m,window=T,support=spec['support'],module=spec['module'],direction=spec['direction'],radius_m=spec['radius_m'],
            rows=len(residual),train_rows=len(payload['train_residual']),heldout_rows=len(payload['heldout_residual']),
            full_fit_success=full['success'],full_converged=full['converged'],full_iterations=full['iterations'],full_accepted_steps=full['accepted_steps'],
            train_fit_success=tr['success'],train_converged=tr['converged'],train_iterations=tr['iterations'],
            initial_objective=full['initial_objective'],objective=full['final_objective'],
            full_residual_rmse_mps=float(np.sqrt(np.mean(residual**2))),train_residual_rmse_mps=r.get('train_residual_rmse_mps'),heldout_raw_rmse_mps=r.get('raw_validation_rmse_mps'),heldout_trimmed_rmse_mps=r.get('trimmed_validation_rmse_mps'),
            b0_mps=r['beta0_estimated_mps'],bdot_mps2=r['beta_dot_estimated_mps2'],speed_mps=np.linalg.norm(v),
            endpoint_error_m=np.linalg.norm(err),initial_to_final_position_m=np.linalg.norm(q-np.array(spec['initial_ecef_m'])),
            p0_parameter_shift_from_init_m=np.linalg.norm(x[:3]-np.array(spec['initial_ecef_m'])),p0_error_m=np.linalg.norm(x[:3]-vec(refs['origin'],'primary_p')),
            native_numerical_success=r['numerical_success'],native_physical_plausible=r['physical_plausible'],native_quality_pass=r['quality_pass'],native_failure_reason=r.get('failure_reason',''),
            clock_20_shadow_fail=bool(abs(r.get('beta0_estimated_mps',np.nan))>20),clock_30_shadow_fail=bool(abs(r.get('beta0_estimated_mps',np.nan))>30),
            bdot_general_0p1_shadow_fail=bool(abs(r.get('beta_dot_estimated_mps2',np.nan))>.1),bdot_dynamic_0p02_trigger=bool(abs(r.get('beta_dot_estimated_mps2',np.nan))>.02),cascade_0p02_rule='NOT_APPLICABLE_M0_M4_NO_REFINEMENT',
            native_normal_condition=native['normal_condition_number'],native_covariance_proxy_m=native['position_cov_sqrt_trace_m'],
            solution_position_condition=d['position_condition'],solution_lambda_min=d['lambda_min'],solution_weak_proxy_unit_m=d['weak_position_std_proxy_m'],
            solution_weak_proxy_RSS_scaled_m=d['weak_position_std_proxy_m']*np.sqrt(sigma2),solution_covariance_trace_unit_m2=d['covariance_pinv_trace_m2'],
            objective_gradient_infinity_norm=float(np.max(abs(J.T@residual))),position_gradient_norm=float(np.linalg.norm(J[:,:3].T@residual)),
            error_along_reference_weak_m=err@weak,error_along_reference_strong_m=err@strong,error_weak_fraction=(err@weak)**2/(err@err) if err@err else None,
            report_epoch_ns=ORIGIN+T*10**9,native_last_epoch_offset_from_T_s=-.1,source=str(path),elapsed_s=payload['elapsed_s'])
        for prefix,arr in [('initial',spec['initial_ecef_m']),('offset',spec['offset_ecef_m']),('p0',x[:3]),('velocity',v),('qT',q),('error',err),('solution_weak_ecef',d['weak_ecef'])]:
            for ax,z in zip('xyz',arr):row[prefix+'_'+ax]=z
        rows.append(row);points[spec['fit_id']]=q
        models[spec['fit_id']]=dict(state=x,conditional_position_information=S,projected_position_J=C,conditional_position_eigenvectors=dirs,reference_weak=weak,reference_strong=strong,local_info=d)
        expected=.5*float(residual@residual)
        residual_checks.append(dict(fit_id=spec['fit_id'],stored_objective=full['final_objective'],recomputed_half_rss=expected,difference=expected-full['final_objective']))
        assert abs(expected-full['final_objective'])<=1e-8*max(expected,1.)
    # Evaluate native risk checks as shadow diagnostics, never selecting/filtering.
    for row in rows:
        same=[x for x in rows if x['window']==row['window'] and x['direction']==row['direction'] and x['radius_m']==row['radius_m']]
        group=pd.DataFrame([enriched[x['fit_id']] for x in same]);r=enriched[row['fit_id']]
        has_static=any(x['model'] in ['M0','M1'] for x in same)
        if has_static or row['model'] in ['M0','M1']:
            rr=rv.compute_group_references(group);veto,why=rv.severe_risk_veto(r,rr)
            row.update(native_severe_risk_shadow=veto,native_severe_risk_reason=why,qualification_comparator_scope='SAME_WINDOW_DIRECTION_RADIUS_AVAILABLE_CORE_MODELS')
        else:
            # Preserve magnitude/condition checks but do not fabricate a static fit.
            checks=[]
            for col,lim,absolute in [('position_cov_sqrt_trace_m',5000,False),('condition_number',1e14,False),('estimated_speed_mps',300,False),('beta0_estimated_mps',30,True),('beta_dot_estimated_mps2',1,True)]:
                z=r.get(col,np.nan)
                if np.isfinite(z) and (abs(z) if absolute else z)>lim:checks.append(col+'>'+str(lim))
            row.update(native_severe_risk_shadow=None,native_severe_risk_reason=';'.join(checks)+';DYNAMIC_GAIN_NOT_AVAILABLE_NO_MATCHED_STATIC_FIT',qualification_comparator_scope='PARTIAL_NOT_AVAILABLE')
    pairrows=[];clusterrows=[];groups=[]
    for T in [30,120]:
        sub=[r for r in rows if r['window']==T]
        for a,b in combinations(sub,2):
            dr=b['full_residual_rmse_mps']-a['full_residual_rmse_mps'];dv=b['heldout_trimmed_rmse_mps']-a['heldout_trimmed_rmse_mps'];de=b['endpoint_error_m']-a['endpoint_error_m']
            sameinit=a['direction']==b['direction'] and a['radius_m']==b['radius_m']
            type_='SAME_MODEL_INITIALIZATION' if a['model']==b['model'] else 'MATCHED_INITIALIZATION_STATE_PAIR' if sameinit else 'DIFFERENT_MODEL_DIFFERENT_INITIALIZATION'
            pairrows.append(dict(window=T,fit_a=a['fit_id'],fit_b=b['fit_id'],model_a=a['model'],model_b=b['model'],comparison_identity=type_,direction_a=a['direction'],direction_b=b['direction'],radius_a_m=a['radius_m'],radius_b_m=b['radius_m'],position_distance_m=np.linalg.norm(points[b['fit_id']]-points[a['fit_id']]),objective_difference=b['objective']-a['objective'],residual_difference_mps=dr,validation_difference_mps=dv,position_error_difference_m=de,
                residual_position_disagreement=bool(dr*de<0 and abs(dr)>cfg['residual_roundoff_tie_mps'] and abs(de)>cfg['position_roundoff_tie_m']),validation_position_disagreement=bool(dv*de<0 and abs(dv)>cfg['residual_roundoff_tie_mps'] and abs(de)>cfg['position_roundoff_tie_m']),
                full_converged_both=bool(a['full_converged'] and b['full_converged']),qualification_used_for_filtering=False))
        for m in MODELS:
            subm=[r for r in sub if r['model']==m];Q=np.array([points[r['fit_id']] for r in subm]);ids=[r['fit_id'] for r in subm]
            pp=[r for r in pairrows if r['window']==T and r['model_a']==m and r['model_b']==m]
            groups.append(dict(window=T,model=m,fits=len(subm),position_diameter_m=max(r['position_distance_m'] for r in pp),objective_range=max(r['objective'] for r in subm)-min(r['objective'] for r in subm),residual_range_mps=max(r['full_residual_rmse_mps'] for r in subm)-min(r['full_residual_rmse_mps'] for r in subm),endpoint_error_min_m=min(r['endpoint_error_m'] for r in subm),endpoint_error_max_m=max(r['endpoint_error_m'] for r in subm)))
            for r in pp:clusterrows.append(dict(record_type='PAIR_DISTANCE',model=m,**r))
            Z=linkage(Q,method='complete',metric='euclidean');members={i:[ids[i]] for i in range(len(ids))}
            for i,z in enumerate(Z):
                mm=members[int(z[0])]+members[int(z[1])];members[len(ids)+i]=mm
                clusterrows.append(dict(record_type='COMPLETE_LINKAGE_MERGE',window=T,model=m,merge_index=i,left_node=int(z[0]),right_node=int(z[1]),position_distance_m=z[2],members=';'.join(mm),member_count=int(z[3]),cluster_cutoff='NONE_NO_CORRECT_CLUSTER_SELECTED'))
    comparison=[]
    for r30 in [r for r in rows if r['window']==30]:
        r120=next(r for r in rows if r['window']==120 and r['model']==r30['model'] and r['direction']==r30['direction'] and r['radius_m']==r30['radius_m'])
        i30=next(r for r in inf if r['window']==30 and r['model']==r30['model'] and r['velocity']=='PVT_PRIMARY' and r['support']=='NATURAL_SUPPORT');i120=next(r for r in inf if r['window']==120 and r['model']==r30['model'] and r['velocity']=='PVT_PRIMARY' and r['support']=='NATURAL_SUPPORT')
        comparison.append(dict(model=r30['model'],direction=r30['direction'],radius_m=r30['radius_m'],W30_fit=r30['fit_id'],W120_fit=r120['fit_id'],W30_error_m=r30['endpoint_error_m'],W120_error_m=r120['endpoint_error_m'],error_change_m=r120['endpoint_error_m']-r30['endpoint_error_m'],W30_residual_mps=r30['full_residual_rmse_mps'],W120_residual_mps=r120['full_residual_rmse_mps'],W30_validation_mps=r30['heldout_trimmed_rmse_mps'],W120_validation_mps=r120['heldout_trimmed_rmse_mps'],lambda_min_gain=i120['lambda_min']/i30['lambda_min'],weak_proxy_improvement=i30['weak_position_std_proxy_m']/i120['weak_position_std_proxy_m'],W30_converged=r30['full_converged'],W120_converged=r120['full_converged'],window_identity='DIFFERENT_PRESPECIFIED_ENDPOINTS_NESTED_OBSERVATIONS_NATURAL_SUPPORT_CHANGES',direction_identity='WINDOW_SPECIFIC_GEOMETRY_DIRECTION' if r30['direction']!='LEGACY_D0' else 'SAME_FIXED_ECEF_DIRECTION'))
    save(O/'fit_tables.json',{'initialization_fit_results.csv':rows,'directional_basin_results.csv':[r for r in rows if r['module']=='DIRECTIONAL'],'solution_clusters.csv':clusterrows,'residual_position_ordering.csv':pairrows,'W30_W120_comparison.csv':comparison})
    save(O/'solution_local_matrices.json',models)
    save(O/'POSTFIT_EVALUATION.json',dict(completed_utc=now(),fits=len(rows),solver_calls=dict(calls),solver_calls_total=sum(calls.values()),groups=groups,objective_checks=residual_checks,full_converged=sum(r['full_converged'] for r in rows),train_converged=sum(r['train_converged'] for r in rows),numerical_success=sum(r['full_fit_success'] for r in rows),EGSHS_runs=0,new_threshold_selected=False,paper_updated=False))
    print(json.dumps(clean(dict(groups=groups,legacy_main=[{k:r[k] for k in ['window','model','radius_m','endpoint_error_m','full_residual_rmse_mps','heldout_trimmed_rmse_mps','error_weak_fraction','full_converged']} for r in rows if r['module']=='MAIN']))),flush=True)

if __name__=='__main__':main()
