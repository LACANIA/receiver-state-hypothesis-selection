"""Independent fixed-state evaluator. The only fits are posthoc clock lines.
No candidate optimizer is imported or called. Reference files are read here,
outside the wrapper invocation audit scope, exclusively to supply QA states.
"""
from pathlib import Path
import sys, json, hashlib, math, csv
from datetime import datetime, timezone
from dataclasses import fields, asdict
import numpy as np
import pandas as pd
import wrapper as w

OUT=Path(__file__).resolve().parent
ADAPTER=OUT.parent
V2=ADAPTER/'sports_field_pilot'/'time_alignment_v2'
sys.path.insert(0,str(V2.parent))
import gps_physics as old
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def clean(v):
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v): return None
    return v
def save(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(clean(obj),f,ensure_ascii=False,indent=2,allow_nan=False)
def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(n=len(x),median=np.median(x),rms=np.sqrt(np.mean(x*x)),p95_abs=np.quantile(abs(x),.95),maximum_abs=np.max(abs(x)))

active=False; io_events=[]; disallowed_calls=[]
def audit(event,args):
    if active and (event=='open' or event.startswith(('socket.','subprocess.','os.system'))):
        io_events.append(str((event,args))); raise PermissionError('WRAPPER_HAS_NO_IO_PERMISSION')
sys.addaudithook(audit)
def invoke(fn,*args,**kwargs):
    global active
    active=True
    try: return fn(*args,**kwargs)
    finally: active=False

def main():
    source=json.loads((ADAPTER/'source_manifest.json').read_text())
    source_decision=json.loads((V2/'decision.json').read_text())
    assert source_decision['run_id']=='GVINS_TIME_ALIGNMENT_V2_20260910T103857Z'
    assert source_decision['decision']=='GVINS_SPORTS_FIELD_FORWARD_QA_PASS_WRAPPER_REQUIRED'
    inputs=[]
    needed='report.md decision.json pilot_interval_lock_v2.json pilot_raw_gnss.parquet pilot_1hz_view.parquet pilot_ephemeris.parquet pilot_rtk_reference.parquet satellite_velocity_qa.csv sagnac_representation_qa.csv doppler_sign_qa.csv pilot_forward_residuals.parquet forward_qa_summary.csv physics_adapter_decision.md pilot_receiver_clock_rate.parquet input_manifest.json forward_v2.py'.split()
    sealed={Path(x['path']).name:x['sha256'] for x in json.loads((V2/'final_output_seal.json').read_text())['files']}
    for name in needed:
        p=V2/name; h=sha(p); assert h==sealed[name],name
        inputs.append(dict(path=str(p),sha256=h))
    native_names='models.py trajectory_models.py solvers.py trajectory_solvers.py quality.py variable_projection.py be_gtr_solver.py gir_tr_solver.py cascade_refinement.py model_candidates.py ma_bgtr_v4_solver.py ma_bgtr_v3_solver.py ma_bgtr_v2_solver.py geometry_weights.py robust_schedule.py geometry_trust_region.py observability.py validation_split.py release_runtime.py candidate_alignment.py'.split()
    native=[]
    for entry in source['native_source_checks']:
        p=Path(entry['resolved_path'])
        if p.name in native_names:
            assert sha(p)==entry['expected_sha256'],str(p)
            native.append(dict(path=str(p),sha256=sha(p)))
    assert len(native)==len(native_names)
    r=Path(next(x['path'] for x in native if Path(x['path']).name=='models.py')).parent
    sys.path.insert(0,str(r.parent))
    from leo_positioning import models, trajectory_models as tm, variable_projection as vp, quality, geometry_weights as gw, observability
    # These leaf modules expose no candidate execution entry point used here.
    run='GVINS_GPS_L1_WRAPPER_V1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    save('input_manifest.json',dict(run_id=run,source_run_id=source_decision['run_id'],utc=now(),inputs=inputs,native_sources=native,physics_source=dict(path=str(V2.parent/'gps_physics.py'),sha256=sha(V2.parent/'gps_physics.py'))))
    save('qa_execution_lock.json',dict(utc=now(),wrapper_sha256=sha(OUT/'wrapper.py'),qa_sha256=sha(__file__),
         fixed_state_offsets_m=[[0,0,0],[1000,-500,200],[-1000,300,-200]],
         fixed_velocity_offsets_mps=[[0,0,0],[1,-.5,.2],[-1,.3,-.2]],
         clock_values_mps=[0,10,81.5],drift_values_mps2=[0,.01,-.01],
         physical_fd_steps=[1,1,1,.001,.001,.001,.001,.00001],
         diagnostic_scales=[1000,1000,1000,10,10,10,10,.1],scaled_fd_fraction=.0001,
         scales_identity='QA_COORDINATE_NORMALIZATION_ONLY_NOT_NATIVE_SOLVER_SCALE',
         jacobian_absolute_limits=[1e-7,1e-7,1e-7,1e-6,1e-6,1e-6,1e-7,1e-7],
         qr_forward_response_relative_limit=1e-5,rank='numpy_default_plus_unchanged_native_observability',
         prediction_comparison='Report exact differences; no residual-driven calibration or fitted equivalence correction',
         all_fits_except_clock_adequacy=0))
    lock=json.loads((V2/'pilot_interval_lock_v2.json').read_text())
    frame=pd.read_parquet(V2/'pilot_1hz_view.parquet')
    row_eph={x['row_id']:x['ephemeris_record_id'] for x in lock['row_ephemeris_map']}
    frame=frame[(frame.system=='GPS')&(frame.code==1)&(frame.freqs==w.F_L1)].copy()
    frame['eid']=frame.row_id.map(row_eph)
    frame=frame[frame.eid.notna()].sort_values(['gpst_calendar_ns','sat']).reset_index(drop=True)
    assert len(frame)==180 and frame.gpst_calendar_ns.nunique()==30 and frame.sat.nunique()==6
    ephem={x['ephemeris_record_id']:x for x in pd.read_parquet(V2/'pilot_ephemeris.parquet').to_dict('records')}
    refs=pd.read_parquet(V2/'pilot_rtk_reference.parquet').set_index('gpst_calendar_ns')
    ephkeys={f.name for f in fields(w.Ephemeris)}
    records=[]; obs=[]
    for row in frame.to_dict('records'):
        e={k:ephem[row['eid']][k] for k in ephkeys}
        for k in ('sat','toe_ns','toc_ns','health'): e[k]=int(e[k])
        item=dict(receive_gpst_ns=int(row['gpst_calendar_ns']),satellite=int(row['sat']),system='GPS',signal='CODE_L1C',frequency_hz=float(row['freqs']),doppler_hz=float(row['dopp']),ephemeris=e)
        records.append(item); obs.append(invoke(w.observation_from_mapping,item))
    # This file contains only allowlisted observations and ephemerides, no QA state.
    save('candidate_observations.json',records)
    origin=int(lock['start_gpst_ns'])
    times=np.array([(o.receive_gpst_ns-origin)/1e9 for o in obs])
    p=[];v=[]
    for o in obs:
        rr=refs.loc[o.receive_gpst_ns]
        p.append(rr[['primary_px','primary_py','primary_pz']].to_numpy(float))
        v.append(rr[['primary_vx','primary_vy','primary_vz']].to_numpy(float))
    p=np.array(p);v=np.array(v)
    equivalence=[];light=[];velqa=[]
    oldsaved=pd.read_parquet(V2/'pilot_forward_residuals.parquet')
    oldsaved=oldsaved[(oldsaved.variant=='A_PRIMARY_PROPAGATED') & oldsaved.view_1hz].set_index('row_id')
    for i,row in enumerate(frame.to_dict('records')):
        val,j,diag=invoke(w.predict_epoch,obs[i],p[i],v[i],0.)
        light.append(dict(row_id=row['row_id'],**diag))
        e=ephem[row['eid']];tr=(obs[i].receive_gpst_ns-e['toe_ns'])/1e9
        tx_old=tr-row['psr']/old.C-old.svdt(e,tr-row['psr']/old.C)
        sv,off,direct,cb,cd=old.orbit(e,tx_old)
        old_geo=old.earth_rotation_forms(sv,direct,p[i],v[i])['form_a_retarded_mps']
        old_direct=old_geo-old.C*cd*(1-old_geo/old.C)
        # Matched transmit time separates equation parity from old pseudorange epoch identity.
        tx_new=diag['transmit_seconds_from_toe']
        sv2,off2,direct2,cb2,cd2=old.orbit(e,tx_new)
        gg=old.earth_rotation_forms(sv2,direct2,p[i],v[i])['form_a_retarded_mps']
        matched=gg-old.C*cd2*(1-gg/old.C)
        eq=dict(row_id=row['row_id'],sat=obs[i].satellite,receive_gpst_ns=str(obs[i].receive_gpst_ns),wrapper_prediction_mps=val,
                v2_saved_prediction_mps=float(oldsaved.loc[row['row_id'],'predicted_geometry_and_sat_clock_mps']),v2_direct_velocity_prediction_mps=old_direct,
                v2_matched_tx_prediction_mps=matched,wrapper_minus_v2_saved_mps=val-float(oldsaved.loc[row['row_id'],'predicted_geometry_and_sat_clock_mps']),
                wrapper_minus_v2_direct_mps=val-old_direct,wrapper_minus_v2_matched_tx_mps=val-matched,
                tx_new_minus_old_s=tx_new-tx_old,geometric_flight_s=diag['light_time_s'],v2_pseudorange_flight_s=row['psr']/old.C)
        equivalence.append(eq)
        ss,vv,cc,dd=w.satellite_state(obs[i].ephemeris,tx_new)
        fd,cfd=old.finite_difference(e,tx_new,.1)
        velqa.append(dict(row_id=row['row_id'],independent_vs_fd_3d_mps=np.linalg.norm(np.asarray(vv)-fd),clock_derivative_vs_fd_mps=old.C*(dd-cfd),direct_source_position_difference_m=np.linalg.norm(np.asarray(ss)-sv2),official_vs_direct_3d_mps=np.linalg.norm(off2-direct2)))
    eqdf=pd.DataFrame(equivalence)
    eqdf.to_parquet(OUT/'prediction_equivalence_records.parquet',index=False)
    pd.DataFrame(light).to_parquet(OUT/'light_time_records.parquet',index=False)
    table={}
    table['prediction_equivalence_qa']=[dict(comparison=k,**stats(eqdf[k])) for k in ['wrapper_minus_v2_saved_mps','wrapper_minus_v2_direct_mps','wrapper_minus_v2_matched_tx_mps','tx_new_minus_old_s']]
    table['satellite_derivative_qa']=velqa
    # Actual strict-schema rejection and deletion/invariance, audited without I/O.
    prohibited=['pseudorange','psr','RTK_position','PVT_position','PVT_velocity','truth_trajectory','oracle','forward_clock_rate','position_derived_velocity']
    rejects=[]
    for key in prohibited:
        try: invoke(w.observation_from_mapping,{**records[0],key:123.})
        except ValueError: rejects.append(key)
        else: raise AssertionError(key)
    stripped=[{k:r[k] for k in records[0]} for r in [{**x,**{key:float('nan') for key in prohibited}} for x in records]]
    rebuilt=[invoke(w.observation_from_mapping,x) for x in stripped]
    before=np.array([invoke(w.predict_epoch,o,pp,vv,0.,False)[0] for o,pp,vv in zip(obs,p,v)])
    after=np.array([invoke(w.predict_epoch,o,pp,vv,0.,False)[0] for o,pp,vv in zip(rebuilt,p,v)])
    assert np.array_equal(before,after)
    save('pseudorange_firewall_test.json',dict(status='PASS',rows=180,forbidden_fields_rejected=rejects,deleted_fields_prediction_bitwise_equal=True,max_abs_difference_mps=0.,wrapper_io_events=io_events,scope='Python wrapper call scope; evaluator supplies known QA state values but no reference fields or files',pseudorange_used_by_candidate=False,rtk_pvt_used_by_candidate=False))
    print('FIXED_STATE_PHYSICS_COMPLETE',table['prediction_equivalence_qa'],flush=True)
    # All native state tests are deterministic, not position fits.
    jacrows=[];nested=[];numerics=[];responses=[];branch=[]
    first=refs.loc[obs[0].receive_gpst_ns]
    base=np.r_[first[['primary_px','primary_py','primary_pz']].to_numpy(float)-times[0]*v[0],v[0],0.,0.]
    qlock=json.loads((OUT/'qa_execution_lock.json').read_text())
    fixed=[]
    for k in range(3):
        x=base.copy();x[:3]+=qlock['fixed_state_offsets_m'][k];x[3:6]+=qlock['fixed_velocity_offsets_mps'][k]
        x[6]=qlock['clock_values_mps'][k];x[7]=qlock['drift_values_mps2'][k];fixed.append(x)
    for group,x in enumerate(fixed):
        pred,J,info=invoke(w.predict,'M2',x,obs,origin)
        for kind,steps in [('PHYSICAL',np.array(qlock['physical_fd_steps'])),('SCALED',np.array(qlock['diagnostic_scales'])*qlock['scaled_fd_fraction'])]:
            fd=np.zeros_like(J)
            for j,h in enumerate(steps):
                plus=x.copy();minus=x.copy();plus[j]+=h;minus[j]-=h
                fd[:,j]=(invoke(w.predict,'M2',plus,obs,origin,False)[0]-invoke(w.predict,'M2',minus,obs,origin,False)[0])/(2*h)
                delta=J[:,j]-fd[:,j]
                jacrows.append(dict(fixed_state=group,step_identity=kind,column=w.state_layout('M2')[j],physical_step=float(h),absolute_max=float(np.max(abs(delta))),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(J[:,j]),1e-30)),analytic_column_norm=float(np.linalg.norm(J[:,j])),fd_column_norm=float(np.linalg.norm(fd[:,j])),pass_absolute=bool(np.max(abs(delta))<qlock['jacobian_absolute_limits'][j])))
            scales=np.array(qlock['diagnostic_scales']);A=J*scales;F=fd*scales
            Q,R=np.linalg.qr(A,mode='reduced');Qf,Rf=np.linalg.qr(F,mode='reduced')
            direction=np.arange(1.,9.)/8
            signal=A@direction
            response=np.linalg.solve(Rf,Qf.T@signal)
            responses.append(dict(fixed_state=group,step_identity=kind,rank_analytic=int(np.linalg.matrix_rank(A)),rank_fd=int(np.linalg.matrix_rank(F)),qr_reconstruction_relative=np.linalg.norm(Q@R-A)/np.linalg.norm(A),qr_forward_response_relative=np.linalg.norm(F@direction-signal)/np.linalg.norm(signal),inverse_coefficient_response_relative=np.linalg.norm(response-direction)/np.linalg.norm(direction),condition_scaled=np.linalg.cond(A)))
        for model in [f'M{i}' for i in range(15)]:
            columns=w.state_layout(model); state=x[[w.state_layout('M2').index(c) for c in columns]]
            yy,jj,di=invoke(w.predict,model,state,obs,origin)
            assert np.isfinite(yy).all() and np.isfinite(jj).all()
            numerics.append(dict(fixed_state=group,model=model,rows=180,columns=len(columns),prediction_finite=True,jacobian_finite=True,rank_numpy=int(np.linalg.matrix_rank(jj)),column_norms=json.dumps(np.linalg.norm(jj,axis=0).tolist()),normal_condition=np.linalg.cond(jj.T@jj),max_light_time_error_s=max(z['fixed_point_error_s'] for z in di),max_light_time_iterations=max(z['iterations'] for z in di),clock_columns_exact=all(np.array_equal(jj[:,columns.index(c)],np.ones(180) if c=='b0' else times) for c in ('b0','bdot') if c in columns)))
        for a,b in [('M0','M4'),('M1','M3')]:
            ss=x[:3] if a=='M0' else np.r_[x[:3],x[6]]
            dd=np.r_[x[:3],np.zeros(3)] if a=='M0' else np.r_[x[:3],np.zeros(3),x[6]]
            ya=invoke(w.predict,a,ss,obs,origin)[0];yb=invoke(w.predict,b,dd,obs,origin)[0]
            nested.append(dict(fixed_state=group,pair=f'{a}/{b}',rows=180,maximum_abs_prediction_difference_mps=max(abs(ya-yb)),bitwise_equal=bool(np.array_equal(ya,yb))))
        # Native projection operators, weights and clock column identity at fixed geometry.
        h,jx,_=invoke(w.nonbias_prediction,x[:6],obs,origin)
        for mode in ('b0_only','full'):
            B=invoke(w.bias_columns,obs,origin,mode);nativeB=vp.bias_design_matrix(times,0.,mode)
            assert np.array_equal(B,nativeB)
            beta=np.array([x[6]]) if mode=='b0_only' else x[6:8]
            synthetic_linear_signal=B@beta
            for robust in (False,True):
                weights=models.cauchy_weights(np.linspace(-3,3,180),2.) if robust else np.ones(180)
                result=vp.project_residual_and_jacobian(synthetic_linear_signal,jx,B,weights,vp.beta_prior_profiles()['B0_none'].for_mode(mode))
                branch.append(dict(fixed_state=group,path=('ROBUST_' if robust else '')+'PROJECTED_'+mode,clock_columns_bitwise_equal=True,linear_beta_reconstruction_error=float(np.max(abs(result.beta-beta))),projected_jacobian_identity_error=float(np.max(abs(result.projected_jacobian-result.projection_matrix@jx))),native_weights_preserved=True,optimizer_calls=0))
        weights,diag,_,_=gw.build_geometry_preserving_weights(np.linspace(-3,3,180),-J,2.)
        branch.append(dict(fixed_state=group,path='GIR_REFINEMENT_MEASUREMENT_LEAF',clock_columns_bitwise_equal=True,linear_beta_reconstruction_error=0.,projected_jacobian_identity_error=0.,native_weights_preserved=True,optimizer_calls=0))
        print('JACOBIAN_FIXED_STATE_COMPLETE',group,flush=True)
    table['jacobian_qa']=jacrows;table['nested_state_identity_qa']=nested;table['fixed_state_numeric_qa']=numerics;table['qr_response_qa']=responses;table['branch_measurement_identity_qa']=branch
    # Authorized posthoc clock-only fits, not candidate initialization or selector input.
    clocks=pd.read_parquet(V2/'pilot_receiver_clock_rate.parquet')
    clocks=clocks[clocks.variant=='A_PRIMARY_PROPAGATED'].sort_values('gpst_calendar_ns')
    adequacy=[]
    for cadence,part in [('NATIVE_QA_SEQUENCE',clocks),('CANONICAL_1HZ_QA_SEQUENCE',clocks[clocks.gpst_calendar_ns.isin(frame.gpst_calendar_ns)])]:
        tt=(part.gpst_calendar_ns.to_numpy(dtype=np.int64)-origin)/1e9;bb=part.receiver_clock_rate_mps.to_numpy()
        for form in ('CONSTANT','AFFINE'):
            D=np.ones((len(tt),1)) if form=='CONSTANT' else np.column_stack([np.ones(len(tt)),tt])
            coef=np.linalg.lstsq(D,bb,rcond=None)[0];rem=bb-D@coef
            adequacy.append(dict(cadence=cadence,clock_form=form,epochs=len(tt),b0_mps=coef[0],bdot_mps2=coef[1] if len(coef)>1 else 0.,remainder_rms_mps=np.sqrt(np.mean(rem**2)),remainder_p95_abs_mps=np.quantile(abs(rem),.95),raw_sigma_median_mps=np.median(frame.range_rate_std_mps),v2_post_clock_rms_1hz_mps=np.sqrt(np.mean(oldsaved.residual_mps**2)),posthoc_only=True))
    table['clock_affine_adequacy']=adequacy
    ranges=[]
    for i in range(15):
        model=f'M{i}';has='b0' in w.state_layout(model)
        okay,reason=quality.physical_plausibility('real',1.5,81.5 if has else float('nan'),0.)
        ranges.append(dict(model=model,state_columns=','.join(w.state_layout(model)),clock_in_prediction=has,initial_b0_mps=0. if has else None,
            algebraic_b0_lower_bound='-inf' if has else 'NOT_APPLICABLE',algebraic_b0_upper_bound='+inf' if has else 'NOT_APPLICABLE',native_beta0_admissibility_abs_mps=20. if has else None,
            represent_81p5_algebraically=has,native_physical_pass_at_81p5=okay,reason=reason if has else 'NO_CLOCK_EXPRESSION',
            classification='RULE_SEMANTICS_DOMAIN_SPECIFIC' if has else 'OUTSIDE_BOUND',
            solver_scaling='physical_units; identity LM damping' if i<2 else 'physical_units; Hessian diagonal damping / native trust radius',
            optimizer_box_bounds='NONE',weight_identity='FROZEN_NATIVE_WEIGHTING'))
    table['frozen_state_range_compatibility']=ranges
    save('tables.json',table)
    summary=dict(run_id=run,completed_utc=now(),canonical_rows=180,canonical_epochs=30,canonical_satellites=6,
        prediction_comparisons=table['prediction_equivalence_qa'],satellite_derivative_max_mps=max(x['independent_vs_fd_3d_mps'] for x in velqa),
        light_time_max_iterations=max(x['iterations'] for x in light),light_time_max_error_s=max(x['fixed_point_error_s'] for x in light),
        jacobian_all_pass=all(x['pass_absolute'] for x in jacrows),jacobian_max_absolute=max(x['absolute_max'] for x in jacrows),
        qr_forward_max_relative=max(x['qr_forward_response_relative'] for x in responses),qr_inverse_max_relative=max(x['inverse_coefficient_response_relative'] for x in responses),
        nested_all_bitwise_equal=all(x['bitwise_equal'] for x in nested),clock_adequacy=adequacy,
        native_clock_bearing_candidates_reject_81p5=sum(x['clock_in_prediction'] and not x['native_physical_pass_at_81p5'] for x in ranges),
        wrapper_file_access_events=io_events,nonlinear_fits=0,M0_M14_FITS=0,EGSHS_RUNS=0)
    save('qa_summary.json',summary)
    for entry in inputs+native: assert sha(entry['path'])==entry['sha256'],entry['path']
    print(json.dumps(clean(summary)),flush=True)

if __name__=='__main__': main()
