from pathlib import Path
from datetime import datetime, timezone
import sys, json, pickle, time, hashlib
from collections import Counter
import numpy as np
import pandas as pd
from bridge import Bridge, O, A, W, read, save, sha, clean

MODE=sys.argv[1]
b=Bridge(); v4=b.mods['ma_bgtr_v4_solver']; p0=np.array(b.prior['initial_ecef_m']);v0=np.zeros(3)
train,val,split=b.mods['validation_split'].blocked_train_validation(b.obs,train_fraction=.70)

def patch_alias(old,new):
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__name__','')
        if name.startswith('leo_positioning') or name=='release_runtime':
            for key,value in list(vars(mod).items()):
                if value is old: setattr(mod,key,new)

if MODE=='qa':
    tests=[]
    for sub_name,sub in [('full',b.obs),('train',train),('heldout',val)]:
        idx=sub['row_index']; ob=[b.observations[i] for i in idx]
        for model in ['M0','M1','M2','M3','M4']:
            x=np.r_[p0,([] if model in ['M0','M1'] else [0.,0.,0.]),
                    ([7.] if model in ['M1','M2','M3'] else []),([.01] if model=='M2' else [])]
            expected=b.w.predict(model,x,ob,b.origin)
            if model in ['M0','M1']:
                out=b.static_res(x,sub['sat_pos_m'],sub['sat_vel_mps'],sub['meas_mps'],model=='M1')
            else:
                config={'M2':b.mods['trajectory_models'].FULL_CTD_CONFIG,
                        'M3':b.mods['trajectory_models'].NO_BDOT_CONFIG,
                        'M4':b.mods['trajectory_models'].NO_BIAS_CONFIG}[model]
                out=b.dynamic_res(x,sub['time_s'],sub['sat_pos_m'],sub['sat_vel_mps'],sub['meas_mps'],0.,config)
            dp=float(np.max(abs(out[2]-expected[0])));dj=float(np.max(abs(out[1]+expected[1])))
            assert dp<1e-12 and dj<1e-12,(model,sub_name,dp,dj)
            tests.append(dict(model=model,subset=sub_name,rows=len(idx),prediction_max_difference=dp,jacobian_max_difference=dj))
    gate_tests=[]
    for speed in [0.,21.]:
        for beta in [0.,20.,20.01,30.,31.,82.]:
            for drift in [0.,.02,.1,.101]:
                b.bypass=False; strict=b.physical('real',speed,beta,drift)
                b.bypass=True; bypass=b.physical('real',speed,beta,drift)
                expected=[s for s in strict[1].split(';') if s and s!='beta0_limit']
                assert bypass==(not expected,';'.join(expected))
                gate_tests.append(dict(speed=speed,b0=beta,bdot=drift,strict=strict,bypass=bypass))
    b.bypass=False
    # Actual measurement row order and signed residual identity, without truth evaluation.
    save(O/'integration_qa.json',dict(status='PASS',tests=tests,clock_gate_tests=gate_tests,
         split=split,train_rows=train['row_index'],heldout_rows=val['row_index'],alias_patches=b.patches,
         light_time_max_error_s=b.max_light_error,light_time_max_iterations=b.max_light_iterations,
         native_sources_unchanged=all(sha(x['resolved_path'])==x['expected_sha256'] for x in b.sources),
         existing_wrapper_source_unchanged=True,candidate_fits=0,selector_calls=0,
         satellite_arrays_identity='Receive-time broadcast metadata used only for exact row lookup in GNSS leaves; physics recomputed at candidate state.',
         ephemeris_covariance_identity='Native zero_or_not_configured; no broadcast uncertainty calibration is asserted.'))
    print('INTEGRATION_QA=PASS',flush=True);sys.exit(0)

assert read(O/'integration_qa.json')['status']=='PASS'
for item in read(O/'input_manifest.json')['bindings']:
    # Do not read reference bytes in candidate process, even to rehash them.
    if Path(item['path']).name!='pilot_rtk_reference.parquet': assert sha(item['path'])==item['sha256']
assert sha(O/'initialization_prior.json')==read(O/'input_manifest.json')['prior_sha256']

denied=[]
native_paths={str(Path(x['resolved_path']).resolve()).casefold() for x in b.sources}
allowed_inputs={str((W/n).resolve()).casefold() for n in ['wrapper.py','candidate_observations.json','__pycache__']}
forbidden_names={'initialization_provenance_qa_only.json','evaluate.py'}
def audit(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes)): return
    p=Path(args[0]).resolve(); s=str(p).casefold()
    try: in_lab=p.is_relative_to(A.parent)
    except ValueError: in_lab=False
    if not in_lab: return # Python/dependency resources outside the project.
    if p.is_relative_to(O) and p.name not in forbidden_names: return
    if s in native_paths or s in allowed_inputs: return
    denied.append(dict(event=event,path=str(p)))
    raise PermissionError('CANDIDATE_FILE_FIREWALL: '+str(p))
sys.addaudithook(audit)

if MODE in ['strict','bypass']:
    if not (O/'RUN_SEAL.json').exists():
        assert MODE=='strict'
        save(O/'RUN_SEAL.json',dict(status='LOCKED_BEFORE_FIRST_CANDIDATE_FIT',
             utc=datetime.now(timezone.utc).isoformat(),protocol=read(O/'protocol.json'),
             files=[dict(path=str(O/n),sha256=sha(O/n)) for n in
                    ['protocol.json','initialization_prior.json','input_manifest.json','integration_qa.json','bridge.py','run.py']],
             native_sources=b.sources,reference_exception='Authorized fixed common perturbed initialization only',
             candidate_pool=v4.V4_CANDIDATE_MODEL_LIST,planned_top_level_runs=20))
    fitdir=O/'raw'/MODE;fitdir.mkdir(parents=True,exist_ok=True)
    b.bypass=MODE=='bypass'
    models=v4.V4_CANDIDATE_MODEL_LIST if MODE=='strict' else v4.V4_CANDIDATE_MODEL_LIST[10:]
    calls=[];counts=Counter()
    for mod_name,fn_name in [('solvers','solve_lm'),('trajectory_solvers','solve_ctd_lm'),
                              ('be_gtr_solver','solve_be_gtr'),('gir_tr_solver','solve_gir_tr')]:
        original=getattr(b.mods[mod_name],fn_name)
        def counted(*args,_fn=original,_name=fn_name,**kwargs):
            counts[_name]+=1
            result=_fn(*args,**kwargs)
            calls.append(dict(function=_name,result=result))
            return result
        patch_alias(original,counted)
    for model in models:
        short=model.split('_')[0]; target=fitdir/(short+'.pkl')
        if target.exists():
            assert (fitdir/(short+'.seal.json')).exists();assert sha(target)==read(fitdir/(short+'.seal.json'))['sha256']
            print('REUSE '+MODE+' '+short,flush=True);continue
        print('START '+MODE+' '+model,flush=True)
        start=time.monotonic();calls.clear();counts.clear();b.full_capture.clear();b.shadow.clear()
        if short in ['M12','M13','M14']:
            full,diag=v4._cascade_full_row(model,b.obs,p0,v0,'real','sports_field_gps_l1',
                'east_10km','zero_velocity','B0_none')
            row,diag=v4._validation_for_cascade(model,full,diag,train,val,p0,v0,'real')
        else:
            full=b.mods['model_candidates'].run_candidate_model(model,'real','sports_field_gps_l1',b.obs,
                'east_10km',p0,'zero_velocity',v0,'B0_none')
            row=v4._validation_for_standard(model,full,train,val,p0,v0,'B0_none','real');diag={}
        assert not denied,denied
        errors=str(row.get('failure_reason',''))
        if 'exception' in errors.lower():
            save(fitdir/(short+'.execution_error.json'),dict(row=row,counts=counts))
            raise RuntimeError('IMPLEMENTATION_OR_NUMERICAL_EXCEPTION_REQUIRES_REVIEW: '+errors)
        payload=dict(model=model,mode=MODE,row=row,diag=diag,full_capture=b.full_capture.copy(),
             solver_calls=calls.copy(),solver_counts=dict(counts),shadow=dict(b.shadow),
             elapsed_s=time.monotonic()-start,source_run=read(O/'protocol.json')['run_id'])
        with target.open('xb') as f: pickle.dump(payload,f)
        save(fitdir/(short+'.seal.json'),dict(sha256=sha(target),model=model,mode=MODE,
              utc=datetime.now(timezone.utc).isoformat(),solver_counts=counts))
        print('DONE '+MODE+' '+short+' '+json.dumps(clean(dict(numerical_success=row.get('numerical_success'),
          residual=row.get('full_residual_rmse_mps'),b0=row.get('beta0_estimated_mps'),
          bdot=row.get('beta_dot_estimated_mps2'),seconds=payload['elapsed_s'],solver_counts=counts))),flush=True)
    save(fitdir/'execution_summary.json',dict(completed_models=models,denied=denied,
       calls=b.calls,light_time_max_error_s=b.max_light_error,light_time_max_iterations=b.max_light_iterations))
    sys.exit(0)

if MODE=='select':
    rows={};payloads={};events=[]
    for mode in ['strict','bypass']:
        use=[];stored=[]
        for model in v4.V4_CANDIDATE_MODEL_LIST:
            short=model.split('_')[0]; fitmode='bypass' if mode=='bypass' and int(short[1:])>=10 else 'strict'
            path=O/'raw'/fitmode/(short+'.pkl')
            assert sha(path)==read(path.with_suffix('.seal.json'))['sha256']
            with path.open('rb') as f: payload=pickle.load(f)
            b.bypass=mode=='bypass';row=b.requalify(payload['row']) if b.bypass else dict(payload['row'])
            row['fit_source']=fitmode;row['CLOCK_GATE_SHADOW_FAIL']=bool(abs(row.get('beta0_estimated_mps',np.nan))>20.)
            use.append(row);stored.append(payload)
        use,robust=b.mods['robust_gates'].apply_robust_hard_gate(use)
        frame=b.rt._enrich_risk(pd.DataFrame(use),b.obs)
        frame=b.rt._add_observable_outlier_evidence(frame,dict(dataset_type='real'))
        rows[mode]=frame;payloads[mode]=stored
        save(O/(mode+'_preselection.json'),frame.to_dict('records'))
    def tracewrap(fn,name):
        def call(*args,**kwargs):
            ans=fn(*args,**kwargs)
            e=dict(operation=name)
            if args and isinstance(args[0],pd.DataFrame): e['input_models']=args[0]['candidate_model'].tolist()
            if isinstance(ans,pd.Series): e['selected_model']=ans.get('candidate_model')
            elif isinstance(ans,pd.DataFrame): e['remaining_models']=ans['candidate_model'].tolist()
            elif isinstance(ans,tuple):
                e['outputs']=[]
                for x in ans:
                    if isinstance(x,pd.DataFrame): e['outputs'].append(dict(models=x['candidate_model'].tolist()))
                    elif isinstance(x,pd.Series): e['outputs'].append(dict(model=x.get('candidate_model')))
                    elif isinstance(x,(dict,str,bool,int,float,np.generic)): e['outputs'].append(x)
            events.append(clean(e));return ans
        return call
    for module,fn in [('hierarchical_gates','_base_filter'),('hierarchical_gates','_dynamic_gate'),
                      ('family_evidence','choose_near_tie_by_risk'),('family_evidence','best_candidate'),
                      ('ma_bgtr_v71_solver','evaluate_bias_gate_v71'),('ma_bgtr_v71_solver','_refine_gate'),
                      ('ma_bgtr_v71_solver','evaluate_robust_gate_v71')]:
        old=getattr(b.mods[module],fn);patch_alias(old,tracewrap(old,fn))
    selections={}
    for mode,frame in rows.items():
        b.bypass=mode=='bypass';events.clear()
        out=b.mods['ma_bgtr_v71_solver'].select_group_v71(b.rt.selector_input_frame(frame))
        selected,family,bias,robust,reason,low=out
        selections[mode]=dict(model=selected['candidate_model'],reason=reason,low_quality=low,
                             family=family,bias=bias,robust=robust,events=events.copy())
        save(O/(mode+'_selector_trace.json'),selections[mode])
        # Preserve the original v7.2 shell and verify its selection pointer.
        m,r,l=b.rt.select_actual_model(frame);assert m==selected['candidate_model'] and l==low
    with (O/'pretruth_results.pkl').open('xb') as f: pickle.dump(dict(rows=rows,payloads=payloads,selections=selections),f)
    assert not denied
    save(O/'PRE_TRUTH_SEAL.json',dict(utc=datetime.now(timezone.utc).isoformat(),
         identity='INITIALIZATION_ASSISTED_REAL_RF_DIAGNOSTIC',
         exception='RTK-derived common initialization constructed in separate authorized prepare process before fits',
         files=[dict(path=str(p),sha256=sha(p)) for p in [O/'pretruth_results.pkl',O/'strict_selector_trace.json',O/'bypass_selector_trace.json']],
         native_unchanged=all(sha(x['resolved_path'])==x['expected_sha256'] for x in b.sources),
         candidate_reference_reads=0,denied=denied,clock_threshold_selected=False))
    save(O/'POSITION_OUTCOME_FIREWALL_AUDIT.json',dict(candidate_process_reference_reads=0,
         reference_used_to_construct_initialization=True,identity='INITIALIZATION_ASSISTED_REAL_RF_DIAGNOSTIC',
         no_pseudorange_candidate_input=True,no_pvt_velocity_candidate_input=True,
         no_forward_qa_clock_input=True,denied=denied,posthoc_evaluation_not_yet_started=True))
    print('PRE_TRUTH_SEAL_CREATED',flush=True)
