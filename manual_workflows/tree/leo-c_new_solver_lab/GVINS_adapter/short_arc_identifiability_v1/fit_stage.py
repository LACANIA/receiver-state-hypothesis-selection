from common import *
from native_bridge import NativeBridge
import pickle,time
from dataclasses import asdict

def main():
    mode=sys.argv[1];T=int(sys.argv[2]);assert T in [30,120]
    for f in read(O/'INFORMATION_SEAL.json')['files']:
        # Fitting process does not read RTK reference bytes, even for hashing.
        if Path(f['path']).name!='reference_qa_only.json':assert sha(f['path'])==f['sha256']
    b=NativeBridge(T)
    if mode=='qa':
        save(O/f'integration_qa_W{T}.json',b.qa());print(f'QA W{T} PASS',flush=True);return
    assert mode=='fit'
    seal=read(O/'FIT_EXECUTION_SEAL.json')
    for f in seal['files']:assert sha(f['path'])==f['sha256']
    train,val,split=b.mods['validation_split'].blocked_train_validation(b.obs,train_fraction=.7)
    schedule=[s for s in read(O/'fit_schedule.json') if s['window']==T]
    allowed={Path(x['resolved_path']).resolve() for x in b.sources}|{(W/'wrapper.py').resolve()}
    allowed |= {O/n for n in ['candidate_observations.json','fit_schedule.json','RUN_PROTOCOL.json','FIT_EXECUTION_SEAL.json','common.py','native_bridge.py','fit_stage.py']}
    denied=[]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0]).resolve()
        if not p.is_relative_to(A.parent):return
        if p in allowed or p.is_relative_to(O/'raw'):return
        denied.append(str(p));raise PermissionError('FIT_REFERENCE_FIREWALL '+str(p))
    sys.addaudithook(audit)
    for s in schedule:
        target=O/'raw'/f"{s['fit_id']}.pkl";stamp=target.with_suffix('.seal.json')
        if target.exists():
            assert sha(target)==read(stamp)['sha256'];print('REUSE '+s['fit_id'],flush=True);continue
        print('START '+s['fit_id']+' '+s['model']+' W'+str(T)+' '+s['direction']+' R'+str(s['radius_m']),flush=True)
        target.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic();b.calls.clear();b.counter.clear()
        m=next(m for m in b.mods['model_candidates'].CANDIDATE_MODEL_LIST if m.startswith(s['model']+'_'))
        p=np.array(s['initial_ecef_m']);v=np.zeros(3)
        full=b.mods['model_candidates'].run_candidate_model(m,'real','sports_field_gps_l1',b.obs,s['fit_id'],p,'zero_velocity',v,'B0_none')
        row=b.mods['ma_bgtr_v4_solver']._validation_for_standard(m,full,train,val,p,v,'B0_none','real')
        err=str(row.get('failure_reason',''))
        payload=dict(spec=s,full_row=full,row=row,solver_calls=b.calls.copy(),solver_counts=dict(b.counter),elapsed_s=time.monotonic()-start,denied=denied.copy())
        if 'exception' in err.lower() or denied:
            save(O/'raw'/f"{s['fit_id']}.execution_error.json",dict(spec=s,row=row,solver_counts=b.counter,denied=denied))
            raise RuntimeError(err)
        assert len(b.calls)==2,(s,len(b.calls))
        result=b.calls[0]['result'];trainres=b.calls[1]['result']
        payload['full_result']=asdict(result);payload['train_result']=asdict(trainres)
        ys,J,infos=w.predict(s['model'],result.state,b.observations,ORIGIN)
        yt,Jt,_=w.predict(s['model'],trainres.state,[b.observations[i] for i in train['row_index']],ORIGIN)
        yv,Jv,_=w.predict(s['model'],trainres.state,[b.observations[i] for i in val['row_index']],ORIGIN)
        payload.update(full_prediction=ys,full_residual=b.obs['meas_mps']-ys,full_J=J,train_residual=train['meas_mps']-yt,heldout_residual=val['meas_mps']-yv,train_J=Jt,heldout_J=Jv,split=split,light_time=infos)
        with target.open('xb') as f:pickle.dump(payload,f)
        save(stamp,dict(utc=now(),sha256=sha(target),fit_id=s['fit_id'],solver_counts=b.counter))
        print('DONE '+s['fit_id']+' '+json.dumps(clean(dict(converged=result.converged,success=result.success,iterations=result.iterations,full_rmse=row['full_residual_rmse_mps'],validation=row['trimmed_validation_rmse_mps'],elapsed_s=payload['elapsed_s']))),flush=True)
    save(O/'raw'/f'W{T}_completion.json',dict(utc=now(),configurations=len(schedule),denied=denied,light_time_max_error_s=b.light_error,light_time_max_iterations=b.light_iterations,EGSHS_runs=0))

if __name__=='__main__':main()
