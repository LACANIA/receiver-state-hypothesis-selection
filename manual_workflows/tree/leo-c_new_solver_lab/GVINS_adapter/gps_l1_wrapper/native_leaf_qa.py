"""Fixed-state native leaf injection test only; never an optimizer adapter.

Bindings are temporary in-memory function slots and restored in finally.
This checks imported aliases in static, dynamic, projected and refinement
consumers without changing scientific source files or executing solvers.
"""
from pathlib import Path
import importlib, json, sys, hashlib
import numpy as np
import wrapper as w

OUT=Path(__file__).resolve().parent
manifest=json.loads((OUT/'input_manifest.json').read_text())
r=Path(next(x['path'] for x in manifest['native_sources'] if Path(x['path']).name=='models.py')).parent
sys.path.insert(0,str(r.parent))
names=['models','trajectory_models','solvers','trajectory_solvers','be_gtr_solver','gir_tr_solver','cascade_refinement','variable_projection']
modules={n:importlib.import_module('leo_positioning.'+n) for n in names}
records=json.loads((OUT/'candidate_observations.json').read_text())
observations=[w.observation_from_mapping(x) for x in records]
origin=1606691928102000000
time=np.array([(o.receive_gpst_ns-origin)/1e9 for o in observations])
# A deterministic Earth-scale test state, not a fitted or receiver reference state.
x=np.array([-2400000.,5400000.,2400000.,1.,-.5,.2,10.,.01])
y=np.array([o.y_mps for o in observations])
sp=np.zeros((180,3));sv=np.zeros((180,3))
def check(a,b):
    if a is not sp or b is not sv: raise ValueError('TEST_BATCH_IDENTITY_REQUIRED')
def static_res(state,a,b,meas,with_bias,receiver_vel_mps=None):
    check(a,b)
    if receiver_vel_mps is not None and np.any(receiver_vel_mps): raise ValueError('STATIC_NONZERO_VELOCITY')
    pred,jac,_=w.predict('M1' if with_bias else 'M0',state,observations,origin)
    return np.asarray(meas)-pred,-jac,pred
def ctd_res(theta,t,a,b,meas,t0,config=modules['trajectory_models'].FULL_CTD_CONFIG):
    check(a,b)
    if not np.array_equal(t,time) or t0!=0.: raise ValueError('TEST_TIME_IDENTITY')
    model='M2' if config.estimate_bdot else ('M3' if config.estimate_b0 else 'M4')
    pred,jac,_=w.predict(model,theta,observations,origin)
    # Native fixed clock configuration remains separate from estimated columns.
    pred=pred+(0. if config.estimate_b0 else config.fixed_b0_mps)+(0. if config.estimate_bdot else config.fixed_bdot_mps2*time)
    return np.asarray(meas)-pred,-jac,pred
def nonbias(state,t,a,b,t0):
    check(a,b)
    if not np.array_equal(t,time) or t0!=0.: raise ValueError('TEST_TIME_IDENTITY')
    h,j,_=w.nonbias_prediction(state,observations,origin)
    return h,j
originals={modules['models'].residuals_and_jacobian:static_res,
           modules['trajectory_models'].residuals_and_jacobian_ctd:ctd_res,
           modules['be_gtr_solver'].h_and_jacobian_x:nonbias}
patches=[];blocked=[]
def profile(frame,event,arg):
    if event=='call':
        file=frame.f_code.co_filename; fn=frame.f_code.co_name
        if 'leo_positioning' in file and (fn.startswith(('solve_','run_candidate','fit_candidate','select_'))):
            if fn!='solve_beta':
                blocked.append(fn); raise RuntimeError('OPTIMIZER_OR_SELECTOR_CALL_FORBIDDEN')
try:
    for name,module in modules.items():
        for attr,value in list(vars(module).items()):
            for orig,replacement in originals.items():
                if value is orig:
                    patches.append((module,attr,value));setattr(module,attr,replacement)
    sys.setprofile(profile)
    result=[]
    target=w.predict('M2',x,observations,origin)
    for name in ['trajectory_models','trajectory_solvers','gir_tr_solver','cascade_refinement']:
        rr,jj,pp=modules[name].residuals_and_jacobian_ctd(x,time,sp,sv,y,0.)
        result.append(dict(path=name+'.residuals_and_jacobian_ctd',max_prediction_difference_mps=float(np.max(abs(pp-target[0]))),max_jacobian_difference=float(np.max(abs(jj+target[1])))))
    for name in ['models','solvers','cascade_refinement']:
        rr,jj,pp=modules[name].residuals_and_jacobian(x[:3],sp,sv,y,False)
        tt=w.predict('M0',x[:3],observations,origin)
        result.append(dict(path=name+'.residuals_and_jacobian',max_prediction_difference_mps=float(np.max(abs(pp-tt[0]))),max_jacobian_difference=float(np.max(abs(jj+tt[1])))))
    hh,jx=modules['be_gtr_solver'].h_and_jacobian_x(x[:6],time,sp,sv,0.)
    vp=modules['variable_projection']
    B=vp.bias_design_matrix(time,0.,'full')
    for robust in (False,True):
        weights=modules['models'].cauchy_weights(np.linspace(-3,3,180)) if robust else np.ones(180)
        ev=modules['be_gtr_solver'].evaluate_be_gtr(x[:6],time,sp,sv,hh+B@x[6:],0.,weights,vp.beta_prior_profiles()['B0_none'],'full',True)
        result.append(dict(path='be_gtr_solver.evaluate_be_gtr_'+str(robust),max_prediction_difference_mps=float(np.max(abs(ev.h_nonbias-hh))),max_jacobian_difference=float(np.max(abs(ev.reduced_jacobian-ev.projection.projection_matrix@jx)))))
    rr,jj,cond=modules['cascade_refinement'].evaluate_full_state(x,dict(time_s=time,t0_s=0.,sat_pos_m=sp,sat_vel_mps=sv,meas_mps=y))
    result.append(dict(path='cascade_refinement.evaluate_full_state',max_prediction_difference_mps=float(np.max(abs(rr-(y-target[0])))),max_jacobian_difference=float(np.max(abs(jj+target[1])))))
finally:
    sys.setprofile(None)
    for module,attr,orig in patches: setattr(module,attr,orig)
assert all(x['max_prediction_difference_mps']==0. and x['max_jacobian_difference']==0. for x in result)
assert not blocked
for item in manifest['native_sources']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
data=dict(status='PASS_FIXED_STATE_LEAF_IDENTITY',paths=result,
          restored_aliases=len(patches),optimizer_calls=0,forbidden_execution_attempts=blocked,
          scope='Temporary full-batch QA injection; not a production loader, native candidate execution or generalized train/subset integration.',
          scientific_source_modified=False)
with (OUT/'native_leaf_identity_test.json').open('x',encoding='utf-8') as f: json.dump(data,f,indent=2)
print(json.dumps(data))
