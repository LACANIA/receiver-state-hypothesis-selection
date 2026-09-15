from common import *
import importlib
from collections import OrderedDict,Counter

class NativeBridge:
    def __init__(self,T):
        self.sources=check_sources()
        root=Path(next(x['resolved_path'] for x in self.sources if Path(x['resolved_path']).name=='models.py')).parent.parent
        sys.path.insert(0,str(root))
        self.mods={n:importlib.import_module('leo_positioning.'+n) for n in ['models','trajectory_models','solvers','trajectory_solvers','model_candidates','ma_bgtr_v4_solver','ma_bgtr_v3_solver','ma_bgtr_v2_solver','validation_split','influence_diagnostics','uncertainty_diagnostics','quality','risk_veto']}
        self.T=T;self.records=[r for r in read(O/'candidate_observations.json') if r['receive_gpst_ns']<ORIGIN+T*10**9]
        self.observations=[w.observation_from_mapping(r) for r in self.records]
        self.times=np.array([(x.receive_gpst_ns-ORIGIN)/1e9 for x in self.observations])
        states=[w.satellite_state(x.ephemeris,(x.receive_gpst_ns-x.ephemeris.toe_ns)/1e9) for x in self.observations]
        self.pos=np.array([s[0] for s in states]);self.vel=np.array([s[1] for s in states]);self.index={tuple(p):i for i,p in enumerate(self.pos)}
        assert len(self.index)==len(self.observations)
        self.obs=dict(time_s=self.times,t0_s=0.,sat_pos_m=self.pos,sat_vel_mps=self.vel,meas_mps=np.array([x.y_mps for x in self.observations]),satellite_number=np.array([x.satellite for x in self.observations]),row_index=np.arange(len(self.times)))
        self.cache=OrderedDict();self.calls=[];self.counter=Counter();self.light_error=0.;self.light_iterations=0;self.leaf_patches=[]
        self.patch(self.mods['models'].residuals_and_jacobian,self.static_res)
        self.patch(self.mods['trajectory_models'].residuals_and_jacobian_ctd,self.dynamic_res)
        for mn,fn in [('solvers','solve_lm'),('trajectory_solvers','solve_ctd_lm')]:
            old=getattr(self.mods[mn],fn)
            def counted(*args,_old=old,_fn=fn,**kwargs):
                self.counter[_fn]+=1
                res=_old(*args,**kwargs);self.calls.append(dict(function=_fn,result=res));return res
            self.patch(old,counted)
    def patch(self,old,new):
        for mod in list(sys.modules.values()):
            if not getattr(mod,'__name__','').startswith('leo_positioning'):continue
            for name,val in list(vars(mod).items()):
                if val is old:setattr(mod,name,new);self.leaf_patches.append((mod.__name__,name))
    def indices(self,a,b,t=None):
        idx=np.array([self.index[tuple(p)] for p in np.asarray(a)])
        assert np.array_equal(a,self.pos[idx]) and np.array_equal(b,self.vel[idx])
        if t is not None:assert np.array_equal(t,self.times[idx])
        return idx
    def geometric(self,x,idx):
        key=(tuple(idx),np.asarray(x,float).tobytes())
        if key not in self.cache:
            p,j,infos=w.predict('M4',x,[self.observations[i] for i in idx],ORIGIN)
            self.cache[key]=(p,j);self.light_error=max(self.light_error,max(v['fixed_point_error_s'] for v in infos));self.light_iterations=max(self.light_iterations,max(v['iterations'] for v in infos))
            if len(self.cache)>24:self.cache.popitem(last=False)
        return self.cache[key]
    def static_res(self,state,a,b,meas,with_bias,receiver_vel_mps=None):
        assert receiver_vel_mps is None or not np.any(receiver_vel_mps)
        idx=self.indices(a,b);p,j=self.geometric(np.r_[state[:3],np.zeros(3)],idx)
        pred=p+(state[3] if with_bias else 0)
        j=np.column_stack([j[:,:3],np.ones(len(idx))]) if with_bias else j[:,:3]
        return np.asarray(meas)-pred,-j,pred
    def dynamic_res(self,x,t,a,b,meas,t0,config=None):
        tm=self.mods['trajectory_models'];cfg=config or tm.FULL_CTD_CONFIG;assert t0==0.
        p,v,b0,bdot=tm.unpack_state(x,cfg);idx=self.indices(a,b,t);pred,j=self.geometric(np.r_[p,v],idx)
        pred=pred+b0+bdot*np.asarray(t);parts=[j]
        if cfg.estimate_b0:parts.append(np.ones((len(idx),1)))
        if cfg.estimate_bdot:parts.append(np.asarray(t)[:,None])
        return np.asarray(meas)-pred,-np.column_stack(parts),pred
    def qa(self):
        train,val,split=self.mods['validation_split'].blocked_train_validation(self.obs,train_fraction=.7)
        tests=[];p=np.array(read(O/'fit_schedule.json')[0]['initial_ecef_m'])
        for lab,sub in [('full',self.obs),('train',train),('heldout',val)]:
            for m in MODELS:
                x=np.r_[p,([] if m in ['M0','M1'] else [0,0,0]),([3.] if m in ['M1','M2','M3'] else []),([.01] if m=='M2' else [])]
                y,j,_=w.predict(m,x,[self.observations[i] for i in sub['row_index']],ORIGIN)
                if m in ['M0','M1']:rr,jj,yy=self.static_res(x,sub['sat_pos_m'],sub['sat_vel_mps'],sub['meas_mps'],m=='M1')
                else:
                    tm=self.mods['trajectory_models'];cfg={'M2':tm.FULL_CTD_CONFIG,'M3':tm.NO_BDOT_CONFIG,'M4':tm.NO_BIAS_CONFIG}[m]
                    rr,jj,yy=self.dynamic_res(x,sub['time_s'],sub['sat_pos_m'],sub['sat_vel_mps'],sub['meas_mps'],0.,cfg)
                assert np.max(abs(yy-y))<1e-12 and np.max(abs(jj+j))<1e-12
                tests.append(dict(model=m,subset=lab,rows=len(y),prediction_error=float(np.max(abs(yy-y))),jacobian_error=float(np.max(abs(jj+j)))))
        return dict(tests=tests,split=split,full_rows=len(self.times),train_rows=train['row_index'],heldout_rows=val['row_index'],sources=self.sources,optimizer_calls=0,selector_calls=0)
