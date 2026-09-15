"""In-memory native measurement leaves and absolute-clock-only diagnostic hooks."""
from pathlib import Path
from collections import OrderedDict, Counter
import sys, json, hashlib, importlib, inspect
import numpy as np

O=Path(__file__).resolve().parent; A=O.parents[1]; W=A/'gps_l1_wrapper'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def save(p,x): Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf-8')

class Bridge:
    def __init__(self):
        self.protocol=read(O/'protocol.json'); self.origin=self.protocol['origin_ns']
        self.prior=read(O/'initialization_prior.json')
        man=read(A/'source_manifest.json'); self.sources=man['native_source_checks']
        assert all(sha(x['resolved_path'])==x['expected_sha256'] for x in self.sources)
        r=Path(self.sources[0]['resolved_path']).parents[1]
        sys.path[:0]=[str(W),str(r/'scripts'),str(r/'src')]
        self.w=importlib.import_module('wrapper'); self.rt=importlib.import_module('release_runtime')
        self.mods={n:importlib.import_module('leo_positioning.'+n) for n in
           ['models','trajectory_models','solvers','trajectory_solvers','be_gtr_solver','gir_tr_solver',
            'cascade_refinement','model_candidates','ma_bgtr_v2_solver','ma_bgtr_v3_solver',
            'ma_bgtr_v4_solver','quality','risk_veto','hierarchical_gates','ma_bgtr_v71_solver',
            'robust_gates','family_evidence','validation_split','uncertainty_diagnostics']}
        self.records=read(W/'candidate_observations.json')
        self.observations=[self.w.observation_from_mapping(x) for x in self.records]
        self.times=np.array([(x.receive_gpst_ns-self.origin)/1e9 for x in self.observations])
        pos=[];vel=[]
        for x in self.observations:
            p,v,_,_=self.w.satellite_state(x.ephemeris,(x.receive_gpst_ns-x.ephemeris.toe_ns)/1e9)
            pos.append(p);vel.append(v)
        self.pos=np.asarray(pos);self.vel=np.asarray(vel)
        self.index={tuple(p):i for i,p in enumerate(self.pos)}
        assert len(self.index)==180
        self.obs=dict(time_s=self.times,t0_s=0.,sat_pos_m=self.pos,sat_vel_mps=self.vel,
          meas_mps=np.array([x.y_mps for x in self.observations]),
          satellite_number=np.array([x.satellite for x in self.observations]),row_index=np.arange(180),
          sealed_initialization_prior=self.prior)
        self.cache=OrderedDict();self.calls=Counter();self.shadow=Counter();self.bypass=False
        self.max_light_error=0.;self.max_light_iterations=0;self.patches=[];self.full_capture=[]
        self.original_physical=self.mods['quality'].physical_plausibility
        self.original_risk=self.mods['risk_veto'].severe_risk_veto
        self.original_refine=self.mods['hierarchical_gates']._refine_gate
        src=inspect.getsource(self.original_refine)
        old='if speed > 300.0 or beta > 30.0 or bdot > 1.0:'
        assert src.count(old)==1
        namespace=dict(vars(self.mods['hierarchical_gates']))
        exec(compile(src.replace(old,'if speed > 300.0 or bdot > 1.0:'),'<absolute_b0_only_refine_diagnostic>','exec'),namespace)
        self.refine_no_b0=namespace['_refine_gate']
        original_final=self.mods['model_candidates'].finalize_candidate
        def capture_final(*args,**kwargs):
            result=original_final(*args,**kwargs)
            self.full_capture.append(dict(model=args[0]['candidate_model'],positions=np.asarray(args[2]).copy(),
                velocity=np.asarray(args[3]).copy(),b0=args[4],bdot=args[5],residual=np.asarray(args[6]).copy(),
                numerical_success=bool(args[8]),converged=bool(args[9]),row=dict(result)))
            return result
        originals=[(self.mods['models'].residuals_and_jacobian,self.static_res),
            (self.mods['trajectory_models'].residuals_and_jacobian_ctd,self.dynamic_res),
            (self.mods['be_gtr_solver'].h_and_jacobian_x,self.nonbias),
            (self.original_physical,self.physical),(self.original_risk,self.risk),
            (self.original_refine,self.refine),(original_final,capture_final)]
        for mod in list(sys.modules.values()):
            name=getattr(mod,'__name__','')
            if not (name.startswith('leo_positioning') or name=='release_runtime'): continue
            for attr,value in list(vars(mod).items()):
                for old,new in originals:
                    if value is old:
                        self.patches.append((name,attr));setattr(mod,attr,new)

    def indices(self,a,b,t=None,t0=None):
        a=np.asarray(a);b=np.asarray(b)
        idx=np.array([self.index[tuple(p)] for p in a],dtype=int)
        assert np.array_equal(a,self.pos[idx]) and np.array_equal(b,self.vel[idx]),'SUBSET_SATELLITE_IDENTITY'
        if t is not None: assert np.array_equal(t,self.times[idx]) and t0==0.,'SUBSET_TIME_ORIGIN_IDENTITY'
        self.calls['rows_'+str(len(idx))]+=1
        return idx

    def geometric(self,x,idx):
        x=np.asarray(x,dtype=float)
        key=(tuple(idx),x.tobytes())
        if key not in self.cache:
            p,j,infos=self.w.predict('M4',x,[self.observations[i] for i in idx],self.origin)
            self.calls['wrapper_batches']+=1;self.calls['wrapper_rows']+=len(idx)
            self.max_light_error=max(self.max_light_error,max(a['fixed_point_error_s'] for a in infos))
            self.max_light_iterations=max(self.max_light_iterations,max(a['iterations'] for a in infos))
            self.cache[key]=(p,j)
            if len(self.cache)>24: self.cache.popitem(last=False)
        return self.cache[key]

    def static_res(self,state,a,b,meas,with_bias,receiver_vel_mps=None):
        assert receiver_vel_mps is None or not np.any(receiver_vel_mps)
        idx=self.indices(a,b);p,j=self.geometric(np.r_[state[:3],np.zeros(3)],idx)
        pred=p+(float(state[3]) if with_bias else 0.)
        jac=np.column_stack([j[:,:3],np.ones(len(idx))]) if with_bias else j[:,:3]
        return np.asarray(meas)-pred,-jac,pred

    def dynamic_res(self,theta,t,a,b,meas,t0,config=None):
        m=self.mods['trajectory_models']; config=config or m.FULL_CTD_CONFIG
        p0,v,b0,bdot=m.unpack_state(theta,config)
        idx=self.indices(a,b,t,t0);p,j=self.geometric(np.r_[p0,v],idx)
        pred=p+b0+bdot*(np.asarray(t)-t0);parts=[j]
        if config.estimate_b0: parts.append(np.ones((len(idx),1)))
        if config.estimate_bdot: parts.append((np.asarray(t)-t0)[:,None])
        return np.asarray(meas)-pred,-np.column_stack(parts),pred

    def nonbias(self,state,t,a,b,t0):
        idx=self.indices(a,b,t,t0);p,j=self.geometric(state,idx)
        return p,-j

    def physical(self,*args,**kwargs):
        ok,why=self.original_physical(*args,**kwargs)
        if 'beta0_limit' in why.split(';'):
            self.shadow['physical_20'+('_bypass' if self.bypass else '_strict')]+=1
        if self.bypass:
            reasons=[x for x in why.split(';') if x and x!='beta0_limit']
            return not reasons,';'.join(reasons)
        return ok,why

    def risk(self,*args,**kwargs):
        veto,why=self.original_risk(*args,**kwargs)
        target='abs(beta0_estimated_mps)>30'
        if target in why.split(';'): self.shadow['severe_30']+=1
        if self.bypass:
            reasons=[x for x in why.split(';') if x and x!=target]
            return bool(reasons),';'.join(reasons)
        return veto,why

    def refine(self,*args,**kwargs):
        strict=self.original_refine(*args,**kwargs)
        bypass=self.refine_no_b0(*args,**kwargs)
        if strict!=bypass: self.shadow['secondary_refine_30']+=1
        return bypass if self.bypass else strict

    def requalify(self,row):
        r=dict(row)
        physical,why=self.physical('real',r.get('estimated_speed_mps',np.nan),
            r.get('beta0_estimated_mps',np.nan),r.get('beta_dot_estimated_mps2',np.nan))
        quality,qwhy=self.mods['quality'].quality_gate('real',bool(r['numerical_success']),physical,None)
        r['physical_plausible']=physical;r['quality_pass']=quality
        reasons=[x for x in str(r.get('failure_reason','')).split(';') if x and x not in
            ('beta0_limit','physical_failure','speed_limit','bdot_limit','numerical_failure')]
        r['failure_reason']=';'.join(dict.fromkeys(reasons+[x for x in (why+';'+qwhy).split(';') if x]))
        return r
