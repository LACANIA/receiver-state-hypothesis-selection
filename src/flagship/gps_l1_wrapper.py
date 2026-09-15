"""Deterministic GPS L1 physics only. No file access, optimizer or selector.

Orbit equations follow the bound GPS broadcast propagation (GPLv3/RTKLIB
provenance in the source manifest). z velocity uses the direct derivative.
The analytic Jacobian is evaluated by forward-mode chain-rule differentiation;
finite differences occur only in the separate QA evaluator.
"""
from dataclasses import dataclass, fields
import math
import numpy as np

C = 299792458.0
MU = 3.986005e14
OMEGA = 7.2921151467e-5
F_L1 = 1575420000.0
LIGHT_TIME_INITIAL_S = 0.075
LIGHT_TIME_TOL_S = 1e-13
LIGHT_TIME_MAX_ITER = 12

@dataclass(frozen=True, slots=True)
class Ephemeris:
    sat: int
    toe_ns: int
    toc_ns: int
    toe_tow: float
    health: int
    A: float
    e: float
    i0: float
    omg: float
    OMG0: float
    M0: float
    delta_n: float
    OMG_dot: float
    i_dot: float
    cuc: float
    cus: float
    crc: float
    crs: float
    cic: float
    cis: float
    af0: float
    af1: float
    af2: float

@dataclass(frozen=True, slots=True)
class Observation:
    receive_gpst_ns: int
    satellite: int
    system: str
    signal: str
    frequency_hz: float
    doppler_hz: float
    ephemeris: Ephemeris

    def __post_init__(self):
        if self.system != 'GPS' or self.signal != 'CODE_L1C' or self.frequency_hz != F_L1:
            raise ValueError('GPS_L1_CA_ONLY')
        if type(self.receive_gpst_ns) is not int or type(self.ephemeris.toe_ns) is not int:
            raise TypeError('INTEGER_GNSS_NANOSECONDS_REQUIRED')
        if self.satellite != self.ephemeris.sat or self.ephemeris.health != 0:
            raise ValueError('SATELLITE_OR_HEALTH_MISMATCH')
        if abs((self.receive_gpst_ns-self.ephemeris.toe_ns)/1e9) >= 7200:
            raise ValueError('BOUND_EPHEMERIS_AGE_EXCEEDED')
        if not math.isfinite(self.doppler_hz):
            raise ValueError('NONFINITE_DOPPLER')
        numeric = [getattr(self.ephemeris, f.name) for f in fields(Ephemeris)]
        if not all(math.isfinite(x) for x in numeric) or not (0 <= self.ephemeris.e < 1) or self.ephemeris.A <= 0:
            raise ValueError('INVALID_EPHEMERIS')

    @property
    def y_mps(self):
        return -C*self.doppler_hz/self.frequency_hz

def observation_from_mapping(record):
    """Strict allowlist: reject extras instead of silently consuming them."""
    names = {f.name for f in fields(Observation)}
    if set(record) != names:
        raise ValueError('CANDIDATE_SCHEMA_FIELDS_MISMATCH')
    eph = record['ephemeris']
    if not isinstance(eph, Ephemeris):
        if set(eph) != {f.name for f in fields(Ephemeris)}:
            raise ValueError('EPHEMERIS_SCHEMA_FIELDS_MISMATCH')
        eph = Ephemeris(**eph)
    return Observation(**{**record, 'ephemeris': eph})

class _D:
    """First-order analytic dual scalar, no finite differencing."""
    __slots__ = ('v', 'g')
    def __init__(self, v, g): self.v=float(v); self.g=np.asarray(g,dtype=float)
    def __add__(self, other):
        if isinstance(other,_D): return _D(self.v+other.v,self.g+other.g)
        return _D(self.v+other,self.g)
    __radd__=__add__
    def __neg__(self): return _D(-self.v,-self.g)
    def __sub__(self,other): return self+-other
    def __rsub__(self,other): return other+-self
    def __mul__(self,other):
        if isinstance(other,_D): return _D(self.v*other.v,self.g*other.v+self.v*other.g)
        return _D(self.v*other,self.g*other)
    __rmul__=__mul__
    def __truediv__(self,other):
        if isinstance(other,_D): return _D(self.v/other.v,(self.g-(self.v/other.v)*other.g)/other.v)
        return _D(self.v/other,self.g/other)
    def __rtruediv__(self,other): return _D(other/self.v,-other*self.g/self.v**2)
    def __pow__(self,power): return _D(self.v**power,power*self.v**(power-1)*self.g)

def _value(x): return x.v if isinstance(x,_D) else float(x)
def _sin(x): return _D(math.sin(x.v),math.cos(x.v)*x.g) if isinstance(x,_D) else math.sin(x)
def _cos(x): return _D(math.cos(x.v),-math.sin(x.v)*x.g) if isinstance(x,_D) else math.cos(x)
def _atan2(y,x):
    if not isinstance(x,_D) and not isinstance(y,_D): return math.atan2(y,x)
    base=x if isinstance(x,_D) else y
    if not isinstance(x,_D): x=_D(x,np.zeros_like(base.g))
    if not isinstance(y,_D): y=_D(y,np.zeros_like(base.g))
    return _D(math.atan2(y.v,x.v),(x.v*y.g-y.v*x.g)/(x.v*x.v+y.v*y.v))
def _dot(a,b): return sum(x*y for x,y in zip(a,b))
def _norm(a): return _dot(a,a)**0.5
def _rotate(s,tau):
    co=_cos(OMEGA*tau); si=_sin(OMEGA*tau)
    return [co*s[0]+si*s[1],-si*s[0]+co*s[1],s[2]]

def satellite_state(eph,t):
    """Seconds relative to toe; clock bias in seconds, drift in s/s."""
    tk=t
    if _value(tk)>302400: tk=tk-604800
    elif _value(tk)<-302400: tk=tk+604800
    e=eph.e; a=eph.A; n=math.sqrt(MU/a**3)+eph.delta_n
    mean=eph.M0+n*tk; ek=mean
    for _ in range(30):
        step=(ek-e*_sin(ek)-mean)/(1-e*_cos(ek)); ek=ek-step
        if abs(_value(step))<1e-14: break
    else: raise ArithmeticError('KEPLER_NONCONVERGENCE')
    se=_sin(ek); ce=_cos(ek); ed=n/(1-e*ce)
    vd=math.sqrt(1-e*e)*ed/(1-e*ce)
    phi=_atan2(math.sqrt(1-e*e)*se,ce-e)+eph.omg
    s2=_sin(2*phi); c2=_cos(2*phi)
    u=phi+eph.cus*s2+eph.cuc*c2
    r=a*(1-e*ce)+eph.crs*s2+eph.crc*c2
    inc=eph.i0+eph.i_dot*tk+eph.cis*s2+eph.cic*c2
    ud=vd+2*vd*(eph.cus*c2-eph.cuc*s2)
    rd=a*e*ed*se+2*vd*(eph.crs*c2-eph.crc*s2)
    ind=eph.i_dot+2*vd*(eph.cis*c2-eph.cic*s2)
    su=_sin(u); cu=_cos(u); si=_sin(inc); ci=_cos(inc)
    x=r*cu; y=r*su; xd=rd*cu-r*ud*su; yd=rd*su+r*ud*cu
    od=eph.OMG_dot-OMEGA
    om=eph.OMG0+od*tk-OMEGA*eph.toe_tow; so=_sin(om); co=_cos(om)
    pos=[x*co-y*ci*so,x*so+y*ci*co,y*si]
    t1=xd-y*od*ci; t2=x*od+yd*ci-y*ind*si
    vel=[t1*co-t2*so,t1*so+t2*co,yd*si+y*ind*ci]
    dt=t+(eph.toe_ns-eph.toc_ns)/1e9
    rel=-2*math.sqrt(MU*a)*e/C**2
    clock=eph.af0+eph.af1*dt+eph.af2*dt*dt+rel*se
    drift=eph.af1+2*eph.af2*dt+rel*ce*ed
    return pos,vel,clock,drift

def _physics(obs,p,v):
    tr=(obs.receive_gpst_ns-obs.ephemeris.toe_ns)/1e9
    tau=LIGHT_TIME_INITIAL_S
    for it in range(1,LIGHT_TIME_MAX_ITER+1):
        s,_,_,_=satellite_state(obs.ephemeris,tr-tau)
        sr=_rotate(s,tau)
        new=_norm([a-b for a,b in zip(sr,p)])/C
        change=abs(_value(new)-_value(tau)); tau=new
        # At least five iterations also converge the analytic derivative.
        if it>=5 and change<LIGHT_TIME_TOL_S: break
    else: raise ArithmeticError('LIGHT_TIME_NONCONVERGENCE')
    s,vs,cb,cd=satellite_state(obs.ephemeris,tr-tau)
    sr=_rotate(s,tau); vv=_rotate(vs,tau)
    d=[a-b for a,b in zip(sr,p)]; rho=_norm(d); u=[x/rho for x in d]
    numer=_dot(u,[a-b for a,b in zip(vv,v)])
    jz=[-sr[1],sr[0],0.0]
    denom=1+_dot(u,[a+OMEGA*b for a,b in zip(vv,jz)])/C
    rate=numer/denom
    sat_clock=-C*cd*(1-rate/C)
    result=rate+sat_clock
    info={'light_time_s':_value(tau),'iterations':it,'fixed_point_error_s':abs(_value(rho)/C-_value(tau)),
          'transmit_seconds_from_toe':_value(tr-tau),'geometric_rate_mps':_value(rate),
          'satellite_clock_term_mps':_value(sat_clock),'satellite_clock_bias_s':_value(cb),
          'satellite_clock_drift_s_per_s':_value(cd)}
    return result,info

def predict_epoch(observation,position_m,velocity_mps,receiver_clock_mps=0.0,jacobian=True):
    """QA/measurement leaf: state components at this receive epoch, not files."""
    if type(observation) is not Observation: raise TypeError('STRICT_OBSERVATION_REQUIRED')
    state=np.r_[position_m,velocity_mps,receiver_clock_mps].astype(float)
    if state.shape!=(7,) or not np.isfinite(state).all(): raise ValueError('INVALID_STATE')
    x=[_D(z,np.eye(7)[i]) for i,z in enumerate(state)] if jacobian else state.tolist()
    val,info=_physics(observation,x[:3],x[3:6]); val=val+x[6]
    return _value(val), (val.g.copy() if jacobian else None),info

STATIC={'M0','M1'}
NO_CLOCK={'M0','M4'}
BIAS_ONLY={'M1','M3','M5','M8'}
PROJECTED={'M5','M6','M8','M9'}
def state_layout(model):
    if model not in {f'M{i}' for i in range(15)}: raise ValueError('UNKNOWN_MODEL')
    cols=['p0_x','p0_y','p0_z']
    if model not in STATIC: cols+=['v_x','v_y','v_z']
    if model not in NO_CLOCK: cols+=['b0']
    if model not in STATIC and model not in NO_CLOCK and model not in BIAS_ONLY: cols+=['bdot']
    return cols

def predict(model,state,observations,time_origin_gpst_ns,jacobian=True):
    """Physical packed state; projected beta packed only for reconstruction QA.

    For M5/M6/M8/M9 optimization space remains six-dimensional; use
    nonbias_prediction() and unchanged native bias_design_matrix separately.
    M14 final state is native full CTD after its static-initialized refinement.
    """
    cols=state_layout(model); x=np.asarray(state,dtype=float)
    if x.shape!=(len(cols),) or not np.isfinite(x).all(): raise ValueError('STATE_LAYOUT_MISMATCH')
    if type(time_origin_gpst_ns) is not int: raise TypeError('INTEGER_ORIGIN_REQUIRED')
    pred=[]; jac=[]; diagnostic=[]
    for obs in observations:
        dt=(obs.receive_gpst_ns-time_origin_gpst_ns)/1e9
        v=np.zeros(3) if model in STATIC else x[3:6]
        p=x[:3] if model in STATIC else x[:3]+dt*v
        b=x[cols.index('b0')] if 'b0' in cols else 0.0
        d=x[cols.index('bdot')] if 'bdot' in cols else 0.0
        value,g,info=predict_epoch(obs,p,v,0.0,jacobian)
        # Clock columns stay exactly additive and independent of light time.
        pred.append(value+b+d*dt)
        if jacobian:
            row=list(g[:3])
            if model not in STATIC: row+=list(g[3:6]+dt*g[:3])
            if 'b0' in cols: row+=[1.0]
            if 'bdot' in cols: row+=[dt]
            jac.append(row)
        diagnostic.append(info)
    return np.asarray(pred),np.asarray(jac) if jacobian else None,diagnostic

def residuals_and_jacobian(model,state,observations,time_origin_gpst_ns):
    pred,jac,info=predict(model,state,observations,time_origin_gpst_ns)
    return np.array([o.y_mps for o in observations])-pred,-jac,pred,info

def nonbias_prediction(state_x,observations,time_origin_gpst_ns):
    pred,jac,info=predict('M4',state_x,observations,time_origin_gpst_ns)
    return pred,-jac,info

def bias_columns(observations,time_origin_gpst_ns,mode):
    dt=np.array([(o.receive_gpst_ns-time_origin_gpst_ns)/1e9 for o in observations])
    if mode=='full': return np.column_stack([np.ones(len(dt)),dt])
    if mode=='b0_only': return np.ones((len(dt),1))
    if mode=='none': return np.zeros((len(dt),0))
    raise ValueError('UNKNOWN_CLOCK_PROJECTION_MODE')
