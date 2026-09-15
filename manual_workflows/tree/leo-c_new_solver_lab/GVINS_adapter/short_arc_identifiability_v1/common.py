from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys
import numpy as np

O=Path(__file__).resolve().parent
A=O.parent
P=A/'sports_field_pilot'
V2=P/'time_alignment_v2'
OLD=A/'sports_field_30s_diagnostic_v1/initialization_assisted'
W=A/'gps_l1_wrapper'
ORIGIN=1606691928102000000
WINDOWS=[30,60,120,300]
MODELS=['M0','M1','M2','M3','M4']
R616=float(np.linalg.norm([500.,-300.,200.]))
sys.path.insert(0,str(W))
import wrapper as w

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(clean(x),f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def vec(r,prefix): return np.array([r[prefix+a] for a in 'xyz'],float)
def enu(p):
    lon=np.arctan2(p[1],p[0]);lat=np.arctan2(p[2],np.hypot(*p[:2])*(1-6.69437999014e-3))
    for _ in range(8):
        n=6378137/np.sqrt(1-6.69437999014e-3*np.sin(lat)**2)
        lat=np.arctan2(p[2]+6.69437999014e-3*n*np.sin(lat),np.hypot(*p[:2]))
    sl,cl=np.sin(lon),np.cos(lon);sf,cf=np.sin(lat),np.cos(lat)
    return np.array([[-sl,cl,0],[-sf*cl,-sf*sl,cf],[cf*cl,cf*sl,sf]])
def signfix(v): return v if v[np.argmax(abs(v))]>=0 else -v
def rank(x):
    return int(np.linalg.matrix_rank(x)) if x.size else 0
def information(Jp,Jn,frame):
    # All M0-M4 native fits use unit row weights. No residual-derived variance
    # or Doppler std is used to alter this common-point information matrix.
    if Jn.shape[1]:
        u,s,_=np.linalg.svd(Jn,full_matrices=False);rn=rank(Jn)
        C=Jp-u[:,:rn]@(u[:,:rn].T@Jp)
    else: rn=0;C=Jp.copy()
    _,s,vt=np.linalg.svd(C,full_matrices=False)
    vals=(s*s)[::-1];dirs=vt[::-1].T
    dirs=np.column_stack([signfix(dirs[:,i]) for i in range(3)])
    S=C.T@C
    tol=max(C.shape)*np.finfo(float).eps*(s[0] if len(s) else 0)
    support=s>tol;rs=int(np.sum(support));iv=np.zeros_like(s);iv[support]=1/(s[support]**2)
    cov=(vt.T*iv)@vt
    weakest=dirs[:,0];strongest=dirs[:,2]
    d=dict(rows=len(Jp),rank_J=rank(np.column_stack([Jp,Jn])),rank_Jn=rn,rank_Sp=rs,
        lambda_min=vals[0],lambda_mid=vals[1],lambda_max=vals[2],
        position_condition=vals[-1]/vals[0] if rs==3 else np.inf,
        covariance_pinv_trace_m2=np.trace(cov),position_sqrt_trace_proxy_m=np.sqrt(np.trace(cov)),
        largest_position_std_proxy_m=1/s[-1] if rs==3 else np.inf,
        weak_position_std_proxy_m=1/s[-1] if rs==3 else np.inf,
        weak_visible_signature_per_s=np.linalg.norm(C@weakest),strong_visible_signature_per_s=np.linalg.norm(C@strongest),
        position_information_trace_retention=np.sum(C*C)/np.sum(Jp*Jp),
        raw_position_jacobian_singular_values=np.linalg.svd(Jp,compute_uv=False),
        weak_ecef=weakest,strong_ecef=strongest,weak_enu=frame@weakest,strong_enu=frame@strongest,
        proxy_identity='UNIT_1_MPS_NOISE_SCALE_UNCALIBRATED_LOCAL_PROXY',
        rank_identity='SVD_MACHINE_PRECISION_ON_PROJECTED_J; NO_NATIVE_GATE_CHANGE',
        covariance_status='FULL_SUPPORTED' if rs==3 else 'PSEUDOINVERSE_OMITS_NULLSPACE_UNBOUNDED')
    return d,S,C,dirs
def design(g,t,T,model):
    jp=g[:,:3];parts=[]
    if model not in ['M0','M1']:parts.append(g[:,3:6]+(t-T)[:,None]*jp)
    if model in ['M1','M2','M3']:parts.append(np.ones((len(t),1)))
    if model=='M2':parts.append(t[:,None])
    return jp,np.column_stack(parts) if parts else np.empty((len(t),0))
def check_sources():
    src=read(A/'source_manifest.json')['native_source_checks']
    for f in src: assert sha(f['resolved_path'])==f['expected_sha256'],f['resolved_path']
    return src
