from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys, importlib.util
import numpy as np

O=Path(__file__).resolve().parent
A=O.parent
S=A/'short_arc_identifiability_v1'
W=A/'gps_l1_wrapper'
V2=A/'sports_field_pilot/time_alignment_v2'
sp=importlib.util.spec_from_file_location('saved_information_math',S/'common.py')
arc=importlib.util.module_from_spec(sp);sp.loader.exec_module(arc)
w=arc.w
ORIGIN=arc.ORIGIN
WINDOWS=[30,60,120]
SAME6=[2,5,6,13,29,30]
now=arc.now
sha=arc.sha
read=arc.read
clean=arc.clean
vec=arc.vec
enu=arc.enu
save=arc.save

def rms(x):
    x=np.asarray(x,float)
    return float(np.sqrt(np.mean(x*x))) if x.size else np.nan

def stats(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)) if len(x) else None,
        rms=rms(x),mse=float(np.mean(x*x)) if len(x) else None,
        abs_p95=float(np.quantile(abs(x),.95)) if len(x) else None,
        max_abs=float(np.max(abs(x))) if len(x) else None)

def project(X,y):
    X=np.asarray(X,float);y=np.asarray(y,float)
    if not X.shape[1]:return np.zeros_like(y)
    u,s,_=np.linalg.svd(X,full_matrices=False)
    tol=max(X.shape)*np.finfo(float).eps*s[0]
    u=u[:,s>tol]
    return u@(u.T@y)

def cosine(a,b,axis=False):
    a=np.asarray(a,float);b=np.asarray(b,float)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    if not np.isfinite(den) or den==0:return np.nan
    q=float(a@b/den)
    return float(np.degrees(np.arccos(np.clip(abs(q) if axis else q,-1,1))))

def autocorr(t,x,lag):
    # Only match actual consecutive integer-second slots; do not bridge gaps.
    t=np.asarray(t,float);x=np.asarray(x,float);slots=np.rint(t-t[0]).astype(int)
    pairs=[(i,j) for i in range(len(x)) for j in [np.searchsorted(slots,slots[i]+lag)] if j<len(x) and slots[j]==slots[i]+lag and np.isfinite(x[i]) and np.isfinite(x[j])]
    if len(pairs)<3:return None
    y,z=np.array([(x[i],x[j]) for i,j in pairs]).T
    if np.std(y)==0 or np.std(z)==0:return None
    return float(np.corrcoef(y,z)[0,1])

def derivative(t,v):
    t=np.asarray(t,float);v=np.asarray(v,float);out=np.full_like(v,np.nan)
    for i in range(1,len(t)-1):
        if np.isfinite(v[i-1:i+2]).all() and np.max(np.diff(t[i-1:i+2]))<=1.001:
            out[i]=(v[i+1]-v[i-1])/(t[i+1]-t[i-1])
    return out

def row_clock(t,y,degree):
    # Numerical centering/scaling only; return original-origin physical coefficients.
    center=float(np.mean(t));scale=float(max(np.ptp(t),1))
    z=(t-center)/scale;X=np.column_stack([z**k for k in range(degree+1)])
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    pred=X@b
    if degree==0:physical=[b[0],0.,0.]
    elif degree==1:physical=[b[0]-b[1]*center/scale,b[1]/scale,0.]
    else:physical=[b[0]-b[1]*center/scale+b[2]*center**2/scale**2,b[1]/scale-2*b[2]*center/scale**2,b[2]/scale**2]
    return pred,np.asarray(physical)
