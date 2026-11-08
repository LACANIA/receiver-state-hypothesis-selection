from common28 import *
import numpy as np
from generator28 import orbit_functions
def geometry_pool(decl):
    frozen,models=bind();of=orbit_functions(BASE);records=[];engineering=[]
    for row in [r for r in decl if r['record_type']=='GEOMETRY_POOL']:
        gid=row['record_id'];lat=float(row['lat_deg']);lon=float(row['lon_deg']);start=float(row['start_s'])
        la,lo=np.deg2rad([lat,lon]);up=np.array([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)])
        anchor=6378137.*up;east=np.array([-np.sin(lo),np.cos(lo),0]);sat=[];sv=[];sid=[];vis=[]
        for layer in [0,.5]:
          fun=of['orbit'] if layer==0 else of['orbit_add']
          for j in range(12):
           for k in range(20):
            pp,vv=zip(*[fun(start+t,j,k) for t in range(31)])
            pp=np.array(pp);vv=np.array(vv);delta=pp-anchor
            elev=np.degrees(np.arcsin(delta@up/np.linalg.norm(delta,axis=1)))
            name=('' if layer==0 else 'ADD_')+f'P{j:02}S{k:02}'
            ok=bool(np.all(elev>=10));vis.append(dict(id=name,min_elevation_deg=float(elev.min()),all_elevations_deg=elev.tolist(),eligible=ok))
            if ok:sat.extend(pp);sv.extend(vv);sid.extend([name]*31)
        tau=np.tile(np.arange(31.),len(sid)//31)
        g=dict(sat=np.array(sat).reshape(-1,3),sv=np.array(sv).reshape(-1,3),satellite_id=np.array(sid),tau=tau,anchor=anchor,east=east,
          T=30.,time_origin=start,output_epoch=start+30.,train=np.flatnonzero(tau.astype(int)%5!=0),test=np.flatnonzero(tau.astype(int)%5==0))
        windows={}
        for name,lower in [('W30',0),('W20',10),('W10',20)]:
            ix=np.flatnonzero((tau>=lower)&(tau<=30));gw=dict(g)
            for key in ['sat','sv','satellite_id','tau']:gw[key]=g[key][ix]
            gw['train']=np.flatnonzero(gw['tau'].astype(int)%5!=0);gw['test']=np.flatnonzero(gw['tau'].astype(int)%5==0)
            assert np.array_equal(ix[gw['train']],np.intersect1d(ix,g['train']))
            assert np.array_equal(ix[gw['test']],np.intersect1d(ix,g['test']))
            windows[name]=dict(rows=ix.tolist(),train=gw['train'].tolist(),test=gw['test'].tolist(),n_rows=len(ix),n_train=len(gw['train']),n_test=len(gw['test']))
        diag={}
        try:diag=frozen.local_diag(np.r_[anchor,np.zeros(5)],models[3],g,g['train'],.3)
        except (ValueError,IndexError,np.linalg.LinAlgError) as ex:diag={'diagnostic_error':str(ex)}
        V=diag.get('position_variance_proxy')
        numeric=diag.get('rank_normalized')==8 and diag.get('QR_physical_response_relative',float('inf'))<=1e-8
        support=V if numeric and V is not None and np.isfinite(V) else None
        gp=ROOT/'geometry'/f'{gid}.npz';gp.parent.mkdir(exist_ok=True);assert not gp.exists();np.savez_compressed(gp,**g)
        records.append(dict(pool_id=gid,lat_deg=lat,lon_deg=lon,start_s=start,report_epoch_s=start+30,anchor=anchor,east=east,
          satellites=list(dict.fromkeys(sid)),n_satellites=len(set(sid)),visibility=vis,windows=windows,position_support_m2=support,
          common_anchor_CV_AFFINE=diag,path=str(gp),sha256=sha(gp),status='AVAILABLE' if len(tau) else 'GEOMETRY_UNAVAILABLE',
          row_ids=[f'{gid}|{s}|{int(t)}' for s,t in zip(sid,tau)]))
    ordered=sorted(records,key=lambda r:(r['position_support_m2'] is None,r['position_support_m2'] if r['position_support_m2'] is not None else 0,r['pool_id']))
    selected={}
    for i,pos in enumerate([1,3,6],1):
        rr=ordered[pos-1];rr['selected_slot']=f'MO_DEV_G{i}';rr['selected_order']=pos;selected[rr['selected_slot']]=rr
    return records,ordered,selected
