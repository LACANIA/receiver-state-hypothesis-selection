from common import *
from itertools import combinations
import pandas as pd

def main():
    assert not (O/'INFORMATION_SEAL.json').exists()
    records=read(O/'candidate_observations.json');obs=[w.observation_from_mapping(r) for r in records]
    rr=read(O/'reference_qa_only.json');refs={r['gpst_calendar_ns']:r for r in rr['reference']}
    t=np.array([(o.receive_gpst_ns-ORIGIN)/1e9 for o in obs]); sats=np.array([o.satellite for o in obs])
    support=read(O/'support_lock.json');tables={k:[] for k in ['window_support_inventory.csv','los_temporal_geometry.csv','local_information_summary.csv','local_information_eigenvectors.csv','nested_information_ordering.csv']}
    geom={};los=[];maxlt=0
    for velocity in ['PVT_PRIMARY','POSITION_DERIVED_SENSITIVITY']:
        deriv=[];valid=[]
        for i,o in enumerate(obs):
            r=refs[o.receive_gpst_ns];p=vec(r,'primary_p');v=vec(r,'primary_v' if velocity=='PVT_PRIMARY' else 'position_derived_v')
            ok=bool(r['reference_eligible']) and np.isfinite(np.r_[p,v]).all();valid.append(ok)
            if not ok:deriv.append(np.full(7,np.nan));continue
            _,g,inf=w.predict_epoch(o,p,v,0.0);deriv.append(g);maxlt=max(maxlt,inf['fixed_point_error_s'])
            if velocity=='PVT_PRIMARY':
                tau=inf['light_time_s'];sp,_,_,_=w.satellite_state(o.ephemeris,(o.receive_gpst_ns-o.ephemeris.toe_ns)/1e9-tau)
                sp=np.array(w._rotate(sp,tau));u=(sp-p)/np.linalg.norm(sp-p);los.append(dict(index=i,time_s=t[i],satellite=o.satellite,los_ecef=u.tolist(),range_m=np.linalg.norm(sp-p)))
        geom[velocity]=(np.array(deriv),np.array(valid))
    los_by={r['index']:r for r in los}
    allinfos=[];store={};weakdirs={}
    for T in WINDOWS:
        mask=t<T;refmask=geom['PVT_PRIMARY'][1]&mask;fr=enu(vec(rr['endpoints'][str(T)],'primary_p'))
        natural=sorted(set(sats[mask]));n_epochs=len(set(t[mask]))
        tables['window_support_inventory.csv'].append(dict(window=T,support='NATURAL_SUPPORT',status='AVAILABLE',rows=int(mask.sum()),epochs=n_epochs,satellites=len(natural),satellite_list=';'.join(map(str,natural)),reference_rows=int(refmask.sum()),planned_seconds=T,report_epoch_ns=ORIGIN+T*10**9,last_observation_ns=max(o.receive_gpst_ns for o,m in zip(obs,mask) if m),reference_epoch_coverage=int(len(set(t[refmask]))),source='candidate_observations.json; row_lineage.json'))
        tables['window_support_inventory.csv'].append(dict(window=T,support='COMMON_SATELLITE_SUPPORT',status=support['status'],rows=None if len(support['common_satellites'])<4 else int((mask&np.isin(sats,support['common_satellites'])).sum()),epochs=n_epochs,satellites=len(support['common_satellites']),satellite_list=';'.join(map(str,support['common_satellites'])),reference_rows=None,planned_seconds=T,report_epoch_ns=ORIGIN+T*10**9,source='support_lock.json; intersection over all 3000 native epochs'))
        for sat in natural:
            ids=np.where(mask&(sats==sat)&geom['PVT_PRIMARY'][1])[0];ls=np.array([los_by[i]['los_ecef'] for i in ids]);tt=t[ids]
            if len(ids)<2:continue
            angles=np.arccos(np.clip(np.sum(ls[1:]*ls[:-1],axis=1),-1,1));deriv=np.linalg.norm(np.diff(ls,axis=0),axis=1)/np.diff(tt)
            moment=ls.T@ls/len(ls);planvals=np.linalg.eigvalsh(moment)
            jpvals=np.linalg.svd(geom['PVT_PRIMARY'][0][ids,:3],compute_uv=False)
            tables['los_temporal_geometry.csv'].append(dict(window=T,satellite=sat,support='NATURAL_SUPPORT',rows=len(ids),first_s=tt[0],last_s=tt[-1],endpoint_los_angle_deg=np.degrees(np.arccos(np.clip(ls[0]@ls[-1],-1,1))),accumulated_los_angle_deg=np.degrees(sum(angles)),los_derivative_median_per_s=np.median(deriv),los_derivative_max_per_s=max(deriv),los_moment_lambda_min=planvals[0],los_moment_lambda_mid=planvals[1],los_moment_lambda_max=planvals[2],position_J_smax=jpvals[0],position_J_smid=jpvals[1],position_J_smin=jpvals[2]))
        for vel,(g,valid) in geom.items():
            for sname in ['NATURAL_SUPPORT','COMMON_SATELLITE_SUPPORT']:
                if sname=='COMMON_SATELLITE_SUPPORT' and len(support['common_satellites'])<4:
                    for m in MODELS:tables['local_information_summary.csv'].append(dict(window=T,model=m,support=sname,velocity=vel,status='COMMON_SUPPORT_UNAVAILABLE',rows=None))
                    continue
                mm=mask&valid
                if sname=='COMMON_SATELLITE_SUPPORT':mm &= np.isin(sats,support['common_satellites'])
                local={}
                for m in MODELS:
                    a,n=design(g[mm],t[mm],T,m);d,S,C,dirs=information(a,n,fr)
                    key=f'{T}_{sname}_{vel}_{m}';store[key]=dict(S=S.tolist(),Jp=a.tolist(),Jn=n.tolist(),projected_position_J=C.tolist())
                    row=dict(window=T,model=m,support=sname,velocity=vel,status='AVAILABLE',satellites=len(set(sats[mm])),report_time_s=T,evaluation_point='COMMON_REFERENCE_TRAJECTORY_TANGENT',**d)
                    allinfos.append(row);local[m]=(S,n,a)
                    flat={k:v for k,v in row.items() if not isinstance(v,np.ndarray)}
                    for j,x in enumerate(d['raw_position_jacobian_singular_values']):flat[f'raw_position_J_s{j}']=x
                    tables['local_information_summary.csv'].append(flat)
                    for label in ['weak','strong']:
                        ev=dict(window=T,model=m,support=sname,velocity=vel,direction=label)
                        for ax,x in zip('xyz',d[label+'_ecef']):ev['ecef_'+ax]=x
                        for ax,x in zip(['east','north','up'],d[label+'_enu']):ev['enu_'+ax]=x
                        tables['local_information_eigenvectors.csv'].append(ev)
                    if m=='M2' and T in [30,120] and vel=='PVT_PRIMARY' and sname=='NATURAL_SUPPORT':weakdirs[T]=dict(weak=d['weak_ecef'],strong=d['strong_ecef'])
                for m1,m2 in combinations(MODELS,2):
                    S1,n1,a1=local[m1];S2,n2,a2=local[m2]
                    contained12=rank(np.column_stack([n2,n1]))==rank(n2)
                    contained21=rank(np.column_stack([n1,n2]))==rank(n1)
                    if not contained12 and not contained21:
                        entry=dict(status='NOT_COMPARABLE',reason='NEITHER_NUISANCE_SPACE_CONTAINS_OTHER',lower=None,higher=None,min_information_difference=None)
                    else:
                        lo,hi=(m1,m2) if contained12 else (m2,m1)
                        delta=local[lo][0]-local[hi][0];ev=np.linalg.eigvalsh(delta);tol=1e-10*max(np.linalg.norm(local[lo][0],2),np.linalg.norm(local[hi][0],2))
                        assert ev[0]>=-tol,(T,vel,lo,hi,ev,tol)
                        entry=dict(status='PASS',reason='COMMON_PHYSICAL_POINT_ROWS_WEIGHTS_POSITION_BLOCK_NESTED_NUISANCE',lower=lo,higher=hi,min_information_difference=ev[0],roundoff_tolerance=tol)
                    tables['nested_information_ordering.csv'].append(dict(window=T,support=sname,velocity=vel,pair=m1+'__'+m2,**entry))
    for row in tables['local_information_summary.csv']:
        if row['status']!='AVAILABLE':continue
        base=next(x for x in allinfos if x['window']==30 and x['model']==row['model'] and x['support']==row['support'] and x['velocity']==row['velocity'])
        cur=next(x for x in allinfos if x['window']==row['window'] and x['model']==row['model'] and x['support']==row['support'] and x['velocity']==row['velocity'])
        row['lambda_min_growth_vs_W30']=row['lambda_min']/base['lambda_min']
        row['weak_proxy_improvement_vs_W30']=base['weak_position_std_proxy_m']/row['weak_position_std_proxy_m']
        row['weak_direction_rotation_vs_W30_deg']=np.degrees(np.arccos(np.clip(abs(cur['weak_ecef']@base['weak_ecef']),-1,1)))
    save(O/'information_tables.json',tables);save(O/'local_matrices.json',store);save(O/'los_records.json',los)
    save(O/'information_full.json',allinfos)
    specs=[];p0=np.array(read(O/'initialization_contract.json')['P_REF_0']);d0=np.array([500,-300,200])/R616
    for T in [30,120]:
        for m in ['M1','M2','M0','M3','M4']:
            radii=[0,R616,1000,5000,10000] if m in ['M1','M2'] else [0,R616,10000]
            for r in radii:specs.append(dict(window=T,model=m,direction='LEGACY_D0',radius_m=r,unit_direction=d0,initial_ecef_m=p0+r*d0,offset_ecef_m=r*d0,support='NATURAL_SUPPORT',module='MAIN' if m in ['M1','M2'] else 'CORE_SUPPLEMENT'))
        for lab in ['weak','strong']:
            for r in [R616,1000,5000,10000]:
                dd=weakdirs[T][lab]
                specs.append(dict(window=T,model='M2',direction=lab.upper(),radius_m=r,unit_direction=dd,initial_ecef_m=p0+r*dd,offset_ecef_m=r*dd,support='NATURAL_SUPPORT',module='DIRECTIONAL'))
    for i,spec in enumerate(specs):
        spec['fit_id']=f'F{i+1:03d}';assert abs(np.linalg.norm(np.array(spec['initial_ecef_m'])-p0)-spec['radius_m'])<1e-8
    assert len(specs)==54
    save(O/'fit_schedule.json',specs)
    seal=['RUN_PROTOCOL.json','support_lock.json','candidate_observations.json','row_lineage.json','reference_qa_only.json','initialization_contract.json','information_tables.json','information_full.json','local_matrices.json','los_records.json','fit_schedule.json','common.py','prepare.py','information_stage.py']
    save(O/'INFORMATION_SEAL.json',dict(utc=now(),status='INFORMATION_COMPLETE_BEFORE_ANY_OPTIMIZER',files=[dict(path=str(O/n),sha256=sha(O/n)) for n in seal],optimizer_calls=0,selector_calls=0,max_light_time_error_s=maxlt,common_support=support['status']))
    print(json.dumps(clean([x for x in tables['local_information_summary.csv'] if x['model']=='M2' and x['velocity']=='PVT_PRIMARY' and x['status']=='AVAILABLE'])),flush=True)

if __name__=='__main__':main()
