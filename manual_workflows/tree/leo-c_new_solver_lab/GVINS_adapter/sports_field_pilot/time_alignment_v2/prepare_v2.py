"""Cache-only data selection/reference preparation, without forward residuals."""
from pathlib import Path
import sys,json,hashlib,math,collections
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
OUT=Path(__file__).resolve().parent; SRC=OUT.parent; ADAPTER=SRC.parent
sys.path.insert(0,str(ADAPTER));import extract_gnss as ex
sys.path.insert(0,str(SRC));from scan_and_select import ephem_for
def now():return datetime.now(timezone.utc).isoformat()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def save(n,v):ex.save_json(OUT/n,clean(v))
def stats(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return {'n':0}
    return dict(n=len(a),minimum=float(a.min()),p5=float(np.quantile(a,.05)),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()),rms=float(np.sqrt(np.mean(a*a))))
def put(n,rows):pq.write_table(pa.Table.from_pylist(clean(rows)),OUT/n,compression='zstd')
def vector(r,prefix):return np.array([r[prefix+a] for a in 'xyz'])
def reframe(r):
    lat,lon=np.deg2rad([r['latitude'],r['longitude']]);a=6378137.;e2=6.69437999014e-3
    n=a/np.sqrt(1-e2*np.sin(lat)**2)
    p=np.array([(n+r['altitude'])*np.cos(lat)*np.cos(lon),(n+r['altitude'])*np.cos(lat)*np.sin(lon),(n*(1-e2)+r['altitude'])*np.sin(lat)])
    # Direct NED-to-ECEF, independent layout check against cached ENU path.
    north=np.array([-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)])
    east=np.array([-np.sin(lon),np.cos(lon),0.]);down=np.array([-np.cos(lat)*np.cos(lon),-np.cos(lat)*np.sin(lon),-np.sin(lat)])
    v=r['vel_n']*north+r['vel_e']*east+r['vel_d']*down
    assert np.linalg.norm(p-vector(r,'ecef_p'))<1e-8
    assert np.linalg.norm(v-vector(r,'ecef_v'))<1e-12
    return p,v
def normgap(ns):return 99000000<=int(ns)<=101000000
def main():
    assert not (OUT/'pilot_interval_lock_v2.json').exists(),'NO_OVERWRITE_LOCK'
    cache=json.loads((SRC/'cache/scan_complete.json').read_text());manifest=json.loads((ADAPTER/'source_manifest.json').read_text())
    checks=[]
    for f in cache['files']+[x for repo in manifest['repositories'] for x in repo['files']]:
        actual=ex.sha(f['path']);assert actual==f['sha256'],f['path'];checks.append(dict(path=f['path'],sha256=actual))
    v1names=['download_manifest.json','bag_inventory.json','reference_time_disposition.json','time_alignment_diagnostics.parquet','epoch_data_eligibility.parquet','physics_adapter_decision.md','report.md','decision.json','pilot_interval_lock.json','gps_physics.py','forward_qa.py','physics_execution_lock.json']
    for n in v1names:checks.append(dict(path=str(SRC/n),sha256=ex.sha(SRC/n)))
    save('input_manifest.json',dict(source_run_id='GVINS_SPORTS_FIELD_QA_V1_20260910T092455Z',run_id='GVINS_TIME_ALIGNMENT_V2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),verified_utc=now(),files=checks,bag_rescans=0,source_extensions=[dict(path=str(p),sha256=ex.sha(p),upstream_commit='7961ab78d93f63077445698c16186b7bfb37fd14',identity='SUPPLEMENTAL_PUBLIC_SOURCE_NOT_CAPTURE_BINARY_PROVENANCE') for p in sorted((OUT/'sources').glob('*'))]))
    save('pre_alignment_lock.json',dict(created_utc=now(),contract_sha256=ex.sha(OUT/'reference_propagation_contract.md'),prepare_sha256=ex.sha(__file__),normal_offset_ns=[2000000,3000000],normal_pvt_gap_ns=[99000000,101000000],position_derivative=dict(degree=2,points=11,half_span_nominal_s=.5),primary_reference='NEAREST_FIXED_PVT_CV_PROPAGATED_TO_RECORDED_RAW_LABEL',forward_residuals_computed=False))
    pvts=pq.read_table(SRC/'cache/all_pvt.parquet').to_pylist();pvts.sort(key=lambda x:x['gpst_calendar_ns'])
    pt=np.array([r['gpst_calendar_ns'] for r in pvts],np.int64);assert np.all(np.diff(pt)>0)
    posvel=[reframe(r) for r in pvts];ps=np.array([x[0] for x in posvel]);vs=np.array([x[1] for x in posvel])
    fixed=np.array([r['valid_fix']==1 and r['carr_soln']==2 and np.all(np.isfinite(np.r_[p,v])) for r,(p,v) in zip(pvts,posvel)])
    raw_epochs=pq.read_table(SRC/'cache/raw_epochs.parquet').to_pylist();ts=np.array([r['gpst_calendar_ns'] for r in raw_epochs],np.int64);assert np.all(np.diff(ts)>0)
    gps=pq.read_table(SRC/'cache/full_raw_gnss.parquet',filters=[('system','==','GPS'),('freqs','==',1575420000.),('code','==',1)]).to_pylist()
    eps=json.loads((SRC/'cache/all_ephemerides.json').read_text());bygps=collections.defaultdict(list)
    for e in eps:
        if e['topic'].endswith('/ephem') and 1<=e['sat']<=32:bygps[e['sat']].append(e)
    rawmap=collections.defaultdict(list);ephmap={};satsets=collections.defaultdict(set)
    for r in gps:
        t=r['gpst_calendar_ns'];rawmap[t].append(r)
        if r['doppler_numeric_available'] and r['pseudorange_valid_bit'] and r['psr'] is not None and r['psr']>0:
            e=ephem_for(r,bygps)
            if e is not None:ephmap[r['row_id']]=e;satsets[t].add(r['sat'])
    times=[];refs={};velqa=[];sensitivity=[]
    for raw in raw_epochs:
        t=raw['gpst_calendar_ns'];j=int(np.searchsorted(pt,t));near=min(range(max(0,j-1),min(len(pt),j+1)),key=lambda k:(abs(int(pt[k])-t),int(pt[k])))
        dt=(t-int(pt[near]))/1e9;edge=j==0 or j==len(pt);bracket_ok=not edge and normgap(pt[j]-pt[j-1])
        adjacent=0<near<len(pt)-1 and normgap(pt[near]-pt[near-1]) and normgap(pt[near+1]-pt[near])
        cat='SEQUENCE_EDGE' if edge else 'PVT_GAP' if not bracket_ok else 'NORMAL_OFFSET' if t-int(pt[near]) in (2000000,3000000) else 'OTHER'
        eligible=cat=='NORMAL_OFFSET' and adjacent and fixed[near] and fixed[j-1] and fixed[j] and not raw['mixed_epoch_message'] and len(satsets[t])>=4
        reasons=[]
        if cat!='NORMAL_OFFSET':reasons.append(cat)
        if not adjacent:reasons.append('REFERENCE_NEIGHBOR_GAP_OR_EDGE')
        if not fixed[near]:reasons.append('REFERENCE_NOT_FIXED')
        if len(satsets[t])<4:reasons.append('GPS_EPHEMERIS_COVERAGE')
        times.append(dict(gpst_calendar_ns=t,bag_record_ns=raw['bag_record_ns'],nearest_pvt_gpst_ns=int(pt[near]),nearest_pvt_bag_ns=pvts[near]['bag_record_ns'],raw_minus_pvt_ms=dt*1000,classification=cat,pvt_bracket_gap_s=None if edge else (int(pt[j])-int(pt[j-1]))/1e9,eligible=bool(eligible),reasons=';'.join(reasons),gps_causal_satellites=len(satsets[t]),seconds_from_start=(t-int(ts[0]))/1e9))
        p=ps[near]+dt*vs[near];v=vs[near];hpos=hvel=lpos=lvel=None;pdvel=None
        if bracket_ok and fixed[j-1] and fixed[j]:
            k=j-1;h=(int(pt[j])-int(pt[k]))/1e9;u=(t-int(pt[k]))/1e9/h
            # Use centred position differences to reduce ECEF cancellation.
            dp=ps[j]-ps[k]
            hpos=ps[k]+(-2*u**3+3*u*u)*dp+(u**3-2*u*u+u)*h*vs[k]+(u**3-u*u)*h*vs[j]
            hvel=(-6*u*u+6*u)/h*dp+(3*u*u-4*u+1)*vs[k]+(3*u*u-2*u)*vs[j]
            lpos=ps[k]+u*dp;lvel=dp/h
        if near>=5 and near+5<len(pt):
            ids=np.arange(near-5,near+6)
            if all(normgap(x) for x in np.diff(pt[ids])) and fixed[ids].all():
                tt=(pt[ids]-pt[near])/1e9;design=np.column_stack([np.ones(11),tt,tt*tt]);coef=np.linalg.lstsq(design,ps[ids]-ps[near],rcond=None)[0];pdvel=coef[1]+2*dt*coef[2]
        ref={**pvts[near],'gpst_calendar_ns':t,'pvt_gpst_calendar_ns':int(pt[near]),'raw_minus_pvt_s':dt,'propagation_distance_m':float(np.linalg.norm(dt*v)),'primary_position_role':'RTK_FIXED_POSITION_REFERENCE','primary_velocity_role':'DEVICE_INTERNAL_PVT_VELOCITY_QA_REFERENCE','time_alignment_method':'STATE_PROPAGATION_TO_MEASUREMENT_EPOCH','nominal_raw_time_approximation':True,'bracket_lower_ns':None if edge else int(pt[j-1]),'bracket_upper_ns':None if edge else int(pt[j]),'reference_eligible':bool(eligible)}
        for prefix,arr in [('primary_p',p),('primary_v',v),('pvt_p',ps[near]),('pvt_v',v),('hermite_p',hpos),('hermite_v',hvel),('linear_p',lpos),('linear_v',lvel),('position_derived_v',pdvel)]:
            for ax,val in zip('xyz',[None]*3 if arr is None else arr):ref[prefix+ax]=val
        refs[t]=ref
        if hpos is not None:
            sensitivity.append(dict(gpst_calendar_ns=t,classification=cat,propagation_distance_m=ref['propagation_distance_m'],hermite_minus_primary_position_m=np.linalg.norm(hpos-p),linear_minus_primary_position_m=np.linalg.norm(lpos-p),hermite_minus_primary_velocity_mps=np.linalg.norm(hvel-v),linear_minus_primary_velocity_mps=np.linalg.norm(lvel-v)))
        if pdvel is not None:
            nv=np.linalg.norm(v);nd=np.linalg.norm(pdvel);angle=math.degrees(math.acos(np.clip(v@pdvel/nv/nd,-1,1))) if nv>0 and nd>0 else None
            velqa.append(dict(gpst_calendar_ns=t,classification=cat,pvt_speed_mps=nv,position_derived_speed_mps=nd,vector_difference_mps=np.linalg.norm(pdvel-v),signed_speed_difference_mps=nd-nv,direction_difference_deg=angle,**{'pvt_v'+ax:val for ax,val in zip('xyz',v)},**{'position_derived_v'+ax:val for ax,val in zip('xyz',pdvel)}))
    put('raw_pvt_time_records.parquet',times);put('all_reference_alignment.parquet',list(refs.values()));put('position_derived_velocity_records.parquet',velqa)
    frame=pd.DataFrame(times);normal=frame[frame.classification=='NORMAL_OFFSET'];modes=frame.groupby(['classification','raw_minus_pvt_ms']).size()
    changes=normal.loc[normal.raw_minus_pvt_ms.diff().fillna(0)!=0,['gpst_calendar_ns','raw_minus_pvt_ms','seconds_from_start']].to_dict('records')
    dist=[dict(classification=k[0],raw_minus_pvt_ms=k[1],epochs=int(n),full_denominator=len(frame)) for k,n in modes.items()]
    offstats=dict(all=stats(frame.raw_minus_pvt_ms),normal=stats(normal.raw_minus_pvt_ms),classification_counts=frame.classification.value_counts().to_dict(),normal_modes=normal.raw_minus_pvt_ms.value_counts().to_dict(),normal_mode_changes=changes,first_60s=stats(normal[normal.seconds_from_start<60].raw_minus_pvt_ms),last_60s=stats(normal[normal.seconds_from_start>normal.seconds_from_start.max()-60].raw_minus_pvt_ms),normal_linear_drift_ms_per_s=float(np.polyfit(normal.seconds_from_start,normal.raw_minus_pvt_ms,1)[0]),bag_difference_ms=stats((frame.bag_record_ns-frame.nearest_pvt_bag_ns)/1e6),bag_delta_offset_correlation=float(np.corrcoef(normal.raw_minus_pvt_ms,(normal.bag_record_ns-normal.nearest_pvt_bag_ns)/1e6)[0,1]))
    save('offset_statistics.json',offstats)
    chosen=None
    for i,t in enumerate(ts):
        j=int(np.searchsorted(ts,t+30_000_000_000))
        if j<=i or ts[j-1]-t<29_900_000_000:continue
        if not all(x['eligible'] for x in times[i:j]) or not all(normgap(x) for x in np.diff(ts[i:j])):continue
        common=set.intersection(*(satsets[int(x)] for x in ts[i:j]))
        if len(common)<4:continue
        chosen=(int(t),int(t+30_000_000_000),i,j,common);break
    assert chosen is not None,'NO_ELIGIBLE_V2_30S_INTERVAL'
    start,stop,i,j,common=chosen
    rows=pq.read_table(SRC/'cache/full_raw_gnss.parquet',filters=[('gpst_calendar_ns','>=',start),('gpst_calendar_ns','<',stop)]).to_pylist()
    primary=[r for r in rows if r['system']=='GPS' and r['code']==1 and r['freqs']==1575420000.]
    chosenrefs=[refs[int(t)] for t in ts[i:j]];used={ephmap[r['row_id']]['ephemeris_record_id']:ephmap[r['row_id']] for r in primary if r['row_id'] in ephmap}
    hz={}
    for sec in range((start+999999999)//10**9,(stop-1)//10**9+1):
        t=int(min(ts[i:j],key=lambda t:(abs(int(t)-sec*10**9),int(t))));assert t not in hz;hz[t]=sec*10**9
    onehz=[dict(r,canonical_gpst_ns=hz[r['gpst_calendar_ns']],canonical_time_offset_s=(r['gpst_calendar_ns']-hz[r['gpst_calendar_ns']])/1e9) for r in rows if r['gpst_calendar_ns'] in hz]
    lock=dict(status='LOCKED_BEFORE_FORWARD_RESIDUALS',locked_utc=now(),start_gpst_ns=start,end_gpst_ns_exclusive=stop,start_gps_week=primary[0]['gps_week'],start_gps_tow_s=primary[0]['gps_tow_s'],end_gps_tow_s=primary[0]['gps_tow_s']+30,bag_start_ns=min(r['bag_record_ns'] for r in rows),bag_end_ns=max(r['bag_record_ns'] for r in rows),native_epochs=j-i,native_all_signal_rows=len(rows),gps_l1_signal_rows=len(primary),gps_l1_forward_eligible_rows=sum(r['row_id'] in ephmap for r in primary),satellites=sorted({r['sat'] for r in primary}),persistent_qualified_satellites=sorted(common),ephemeris_ids=sorted(used),row_ephemeris_map=[dict(row_id=r['row_id'],ephemeris_record_id=ephmap[r['row_id']]['ephemeris_record_id'] if r['row_id'] in ephmap else None) for r in primary],one_hz_epoch_mapping=hz,selection_reason='FIRST_CHRONOLOGICAL_DATA_ONLY_WINDOW_WITH_PERSISTENT_FOUR_GPS_L1_AND_CONTINUOUS_FIXED_PVT',prior_start_epochs_rejected=i,reference_fixed_fraction=1,reference_gap_count=0,prior_exact_epoch_check='FAIL_PRESERVED',contract_sha256=ex.sha(OUT/'reference_propagation_contract.md'))
    save('pilot_interval_lock_v2.json',lock)
    for n,table in [('pilot_raw_gnss.parquet',rows),('pilot_1hz_view.parquet',onehz),('pilot_rtk_reference.parquet',chosenrefs),('pilot_ephemeris.parquet',list(used.values()))]:put(n,table)
    for row in sensitivity:row['in_pilot']=start<=row['gpst_calendar_ns']<stop
    for row in velqa:row['in_pilot']=start<=row['gpst_calendar_ns']<stop
    save('alignment_table_data.json',dict(raw_pvt_offset_distribution=dist,reference_alignment_sensitivity=sensitivity,position_vs_pvt_velocity_qa=velqa))
    save('extraction_seal_v2.json',dict(utc=now(),files=[dict(path=str(OUT/n),sha256=ex.sha(OUT/n)) for n in ['pilot_interval_lock_v2.json','pilot_raw_gnss.parquet','pilot_1hz_view.parquet','pilot_rtk_reference.parquet','pilot_ephemeris.parquet','pre_alignment_lock.json','reference_propagation_contract.md']]))
    print(json.dumps(clean(dict(offset_stats=offstats,pilot={k:v for k,v in lock.items() if k not in ('row_ephemeris_map','one_hz_epoch_mapping')},propagation_distance_pilot_m=stats([r['propagation_distance_m'] for r in chosenrefs]),velocity_pilot=stats([r['vector_difference_mps'] for r in velqa if r['in_pilot']])))),flush=True)
if __name__=='__main__':main()
