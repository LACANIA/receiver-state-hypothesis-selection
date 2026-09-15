"""Independent physical QA at RTK reference; only common clock offset is fitted."""
import collections,json,math,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from gps_physics import C,OMEGA,STEPS,orbit,svdt,finite_difference,earth_rotation_forms,corrected_distance,rotation,jz
OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT.parent));import extract_gnss as ex
from scan_and_select import gps_l1
def now():return datetime.now(timezone.utc).isoformat()
def save(n,v):ex.save_json(OUT/n,v)
def clean(obj):
    if isinstance(obj,dict):return {str(k):(str(v) if str(k).endswith('_ns') and isinstance(v,(int,np.integer)) else clean(v)) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.generic):return clean(obj.item())
    if isinstance(obj,float) and not math.isfinite(obj):return None
    return obj
def summary(g):
    r=g['residual_mps'].to_numpy(dtype=float);n=len(r)
    return {'valid_observations':n,'satellites':int(g.sat.nunique()),'epochs':int(g.gpst_calendar_ns.nunique()),'median_residual_mps':float(np.median(r)),'median_absolute_residual_mps':float(np.median(np.abs(r))),'rms_residual_mps':float(np.sqrt(np.mean(r*r))),'p95_absolute_residual_mps':float(np.quantile(np.abs(r),.95)),'max_absolute_residual_mps':float(np.max(np.abs(r))),
      'median_before_common_offset_mps':float(g.pre_offset_residual_mps.median()),'rms_before_common_offset_mps':float(np.sqrt(np.mean(g.pre_offset_residual_mps**2))),
      'median_sigma_y_mps':float(g.range_rate_std_mps.median()),'abs_residual_cn0_spearman':float(spearmanr(np.abs(r),g.CN0,nan_policy='omit').statistic) if g.CN0.nunique()>1 else None,'abs_residual_elevation_spearman':float(spearmanr(np.abs(r),g.elevation_deg,nan_policy='omit').statistic) if g.elevation_deg.nunique()>1 else None}
def run():
    lock=json.loads((OUT/'pilot_interval_lock.json').read_text('utf-8'));physics=json.loads((OUT/'physics_execution_lock.json').read_text('utf-8'))
    assert lock['status']=='LOCKED_BEFORE_FORWARD_RESIDUALS'
    assert ex.sha(OUT/'gps_physics.py')==physics['physical_code_sha256'],'PHYSICS_SOURCE_CHANGED'
    seal=json.loads((OUT/'extraction_seal.json').read_text('utf-8'))
    assert all(ex.sha(x['path'])==x['sha256'] for x in seal['files'])
    if (OUT/'pilot_forward_residuals.parquet').exists():raise RuntimeError('EXISTING_FORWARD_OUTPUT_NO_OVERWRITE')
    save('forward_execution_seal.json',{'utc':now(),'sources':[{'path':str(OUT/n),'sha256':ex.sha(OUT/n)} for n in ('forward_qa.py','gps_physics.py','physical_definitions.md','physics_execution_lock.json','pilot_interval_lock.json','extraction_seal.json','download_manifest.json')],'receiver_reference_role':'RTK_FIXED_POSTHOC_PHYSICAL_QA_REFERENCE','candidate_runs':0,'egshs_runs':0,'positioning_experiments':0})
    raw=pq.read_table(OUT/'pilot_raw_gnss.parquet').to_pylist();raw=[r for r in raw if gps_l1(r,physics['signal_codes'])]
    refs={r['gpst_calendar_ns']:r for r in pq.read_table(OUT/'pilot_rtk_reference.parquet').to_pylist()}
    eps={e['ephemeris_record_id']:e for e in pq.read_table(OUT/'pilot_ephemeris.parquet').to_pylist()}
    roweph={r['row_id']:r['ephemeris_record_id'] for r in lock['row_ephemeris_map']}
    hzepochs={int(k) for k in lock['one_hz_epoch_mapping']}
    records=[];states=[];velocity=[];sagnac=[];unavailable=[];tested_eph=set();start=time.monotonic()
    for row in raw:
        eid=roweph[row['row_id']]
        if eid is None:unavailable.append({'row_id':row['row_id'],'reason':'NO_PRELOCKED_USABLE_EPHEMERIS_OR_OBSERVABLE'});continue
        e=eps[eid];ref=refs[row['gpst_calendar_ns']]
        r=np.array([ref['ecef_p'+a] for a in 'xyz']);vr=np.array([ref['ecef_v'+a] for a in 'xyz'])
        tr=(row['gpst_calendar_ns']-e['toe_ns'])/1e9
        tf=row['psr']/C;tx0=tr-tf;clock0=svdt(e,tx0);tx=tx0-clock0
        s,va,vc,clock,drift=orbit(e,tx)
        fd={h:finite_difference(e,tx,h) for h in STEPS};vf,cf=fd[.1]
        u=(s-r)/np.linalg.norm(s-r);rad=s/np.linalg.norm(s)
        geom=earth_rotation_forms(s,vf,r,vr);geoman=earth_rotation_forms(s,va,r,vr)
        lat=math.radians(ref['latitude']);lon=math.radians(ref['longitude']);up=np.array([math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)])
        el=math.degrees(math.asin(float(np.clip(geom['rotated_los']@up,-1,1))))
        base={'row_id':row['row_id'],'gpst_calendar_ns':row['gpst_calendar_ns'],'gps_week':row['gps_week'],'gps_tow_s':row['gps_tow_s'],'sat':row['sat'],'ephemeris_record_id':eid}
        dt_clock=tx+(e['toe_ns']-e['toc_ns'])/1e9
        poly_drift=e['af1']+2*e['af2']*dt_clock
        for h,(vv,cc) in fd.items():
            diff=va-vv
            velocity.append({**base,'step_s':h,'official_minus_fd_3d_mps':float(np.linalg.norm(diff)),'official_minus_fd_radial_mps':float(rad@diff),'official_minus_fd_los_mps':float(u@diff),'official_minus_fd_predicted_doppler_mps':geoman['form_a_retarded_mps']-earth_rotation_forms(s,vv,r,vr)['form_a_retarded_mps'],
              'corrected_analytic_minus_fd_3d_mps':float(np.linalg.norm(vc-vv)),'fd_minus_h0p1_3d_mps':float(np.linalg.norm(vv-vf)),'analytic_minus_fd_clock_drift_s_per_s':drift-cc,'clock_difference_range_rate_mps':C*(drift-cc),'analytic_polynomial_clock_drift_s_per_s':poly_drift,'relativistic_clock_drift_s_per_s':drift-poly_drift,'polynomial_fd_clock_drift_s_per_s':((e['af0']+e['af1']*(dt_clock+h)+e['af2']*(dt_clock+h)**2)-(e['af0']+e['af1']*(dt_clock-h)+e['af2']*(dt_clock-h)**2))/(2*h),'official_vz_mps':float(va[2]),'corrected_derivative_vz_mps':float(vc[2]),'source_expression':'official: ydot*sin(i)+ydot*idot*cos(i); chain-rule: ydot*sin(i)+y*idot*cos(i)'})
        sag={**base,**{k:v for k,v in geom.items() if not isinstance(v,np.ndarray)},'a_minus_b_epoch_mps':geom['form_a_epoch_mps']-geom['form_b_epoch_mps'],'a_minus_b_retarded_mps':geom['form_a_retarded_mps']-geom['form_b_retarded_mps'],'naive_minus_full_retarded_mps':geom['naive_rotated_mps']-geom['form_a_retarded_mps'],'naive_minus_epoch_mps':geom['naive_rotated_mps']-geom['form_a_epoch_mps']}
        if eid not in tested_eph:
            sag['implicit_distance_fd_h0p1_mps']=(corrected_distance(e,tx,r,vr,.1)-corrected_distance(e,tx,r,vr,-.1))/.2
            sag['implicit_distance_fd_minus_form_a_mps']=sag['implicit_distance_fd_h0p1_mps']-geom['form_a_retarded_mps'];tested_eph.add(eid)
        else:sag['implicit_distance_fd_h0p1_mps']=None;sag['implicit_distance_fd_minus_form_a_mps']=None
        sagnac.append(sag)
        sr,ar,cr,cbr,cdr=orbit(e,tr);vfr,cfr=finite_difference(e,tr,.1);ur=(sr-r)/np.linalg.norm(sr-r)
        ga=geom['form_a_retarded_mps'];g_analytic=geoman['form_a_retarded_mps']
        v_rotated=rotation(geom['geometric_flight_s'])@vf
        v_effective=v_rotated*(1-ga/C)-OMEGA*jz(geom['rotated_satellite_position'])*ga/C
        sag['effective_velocity_reconstruction_difference_mps']=float(geom['rotated_los']@(v_effective-vr))-ga
        predictions={
          'A_FULL_FD':ga-C*cf*(1-ga/C),
          'B_WRONG_SIGN':ga-C*cf*(1-ga/C),
          'C_RECEIVE_TIME_NO_EARTH_ROTATION':float(ur@(vfr-vr))-C*cfr,
          'D_FULL_OFFICIAL_ANALYTIC':g_analytic-C*drift*(1-g_analytic/C),
          'E_GVINS_EPOCH_CONVENTION_FD':geom['form_b_epoch_mps']-C*cf,
          'F_FULL_NO_SATELLITE_CLOCK':ga}
        for variant,pred in predictions.items():
            y=row['range_rate_raw_mps']*(-1 if variant=='B_WRONG_SIGN' else 1)
            records.append({**row,**base,'variant':variant,'view_1hz':row['gpst_calendar_ns'] in hzepochs,'observed_y_mps':y,'predicted_geometry_and_sat_clock_mps':pred,'pre_offset_residual_mps':y-pred,'elevation_deg':el,'psr_std_le2_and_dopp_std_le2':bool(row['psr_std'] is not None and row['psr_std']<=2 and row['dopp_std']<=2),'quality_set':'ALL_FINITE_CAUSAL_EPHEMERIS_RTK_FIXED','reference_vel_acc_mps':ref['vel_acc'],'reference_h_acc_m':ref['h_acc'],'raw_pvt_time_difference_s':0.})
        states.append({**base,'bag_record_ns':row['bag_record_ns'],'receive_seconds_from_toe':tr,'transmit_seconds_from_toe':tx,'pseudorange_flight_s':tf,'preliminary_satellite_clock_bias_s':clock0,'transmit_gpst_ns_rounded':int(row['gpst_calendar_ns']-round((tf+clock0)*1e9)),'ephemeris_age_s':tr,'clock_polynomial_seconds_from_toc':tx+(e['toe_ns']-e['toc_ns'])/1e9,'satellite_clock_bias_s':clock,'satellite_clock_drift_fd_s_per_s':cf,'satellite_clock_drift_analytic_s_per_s':drift,
          **{f'receive_frame_sat_position_{a}_m':float(x) for a,x in zip('xyz',geom['rotated_satellite_position'])},**{f'receive_frame_rotated_velocity_{a}_mps':float(x) for a,x in zip('xyz',v_rotated)},**{f'rtk_conditioned_effective_velocity_{a}_mps':float(x) for a,x in zip('xyz',v_effective)},'rotated_and_effective_state_role':'RTK_CONDITIONED_PHYSICAL_QA_ONLY_NOT_EXOGENOUS_CANDIDATE_INPUT',
          **{f'sat_position_{a}_m':float(x) for a,x in zip('xyz',s)},**{f'sat_velocity_fd_{a}_mps':float(x) for a,x in zip('xyz',vf)},**{f'sat_velocity_official_{a}_mps':float(x) for a,x in zip('xyz',va)},**{f'sat_velocity_chain_rule_{a}_mps':float(x) for a,x in zip('xyz',vc)},'elevation_deg':el,'geometric_range_rate_mps':ga,'sat_clock_range_rate_correction_mps':-C*cf*(1-ga/C)})
    df=pd.DataFrame(records);clocks=[]
    for (variant,t),g in df.groupby(['variant','gpst_calendar_ns'],sort=False):
        w=1/g.range_rate_std_mps.to_numpy()**2;b=float(np.dot(w,g.pre_offset_residual_mps)/sum(w))
        df.loc[g.index,'receiver_clock_rate_mps']=b;df.loc[g.index,'residual_mps']=g.pre_offset_residual_mps-b
        clocks.append({'variant':variant,'gpst_calendar_ns':int(t),'receiver_clock_rate_mps':b,'receiver_fractional_clock_rate_s_per_s':b/C,'sigma_only_common_offset_se_mps':float(1/math.sqrt(sum(w))),'satellites':int(g.sat.nunique()),'observations':len(g),'weighted_residual_sum':float(np.dot(w,g.pre_offset_residual_mps-b))})
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),OUT/'pilot_forward_residuals.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(states),OUT/'pilot_satellite_states.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(clocks),OUT/'pilot_receiver_clock_rate.parquet',compression='zstd')
    summaries=[];persat=[]
    for view,viewdf in [('NATIVE',df),('CANONICAL_1HZ',df[df.view_1hz])]:
        for variant,g in viewdf.groupby('variant'):
            summaries.append({'view':view,'variant':variant,'quality_set':'ALL_FINITE_CAUSAL_EPHEMERIS_RTK_FIXED',**summary(g)})
            for sat,gg in g.groupby('sat'):persat.append({'view':view,'variant':variant,'sat':int(sat),**summary(gg)})
    pq.write_table(pa.Table.from_pylist(clean(persat)),OUT/'per_satellite_forward_summary.parquet',compression='zstd')
    # Source-filter-compatible numeric subview, NOT claimed as full GVINS acceptance:
    # full native preprocessing also depends on tracking history and estimator readiness.
    filtered=[]
    for (variant,t),g in df[df.psr_std_le2_and_dopp_std_le2].groupby(['variant','gpst_calendar_ns']):
        if len(g)<2:continue
        g=g.copy();w=1/g.range_rate_std_mps**2;b=float(np.dot(w,g.pre_offset_residual_mps)/sum(w));g['receiver_clock_rate_mps']=b;g['residual_mps']=g.pre_offset_residual_mps-b;filtered.append(g)
    if filtered:
        fdf=pd.concat(filtered)
        for variant,g in fdf.groupby('variant'):summaries.append({'view':'NATIVE','variant':variant,'quality_set':'OFFICIAL_STD_FILTER_ONLY_NOT_FULL_NATIVE_ACCEPTANCE',**summary(g)})
        pq.write_table(pa.Table.from_pandas(fdf,preserve_index=False),OUT/'official_std_only_residuals.parquet',compression='zstd')
    sign=[r for r in summaries if r['variant'] in ('A_FULL_FD','B_WRONG_SIGN')]
    save('qa_table_data.json',clean({'satellite_velocity_qa':velocity,'sagnac_representation_qa':sagnac,'doppler_sign_qa':sign,'forward_qa_summary':summaries}))
    qa={'completed_utc':now(),'forward_elapsed_s':time.monotonic()-start,'primary_rows':len(states),'primary_satellites':len({r['sat'] for r in states}),'primary_epochs':len({r['gpst_calendar_ns'] for r in states}),'unavailable_rows':unavailable,'all_ephemerides_tested':sorted(tested_eph),'physical_variants':list(predictions),'summary':clean(summaries),'counts':{'m0_m14_runs':0,'egshs_runs':0,'new_positioning_experiments':0,'common_clock_offsets_primary':len({r['gpst_calendar_ns'] for r in states})},'maximum_chain_rule_minus_fd_h0p1_mps':max(r['corrected_analytic_minus_fd_3d_mps'] for r in velocity if r['step_s']==.1),'maximum_sagnac_forms_retarded_difference_mps':max(abs(r['a_minus_b_retarded_mps']) for r in sagnac),'maximum_naive_rotate_difference_mps':max(abs(r['naive_minus_full_retarded_mps']) for r in sagnac),'maximum_implicit_range_fd_difference_mps':max(abs(r['implicit_distance_fd_minus_form_a_mps']) for r in sagnac if r['implicit_distance_fd_minus_form_a_mps'] is not None)}
    save('forward_qa_execution.json',clean(qa));print(json.dumps(clean(qa)),flush=True)
if __name__=='__main__':run()
