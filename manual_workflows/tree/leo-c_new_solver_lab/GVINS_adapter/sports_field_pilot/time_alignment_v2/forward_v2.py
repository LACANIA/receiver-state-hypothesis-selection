"""Real GPS L1 forward QA with prelocked references. No positioning engine."""
from pathlib import Path
import sys,json,math,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
OUT=Path(__file__).resolve().parent;SRC=OUT.parent
sys.path.insert(0,str(OUT));from prepare_v2 import ex,now,save,clean,put,stats,vector
sys.path.insert(0,str(SRC))
from gps_physics import C,OMEGA,STEPS,orbit,svdt,finite_difference,earth_rotation_forms,corrected_distance,rotation,jz
from forward_qa import summary
def main():
    assert not (OUT/'pilot_forward_residuals.parquet').exists(),'NO_OVERWRITE_FORWARD'
    lock=json.loads((OUT/'pilot_interval_lock_v2.json').read_text());seal=json.loads((OUT/'extraction_seal_v2.json').read_text())
    assert lock['status']=='LOCKED_BEFORE_FORWARD_RESIDUALS'
    for f in seal['files']:assert ex.sha(f['path'])==f['sha256'],f['path']
    old=json.loads((SRC/'physics_execution_lock.json').read_text());assert ex.sha(SRC/'gps_physics.py')==old['physical_code_sha256']
    # Verify the Hermite basis and derivative for a constant acceleration fixture.
    t=.002;h=.1;p0=np.array([1.,2.,3.]);v0=np.array([.5,-1.,2.]);acc=np.array([.3,.2,-.1]);p1=p0+h*v0+.5*h*h*acc;v1=v0+h*acc;u=t/h;dp=p1-p0
    ph=p0+(-2*u**3+3*u*u)*dp+(u**3-2*u*u+u)*h*v0+(u**3-u*u)*h*v1
    vh=(-6*u*u+6*u)/h*dp+(3*u*u-4*u+1)*v0+(3*u*u-2*u)*v1
    assert np.linalg.norm(ph-(p0+t*v0+.5*t*t*acc))<1e-12 and np.linalg.norm(vh-(v0+t*acc))<1e-12
    save('pre_forward_seal_v2.json',dict(utc=now(),files=[dict(path=str(p),sha256=ex.sha(p)) for p in [Path(__file__),SRC/'gps_physics.py',SRC/'forward_qa.py',OUT/'prepare_v2.py',OUT/'reference_propagation_contract.md',OUT/'RAWX_PVT_TIME_SEMANTICS.md',OUT/'extraction_seal_v2.json']],primary='PRIMARY_PROPAGATED',velocity='FD_H0P1_CONDITIONAL_CHAIN_RULE_CHECK',physical_reference_time='RECORDED_RAW_RECEIVER_LOCAL_GPS_ALIGNED_LABEL',hermite_fixture_checked=True,positioning_experiments=0))
    raw=pq.read_table(OUT/'pilot_raw_gnss.parquet').to_pylist();raw=[r for r in raw if r['system']=='GPS' and r['code']==1 and r['freqs']==1575420000.]
    refs={r['gpst_calendar_ns']:r for r in pq.read_table(OUT/'pilot_rtk_reference.parquet').to_pylist()};eps={e['ephemeris_record_id']:e for e in pq.read_table(OUT/'pilot_ephemeris.parquet').to_pylist()};roweph={r['row_id']:r['ephemeris_record_id'] for r in lock['row_ephemeris_map']};hz={int(k) for k in lock['one_hz_epoch_mapping']}
    records=[];states=[];vel=[];sags=[];align=[];excluded=[];tested=set();started=time.monotonic()
    for row in raw:
        eid=roweph[row['row_id']]
        if eid is None:excluded.append(dict(row_id=row['row_id'],sat=row['sat'],gpst_calendar_ns=row['gpst_calendar_ns'],reason='NO_PRELOCKED_CAUSAL_HEALTHY_EPHEMERIS_OR_VALID_OBSERVABLE'));continue
        e=eps[eid];ref=refs[row['gpst_calendar_ns']];r=vector(ref,'primary_p');vr=vector(ref,'primary_v')
        tr=(row['gpst_calendar_ns']-e['toe_ns'])/1e9;tf=row['psr']/C;clock0=svdt(e,tr-tf);tx=tr-tf-clock0
        s,va,vc,cb,cd=orbit(e,tx);fd={h:finite_difference(e,tx,h) for h in STEPS};vf,cfd=fd[.1]
        # Acceptance tests depend only on differentiating the satellite equations.
        assert np.linalg.norm(vc-vf)<1e-4,'SATELLITE_DERIVATIVE_NOT_CONFIRMED'
        assert np.linalg.norm(fd[.01][0]-vf)<1e-4,'SATELLITE_FD_SMALL_STEP_DISAGREEMENT'
        geom=earth_rotation_forms(s,vf,r,vr);geomoff=earth_rotation_forms(s,va,r,vr);u=(s-r)/np.linalg.norm(s-r);rad=s/np.linalg.norm(s)
        lat,lon=np.deg2rad([ref['latitude'],ref['longitude']]);up=np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]);el=math.degrees(math.asin(np.clip(geom['rotated_los']@up,-1,1)))
        base=dict(row_id=row['row_id'],gpst_calendar_ns=row['gpst_calendar_ns'],gps_week=row['gps_week'],gps_tow_s=row['gps_tow_s'],sat=row['sat'],ephemeris_record_id=eid)
        dt=tx+(e['toe_ns']-e['toc_ns'])/1e9;poly=e['af1']+2*e['af2']*dt
        for h,(vv,cc) in fd.items():
            diff=va-vv;geomh=earth_rotation_forms(s,vv,r,vr)
            vel.append(dict(**base,step_s=h,official_minus_fd_3d_mps=np.linalg.norm(diff),official_minus_fd_radial_mps=rad@diff,official_minus_fd_los_mps=u@diff,official_minus_fd_prediction_mps=geomoff['form_a_retarded_mps']-geomh['form_a_retarded_mps'],independent_analytic_minus_fd_3d_mps=np.linalg.norm(vc-vv),independent_analytic_minus_fd_los_mps=u@(vc-vv),fd_minus_h0p1_3d_mps=np.linalg.norm(vv-vf),analytic_minus_fd_clock_s_per_s=cd-cc,clock_difference_range_rate_mps=C*(cd-cc),polynomial_clock_derivative_s_per_s=poly,polynomial_fd_s_per_s=((e['af0']+e['af1']*(dt+h)+e['af2']*(dt+h)**2)-(e['af0']+e['af1']*(dt-h)+e['af2']*(dt-h)**2))/(2*h),relativistic_clock_derivative_s_per_s=cd-poly,official_vz_mps=va[2],independent_vz_mps=vc[2]))
        sag={**base,**{k:v for k,v in geom.items() if not isinstance(v,np.ndarray)},'a_minus_b_epoch_mps':geom['form_a_epoch_mps']-geom['form_b_epoch_mps'],'a_minus_b_retarded_mps':geom['form_a_retarded_mps']-geom['form_b_retarded_mps'],'simple_rotation_minus_retarded_mps':geom['naive_rotated_mps']-geom['form_a_retarded_mps'],'simple_rotation_minus_epoch_mps':geom['naive_rotated_mps']-geom['form_a_epoch_mps']}
        if eid not in tested:
            sag['light_cone_distance_fd_mps']=(corrected_distance(e,tx,r,vr,.1)-corrected_distance(e,tx,r,vr,-.1))/.2
            sag['distance_fd_minus_retarded_mps']=sag['light_cone_distance_fd_mps']-geom['form_a_retarded_mps'];tested.add(eid)
        else:sag['light_cone_distance_fd_mps']=None;sag['distance_fd_minus_retarded_mps']=None
        sags.append(sag)
        sr,_,_,_,_=orbit(e,tr);vfr,cfr=finite_difference(e,tr,.1);ur=(sr-r)/np.linalg.norm(sr-r)
        ga=geom['form_a_retarded_mps'];go=geomoff['form_a_retarded_mps'];pred=ga-C*cfd*(1-ga/C)
        preds={'A_PRIMARY_PROPAGATED':pred,'B_WRONG_SIGN':pred,'C_RECEIVE_TIME_NO_ROTATION':float(ur@(vfr-vr))-C*cfr,'D_OFFICIAL_ANALYTIC':go-C*cd*(1-go/C),'E_GVINS_EPOCH_CONVENTION':geom['form_b_epoch_mps']-C*cfd,'F_NO_SATELLITE_CLOCK':ga,'G_SIMPLE_ROTATED_STATES':geom['naive_rotated_mps']-C*cfd}
        for key,pp,pv in [('H_BRACKET_HERMITE','hermite_p','hermite_v'),('I_BRACKET_LINEAR','linear_p','linear_v'),('J_POSITION_DERIVED_VELOCITY','primary_p','position_derived_v'),('K_UNPROPAGATED_PVT','pvt_p','pvt_v')]:
            if ref[pv+'x'] is None:continue
            rr=vector(ref,pp);vv=vector(ref,pv);g=earth_rotation_forms(s,vf,rr,vv)['form_a_retarded_mps'];preds[key]=g-C*cfd*(1-g/C)
            align.append(dict(**base,reference_variant=key,reference_position_difference_m=np.linalg.norm(rr-r),reference_velocity_difference_mps=np.linalg.norm(vv-vr),prediction_minus_primary_mps=preds[key]-pred,raw_sigma_y_mps=row['range_rate_std_mps'],propagation_distance_m=ref['propagation_distance_m'],raw_minus_pvt_s=ref['raw_minus_pvt_s']))
        for variant,prediction in preds.items():
            y=row['range_rate_raw_mps']*(-1 if variant=='B_WRONG_SIGN' else 1)
            records.append({**row,**base,'variant':variant,'view_1hz':row['gpst_calendar_ns'] in hz,'observed_y_mps':y,'predicted_geometry_and_sat_clock_mps':prediction,'pre_offset_residual_mps':y-prediction,'elevation_deg':el,'raw_minus_pvt_s':ref['raw_minus_pvt_s'],'reference_propagation_distance_m':ref['propagation_distance_m'],'psr_std_le2_and_dopp_std_le2':bool(row['psr_std'] is not None and row['psr_std']<=2 and row['dopp_std']<=2),'quality_set':'ALL_FINITE_CAUSAL_EPHEMERIS_FIXED_REFERENCE'})
        states.append(dict(**base,receive_seconds_from_toe=tr,transmit_seconds_from_toe=tx,pseudorange_flight_s=tf,geometric_flight_s=geom['geometric_flight_s'],clock_correction_s=clock0,ephemeris_signed_age_s=tr,ephemeris_absolute_age_s=abs(tr),clock_bias_s=cb,clock_drift_fd_s_per_s=cfd,clock_drift_analytic_s_per_s=cd,clock_polynomial_seconds_from_toc=dt,clock_range_rate_correction_mps=-C*cfd*(1-ga/C),**{'sat_p'+ax:x for ax,x in zip('xyz',s)},**{'sat_fd_v'+ax:x for ax,x in zip('xyz',vf)},**{'sat_official_v'+ax:x for ax,x in zip('xyz',va)},**{'sat_independent_v'+ax:x for ax,x in zip('xyz',vc)}))
    df=pd.DataFrame(records);clocks=[]
    for (variant,t),g in df.groupby(['variant','gpst_calendar_ns'],sort=False):
        w=1/g.range_rate_std_mps.to_numpy()**2;b=float(np.dot(w,g.pre_offset_residual_mps)/sum(w));res=g.pre_offset_residual_mps-b
        df.loc[g.index,'receiver_clock_rate_mps']=b;df.loc[g.index,'residual_mps']=res
        clocks.append(dict(variant=variant,gpst_calendar_ns=int(t),receiver_clock_rate_mps=b,receiver_fractional_clock_rate_s_per_s=b/C,observations=len(g),weighted_residual_mean_mps=float(np.dot(w,res)/sum(w))))
    put('pilot_forward_residuals.parquet',df.to_dict('records'));put('pilot_satellite_states.parquet',states);put('pilot_receiver_clock_rate.parquet',clocks);put('unavailable_gps_l1_rows.parquet',excluded);put('alignment_forward_sensitivity.parquet',align)
    summaries=[];satstats=[]
    for view,part in [('NATIVE',df),('CANONICAL_1HZ',df[df.view_1hz])]:
        for var,g in part.groupby('variant'):
            summaries.append(dict(view=view,variant=var,quality_set='ALL_FINITE_CAUSAL_EPHEMERIS_FIXED_REFERENCE',**summary(g)))
            for sat,sg in g.groupby('sat'):satstats.append(dict(view=view,variant=var,sat=int(sat),**summary(sg)))
    # These rows are a source uncertainty-filter subview, not a new primary rule.
    filtered=[]
    for (var,t),g in df[df.psr_std_le2_and_dopp_std_le2].groupby(['variant','gpst_calendar_ns']):
        if len(g)<2:continue
        g=g.copy();w=1/g.range_rate_std_mps**2;b=float(np.dot(w,g.pre_offset_residual_mps)/sum(w));g['receiver_clock_rate_mps']=b;g['residual_mps']=g.pre_offset_residual_mps-b;filtered.append(g)
    if filtered:
        fdf=pd.concat(filtered);put('official_std_filter_only_residuals.parquet',fdf.to_dict('records'))
        for var,g in fdf.groupby('variant'):summaries.append(dict(view='NATIVE',variant=var,quality_set='OFFICIAL_STD_FILTER_ONLY_NOT_FULL_NATIVE_ACCEPTANCE',**summary(g)))
    put('per_satellite_forward_summary.parquet',satstats)
    tables=json.loads((OUT/'alignment_table_data.json').read_text())
    # Full sequence reference alignment remains in its table; forward sensitivity
    # has its own per-observation file to preserve different statistical units.
    tables.update(satellite_velocity_qa=vel,sagnac_representation_qa=sags,doppler_sign_qa=[r for r in summaries if r['variant'] in ('A_PRIMARY_PROPAGATED','B_WRONG_SIGN')],forward_qa_summary=summaries,alignment_forward_sensitivity=align)
    # Exact integer nanoseconds exported as strings; Parquet keeps int64.
    def csvclean(v):
        if isinstance(v,dict):return {k:str(x) if k.endswith('_ns') and isinstance(x,(int,np.integer)) else csvclean(x) for k,x in v.items()}
        if isinstance(v,list):return [csvclean(x) for x in v]
        return clean(v)
    save('qa_table_data.json',csvclean(tables))
    primary=df[df.variant=='A_PRIMARY_PROPAGATED'];clockdf=pd.DataFrame(clocks);baseclock=clockdf[clockdf.variant=='A_PRIMARY_PROPAGATED'].sort_values('gpst_calendar_ns');adf=pd.DataFrame(align)
    result=dict(completed_utc=now(),elapsed_s=time.monotonic()-started,primary_rows=len(primary),primary_epochs=primary.gpst_calendar_ns.nunique(),primary_satellites=primary.sat.nunique(),excluded_gps_l1_rows=len(excluded),upstream_velocity_formula_discrepancy=True,velocity_official_minus_fd_h0p1=stats([v['official_minus_fd_3d_mps'] for v in vel if v['step_s']==.1]),velocity_independent_minus_fd_h0p1=stats([v['independent_analytic_minus_fd_3d_mps'] for v in vel if v['step_s']==.1]),sagnac_a_minus_b=stats([v['a_minus_b_retarded_mps'] for v in sags]),simple_rotated_minus_retarded=stats([v['simple_rotation_minus_retarded_mps'] for v in sags]),distance_fd_minus_retarded=stats([v['distance_fd_minus_retarded_mps'] for v in sags if v['distance_fd_minus_retarded_mps'] is not None]),receiver_clock_rate_mps=stats(baseclock.receiver_clock_rate_mps),receiver_clock_rate_adjacent_difference_mps=stats(np.diff(baseclock.receiver_clock_rate_mps)),alignment_prediction_difference={v:stats(g.prediction_minus_primary_mps) for v,g in adf.groupby('reference_variant')},summaries=summaries,m0_m14_runs=0,egshs_runs=0,positioning_experiments=0)
    save('forward_qa_execution_v2.json',result);print(json.dumps(clean(result)),flush=True)
if __name__=='__main__':main()
