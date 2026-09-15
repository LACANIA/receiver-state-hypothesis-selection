from common import *
from dataclasses import fields
from collections import defaultdict
import pandas as pd
import pyarrow.parquet as pq

def main():
    assert not (O/'RUN_PROTOCOL.json').exists(),'EXISTING_PROTOCOL_NOT_OVERWRITTEN'
    src=check_sources(); old=read(OLD/'decision.json')
    assert old['run_id']=='GVINS_CLOCK_BYPASS_INIT_ASSISTED_20260910T134437Z'
    bindings=[]
    for f in read(P/'cache/scan_complete.json')['files']:
        assert sha(f['path'])==f['sha256'],f['path'];bindings.append(f)
    v2seal={Path(f['path']).resolve():f['sha256'] for f in read(V2/'final_output_seal.json')['files']}
    for name in ['all_reference_alignment.parquet','pre_alignment_lock.json','reference_propagation_contract.md']:
        fp=V2/name;actual=sha(fp);assert actual==v2seal[fp.resolve()];bindings.append(dict(path=str(fp),sha256=actual))
    for p in [W/'wrapper.py',W/'candidate_observations.json',OLD/'decision.json',OLD/'verification.json',OLD/'RUN_SEAL.json']:
        bindings.append(dict(path=str(p),sha256=sha(p)))
    save(O/'RUN_PROTOCOL.json',dict(task='GVINS_GPS_L1_SHORT_ARC_IDENTIFIABILITY_AND_INITIALIZATION_REVIEW_V1',
        run_id='GVINS_SHORT_ARC_V1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        source_run=old['run_id'],locked_utc=now(),identity='INITIALIZATION_ASSISTED_REAL_RF_IDENTIFIABILITY_DIAGNOSTIC',
        origin_ns=ORIGIN,gps_week=2134,tow_start=83928.102,windows=WINDOWS,interval='[start,start+T)',
        report_epoch='NOMINAL_END_T; also save native last-row epoch and position; never equate them',
        canonical='Nearest raw epoch to each integer GPST second inside fixed window; ties earlier; no residual input',
        candidate_row_eligibility='GPS CODE_L1C 1575420000; finite native Doppler; causal healthy eph age<7200s; strict wrapper schema. No pseudorange filter or input.',
        support='NATURAL and intersection over EVERY raw epoch of W300; intersection >=4 required for COMMON',
        reference='Existing STATE_PROPAGATION_TO_MEASUREMENT_EPOCH; preserve normal 2/3ms and continuous RTK-fixed contract; reference gaps excluded from matrices, not raw support or fits',
        information_primary='COMMON_REFERENCE_TRAJECTORY_TANGENT: actual per-epoch RTK position and PVT velocity, perturbation dr=dq+(t-T)dv. M0/M1 constrain dv=0; this is not an assertion that true trajectory is static/CV.',
        velocity_sensitivity='Existing 11 point quadratic position derivative, +/-0.5s; same reference rows; no new derivative tuning',
        nuisance={'M0':[],'M1':['b0'],'M2':['vx','vy','vz','b0','bdot'],'M3':['vx','vy','vz','b0'],'M4':['vx','vy','vz']},
        weighting='FROZEN_NATIVE_UNIT_ROW_WEIGHTS; unit sigma=1m/s for dimensional reference proxies. Native solution RSS/dof scaled covariance saved separately.',
        matrix_engine='SVD projection then SVD of projected position Jacobian; machine-precision rank. No normal-inverse cancellation, no altered native rank tolerances.',
        fit_windows=[30,120],fit_support='NATURAL_SUPPORT_ONLY',main_fits=20,core_supplement_fits=18,directional_fits=16,
        planned_top_level_configurations=54,planned_full_solver_calls=54,planned_native_training_solver_calls=54,
        internal_count_identity='Each configuration has native full fit and separate native 70/30 training fit for heldout prediction, total planned 108 solver calls; no scientific restarts.',
        initial_radii=[0,R616,1000,5000,10000],legacy_direction=np.array([500,-300,200])/R616,
        eigenvector_sign='largest absolute ECEF component positive',velocity_initial=[0,0,0],clock_initial=[0,0],prior='B0_none',
        clustering='Complete-linkage position hierarchy, all pairwise distances. No scientifically selected cut or correctness label; distinguish numerical endpoint scatter from separate well-separated minima.',
        no_selector=True,EGSHS_runs=0,no_new_threshold=True,paper_updated=False,sources=bindings,native_sources=src))
    stop=ORIGIN+300_000_000_000
    raw=pq.read_table(P/'cache/full_raw_gnss.parquet',filters=[('gpst_calendar_ns','>=',ORIGIN),('gpst_calendar_ns','<',stop),('system','==','GPS'),('code','==',1),('freqs','==',w.F_L1)]).to_pandas()
    epochs=pq.read_table(P/'cache/raw_epochs.parquet',filters=[('gpst_calendar_ns','>=',ORIGIN),('gpst_calendar_ns','<',stop)]).to_pandas()
    ts=epochs.gpst_calendar_ns.to_numpy(np.int64);assert np.all(np.diff(ts)>0)
    eps=defaultdict(list)
    for e in read(P/'cache/all_ephemerides.json'):
        if e['topic'].endswith('/ephem') and 1<=e['sat']<=32:eps[e['sat']].append(e)
    ephfields={f.name for f in fields(w.Ephemeris)}
    cleanrows=[];lineage=[];satsets={int(t):set() for t in ts};excluded=[]
    for r in raw.to_dict('records'):
        candidates=[e for e in eps[r['sat']] if e['health']==0 and e['bag_record_ns']<=r['bag_record_ns'] and abs(e['toe_ns']-r['gpst_calendar_ns'])<7200_000_000_000]
        if not r['doppler_numeric_available'] or not np.isfinite(r['dopp']) or not candidates:
            excluded.append(dict(row_id=r['row_id'],reason='DOPPLER_OR_EPHEMERIS_UNAVAILABLE'));continue
        e=min(candidates,key=lambda e:(abs(e['toe_ns']-r['gpst_calendar_ns']),-e['bag_record_ns'],e['ephemeris_record_id']))
        rec=dict(receive_gpst_ns=int(r['gpst_calendar_ns']),satellite=int(r['sat']),system='GPS',signal='CODE_L1C',frequency_hz=w.F_L1,doppler_hz=r['dopp'],ephemeris={k:e[k] for k in ephfields})
        try: w.observation_from_mapping(rec)
        except (ValueError,TypeError) as ex:excluded.append(dict(row_id=r['row_id'],reason=str(ex)));continue
        satsets[int(r['gpst_calendar_ns'])].add(int(r['sat']))
        cleanrows.append(rec);lineage.append(dict(row_id=r['row_id'],gpst_ns=int(r['gpst_calendar_ns']),satellite=int(r['sat']),ephemeris_id=e['ephemeris_record_id'],causal_bag_ns=e['bag_record_ns'],raw_bag_ns=r['bag_record_ns'],sigma_y_mps=r['range_rate_std_mps'],native_status=r['status']))
    common=sorted(set.intersection(*satsets.values()))
    cmap={}
    for sec in range((ORIGIN+999999999)//10**9,(stop-1)//10**9+1):
        chosen=int(min(ts,key=lambda t:(abs(int(t)-sec*10**9),int(t))));assert chosen not in cmap
        cmap[chosen]=sec*10**9
    selected=[(r,l) for r,l in zip(cleanrows,lineage) if r['receive_gpst_ns'] in cmap]
    selected.sort(key=lambda q:(q[0]['receive_gpst_ns'],q[0]['satellite']))
    records=[x[0] for x in selected];lines=[dict(x[1],canonical_ns=cmap[x[0]['receive_gpst_ns']],offset_s=(x[0]['receive_gpst_ns']-cmap[x[0]['receive_gpst_ns']])/1e9) for x in selected]
    assert len({(r['receive_gpst_ns'],r['satellite']) for r in records})==len(records)
    oldobs=read(W/'candidate_observations.json')
    new30=[r for r in records if r['receive_gpst_ns']<ORIGIN+30_000_000_000]
    assert new30==oldobs,'W30_NATIVE_INPUT_IDENTITY_CONFLICT'
    refs=pq.read_table(V2/'all_reference_alignment.parquet',filters=[('gpst_calendar_ns','>=',ORIGIN),('gpst_calendar_ns','<=',stop)]).to_pandas()
    refs=refs.set_index('gpst_calendar_ns',drop=False)
    ref0=refs.loc[ORIGIN].to_dict();assert ref0['reference_eligible']
    p0=vec(ref0,'primary_p')
    endpoints={str(T):refs.loc[ORIGIN+T*10**9].to_dict() for T in WINDOWS}
    for T,r in endpoints.items(): assert r['reference_eligible'],('REPORT_EPOCH_REFERENCE_UNAVAILABLE',T)
    reference=[]
    for t in cmap:
        r=refs.loc[t].to_dict();reference.append(r)
    save(O/'candidate_observations.json',records)
    save(O/'row_lineage.json',lines)
    save(O/'reference_qa_only.json',dict(reference=reference,origin=ref0,endpoints=endpoints))
    save(O/'support_lock.json',dict(locked_utc=now(),common_satellites=common,status='AVAILABLE' if len(common)>=4 else 'COMMON_SUPPORT_UNAVAILABLE',native_epochs=len(ts),canonical_epochs=len(cmap),excluded=excluded,native_satellite_counts={str(s):sum(s in z for z in satsets.values()) for s in sorted(set.union(*satsets.values()))},row_sha=sha(O/'candidate_observations.json'),canonical_map=cmap))
    save(O/'initialization_contract.json',dict(locked_utc=now(),P_REF_0=p0,origin_ns=ORIGIN,origin_tow=83928.102,
        propagation_method=ref0['time_alignment_method'],propagation_dt_s=ref0['raw_minus_pvt_s'],
        initialization_identity='AUTHORIZED_RTK_FIXED_ORIGIN_PLUS_PREDEFINED_PERTURBATION',
        legacy_direction=np.array([500,-300,200])/R616,radii=[0,R616,1000,5000,10000],
        v0=[0,0,0],b0=0,bdot=0,old_east_10km='SAVED_REFERENCE_RESULT_NOT_REUSED_AS_NEW_FIT',
        allowed_candidate_prior='ONLY derived initial ECEF vector; no RTK/PVT reference files or velocities'))
    print(json.dumps(dict(rows=len(records),W30=len(new30),common=common,reference_eligible=int(sum(r['reference_eligible'] for r in reference)),epochs=len(reference),P_REF_0=p0.tolist())),flush=True)

if __name__=='__main__':main()
