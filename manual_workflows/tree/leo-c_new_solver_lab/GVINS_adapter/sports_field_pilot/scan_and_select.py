"""Reuse registered extractor types and conversions; inventory then lock first pilot."""
import collections,hashlib,json,math,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rosbags.rosbag1 import Reader
OUT=Path(__file__).resolve().parent;ADAPTER=OUT.parent
sys.path.insert(0,str(ADAPTER));import extract_gnss as ex
BAG=ADAPTER.parent/'GVINS_data/raw/sports_field.bag'
CACHE=OUT/'cache'
def save(name,obj):ex.save_json(OUT/name,obj)
def now():return datetime.now(timezone.utc).isoformat()
def stats(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    if len(a)==0:return {'n':0}
    return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
def gps_l1(row,codes):
    code=codes.get(str(row['code']),{})
    return row['system']=='GPS' and row['freqs']==1575420000.0 and code.get('name','') in {'CODE_L1C','CODE_L1P','CODE_L1W','CODE_L1Y','CODE_L1M','CODE_L1N','CODE_L1S','CODE_L1L'}
def ephem_for(row,eps):
    candidates=[e for e in eps.get(row['sat'],[]) if e['health']==0 and e['bag_record_ns']<=row['bag_record_ns'] and abs(e['toe_ns']-row['gpst_calendar_ns'])<7200*10**9]
    if not candidates:return None
    return min(candidates,key=lambda e:(abs(e['toe_ns']-row['gpst_calendar_ns']),-e['bag_record_ns'],e['ephemeris_record_id']))
def scan():
    d=json.loads((OUT/'download_manifest.json').read_text('utf-8'))
    assert d['status']=='VERIFIED_COMPLETE' and d['actual_sha256']==d['locked_source']['sha256']
    assert BAG.stat().st_size==d['actual_size_bytes']
    CACHE.mkdir(exist_ok=True)
    if (CACHE/'scan_complete.json').exists():return
    if (CACHE/'full_raw_gnss.parquet').exists():raise RuntimeError('UNSEALED_SCAN_EXISTS_REQUIRES_EXPLICIT_RECOVERY')
    store=ex.official_typestore();inv=ex.inventory(BAG,OUT/'bag_inventory.json')
    constellation=collections.Counter();satellite_observations=collections.Counter();satsets=collections.defaultdict(set);signals=collections.Counter();eps=[];pvt=[];rawepochs=[];message_ids=[];optional=collections.Counter();nmsg=0;buffer=[];start=time.monotonic();tick=start
    with Reader(BAG) as reader,pq.ParquetWriter(CACHE/'full_raw_gnss.parquet',ex.RAW_SCHEMA,compression='zstd') as writer:
        selected=ex.checked_connections(reader,store,optional=True)
        for conn,stamp,bytes_ in reader.messages(connections=selected):
            nmsg+=1;msg=store.deserialize_ros1(bytes_,conn.msgtype)
            payload_sha=hashlib.sha256(bytes_).hexdigest()
            message_ids.append({'message_index':nmsg,'bag_record_ns':stamp,'connection_id':conn.id,'topic':conn.topic,'msgtype':conn.msgtype,'md5':conn.digest,'payload_sha256':payload_sha})
            if conn.topic.endswith('/range_meas'):
                for obs in msg.meas:satellite_observations[ex.sat_identity(int(obs.sat))[0]]+=1
                rows=ex.signal_rows(msg,stamp,nmsg)
                epochs=sorted({r['gpst_calendar_ns'] for r in rows})
                rawepochs.extend({'gpst_calendar_ns':epoch,'bag_record_ns':stamp,'message_index':nmsg,'mixed_epoch_message':len(epochs)!=1} for epoch in epochs)
                for row in rows:
                    constellation[row['system']]+=1;satsets[row['system']].add(row['sat']);signals[(row['system'],row['code'],row['freqs'])]+=1
                buffer.extend(rows)
                if len(buffer)>=10000:writer.write_table(pa.Table.from_pylist(buffer,schema=ex.RAW_SCHEMA));buffer=[]
            elif conn.topic.endswith('/receiver_pvt'):pvt.append(ex.pvt_row(msg,stamp))
            elif conn.topic.endswith('/ephem') or conn.topic.endswith('/glo_ephem'):
                e=ex.plain(msg);e.update(bag_record_ns=stamp,topic=conn.topic,ephemeris_record_id=f'EPH_{nmsg:08d}',ros_payload_sha256=payload_sha,connection_id=conn.id,connection_md5=conn.digest)
                e['toe_ns']=ex.gpst_ns(e['toe']['week'],e['toe']['tow'])
                if 'toc' in e:e['toc_ns']=ex.gpst_ns(e['toc']['week'],e['toc']['tow'])
                eps.append(e)
            else:optional[conn.topic]+=1
            if time.monotonic()-tick>15:
                print(json.dumps({'scanned_GNSS_messages':nmsg,'raw_signals':sum(constellation.values()),'elapsed_s':time.monotonic()-start}),flush=True);tick=time.monotonic()
        if buffer:writer.write_table(pa.Table.from_pylist(buffer,schema=ex.RAW_SCHEMA))
    pq.write_table(pa.Table.from_pylist(pvt),CACHE/'all_pvt.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(rawepochs),CACHE/'raw_epochs.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(message_ids),CACHE/'message_record_identity.parquet',compression='zstd')
    ex.save_json(CACHE/'all_ephemerides.json',eps)
    lock=json.loads((OUT/'physics_execution_lock.json').read_text('utf-8'));codes=lock['signal_codes']
    inv.update(body_messages_deserialized=nmsg,raw_signal_rows=sum(constellation.values()),observation_counts_by_constellation=dict(constellation),satellites_by_constellation={k:sorted(v) for k,v in satsets.items()},signal_frequency_distribution=[{'system':k[0],'code':k[1],'frequency_hz':k[2],'count':v,'official_code':codes.get(str(k[1]),{})} for k,v in sorted(signals.items(),key=lambda kv:(kv[0][0],kv[0][1],str(kv[0][2])))],optional_topic_body_counts=dict(optional),scan_completed_utc=now(),scan_elapsed_s=time.monotonic()-start,cache_raw_identity='FULL_SEQUENCE_NATIVE_SIGNALS_NO_RESAMPLING')
    inv['satellite_observation_messages_by_constellation']=dict(satellite_observations)
    inv['observation_count_identity']='observation_counts_by_constellation counts signal rows; satellite_observation_messages_by_constellation counts GnssObsMsg objects before multifrequency flattening'
    save('bag_inventory.json',inv)
    ex.save_json(CACHE/'scan_complete.json',{'completed_utc':now(),'files':[{'path':str(p),'sha256':ex.sha(p)} for p in (CACHE/'full_raw_gnss.parquet',CACHE/'all_pvt.parquet',CACHE/'raw_epochs.parquet',CACHE/'all_ephemerides.json',CACHE/'message_record_identity.parquet')]})

def select():
    lock=json.loads((OUT/'physics_execution_lock.json').read_text('utf-8'));codes=lock['signal_codes']
    eps=json.loads((CACHE/'all_ephemerides.json').read_text('utf-8'));bygps=collections.defaultdict(list)
    for e in eps:
        if e['topic'].endswith('/ephem') and 1<=e['sat']<=32:bygps[e['sat']].append(e)
    pvts=pq.read_table(CACHE/'all_pvt.parquet').to_pylist();pvtmap=collections.defaultdict(list)
    for r in pvts:pvtmap[r['gpst_calendar_ns']].append(r)
    raw=pq.read_table(CACHE/'full_raw_gnss.parquet',filters=[('system','==','GPS'),('freqs','==',1575420000.)]).to_pylist()
    raw=[r for r in raw if gps_l1(r,codes)];rawmap=collections.defaultdict(list)
    for r in raw:rawmap[r['gpst_calendar_ns']].append(r)
    epochs=pq.read_table(CACHE/'raw_epochs.parquet').to_pylist();times=sorted(set(r['gpst_calendar_ns'] for r in epochs));mixed={r['gpst_calendar_ns'] for r in epochs if r['mixed_epoch_message']}
    ordered=np.array([r['gpst_calendar_ns'] for r in epochs],dtype=np.int64);pvtimes=np.array(sorted(pvtmap),dtype=np.int64)
    nearest=[];elig=[];reasoncounts=collections.Counter();chosen_refs={};ephchosen={}
    for t in times:
        j=int(np.searchsorted(pvtimes,t));pool=pvtimes[max(0,j-1):min(len(pvtimes),j+1)]
        nearest.append((int(min(pool,key=lambda x:abs(int(x)-t)))-t)/1e9 if len(pool) else np.nan)
        goodrows=[]
        for row in rawmap[t]:
            reason=None
            if not row['doppler_numeric_available']:reason='NONFINITE_OR_NONPOSITIVE_DOPPLER_SIGMA'
            elif row['psr'] is None or row['psr']<=0 or not row['pseudorange_valid_bit']:reason='PSEUDORANGE_UNAVAILABLE'
            else:
                e=ephem_for(row,bygps)
                if e is None:reason='NO_CAUSAL_HEALTHY_EPHEMERIS'
                else:ephchosen[row['row_id']]=e;goodrows.append(row)
            if reason:reasoncounts[reason]+=1
        refs=pvtmap.get(t,[])
        goodref=[r for r in refs if r['posthoc_fixed_eligible'] and all(r[k] is not None and math.isfinite(r[k]) for k in ('ecef_px','ecef_py','ecef_pz','ecef_vx','ecef_vy','ecef_vz'))]
        # Duplicate references with different physical records cannot be silently resolved.
        ref_ok=len(refs)==1 and len(goodref)==1
        if ref_ok:chosen_refs[t]=goodref[0]
        reason='OK' if ref_ok and len({r['sat'] for r in goodrows})>=4 and t not in mixed else ('REFERENCE_EXACT_FIXED_MISSING' if not ref_ok else 'GPS_L1_CAUSAL_COUNT_OR_EPOCH_IDENTITY')
        elig.append({'gpst_calendar_ns':t,'gps_l1_rows':len(rawmap[t]),'gps_l1_causal_usable_satellites':len({r['sat'] for r in goodrows}),'pvt_exact_matches':len(refs),'fixed_reference_eligible':ref_ok,'eligible':reason=='OK','reason':reason})
    pq.write_table(pa.Table.from_pylist(elig),OUT/'epoch_data_eligibility.parquet',compression='zstd')
    fix_segments=[];cur=[]
    for r in sorted(pvts,key=lambda r:r['gpst_calendar_ns']):
        t=r['gpst_calendar_ns']
        if r['posthoc_fixed_eligible']:
            if cur and (t-cur[-1]>200000000 or t<=cur[-1]):fix_segments.append(cur);cur=[]
            cur.append(t)
        elif cur:fix_segments.append(cur);cur=[]
    if cur:fix_segments.append(cur)
    fixspan=[(x[-1]-x[0])/1e9 for x in fix_segments]
    inv=json.loads((OUT/'bag_inventory.json').read_text('utf-8'))
    inv.update(gnss_gpst_start_ns=times[0] if times else None,gnss_gpst_end_ns=times[-1] if times else None,gnss_epoch_count=len(times),gnss_measurement_interval_s=stats(np.diff(times)/1e9),gnss_nonincreasing_epoch_count=int(np.sum(np.diff(ordered)<=0)),raw_to_nearest_pvt_signed_seconds=stats(nearest),raw_pvt_exact_epoch_count=sum(t in pvtmap for t in times),pvt_count=len(pvts),valid_fix_count=sum(r['valid_fix'] for r in pvts),carr_soln_distribution=dict(collections.Counter(r['carr_soln'] for r in pvts)),rtk_fixed_records=sum(r['posthoc_fixed_eligible'] for r in pvts),rtk_fixed_fraction=sum(r['posthoc_fixed_eligible'] for r in pvts)/len(pvts) if pvts else None,rtk_fixed_duration_between_consecutive_samples_s=sum(fixspan),rtk_fixed_segment_duration_s=stats(fixspan),rtk_fixed_segments=[{'start_gpst_ns':x[0],'end_gpst_ns':x[-1],'sample_count':len(x),'sample_span_s':(x[-1]-x[0])/1e9} for x in fix_segments],gps_l1_row_exclusions=dict(reasoncounts),eligible_epoch_count=sum(r['eligible'] for r in elig),epoch_reasons=dict(collections.Counter(r['reason'] for r in elig)))
    inv.update(raw_to_nearest_pvt_absolute_seconds=stats(np.abs(nearest)),raw_bag_minus_gpst_calendar_seconds=stats([(r['bag_record_ns']-r['gpst_calendar_ns'])/1e9 for r in epochs]),pvt_bag_minus_gpst_calendar_seconds=stats([(r['bag_record_ns']-r['gpst_calendar_ns'])/1e9 for r in pvts]),pvt_fix_type_distribution=dict(collections.Counter(r['fix_type'] for r in pvts)),rtk_fixed_non_3d_fix_type_count=sum(r['posthoc_fixed_eligible'] and r['fix_type']!=3 for r in pvts),gpst_calendar_not_utc=True)
    eligible_spans=[];good_segment=[]
    for entry in elig:
        t=entry['gpst_calendar_ns']
        if good_segment and (not entry['eligible'] or t-good_segment[-1]>200000000):
            eligible_spans.append((good_segment[-1]-good_segment[0])/1e9);good_segment=[]
        if entry['eligible']:good_segment.append(t)
    if good_segment:eligible_spans.append((good_segment[-1]-good_segment[0])/1e9)
    inv['complete_data_eligible_segment_span_s']=stats(eligible_spans)
    save('bag_inventory.json',inv)
    times=np.array(times,dtype=np.int64);ok=np.array([r['eligible'] for r in elig]);badprefix=np.r_[0,np.cumsum(~ok)];chosen=None
    for i,t in enumerate(times):
        j=int(np.searchsorted(times,t+30_000_000_000))
        if j<=i or times[j-1]-t<29_900_000_000 or badprefix[j]!=badprefix[i]:continue
        if np.any(np.diff(times[i:j])>200000000):continue
        chosen=(int(t),int(t+30_000_000_000),i,j);break
    if chosen is None:
        save('pilot_interval_lock.json',{'locked_utc':now(),'status':'NO_ELIGIBLE_30S_PILOT_WINDOW','interval':None,'quality_rules_unchanged':True,'coverage_source':'bag_inventory.json','physics_execution_lock_sha256':ex.sha(OUT/'physics_execution_lock.json')})
        print('NO_ELIGIBLE_30S_PILOT_WINDOW',flush=True);return False
    start,stop,i,j=chosen
    allrows=pq.read_table(CACHE/'full_raw_gnss.parquet',filters=[('gpst_calendar_ns','>=',start),('gpst_calendar_ns','<',stop)]).to_pylist()
    primary=[r for r in allrows if gps_l1(r,codes)];refs=[chosen_refs[int(t)] for t in times[i:j]]
    used={ephchosen[r['row_id']]['ephemeris_record_id']:ephchosen[r['row_id']] for r in primary if r['row_id'] in ephchosen}
    rowmaps=[{'row_id':r['row_id'],'ephemeris_record_id':ephchosen[r['row_id']]['ephemeris_record_id'] if r['row_id'] in ephchosen else None} for r in primary]
    integers=range((start+999999999)//1000000000,(stop-1)//1000000000+1);selected1hz={}
    for second in integers:
        selected=int(min(times[i:j],key=lambda t:(abs(int(t)-second*10**9),int(t))))
        if selected in selected1hz:raise RuntimeError('DUPLICATE_ONE_HZ_EPOCH')
        selected1hz[selected]=second*10**9
    onehz=[dict(r,canonical_gpst_ns=selected1hz[r['gpst_calendar_ns']],canonical_time_offset_s=(r['gpst_calendar_ns']-selected1hz[r['gpst_calendar_ns']])/1e9) for r in allrows if r['gpst_calendar_ns'] in selected1hz]
    interval={'status':'LOCKED_BEFORE_FORWARD_RESIDUALS','locked_utc':now(),'start_gpst_ns':start,'end_gpst_ns_exclusive':stop,'start_gps_week':primary[0]['gps_week'],'start_gps_tow_s':primary[0]['gps_tow_s'],'end_gps_tow_s':primary[0]['gps_tow_s']+30,'bag_start_ns':min(r['bag_record_ns'] for r in allrows),'bag_end_ns':max(r['bag_record_ns'] for r in allrows),'satellites':sorted({r['sat'] for r in primary}),'native_epochs':j-i,'native_all_signal_rows':len(allrows),'gps_l1_signal_rows':len(primary),'gps_l1_forward_eligible_rows':sum(r['row_id'] in ephchosen for r in primary),'rtk_reference_count':len(refs),'rtk_fixed_fraction':1.,'ephemeris_ids':sorted(used),'row_ephemeris_map':rowmaps,'raw_row_ids':[r['row_id'] for r in allrows],'one_hz_epoch_mapping':selected1hz,'one_hz_all_signal_rows':len(onehz),'selection_reason':'FIRST_CHRONOLOGICAL_INTERVAL_MEETING_PRELOCKED_DATA_ONLY_CONTRACT','residuals_accessed_for_selection':False,'physics_execution_lock_sha256':ex.sha(OUT/'physics_execution_lock.json'),'bag_sha256':json.loads((OUT/'download_manifest.json').read_text('utf-8'))['actual_sha256']}
    end_since_gps_epoch_ns=stop-ex.GPS_EPOCH_UNIX_S*10**9
    interval['end_gps_week']=end_since_gps_epoch_ns//(604800*10**9)
    interval['end_gps_tow_s']=(end_since_gps_epoch_ns%(604800*10**9))/1e9
    interval['scan_and_selection_source_sha256']=ex.sha(OUT/'scan_and_select.py')
    save('pilot_interval_lock.json',interval)
    pq.write_table(pa.Table.from_pylist(allrows,schema=ex.RAW_SCHEMA),OUT/'pilot_raw_gnss.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(onehz),OUT/'pilot_1hz_view.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(refs),OUT/'pilot_rtk_reference.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(list(used.values())),OUT/'pilot_ephemeris.parquet',compression='zstd')
    save('extraction_seal.json',{'utc':now(),'files':[{'path':str(OUT/n),'sha256':ex.sha(OUT/n)} for n in ('pilot_interval_lock.json','pilot_raw_gnss.parquet','pilot_1hz_view.parquet','pilot_rtk_reference.parquet','pilot_ephemeris.parquet')]})
    print(json.dumps({k:v for k,v in interval.items() if k not in ('raw_row_ids','row_ephemeris_map','one_hz_epoch_mapping')}),flush=True);return True
if __name__=='__main__':scan();select()
