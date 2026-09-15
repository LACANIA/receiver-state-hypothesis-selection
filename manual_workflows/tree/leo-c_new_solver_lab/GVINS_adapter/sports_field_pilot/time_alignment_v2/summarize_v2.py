"""Summaries and independent identities from saved QA outputs only."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from prepare_v2 import OUT,SRC,stats,save,clean,ex
q=json.loads((OUT/'qa_table_data.json').read_text());f=pq.read_table(OUT/'pilot_forward_residuals.parquet').to_pandas();primary=f[f.variant=='A_PRIMARY_PROPAGATED'].set_index('row_id')
refs=pq.read_table(OUT/'pilot_rtk_reference.parquet').to_pandas();clock=pq.read_table(OUT/'pilot_receiver_clock_rate.parquet').to_pandas();vel=pd.DataFrame(q['satellite_velocity_qa']);sag=pd.DataFrame(q['sagnac_representation_qa']);der=pd.DataFrame(q['position_vs_pvt_velocity_qa']);sens=pd.DataFrame(q['reference_alignment_sensitivity']);align=pq.read_table(OUT/'alignment_forward_sensitivity.parquet').to_pandas()
result={}
result['native_primary']=[x for x in q['forward_qa_summary'] if x['variant']=='A_PRIMARY_PROPAGATED' and x['view']=='NATIVE' and x['quality_set'].startswith('ALL_FINITE')][0]
result['one_hz_primary']=[x for x in q['forward_qa_summary'] if x['variant']=='A_PRIMARY_PROPAGATED' and x['view']=='CANONICAL_1HZ'][0]
result['velocity_by_step']={str(h):{c:stats(g[c]) for c in ['official_minus_fd_3d_mps','official_minus_fd_prediction_mps','independent_analytic_minus_fd_3d_mps','fd_minus_h0p1_3d_mps','clock_difference_range_rate_mps']} for h,g in vel.groupby('step_s')}
result['sagnac']={c:stats(sag[c].dropna()) for c in ['a_minus_b_epoch_mps','a_minus_b_retarded_mps','simple_rotation_minus_retarded_mps','simple_rotation_minus_epoch_mps','distance_fd_minus_retarded_mps']}
result['sagnac_absolute']={c:stats(abs(sag[c].dropna())) for c in ['a_minus_b_epoch_mps','a_minus_b_retarded_mps','simple_rotation_minus_retarded_mps']}
result['propagation_distance_pilot_m']=stats(refs.propagation_distance_m)
result['position_derived_velocity']={label:{c:stats(g[c]) for c in ['vector_difference_mps','signed_speed_difference_mps','direction_difference_deg','pvt_speed_mps']} for label,g in [('ALL_AVAILABLE',der),('PILOT',der[der.in_pilot])]}
result['alignment_reference_pilot']={c:stats(sens[sens.in_pilot][c]) for c in ['hermite_minus_primary_position_m','linear_minus_primary_position_m','hermite_minus_primary_velocity_mps','linear_minus_primary_velocity_mps']}
result['alignment_prediction']={}
for v,g in align.groupby('reference_variant'):
    selected=f[f.variant==v].set_index('row_id').loc[primary.index]
    result['alignment_prediction'][v]=dict(signed=stats(g.prediction_minus_primary_mps),absolute=stats(abs(g.prediction_minus_primary_mps)),relative_to_sigma=stats(abs(g.prediction_minus_primary_mps)/g.raw_sigma_y_mps),post_clock_difference=stats(selected.residual_mps-primary.residual_mps))
result['per_satellite']=[]
for sat,g in primary.groupby('sat'):
    result['per_satellite'].append(dict(sat=int(sat),observations=len(g),median_residual_mps=float(g.residual_mps.median()),rms_residual_mps=float(np.sqrt(np.mean(g.residual_mps**2))),p95_abs_residual_mps=float(np.quantile(abs(g.residual_mps),.95)),median_cn0=float(g.CN0.median()),median_elevation_deg=float(g.elevation_deg.median())))
st=pq.read_table(OUT/'pilot_satellite_states.parquet').to_pandas();result['ephemeris_age_s']=stats(st.ephemeris_absolute_age_s);result['flight_s']=dict(pseudorange=stats(st.pseudorange_flight_s),geometric=stats(st.geometric_flight_s));result['sat_clock_correction_mps']=stats(st.clock_range_rate_correction_mps)
result['excluded_by_satellite']=pq.read_table(OUT/'unavailable_gps_l1_rows.parquet').to_pandas().sat.value_counts().to_dict()
result['physics_prediction_delta_from_primary']={v:stats(g.set_index('row_id').loc[primary.index].predicted_geometry_and_sat_clock_mps-primary.predicted_geometry_and_sat_clock_mps) for v,g in f.groupby('variant')}
# Independent reconciliation of the only fitted scalar, preserving sigma-only weights.
max_clockerr=0.;max_zero=0.
for (variant,t),g in f.groupby(['variant','gpst_calendar_ns']):
    w=1/g.range_rate_std_mps.to_numpy()**2;b=np.sum(w*(g.observed_y_mps-g.predicted_geometry_and_sat_clock_mps))/np.sum(w)
    max_clockerr=max(max_clockerr,float(np.max(abs(g.receiver_clock_rate_mps-b))));max_zero=max(max_zero,abs(float(np.dot(w,g.residual_mps)/sum(w))))
assert max_clockerr<1e-10 and max_zero<1e-9
raw=pq.read_table(OUT/'pilot_raw_gnss.parquet').to_pandas();one=pq.read_table(OUT/'pilot_1hz_view.parquet').to_pandas();rmap=raw.set_index('row_id')
assert one.row_id.is_unique and raw.row_id.is_unique
for c in ['gpst_calendar_ns','sat','freqs','dopp','dopp_std','psr']:
    a=one[c].to_numpy();b=rmap.loc[one.row_id,c].to_numpy();assert np.array_equal(a,b,equal_nan=True),c
assert len(primary)==1800 and len(primary)+len(pq.read_table(OUT/'unavailable_gps_l1_rows.parquet'))==len(raw[(raw.system=='GPS')&(raw.code==1)&(raw.freqs==1575420000.)])
assert (pd.to_numeric(pd.DataFrame(q['raw_pvt_offset_distribution']).epochs).sum()==15103)
result['verification']=dict(common_offset_recomputation_max_error_mps=max_clockerr,weighted_post_clock_mean_max_abs_mps=max_zero,canonical_view_original_row_parity=True,primary_excluded_denominator_reconciled=True,normal_offset_classification_reconciled=True,all_pilot_reference_fixed=bool((refs.valid_fix & (refs.carr_soln==2)).all()))
save('result_details.json',result)
src=json.loads((OUT/'input_manifest.json').read_text())
save('time_semantics.json',dict(source_run_id=src['source_run_id'],run_id=src['run_id'],time_alignment_method='STATE_PROPAGATION_TO_MEASUREMENT_EPOCH',raw_field='UBX-RXM-RAWX.rcvTow',raw_encoding='binary64 seconds, receiver-local approximately GPS-aligned',pvt_field='UBX-NAV-PVT.iTOW',pvt_encoding='uint32 milliseconds of navigation epoch',pvt_week_source='curr_time from RAWX with half-week crossover',pvt_utc_conversion_executed=False,pvt_utc_guard='if(false && ...)',rawx_receiver_clock_reset_field_saved=False,rawx_receiver_local_clock_bias_saved=False,capture_binary_hash_available=False,ublox_driver_originally_in_V1_manifest=False,supplementary_public_driver_commit='7961ab78d93f63077445698c16186b7bfb37fd14',raw_minus_pvt_statistics=json.loads((OUT/'offset_statistics.json').read_text()),reference_time_scope='NOMINAL_RECORDED_RAW_LABEL; precise clock-bias decomposition not identified',bag_timestamp_used_as_GNST=False,physical_delay_inferred_from_label_difference=False,exact_epoch_V1_check='FAIL_PRESERVED',manuscript_modified=False))
print(json.dumps(clean({k:result[k] for k in ['one_hz_primary','velocity_by_step','sagnac_absolute','position_derived_velocity','alignment_reference_pilot','alignment_prediction','per_satellite','ephemeris_age_s','flight_s','excluded_by_satellite','verification']})))
