"""Authorized position-only prior construction, separated from candidate execution."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np
import pyarrow.parquet as pq

O=Path(__file__).resolve().parent
A=O.parents[1]
W=A/'gps_l1_wrapper'
V=A/'sports_field_pilot/time_alignment_v2'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (O/n).open('x',encoding='utf-8') as f: json.dump(v,f,indent=2,allow_nan=False)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
if (O/'initialization_prior.json').exists(): raise RuntimeError('ALREADY_PREPARED')
now=datetime.now(timezone.utc).isoformat()
protocol=dict(task='GVINS_SPORTS_FIELD_30S_CLOCK_GATE_BYPASS_DIAGNOSTIC_V1',
 run_id='GVINS_CLOCK_BYPASS_INIT_ASSISTED_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
 identity='INITIALIZATION_ASSISTED_REAL_RF_DIAGNOSTIC',locked_utc=now,
 source_runs=['GVINS_TIME_ALIGNMENT_V2_20260910T103857Z','GVINS_GPS_L1_WRAPPER_V1_20260910T112243Z'],
 gps_week=2134,tow_start=83928.102,tow_end_exclusive=83958.102,origin_ns=1606691928102000000,
 initialization=dict(anchor='RTK fixed primary position at first canonical measurement epoch',
   perturbation='10000 m local East, using longitude of that ECEF position',
   velocity_mps=[0.,0.,0.],joint_clock_initial=[0.,0.],beta_prior='B0_none',
   applies_to='All models, full/train fits, strict and bypass',
   reference_permission='User explicitly authorized truth-associated perturbed common initialization only',
   note='No velocity propagation from anchor to model origin; approximate prior identity recorded explicitly'),
 run_A='STRICT_FROZEN',run_B='ABSOLUTE_CLOCK_GATE_BYPASS_DIAGNOSTIC',
 shared_models=list(range(10)),fit_path_affected_models=list(range(10,15)),
 top_level_budget=20,clock_bypass=[20.,30.],other_gates_unchanged=True,
 no_threshold_selection=True,no_paper_update=True,independent_validation=False)
write('protocol.json',protocol)  # Before reference position values are read.
wm=read(W/'output_manifest.json'); vs=read(V/'final_output_seal.json')
inputs=[W/'wrapper.py',W/'candidate_observations.json',V/'pilot_rtk_reference.parquet',V/'pilot_interval_lock_v2.json']
bindings=[]
for p in inputs:
    manifest=wm if p.parent==W else vs
    b=next(x for x in manifest['files'] if Path(x['path'])==p)
    assert sha(p)==b['sha256']
    bindings.append(dict(path=str(p),sha256=sha(p)))
records=read(W/'candidate_observations.json')
assert len(records)==180
times=sorted({r['receive_gpst_ns'] for r in records})
assert len(times)==30 and sorted({r['satellite'] for r in records})==[2,5,6,13,29,30]
first=times[0]
cols=['gpst_calendar_ns','primary_px','primary_py','primary_pz','reference_eligible','valid_fix','carr_soln']
ref=pq.read_table(V/'pilot_rtk_reference.parquet',columns=cols,filters=[('gpst_calendar_ns','=',first)]).to_pandas()
assert len(ref)==1 and bool(ref.iloc[0].reference_eligible) and bool(ref.iloc[0].valid_fix) and ref.iloc[0].carr_soln==2
p=ref.iloc[0][['primary_px','primary_py','primary_pz']].to_numpy(float)
lon=np.arctan2(p[1],p[0]); east=np.array([-np.sin(lon),np.cos(lon),0.])
initial=p+10000.*east
write('initialization_prior.json',dict(initial_ecef_m=initial.tolist(),initial_velocity_mps=[0.,0.,0.],
    position_init_label='east_10km',velocity_init_label='zero_velocity',beta_prior_profile='B0_none',
    identity='USER_AUTHORIZED_RTK_PERTURBED_COMMON_INITIALIZATION'))
write('initialization_provenance_qa_only.json',dict(anchor_ecef_m=p.tolist(),anchor_epoch_ns=first,
    east_unit= east.tolist(),perturbation_m=(initial-p).tolist(),perturbation_norm_m=float(np.linalg.norm(initial-p)),
    anchor_minus_model_origin_s=(first-protocol['origin_ns'])/1e9,
    position_columns_read=cols,velocity_columns_read=[],candidate_allowed_file='initialization_prior.json'))
write('input_manifest.json',dict(bindings=bindings,protocol_sha256=sha(O/'protocol.json'),
    prior_sha256=sha(O/'initialization_prior.json'),row_times_ns=times,rows=180,
    report_epoch_ns=times[-1]))
print(json.dumps(dict(status='PRIOR_LOCKED',identity=protocol['identity'],rows=180,
    anchor_epoch_ns=first,report_epoch_ns=times[-1],offset_m=float(np.linalg.norm(initial-p)))))
