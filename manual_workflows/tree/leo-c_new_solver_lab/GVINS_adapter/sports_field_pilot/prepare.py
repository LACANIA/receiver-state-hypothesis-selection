import csv,json,re,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from gps_physics import orbit,finite_difference,earth_rotation_forms,corrected_distance,STEPS
OUT=Path(__file__).resolve().parent;ADAPTER=OUT.parent
sys.path.insert(0,str(ADAPTER));import extract_gnss as ex

def run():
    source=json.loads((ADAPTER/'source_manifest.json').read_text('utf-8'))
    assert source['run_id']=='GVINS_ADAPTER_V1_20260910T083449Z'
    checks=[]
    for repo in source['repositories']:
        for item in repo['files']:
            actual=ex.sha(item['path'])
            checks.append({'path':item['path'],'sha256':actual,'matches':actual==item['sha256']})
    assert all(x['matches'] for x in checks),'SOURCE_HASH_CONFLICT'
    fields=list(csv.DictReader((ADAPTER/'GVINS_GNSS_FIELD_IDENTITY.csv').open(encoding='utf-8')))
    code_source=ADAPTER.parent/'gnss_comm-main/include/gnss_comm/gnss_constant.hpp'
    codes={int(v):{'name':k,'official_comment':note.strip()} for k,v,note in re.findall(r'#define\s+(CODE_\w+)\s+(\d+)\s+/\*\s*(.*?)\*/',code_source.read_text('utf-8'))}
    lock={'task':'GVINS_SPORTS_FIELD_DOWNLOAD_AND_30S_DOPPLER_FORWARD_QA_V1','run_id':'GVINS_SPORTS_FIELD_QA_V1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),'locked_utc':datetime.now(timezone.utc).isoformat(),'source_run_id':source['run_id'],'sources':checks,'field_identity_rows':len(fields),'signal_codes':codes,
      'source_manifest_sha256':ex.sha(ADAPTER/'source_manifest.json'),'extractor_sha256':ex.sha(ADAPTER/'extract_gnss.py'),
      'physical_code_sha256':ex.sha(OUT/'gps_physics.py'),'python':sys.executable,'base_prefix':sys.base_prefix,
      'primary':'SINGLE_CONSTELLATION_SINGLE_BAND_GPS_L1','fd_steps_seconds':list(STEPS),'primary_velocity':'CENTERED_FD_H0p1_WITH_CHAIN_RULE_CROSSCHECK','primary_forward':'FORM_A_RETARDED_PLUS_SATELLITE_CLOCK_RATE',
      'interval':'First raw-epoch start, half-open 30 s, >=4 eligible GPS L1 satellites at each epoch, gaps<=0.2s, span>=29.9s, exact GPST ns PVT join, valid_fix and carr_soln==2',
      'ephemeris':'same satellite, health==0, arrived by observation bag timestamp, abs(toe-observation)<7200; nearest toe; ties most recent arrival then record index',
      'one_hz_rule':'For every GPST integer second lying inside pilot, choose closest raw epoch within pilot; exact distance tie -> earlier epoch. All signals at chosen epoch retained. Save signed offset. No residual access.',
      'noise_weight':'1/sigma_y^2 only; per-epoch weighted common receiver clock-rate offset; no CN0/elevation reweight',
      'amendments_to_predecessor':['Other constellations inventory only, no forward QA','Weighted common offset replaces descriptive median offset'],
      'geometry_time':'GPST throughout. Transmission time from observed pseudorange and pinned satellite clock iteration. Earth-rotation angle from geometric range/c in QA, not clock-biased pseudorange/c.',
      'secondary_convention':'Pinned GVINS epoch-rate first-order Sagnac and original analytic velocity retained as named QA controls.',
      'no_new_positioning_experiments':True,'m0_m14_runs':0,'egshs_runs':0}
    p=OUT/'physics_execution_lock.json'
    if p.exists(): raise FileExistsError(p)
    ex.save_json(p,lock)
    # Deterministic algebra fixture, not a new measured/simulated performance sample.
    eph=dict(A=26560000.,e=.012,delta_n=4e-9,M0=.7,omg=1.1,i0=.95,i_dot=2e-10,OMG0=1.8,OMG_dot=-8e-9,toe_tow=432000.,toe_ns=0,toc_ns=0,cus=2e-6,cuc=-3e-6,crs=20.,crc=230.,cis=1e-7,cic=-2e-7,af0=2e-4,af1=-3e-12,af2=2e-20)
    t=1875.;p,vo,vd,cb,cd=orbit(eph,t);r=np.array([-2410000.,5380000.,2400000.]);vr=np.array([1.,2.,-.5])
    rows=[]
    for h in STEPS:
        vf,cf=finite_difference(eph,t,h)
        rows.append({'h_s':h,'corrected_analytic_minus_fd_norm_mps':float(np.linalg.norm(vd-vf)),'official_minus_fd_norm_mps':float(np.linalg.norm(vo-vf)),'clock_derivative_difference_s_per_s':float(cd-cf)})
    forms=earth_rotation_forms(p,vd,r,vr)
    range_fd=(corrected_distance(eph,t,r,vr,.1)-corrected_distance(eph,t,r,vr,-.1))/.2
    assert max(x['corrected_analytic_minus_fd_norm_mps'] for x in rows)<5e-5
    assert abs(forms['form_a_retarded_mps']-range_fd)<5e-5
    assert abs(forms['form_a_epoch_mps']-forms['form_b_epoch_mps'])<5e-5
    ex.save_json(OUT/'deterministic_checks.json',{'utc':datetime.now(timezone.utc).isoformat(),'identity':'DETERMINISTIC_ALGEBRA_FIXTURE_NO_NOISE_NO_POSITIONING','velocity_clock':rows,'sagnac_forms':{k:float(v) for k,v in forms.items() if not isinstance(v,np.ndarray)},'implicit_distance_fd_mps':range_fd,'status':'PASS'})
    print(json.dumps({'source_files_verified':len(checks),'field_identity_rows':len(fields),'deterministic_checks':'PASS','run_id':lock['run_id']}))
if __name__=='__main__':run()
