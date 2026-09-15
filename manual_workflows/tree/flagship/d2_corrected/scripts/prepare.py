from common import *
guard()
assert not (T/'FROZEN_REVALIDATION_INPUT_MANIFEST.json').exists(),'input manifest already exists'
pre=json.loads((A/'PRE_AUDIT_MANIFEST.json').read_text());supp=json.loads((A/'SUPPLEMENTAL_INPUT_MANIFEST.json').read_text())
known={plain(x['source_path']).lower():x for x in pre['files']+supp}
backup=json.loads((R/'BACKUP_MANIFEST.json').read_text())
records={}
def add(p,role,expected=None,source_epoch='native frozen identity',candidate='M0-M14',binding='prior audit manifest'):
 p=path(p);key=plain(p).lower()
 assert not any(k in key for k in ['dynamic_pool','external_dynamic_data','ca_prototype'])
 old=known.get(key);expected=expected or (old['SHA256'] if old else None)
 h=sha(p)
 if expected:assert h==expected,('source integrity',plain(p),h,expected)
 records[key]={'path':plain(p),'role':role,'size':p.stat().st_size,'SHA256':h,'registered_SHA256':expected,'source_epoch':source_epoch,'target_epoch':110,'candidate_identity':candidate,'selector_identity':'publication v052 frozen M0-M14; SHA bound spec/code','identity_binding':binding,'previous_manifest_match':h==expected if expected else None}
for x in backup['files']:
 p=path(x['source']);rel=Path(x['backup_relative_path']);role='historical_D2_asset'
 if 'observations' in rel.parts:role='observation'
 if 'SEALED_TRUTH_STORE' in rel.parts:role='truth_file_raw_hash_only_before_lock'
 if rel.name=='D2_ALL_CANDIDATE_RECORDS.csv':role='candidate_states'
 if rel.name=='D2_CANDIDATE_RECORDS_NO_TRUTH.csv':role='selector_inputs'
 if rel.name=='D2_EVALUATION_PLAN_LOCK.json':role='evaluation_plan'
 if rel.name=='D2_BATCH_MANIFEST.csv':role='batch_manifest'
 if rel.name=='D2_SEALED_TRUTH_MANIFEST.csv':role='truth_manifest'
 if rel.name.endswith('_SELECTIONS.csv'):role='historical_selection'
 add(p,role,x['sha256'],120 if role in ['candidate_states','selector_inputs','historical_selection'] else 'native','M0-M14', 'BACKUP_MANIFEST source-to-backup identity')
 if len(rel.parts)>1 and rel.parts[1] in ['release_src','selector','original_scripts']:
  dest=T/'source_snapshot'/Path(*rel.parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha(dest)==x['sha256']
required=['PAPER_FLAGSHIP_ROUTE_A_CRITICAL_SOURCE_TO_CLAIM_AUDIT_TOTAL_REPORT.md','PRE_AUDIT_MANIFEST.json','SUPPLEMENTAL_INPUT_MANIFEST.json','CANDIDATE_EPOCH_IDENTITY_MATRIX.csv','D2_SELECTION_FEATURE_EPOCH_AUDIT.csv','D2_OLD_RESULT_CLAIM_IMPACT_MATRIX.csv','D2_DEPENDENCE_STRUCTURE_AUDIT.md','GLOBAL_STATE_SEMANTICS_AUDIT.csv','MANUSCRIPT_CLAIM_INTEGRITY_MATRIX.csv','SOURCE_TO_CLAIM_DEPENDENCY_EDGES.csv']
for n in required:add(A/n,'required_integrity_audit')
add(R/'BACKUP_MANIFEST.json','correction_backup_manifest')
report=[x for x in pre['files']+supp if Path(x['source_path']).name=='D2_EPOCH110_CORRECTION_REPORT.md'];assert len(report)==1
add(report[0]['source_path'],'provisional_correction_report',report[0]['SHA256'])
protocol=[x for x in pre['files']+supp if Path(x['source_path']).name=='PAPER_FLAGSHIP_FINAL_DOMAIN_GENERATION_PROTOCOL.md'];assert len(protocol)==1
add(protocol[0]['source_path'],'formal_output_protocol',protocol[0]['SHA256'])
for name in ['stage_methods.py','d2_common.py']:
 p=W/'tmp/d2_execution_r1'/name;add(p,'frozen_execution_wrapper');dest=T/'source_snapshot/original_scripts'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
freeze=F/'03_working_manuscript/GPS_SOLUTIONS_SUBMISSION_BASELINE_R3_2_1_FROZEN'
for p in freeze.rglob('*'):
 if p.is_file():add(p,'Route_A_protected_artifact')
js('FROZEN_REVALIDATION_INPUT_MANIFEST.json',{'created_utc':utc(),'task':'ROUTE_A_FROZEN_SELECTOR_REVALIDATION','observation_interval_s':[0,120],'formal_output_epoch_s':110,'truth_epoch_s':110,'evaluation_identity':'full-window retrospective estimate reported at 110 s','truth_coordinates_parsed':False,'files':list(records.values()),'Route_B_accessed':False,'authorization_overlay':'Current user authorizes epoch transform/covariance/replay; historical evaluation-plan lock remains unchanged.'})
js('STATISTICAL_SENSITIVITY_PLAN_BEFORE_RESULTS.json',{'created_utc':utc(),'estimand':'equal mean of 92 block paired Lcap1000 differences; all six scenarios retained within block','historical_bootstrap_seed':282172439,'historical_bootstrap_reps':10000,'historical_signflip_seed':2779749054,'historical_signflip_draws':100000,'sensitivity_seed':2026090554,'sensitivity_reps':10000,'procedures':['A historical orbit-time-receiver block','B receiver-cluster ratio bootstrap','C random-realization-cluster ratio bootstrap','D receiver-by-realization two-way pigeonhole ratio bootstrap','E connected receiver-realization component cluster ratio bootstrap'],'roles':['HISTORICAL_PROTOCOL_REPLAY','DEPENDENCE_SENSITIVITY_ONLY'],'intervals':'percentile 95%; all procedures reported; no multiplicity-confirmatory inference','component_graph':'shared receiver and reused random realization; propagation intervals treated as two fixed design settings, not independent random clusters','no_tuning':True})
js('PREPARATION_STATUS.json',{'files':len(records),'source_snapshot_files':len(list((T/'source_snapshot').rglob('*.py'))),'created_utc':utc()})
print('input identities verified',len(records))
