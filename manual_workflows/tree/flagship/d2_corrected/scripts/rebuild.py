from common import *
guard()
assert not (T/'PRE_TRUTH_REVALIDATION_LOCK.json').exists()
stage,selector,evaluator=sources()
from leo_positioning.uncertainty_diagnostics import residual_jacobian_for_row,covariance_diagnostics_from_jacobian
from leo_positioning.risk_veto import severe_risk_veto,compute_group_references
calls={}
def profile(frame,event,arg):
 if event=='call' and 'leo_positioning' in frame.f_code.co_filename:
  name=frame.f_code.co_name;calls[name]=calls.get(name,0)+1
  if name.startswith(('solve','run_candidate','fit_','optimize','levenberg','irls_')):raise RuntimeError('candidate optimization forbidden: '+name)
sys.setprofile(profile)
c=pd.read_csv(byrole('candidate_states')[0]);norm=pd.read_csv(byrole('selector_inputs')[0]);batch=pd.read_csv(byrole('batch_manifest')[0])
D2=byrole('batch_manifest')[0].parent
spec=json.loads((T/'source_snapshot/selector/ma_bgtr_mode_a_selector_spec_v052.json').read_text())
lockpath=next(path(x['path']) for x in inputs()['files'] if x['path'].endswith('D2_METHOD_OUTPUT_LOCK.json'))
methodlock=json.loads(lockpath.read_text())
history={}
for method in evaluator.METHOD_ORDER:
 key=method.removeprefix('ALWAYS_')+'_sha256'
 matches=[path(x['path']) for x in inputs()['files'] if x['SHA256']==methodlock[key]]
 assert matches,(method,'method lock binding')
 history[method]=pd.read_csv(matches[0]).set_index('batch_id')
assert len(c)==8280 and len(batch)==552 and len(norm)==8280
assert c.groupby('batch_id').candidate_model.nunique().eq(15).all()
assert c.beta_prior_profile.eq('B0_none').all()
POS=['final_ecef_x_m','final_ecef_y_m','final_ecef_z_m'];VEL=['estimated_vx_mps','estimated_vy_mps','estimated_vz_mps']
output={k:[] for k in evaluator.METHOD_ORDER};audit=[];risks=[];replays=[];features=[];states=[];parity=[];specs=[];oldnorms=[]
def equiv(a,b):
 if (pd.isna(a) or str(a).lower() in ['','nan']) and (pd.isna(b) or str(b).lower() in ['','nan']):return True
 if isinstance(a,(int,float,np.number,bool)) and isinstance(b,(int,float,np.number,bool)):return bool(np.isclose(a,b,rtol=1e-8,atol=1e-8,equal_nan=True))
 return str(a)==str(b)
def verifyframe(old,new,columns,label):
 old=old.set_index('candidate_model');new=new.set_index('candidate_model')
 for model in old.index:
  for field in columns:
   if field=='candidate_model':continue
   assert equiv(old.loc[model,field],new.loc[model,field]),(label,model,field,old.loc[model,field],new.loc[model,field])
def admitted(g):return sorted(selector.base_filter(g,selector.best(g,selector.STATIC),spec['thresholds']).candidate_model.tolist())
for n,(bid,g) in enumerate(c.groupby('batch_id',sort=True)):
 g=g.copy();ngold=norm.loc[norm.group_id==bid].copy();modified=g.copy();arr={}
 with np.load(D2/'observations'/f'{bid}_OBS.npz',allow_pickle=False) as z:
  obs={k:np.array(z[k]) for k in ['time_s','sat_pos_m','sat_vel_mps','meas_mps','satellite_number']};obs['t0_s']=float(z['t0_s'])
  public={'batch_id':bid,'scenario':str(z['scenario'].item()),'observation_count':len(obs['time_s'])};sigma=float(z['sigma_mps'])
 assert obs['t0_s']==0 and np.min(obs['time_s'])==0 and np.max(obs['time_s'])==120
 refs=compute_group_references(g)
 for idx,row in g.iterrows():
  model=row.candidate_model;mid=model.split('_')[0];static=mid in ['M0','M1']
  r,j,w,pred,family=residual_jacobian_for_row(row.to_dict(),obs)
  assert static==(family=='static')
  old=covariance_diagnostics_from_jacobian(r,j,w,family,120)
  new=covariance_diagnostics_from_jacobian(r,j,w,family,110)
  rel=abs(old['position_cov_sqrt_trace_m']-row.position_cov_sqrt_trace_m)/max(abs(row.position_cov_sqrt_trace_m),1e-12)
  assert rel<1e-7,(bid,model,'old covariance parity',rel)
  rmse=float(np.sqrt(np.mean(r*r)));assert abs(rmse-row.full_residual_rmse_mps)<1e-7,(bid,model,'residual parity')
  for field,val in old.items():
   if field in row.index and field not in ['normal_condition_number','min_singular_value','max_singular_value']:
    assert equiv(row[field],val),(bid,model,field,row[field],val)
  oldflag,oldreason=severe_risk_veto(row,refs);assert oldflag==bool(row.severe_risk_veto),(bid,model,'old risk parity')
  assert equiv(oldreason,row.severe_risk_reason),(bid,model,'old risk reason parity')
  H=j.T@(w[:,None]*j);pinv=False
  try:P0=np.linalg.inv(H)
  except np.linalg.LinAlgError:P0=np.linalg.pinv(H,rcond=1e-10);pinv=True
  sigma2=max(float(np.sum(w*r*r)/max(len(r)-j.shape[1],1)),1e-12);P0*=sigma2
  dim=j.shape[1];Q120=np.eye(dim);Q110=np.eye(dim);Fmat=np.eye(dim)
  if not static:
   Q120[:3,3:6]=120*np.eye(3);Q110[:3,3:6]=110*np.eye(3);Fmat[:3,3:6]=-10*np.eye(3)
  P120=Q120@P0@Q120.T;P110=Fmat@P120@Fmat.T;direct=Q110@P0@Q110.T
  p120=row[POS].to_numpy(float);v=row[VEL].to_numpy(float)
  x120=p120.copy() if static else np.r_[p120,v]
  if dim>len(x120):x120=np.r_[x120,row.beta0_estimated_mps]
  if dim>len(x120):x120=np.r_[x120,row.beta_dot_estimated_mps2]
  assert len(x120)==dim
  x110=Fmat@x120
  if not static:assert np.allclose(x110[:3],p120-10*v,rtol=0,atol=1e-8)
  assert np.isfinite(x110).all() and np.isfinite(P110).all()
  # Diagnostic tolerance only; it never clips eigenvalues or changes the frozen inverse/rank rules.
  scale=max(float(np.max(np.abs(P110))),1.);sym=float(np.max(np.abs(P110-P110.T)));eig=np.linalg.eigvalsh((P110+P110.T)/2)
  psd=bool(eig[0]>=-1e-9*scale);assert psd and sym/scale<1e-7
  transfer=float(np.max(np.abs(P110-direct))/scale);assert transfer<1e-9
  trace=float(np.trace(P110[:3,:3]));assert np.isclose(trace,new['position_cov_trace'],rtol=1e-8)
  # Preserve exact historical scalar evaluation order for the frozen selector.
  for field in ['position_cov_trace','position_cov_sqrt_trace_m','covariance_available']:modified.at[idx,field]=new[field]
  modified.loc[idx,POS]=x110[:3]
  sv=np.linalg.svd(H,compute_uv=False);ranktol=max(float(sv.max()),1.)*1e-10
  audit.append({'candidate':model,'batch':bid,'source_covariance_status':'FIXED_ESTIMATE_HISTORICAL_DIAGNOSTIC_RECONSTRUCTION','F_identity':mid+'_mixed_clock_t0_position120_to110','state_dimension':dim,'fitted_dimension':stage.MODEL_DOF[model],'symmetry_error':sym,'symmetry_relative_error':sym/scale,'minimum_eigenvalue':float(eig[0]),'PSD_status':'PASS_RELATIVE_DIAGNOSTIC_TOLERANCE' if psd else 'FAIL','PSD_diagnostic_tolerance':1e-9*scale,'position_covariance_trace_before':float(np.trace(P120[:3,:3])),'position_covariance_trace_after':trace,'finite_status':True,'rank_status':int(old['effective_rank']),'rank_tolerance_frozen':ranktol,'pseudoinverse_actual':pinv,'covariance120_relative_reproduction_error':rel,'full_joint_transform_relative_error':transfer,'residual_RMSE_absolute_reproduction_error':abs(rmse-row.full_residual_rmse_mps),'clock_reference':'b0 at t0=0 retained; all cross covariance terms propagated','matrix_file':plain(T/'joint_covariance'/f'{bid}.npz'),'matrix_prefix':mid})
  for key,value in {'r':r,'J':j,'w':w,'normal':H,'sigma2':np.array(sigma2),'P0':P0,'P120':P120,'P110':P110,'F':Fmat,'x120':x120,'x110':x110}.items():arr[mid+'_'+key]=value
  states.append({'batch_id':bid,'candidate_model':model,'source_epoch':120,'target_epoch':110,**dict(zip(POS,x110[:3])),'stored_p120_x':p120[0],'stored_p120_y':p120[1],'stored_p120_z':p120[2],**dict(zip(VEL,v)),'clock_epoch':0,'matrix_prefix':mid,'matrix_file':plain(T/'joint_covariance'/f'{bid}.npz')})
  if n==0:
   layout='p_ECEF[0:3] m'+('; v_ECEF[3:6] m/s' if not static else '')
   if dim in [4,7,8]:layout+='; b0['+str(3 if static else 6)+'] m/s at t0=0'
   if dim==8:layout+='; bdot[7] m/s^2'
   specs.append({'candidate':model,'state_layout':layout,'fitted_state_dimension':stage.MODEL_DOF[model],'historical_diagnostic_dimension':dim,'source_epoch':120,'target_epoch':110,'transform_matrix_available':True,'position_transform':'identity' if static else 'p110=p120-10*v_est','velocity_transform':'no fitted velocity' if static else 'identity; constant ECEF velocity','clock_transform':'identity on native b0 at t0=0 and drift; position-clock cross covariance propagated','static_or_dynamic':'static' if static else 'constant_velocity','acceleration_identity':'absent','covariance_epoch':'native P0 at 0; full mixed-state P120 -> P110','reported_endpoint_identity':'120 s persisted estimated_positions[-1] -> 110 s full-window retrospective','position_columns':','.join(POS),'velocity_columns':','.join(VEL) if not static else 'none','notes':'Historical 8-coordinate diagnostic proxy includes non-fitted drift coordinate; actual b0-only fit is 7-dimensional. Frozen covariance semantics retained; not a calibrated native-fit covariance.' if mid in ['M5','M8'] else 'Historical covariance proxy; no statistical calibration claim.'})
 (T/'joint_covariance').mkdir(exist_ok=True);np.savez_compressed(T/'joint_covariance'/f'{bid}.npz',**arr)
 refs110=compute_group_references(modified)
 for idx,row in modified.iterrows():
  flag,reason=severe_risk_veto(row,refs110);modified.at[idx,'severe_risk_veto']=flag;modified.at[idx,'severe_risk_reason']=reason
 oldregen=stage.apply_common_eligibility(pd.DataFrame([stage.normalized_candidate(row,public) for _,row in g.iterrows()]))
 verifyframe(ngold,oldregen,[x for x in stage.NO_TRUTH_COLUMNS if x in ngold.columns],'historical normalized fields')
 ng=stage.apply_common_eligibility(pd.DataFrame([stage.normalized_candidate(row,public) for _,row in modified.iterrows()]))
 orig=selector.select_group(ngold.copy(),spec);replay=selector.select_group(ng.copy(),spec);h=history['EGSHS'].loc[bid]
 for field in ['selected_candidate','status','selection_reason','low_quality']:assert equiv(orig[field],h[field]),(bid,field,'selector parity')
 if 'near_tie_members' in h:assert equiv(';'.join(orig['near_tie_members']),h.near_tie_members),(bid,'near tie parity')
 aset0=admitted(ngold);aset1=admitted(ng);changed=orig['selected_candidate']!=replay['selected_candidate']
 riskschanged=int((g.severe_risk_veto.to_numpy()!=modified.severe_risk_veto.to_numpy()).sum())
 replays.append({'batch':bid,'historical_selected_candidate':orig['selected_candidate'],'110s_replay_selected_candidate':replay['selected_candidate'],'selection_changed':changed,'historical_status':orig['status'],'replay_status':replay['status'],'admission_set_change':aset0!=aset1,'historical_admission_set':';'.join(aset0),'110s_admission_set':';'.join(aset1),'risk_flag_change':riskschanged>0,'risk_flag_changed_count':riskschanged,'near_tie_change':orig['near_tie_members']!=replay['near_tie_members'],'historical_near_tie':';'.join(orig['near_tie_members']),'110s_near_tie':';'.join(replay['near_tie_members']),'fallback_change':orig['low_quality']!=replay['low_quality'],'historical_reason':orig['selection_reason'],'reason_trace':json.dumps(replay,default=str),'historical_parity':True})
 for idx,row in modified.iterrows():
  model=row.candidate_model;oldrow=g.loc[idx];target=ng.loc[ng.candidate_model==model].iloc[0]
  risks.append({'batch':bid,'candidate':model,'historical_flag':bool(oldrow.severe_risk_veto),'110s_flag':bool(row.severe_risk_veto),'changed':bool(oldrow.severe_risk_veto)!=bool(row.severe_risk_veto),'source_field':'position_cov_sqrt_trace_m and unchanged frozen native risk inputs','historical_reason':oldrow.severe_risk_reason,'110s_reason':row.severe_risk_reason,'historical_covariance_sqrt_trace_m':oldrow.position_cov_sqrt_trace_m,'110s_covariance_sqrt_trace_m':row.position_cov_sqrt_trace_m,'selector_consequence':f'admission {model in aset0}->{model in aset1}; selected {orig["selected_candidate"]}->{replay["selected_candidate"]}'})
  features.append({'batch_id':bid,**target.to_dict(),'position_cov_trace':row.position_cov_trace,'covariance_available':bool(row.covariance_available),'source_epoch':120,'target_epoch':110})
 oldnorms.append(ngold)
 full=modified.drop(columns=[x for x in ng.columns if x in modified and x not in ['candidate_model']],errors='ignore').merge(ng,on='candidate_model',validate='one_to_one')
 full['batch_id']=bid
 # Restored metadata from normalized table; persisted transformed position never enters risk or validation.
 for method in evaluator.METHOD_ORDER:
  if method=='EGSHS':out=stage.selected_output_row(bid,replay['selected_candidate'],full,replay['status'],replay['selection_reason'],replay['low_quality'])
  elif method.startswith('ALWAYS_'):out=stage.fixed_selection(bid,next(x for x in stage.CANDIDATES if x.startswith(method[7:]+'_')),full)
  else:
   oldic=stage.information_criterion_selection(bid,method,sigma,ngold,g.merge(ngold[['candidate_model','fixed_numeric_valid']],on='candidate_model',suffixes=('','_norm')))
   assert oldic['selected_candidate']==history[method].loc[bid,'selected_candidate'],(bid,method,'IC parity')
   out=stage.information_criterion_selection(bid,method,sigma,ng,full)
  out['output_epoch_s']=110
  selected=modified.loc[modified.candidate_model==out['selected_candidate']]
  out['selected_position_cov_sqrt_trace_m']=float(selected.iloc[0].position_cov_sqrt_trace_m) if len(selected) else np.nan
  out['selected_severe_risk_veto']=bool(selected.iloc[0].severe_risk_veto) if len(selected) else False
  output[method].append(out)
  parity.append({'batch':bid,'method':method,'historical_selected_candidate':history[method].loc[bid,'selected_candidate'],'replayed_selected_candidate':out['selected_candidate'],'selection_changed':history[method].loc[bid,'selected_candidate']!=out['selected_candidate'],'historical_status':history[method].loc[bid,'status'],'replayed_status':out['status']})
 if n%50==0:print('frozen state/covariance/selector batch',n+1,'/552',flush=True)
sys.setprofile(None)
csv('D2_CANDIDATE_STATE_TRANSFORM_SPEC.csv',specs);ad=csv('D2_COVARIANCE_PROPAGATION_AUDIT.csv',audit)
fd=csv('D2_SELECTION_FEATURE_110S_REBUILD.csv',features);csv('D2_RISK_FLAG_BEFORE_AFTER.csv',risks);rd=csv('D2_FROZEN_SELECTOR_REPLAY.csv',replays);csv('D2_110S_CANDIDATE_STATES.csv',states);csv('D2_ALL_METHOD_SELECTION_PARITY.csv',parity)
oldall=pd.concat(oldnorms).set_index(['group_id','candidate_model']);newall=fd.set_index(['group_id','candidate_model']);classes=[]
epoch={'position_cov_sqrt_trace_m','position_covariance_proxy_m','position_cov_trace'}
recomputed={'severe_risk_veto','severe_risk_reason','base_filter_pass','base_filter_reason','selectable','fixed_numeric_valid','state_finite','covariance_available'}
missing={'position_bootstrap_spread_m','bootstrap_spread_m'}
for field in fd.columns:
 if field in ['group_id','candidate_model']:continue
 category='TRANSFORMED_TO_110' if field in epoch else 'RECOMPUTED_AT_FIXED_ESTIMATE' if field in recomputed else 'UNAVAILABLE_AT_110' if field in missing else 'UNCHANGED_BY_EPOCH' if field in stage.NO_TRUTH_COLUMNS else 'NOT_USED_BY_SELECTOR'
 changedcount=sum(not equiv(a,b) for a,b in zip(oldall.loc[newall.index,field],newall[field])) if field in oldall.columns else None
 classes.append({'field':field,'classification':category,'records':len(fd),'changed_count':changedcount,'missing_count':int(fd[field].isna().sum()),'source':'frozen stage_methods normalization + field epoch audit','identity_note':'Native t0 information/condition/clock, fitted validation rows and model status retained; covariance aliases reflect110; pre-existing missing bootstrap kept with original missing policy.'})
csv('D2_SELECTION_FIELD_CLASSIFICATION.csv',classes)
for method,rows in output.items():csv('method_outputs/D2_'+method+'_SELECTIONS.csv',rows)
js('NO_OPTIMIZATION_CALL_TRACE.json',{'candidate_refits':0,'prohibited_solver_calls':0,'executed_leo_positioning_functions':calls,'guard':'solver/fit/optimization entry names rejected; only saved-state residual/Jacobian diagnostics and frozen selection helpers executed'})
js('REBUILD_CHECKS.json',{'candidate_records':len(ad),'batches':len(rd),'candidate_refits':0,'threshold_changes':0,'rule_changes':0,'joint_covariance_110s_complete':len(ad)==8280 and bool(ad.finite_status.all()),'epoch_dependent_fields_complete':True,'preexisting_bootstrap_unavailable':True,'frozen_selector_replay_complete':len(rd)==552,'selection_changed_count':int(rd.selection_changed.sum()),'historical_selector_parity_count':int(rd.historical_parity.sum()),'risk_flag_changes':sum(x['changed'] for x in risks),'max_covariance120_reproduction_relative_error':float(ad.covariance120_relative_reproduction_error.max()),'M5_M8_covariance_identity':'full historical 8-coordinate diagnostic proxy retained; 7 fitted coordinates distinguished; no covariance calibration claim','created_utc':utc()})
lockfiles=[p for p in T.rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.npz','.py']]
js('PRE_TRUTH_REVALIDATION_LOCK.json',{'created_utc':utc(),'truth_coordinates_parsed_this_run':False,'result_identity':'D2_FROZEN_SELECTOR_REVALIDATION_110S','files':[{'path':plain(p),'SHA256':sha(p),'size':p.stat().st_size} for p in lockfiles]})
print('selector and state outputs locked before truth evaluation',len(rd))
