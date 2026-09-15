from common import *
guard()
stage,selector,ev=sources()
lock=json.loads((T/'PRE_TRUTH_REVALIDATION_LOCK.json').read_text());assert not lock['truth_coordinates_parsed_this_run']
for item in lock['files']:assert sha(path(item['path']))==item['SHA256'],item['path']
assert not (T/'EVALUATION_COMPLETED.json').exists()
batch=pd.read_csv(byrole('batch_manifest')[0]);tm=pd.read_csv(byrole('truth_manifest')[0]);plan=json.loads(byrole('evaluation_plan')[0].read_text())
assert len(batch)==552 and len(tm)==552
methodpaths={}
for method in ev.METHOD_ORDER:
 p=T/'method_outputs'/f'D2_{method}_SELECTIONS.csv';d=pd.read_csv(p);assert d.output_epoch_s.eq(110).all()
 csv('evaluation_inputs/'+p.name,d.drop(columns='output_epoch_s'));methodpaths[method]=T/'evaluation_inputs'/p.name
js('EVALUATION_INPUT_ADAPTER_LOCK.json',{'created_utc':utc(),'operation':'drop redundant output_epoch_s column solely to avoid pandas merge suffix; all saved states/selection/status unchanged','files':[{'path':plain(p),'SHA256':sha(p)} for p in methodpaths.values()]})
first=utc();truth=[];truthpaths={x['SHA256']:path(x['path']) for x in inputs()['files'] if x['role']=='truth_file_raw_hash_only_before_lock'}
for item in tm.sort_values('batch_id').itertuples(index=False):
 p=truthpaths[str(item.truth_record_sha256)];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item.truth_record_sha256
 v=json.loads(raw);assert v['batch_id']==item.batch_id and v['output_epoch_s']==110
 core=dict(v);embedded=core.pop('truth_record_sha256');assert hashlib.sha256(ev.canonical_json_bytes(core)).hexdigest()==embedded
 pos=np.array(v['output_epoch_truth_position_ecef_m']);assert pos.shape==(3,) and np.isfinite(pos).all()
 truth.append({'batch_id':item.batch_id,'truth_record_id':item.truth_record_id,'block_id_truth':v['block_id'],'truth_x_m':pos[0],'truth_y_m':pos[1],'truth_z_m':pos[2],'output_epoch_s':110,'truth_file':plain(p),'truth_file_SHA256':item.truth_record_sha256,'truth_core_SHA256':embedded})
truth=pd.DataFrame(truth);joined=batch.merge(truth,on=['batch_id','truth_record_id'],validate='one_to_one');assert joined.block_id.eq(joined.block_id_truth).all()
joined['scenario_code']=joined.scenario_family.str.extract(r'(H[0-5])',expand=False);assert joined.scenario_code.notna().all()
csv('D2_TRUTH_JOIN_AUDIT.csv',truth);js('TRUTH_READ_EVENT.json',{'first_truth_coordinate_read_utc_this_run':first,'pre_truth_lock_utc':lock['created_utc'],'truth_records':552,'truth_epoch':110,'new_independent_blind_experiment':False})
long=pd.concat([ev.evaluate_method(m,p,joined) for m,p in methodpaths.items()],ignore_index=True)
assert len(long)==3864
block=long.groupby(['block_id','receiver_id','interval_id','method'],as_index=False).agg(scenario_count=('batch_id','size'),finite_outputs=('method_failure',lambda x:int((~x).sum())),failures=('method_failure','sum'),block_Lcap100=('Lcap100','mean'),block_Lcap500=('Lcap500','mean'),block_Lcap1000=('Lcap1000','mean'),block_Lcap2000=('Lcap2000','mean'),block_mean_error_m_finite_only=('position_error_m','mean'))
assert len(block)==644 and block.scenario_count.eq(6).all()
summary=ev.method_summary(long,block);summary['result_identity']='D2_FROZEN_SELECTOR_REVALIDATION_110S';summary['validation_status']='PROVISIONAL_REVALIDATION_SUMMARY'
for m in ev.METHOD_ORDER:
 d=pd.read_csv(T/'method_outputs'/f'D2_{m}_SELECTIONS.csv');mask=summary.method==m
 summary.loc[mask,'selected_covariance_sqrt_trace_m_median']=d.selected_position_cov_sqrt_trace_m.median();summary.loc[mask,'selected_risk_flag_count']=d.selected_severe_risk_veto.sum()
csv('D2_110S_BATCH_ERROR_TABLE.csv',long);csv('D2_110S_BLOCK_METRICS.csv',block);csv('D2_110S_METHOD_SUMMARY.csv',summary)
scenario=ev.scenario_summary(long);csv('D2_110S_SCENARIO_SUMMARY.csv',scenario)
csv('D2_110S_SELECTION_CONDITIONED_OUTCOMES.csv',ev.selection_outcomes(long));csv('D2_110S_CHANGED_FROM_M2_ANALYSIS.csv',ev.changed_from_m2(long))
diff={name:ev.block_difference_table(block,batch,'ALWAYS_'+name) for name in ['M2','M12']}
csv('D2_110S_BLOCK_DIFFERENCES.csv',pd.concat(diff.values(),ignore_index=True));lobo=ev.lobo_table(diff);csv('D2_110S_LEAVE_ONE_BLOCK_OUT.csv',lobo)
boot=ev.bootstrap_results(diff['M2'].difference.to_numpy(),diff['M12'].difference.to_numpy(),plan['bootstrap']['seed'],plan['bootstrap']['repetitions']);boot['role']='HISTORICAL_PROTOCOL_REPLAY';js('D2_110S_HISTORICAL_BOOTSTRAP.json',boot)
sign=ev.signflip_results(diff['M2'].difference.to_numpy(),diff['M12'].difference.to_numpy(),plan['sign_flip']['seed'],plan['sign_flip']['draws']);sign['role']='HISTORICAL_PROTOCOL_REPLAY';js('D2_110S_HISTORICAL_SIGNFLIP.json',sign)
paired=[]
for name,d in diff.items():
 y=d.difference.to_numpy();loo=(y.sum()-y)/(len(y)-1);base=float(d['ALWAYS_'+name+'_block_Lcap1000'].mean());bb=boot['Delta_'+name]
 paired.append({'comparison':'EGSHS_minus_'+name,'point_difference':y.mean(),'relative_difference':y.mean()/base,'relative_difference_denominator':'comparator block-equal Lcap1000','blocks_favorable':int((y<-1e-12).sum()),'blocks_tied':int((np.abs(y)<=1e-12).sum()),'blocks_unfavorable':int((y>1e-12).sum()),'leave_one_block_min':loo.min(),'leave_one_block_max':loo.max(),'largest_favorable_block':d.iloc[np.argmin(y)].block_id,'remove_largest_favorable_block':loo[np.argmin(y)],'historical_bootstrap_lower95':bb['percentile_2_5'],'historical_bootstrap_upper95':bb['percentile_97_5'],'historical_signflip_p_like':sign['Delta_'+name]['two_sided_p_like_plus_one'],'inference_role':'HISTORICAL_PROTOCOL_REPLAY; dependence sensitivities reported separately'})
csv('D2_110S_PAIRED_COMPARISON.csv',paired)
# Crossed receiver/random-realization graph reconstructed from original design identities.
blocks=batch[['block_id','receiver_id','interval_id','realization_index']].drop_duplicates().sort_values('block_id');assert len(blocks)==92
parent={}
def find(x):
 parent.setdefault(x,x)
 if parent[x]!=x:parent[x]=find(parent[x])
 return parent[x]
def union(x,y):parent[find(x)]=find(y)
for row in blocks.itertuples(index=False):union('receiver:'+row.receiver_id,'realization:'+str(row.realization_index))
roots=sorted({find(x) for x in parent});ids={x:f'COMPONENT_{i+1}' for i,x in enumerate(roots)}
blocks['component']=[ids[find('receiver:'+x)] for x in blocks.receiver_id];csv('D2_DEPENDENCE_BLOCK_REGISTRY.csv',blocks)
edges=[]
for r in batch.itertuples(index=False):
 for rel,target in [('receiver',r.receiver_id),('random_realization',r.realization_index),('within_block_scenario',r.scenario_family),('source_holdout',r.source_holdout_id),('fixed_propagation_interval',r.interval_id)]:edges.append({'batch':r.batch_id,'block':r.block_id,'relation':rel,'target':target})
csv('D2_DEPENDENCE_GRAPH_EDGES.csv',edges)
sp=json.loads((T/'STATISTICAL_SENSITIVITY_PLAN_BEFORE_RESULTS.json').read_text());rng=np.random.default_rng(sp['sensitivity_seed']);reps=sp['sensitivity_reps'];sensitivity=[]
for label,table in diff.items():
 d=table.merge(blocks,on=['block_id','receiver_id','interval_id'],validate='one_to_one');y=d.difference.to_numpy();codes={}
 for col in ['receiver_id','realization_index','component']:codes[col]=pd.factorize(d[col],sort=True)
 rc,ru=codes['receiver_id'];sc,su=codes['realization_index'];cc,cu=codes['component'];assert len(ru)==48 and len(su)==10
 draws={k:[] for k in ['B_receiver_cluster','C_random_realization_cluster','D_two_way_receiver_random_realization','E_connected_component_cluster']};empty=0
 for _ in range(reps):
  rw=rng.multinomial(len(ru),np.ones(len(ru))/len(ru));sw=rng.multinomial(len(su),np.ones(len(su))/len(su));cw=rng.multinomial(len(cu),np.ones(len(cu))/len(cu))
  for name,weights in [('B_receiver_cluster',rw[rc]),('C_random_realization_cluster',sw[sc]),('D_two_way_receiver_random_realization',rw[rc]*sw[sc]),('E_connected_component_cluster',cw[cc])]:
   if weights.sum():draws[name].append(float(np.sum(y*weights)/weights.sum()))
   else:empty+=1
 bb=boot['Delta_'+label];sensitivity.append({'comparison':'EGSHS_minus_'+label,'procedure':'A_original_orbit_time_receiver_block','point_difference':y.mean(),'lower95':bb['percentile_2_5'],'upper95':bb['percentile_97_5'],'bootstrap_mean':bb['bootstrap_mean'],'replications':10000,'seed':plan['bootstrap']['seed'],'receiver_clusters':len(ru),'realization_clusters':len(su),'resampling_clusters':92,'role':'HISTORICAL_PROTOCOL_REPLAY','statistical_qualification':'block bootstrap treats blocks as exchangeable; crossed dependence not resolved'})
 for name,vals in draws.items():
  arr=np.array(vals);lo,hi=np.quantile(arr,[.025,.975]);sensitivity.append({'comparison':'EGSHS_minus_'+label,'procedure':name,'point_difference':y.mean(),'lower95':lo,'upper95':hi,'bootstrap_mean':arr.mean(),'replications':len(arr),'seed':sp['sensitivity_seed'],'receiver_clusters':len(ru),'realization_clusters':len(su),'resampling_clusters':len(ru) if name.startswith('B') else len(su) if name.startswith('C') else len(cu) if name.startswith('E') else '48 x 10','role':'DEPENDENCE_SENSITIVITY_ONLY','statistical_qualification':'conditional on two fixed propagation intervals; limited realization/component count; no confirmed independent-trial inference','empty_two_way_draws':empty})
 np.savez_compressed(T/f'D2_{label}_DEPENDENCE_BOOTSTRAP_DRAWS.npz',**{k:np.array(v) for k,v in draws.items()})
csv('D2_110S_DEPENDENCE_SENSITIVITY.csv',sensitivity)
js('D2_DEPENDENCE_STRUCTURE.json',{'eligible_evaluation_blocks':len(blocks),'receiver_identities':len(ru),'propagation_intervals':int(blocks.interval_id.nunique()),'within_block_scenarios':6,'reused_random_realizations':len(su),'receiver_realization_components':len(cu),'component_sizes':blocks.groupby('component').size().to_dict(),'source_holdout_ids':int(batch.source_holdout_id.nunique()),'components_conditional_on_fixed_intervals':True,'all_blocks_independent':False})
contribution=[];removals=[]
pivot=long.pivot(index=['batch_id','block_id','scenario_code'],columns='method',values='Lcap1000')
for comp in ['ALWAYS_M2','ALWAYS_M12']:
 delta=(pivot.EGSHS-pivot[comp]).groupby('scenario_code').mean();net=delta.mean();den=delta.abs().sum()
 for scene,v in delta.items():contribution.append({'comparator':comp,'scenario':scene,'scenario_mean_difference':v,'signed_contribution':v/6,'signed_share_of_net_difference':v/(6*net) if abs(net)>1e-12 else np.nan,'absolute_contribution':abs(v)/6,'absolute_share_of_scenario_contrasts':abs(v)/den if den>1e-12 else np.nan,'overall_difference':net,'remove_H0':delta.drop('H0').mean(),'remove_H5':delta.drop('H5').mean(),'remove_H0_and_H5':delta.drop(['H0','H5']).mean(),'interpretation':'scenario contribution is descriptive accounting, not causal attribution'})
 for name,removed in [('remove_H0',['H0']),('remove_H5',['H5']),('remove_H0_and_H5',['H0','H5'])]:
  q=pivot.loc[~pivot.index.get_level_values('scenario_code').isin(removed)]
  removals.append({'comparator':comp,'scope':name,'remaining_scenarios':6-len(removed),'blocks':q.index.get_level_values('block_id').nunique(),'EGSHS_Lcap1000':q.EGSHS.groupby('block_id').mean().mean(),'comparator_Lcap1000':q[comp].groupby('block_id').mean().mean(),'difference':(q.EGSHS-q[comp]).groupby('block_id').mean().mean(),'role':'DESCRIPTIVE_SENSITIVITY_ONLY'})
csv('D2_110S_SCENARIO_CONTRIBUTION.csv',contribution);csv('D2_110S_SCENARIO_REMOVAL.csv',removals)
metrics=['block_equal_Lcap1000','batch_median_error_m_finite_only','batch_p75_error_m_finite_only','batch_p95_error_m_finite_only','RMSE_m_finite_only','batch_max_error_m_finite_only','gt_1000m_count_all_batches','gt_2000m_count_all_batches','failures']
tail=summary.loc[summary.method.isin(['EGSHS','ALWAYS_M2','ALWAYS_M12','BIC']),['method']+metrics];csv('D2_110S_TAIL_COMPARISON.csv',tail)
ss=summary.set_index('method');bic=[]
for metric in metrics:
 bic.append({'metric':metric,'EGSHS':ss.loc['EGSHS',metric],'BIC':ss.loc['BIC',metric],'EGSHS_minus_BIC':ss.loc['EGSHS',metric]-ss.loc['BIC',metric],'BIC_lowest_Lcap_among_compatible_IC_selectors':bool(ss.loc['BIC','block_equal_Lcap1000']==ss.loc[['AIC','FSC','BIC'],'block_equal_Lcap1000'].min()),'scope':'AIC/FSC/BIC all use frozen M0-M4 compatible subset; EGSHS targets full heterogeneous candidate pool'})
csv('D2_110S_BIC_COMPARISON.csv',bic)
js('EVALUATION_COMPLETED.json',{'created_utc':utc(),'batch_method_records':len(long),'block_method_records':len(block),'dependence_procedures':5,'dependence_comparisons':2,'dependence_sensitivity_complete':len(sensitivity)==10 and all(x['replications']==10000 for x in sensitivity),'scenario_decomposition_complete':True,'truth_110_verified':True,'evaluation_script_SHA256':sha(Path(__file__))})
print('evaluation and five dependence procedures completed',len(long))
