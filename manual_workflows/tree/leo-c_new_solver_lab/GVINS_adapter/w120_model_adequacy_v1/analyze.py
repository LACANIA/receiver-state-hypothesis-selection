from common import *
import pandas as pd

TABLE_NAMES=['reference_motion_model_adequacy.csv','support_control_inventory.csv','oracle_residual_decomposition.csv','clock_temporal_adequacy.csv','trajectory_signature_projection.csv','weak_direction_motion_alignment.csv','satellite_support_effect.csv','m2_oracle_comparison.csv','residual_blindness_interpretation.csv']

def motion_characterization(T,refs,origin,end):
    t=np.array([(r['gpst_calendar_ns']-ORIGIN)/1e9 for r in refs]);p=np.array([vec(r,'primary_p') for r in refs])
    center=float(t.mean());base=p.mean(axis=0);X=np.column_stack([np.ones(len(t)),t-center]);coef=np.linalg.lstsq(X,p-base,rcond=None)[0]
    vcv=coef[1];p0=base+coef[0]-center*vcv;pcv=p0+t[:,None]*vcv
    dev=np.linalg.norm(p-pcv,axis=1);frame=enu(vec(end,'primary_p'))
    rows=[];profiles={}
    for velocity,prefix in [('PVT_PRIMARY','primary_v'),('POSITION_DERIVED_SENSITIVITY','position_derived_v')]:
        v=np.array([vec(r,prefix) for r in refs]);valid=np.isfinite(v).all(axis=1);a=derivative(t,v);speed=np.linalg.norm(v,axis=1)
        vdev=np.linalg.norm(v-vcv,axis=1);meanv=np.mean(v[valid],axis=0);vary=np.linalg.norm(v-meanv,axis=1)
        accel=np.linalg.norm(a,axis=1);curv=np.full(len(t),np.nan);heading=np.full(len(t),np.nan);normal=np.full_like(a,np.nan)
        for i in range(len(t)):
            ve=frame@v[i]
            if valid[i] and np.hypot(ve[0],ve[1])>0:heading[i]=np.arctan2(ve[0],ve[1])
            if speed[i]>0 and np.isfinite(a[i]).all():
                normal[i]=a[i]-v[i]*(a[i]@v[i])/speed[i]**2
                curv[i]=np.linalg.norm(np.cross(v[i],a[i]))/speed[i]**3
        dh=[]
        for i in range(1,len(t)):
            if t[i]-t[i-1]<=1.001 and np.isfinite(heading[i-1:i+1]).all():dh.append(np.arctan2(np.sin(heading[i]-heading[i-1]),np.cos(heading[i]-heading[i-1])))
        row=dict(window=T,velocity=velocity,epochs=len(t),velocity_epochs=int(valid.sum()),sample_first_s=t[0],sample_last_s=t[-1],sample_span_s=t[-1]-t[0],
            sampled_displacement_m=np.linalg.norm(p[-1]-p[0]),nominal_endpoint_displacement_m=np.linalg.norm(vec(end,'primary_p')-vec(origin,'primary_p')),
            path_length_m=np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)),nominal_displacement_mean_speed_mps=np.linalg.norm(vec(end,'primary_p')-vec(origin,'primary_p'))/T,
            mean_velocity_norm_mps=np.linalg.norm(meanv),cv_speed_mps=np.linalg.norm(vcv),speed_median_mps=np.nanmedian(speed),speed_p95_mps=np.nanquantile(speed,.95),speed_max_mps=np.nanmax(speed),
            velocity_variation_rms_mps=np.sqrt(np.nanmean(vary*vary)),acceleration_proxy_rms_mps2=np.sqrt(np.nanmean(accel*accel)),acceleration_proxy_p95_mps2=np.nanquantile(accel,.95),
            heading_total_variation_deg=np.degrees(np.sum(abs(np.array(dh)))),heading_signed_change_deg=np.degrees(np.sum(dh)),curvature_proxy_median_per_m=np.nanmedian(curv),curvature_proxy_p95_per_m=np.nanquantile(curv,.95),
            position_CV_deviation_rms_m=rms(dev),position_CV_deviation_p95_m=np.quantile(dev,.95),position_CV_deviation_max_m=np.max(dev),
            velocity_CV_deviation_rms_mps=np.sqrt(np.nanmean(vdev*vdev)),velocity_CV_deviation_p95_mps=np.nanquantile(vdev,.95),
            CV_endpoint_error_m=np.linalg.norm(p0+T*vcv-vec(end,'primary_p')),CV_trajectory_fit='POSITION_ONLY_OLS_ONE_ROW_PER_EPOCH_NOT_DOPPLER_POSITIONING',
            acceleration_identity='CENTERED_DIFFERENCE_OF_CANONICAL_VELOCITY; edges/gaps missing',curvature_identity='norm(v cross a)/norm(v)^3; zero speed undefined; low speed can amplify noise')
        for k,val in [('mean_velocity',meanv),('cv_p0',p0),('cv_velocity',vcv)]:row.update({k+'_'+c:val[j] for j,c in enumerate('xyz')})
        rows.append(row);profiles[velocity]=dict(t=t,p=p,v=v,a=a,normal=normal,heading=heading,curvature=curv,position_deviation=dev,velocity_deviation=vdev)
    return rows,profiles,p0,vcv

def physical(obs,p,v):
    out=[];jac=[];meta=[]
    for o,pp,vv in zip(obs,p,v):
        if not np.isfinite(vv).all():out.append(np.nan);jac.append(np.full(7,np.nan));meta.append(None);continue
        y,g,m=w.predict_epoch(o,pp,vv,0.,True);out.append(y);jac.append(g);meta.append(m)
    return np.array(out),np.array(jac),meta

def main():
    assert not (O/'analysis_tables.json').exists()
    protocol=read(O/'RUN_PROTOCOL.json');assert protocol['new_candidate_fits']==0
    for x in protocol['source_bindings']:assert sha(x['path'])==x['sha256'],x['path']
    arc.check_sources()
    save(O/'ANALYSIS_EXECUTION_SEAL.json',dict(utc=now(),run_id=protocol['run_id'],files=[dict(path=str(O/n),sha256=sha(O/n)) for n in ['RUN_PROTOCOL.json','common.py','analyze.py']],new_candidate_fits=0))
    observations=read(S/'candidate_observations.json');lineage=read(S/'row_lineage.json');references=read(S/'reference_qa_only.json');fits=read(S/'fit_tables.json')['initialization_fit_results.csv'];oldinf=read(S/'information_full.json')
    tables={n:[] for n in TABLE_NAMES};details={};oracle_rows=[];per_sat=[];temporal=[];clocks=[];physics_meta=[];checks=[];support_info={};decomps={};budget=[]
    line={(r['gpst_ns'],r['satellite']):r for r in lineage};refmap={r['gpst_calendar_ns']:r for r in references['reference']}
    for T in WINDOWS:
        print(f'WINDOW_START=W{T}',flush=True)
        rr=[r for r in references['reference'] if r['gpst_calendar_ns']<ORIGIN+T*10**9 and r['reference_eligible']]
        mr,profiles,p0,vcv=motion_characterization(T,rr,references['origin'],references['endpoints'][str(T)])
        tables['reference_motion_model_adequacy.csv']+=mr;details[f'motion_{T}']=profiles
        rec=[r for r in observations if r['receive_gpst_ns']<ORIGIN+T*10**9];obs=[w.observation_from_mapping(r) for r in rec]
        epochs=np.array([o.receive_gpst_ns for o in obs],dtype=np.int64);t=(epochs-ORIGIN)/1e9;sat=np.array([o.satellite for o in obs]);y=np.array([o.y_mps for o in obs]);sigma=np.array([line[(o.receive_gpst_ns,o.satellite)]['sigma_y_mps'] for o in obs])
        refrows=[refmap[o.receive_gpst_ns] for o in obs];pr=np.array([vec(r,'primary_p') for r in refrows]);vr=np.array([vec(r,'primary_v') for r in refrows]);vsens=np.array([vec(r,'position_derived_v') for r in refrows]);ph=np.array([vec(r,'hermite_p') for r in refrows]);vh=np.array([vec(r,'hermite_v') for r in refrows]);pc=p0+t[:,None]*vcv;vc=np.tile(vcv,(len(t),1))
        gr,jr,metar=physical(obs,pr,vr);gc,jc,metac=physical(obs,pc,vc);gs,js,metas=physical(obs,pr,vsens);gh,_,_=physical(obs,ph,vh)
        for r,meta in zip(rec,metar):physics_meta.append(dict(window=T,epoch_ns=r['receive_gpst_ns'],satellite=r['satellite'],**meta))
        refs_for_prediction={'PVT_PRIMARY':(gr,jr,vr),'POSITION_DERIVED_SENSITIVITY':(gs,js,vsens)}
        supports={'SUPPORT_A_NATURAL':np.ones(len(t),bool),'SUPPORT_B_W30_SATELLITES':np.isin(sat,SAME6)}
        if T==120:supports['SUPPORT_C_W120_WITHOUT_PRN15']=sat!=15
        if T==120:assert np.array_equal(supports['SUPPORT_B_W30_SATELLITES'],supports['SUPPORT_C_W120_WITHOUT_PRN15'])
        frame=enu(vec(references['endpoints'][str(T)],'primary_p'))
        for support,mask0 in supports.items():
            idx=np.flatnonzero(mask0);u,ct=np.unique(epochs[idx],return_counts=True)
            selectedline=[line[(int(epochs[i]),int(sat[i]))] for i in idx]
            tables['support_control_inventory.csv'].append(dict(window=T,support=support,rows=len(idx),epochs=len(u),satellites=len(set(sat[idx])),prns=sorted(set(sat[idx])),min_rows_per_epoch=min(ct),max_rows_per_epoch=max(ct),
                first_s=min(t[idx]),last_s=max(t[idx]),PRN15_rows=int(sum(sat[idx]==15)),missing_from_complete_same6=int(6*len(u)-sum(np.isin(sat[idx],SAME6))),
                causal_ephemeris=True,max_abs_ephemeris_age_s=max(abs((obs[i].receive_gpst_ns-obs[i].ephemeris.toe_ns)/1e9) for i in idx),
                input_row_identity_sha256=hashlib.sha256(json.dumps([(q['row_id'],q['ephemeris_id']) for q in selectedline]).encode()).hexdigest(),
                redundancy='EXACT_ALIAS_OF_SUPPORT_B' if support.startswith('SUPPORT_C') else ('EXACT_ROWS_AS_A' if support.startswith('SUPPORT_B') and T<120 else 'DISTINCT')))
            for velocity,(gp,jp,vp) in refs_for_prediction.items():
                mask=mask0&np.isfinite(gp);ix=np.flatnonzero(mask);tt=t[ix];ss=sat[ix];yy=y[ix];sig=sigma[ix];ee=epochs[ix]
                un,inv,counts=np.unique(ee,return_inverse=True,return_counts=True);te=(un-ORIGIN)/1e9
                meanA=np.bincount(inv,weights=yy-gp[ix])/counts;meanC=np.bincount(inv,weights=yy-gc[ix])/counts
                bB,bcoef=row_clock(tt,yy-gp[ix],1);bD,dcoef=row_clock(tt,yy-gc[ix],1)
                rA=yy-gp[ix]-meanA[inv];rB=yy-gp[ix]-bB;rC=yy-gc[ix]-meanC[inv];rD=yy-gc[ix]-bD
                case_res={'A':rA,'B':rB,'C':rC,'D':rD};case_cl={'A':meanA[inv],'B':bB,'C':meanC[inv],'D':bD}
                case_geom={'A':gp[ix],'B':gp[ix],'C':gc[ix],'D':gc[ix]}
                key=f'{T}_{support}_{velocity}'
                for case,res in case_res.items():
                    st=stats(res);row=dict(window=T,support=support,velocity=velocity,case=case,rows=len(ix),epochs=len(un),satellites=len(set(ss)),residual_median_mps=st['median'],residual_rms_mps=st['rms'],residual_mse_mps2=st['mse'],residual_abs_p95_mps=st['abs_p95'],residual_abs_max_mps=st['max_abs'],
                        sigma_median_mps=np.median(sig),sigma_rms_mps=rms(sig),normalized_residual_rms=rms(res/sig),clock_b0_mps=bcoef[0] if case=='B' else dcoef[0] if case=='D' else None,clock_bdot_mps2=bcoef[1] if case=='B' else dcoef[1] if case=='D' else None,
                        weighting='NATIVE_UNIT_ROWS',identity='ORACLE_POSTHOC_FIXED_TRAJECTORY_CLOCK_DIAGNOSTIC_NOT_POSITIONING')
                    tables['oracle_residual_decomposition.csv'].append(row)
                    for s in sorted(set(ss)):
                        z=ss==s;stt=stats(res[z]);per_sat.append(dict(window=T,support=support,velocity=velocity,case=case,prn=int(s),**stt,lag1=autocorr(tt[z],res[z],1),lag5=autocorr(tt[z],res[z],5)))
                    bins=np.floor(tt/10).astype(int)
                    for b in sorted(set(bins)):
                        z=bins==b;temporal.append(dict(window=T,support=support,velocity=velocity,case=case,bin_start_s=int(b*10),**stats(res[z]),mean_mps=np.mean(res[z])))
                metric={c:dict(rms=rms(z),mse=np.mean(z*z)) for c,z in case_res.items()}
                br=dict(window=T,support=support,velocity=velocity)
                for measure in ['rms','mse']:
                    a,b,c,d=[metric[k][measure] for k in 'ABCD']
                    br.update({f'{measure}_clock_form_cost':b-a,f'{measure}_motion_model_cost':c-a,f'{measure}_combined_model_cost':d-a,f'{measure}_interaction':d-b-c+a})
                budget.append(br);decomps[key]=dict(metric=metric,bcoef=bcoef,dcoef=dcoef,rows=len(ix))
                for degree,name in [(0,'CONSTANT'),(1,'AFFINE'),(2,'QUADRATIC_DIAGNOSTIC_NOT_A_CANDIDATE')]:
                    cb,coeff=row_clock(tt,meanA[inv],degree);crem=meanA[inv]-cb;epochsrem=np.bincount(inv,weights=crem)/counts
                    crow=dict(window=T,support=support,velocity=velocity,form=name,epochs=len(un),rows=len(ix),b0_mps=coeff[0],bdot_mps2=coeff[1],quadratic_coefficient_mps3=coeff[2],
                        row_weighted_remainder_rms_mps=rms(crem),epoch_remainder_rms_mps=rms(epochsrem),epoch_remainder_abs_p95_mps=np.quantile(abs(epochsrem),.95),epoch_remainder_abs_max_mps=np.max(abs(epochsrem)),
                        lag1=autocorr(te,epochsrem,1),lag5=autocorr(te,epochsrem,5),common_clock_min_mps=min(meanA),common_clock_max_mps=max(meanA),clock_series_identity='QA_COMMON_COMPONENT_INCLUDES_MEASUREMENT_AND_REFERENCE_ERRORS')
                    binmeans=[np.mean(epochsrem[(te>=b)&(te<b+10)]) for b in np.arange(0,T,10) if np.any((te>=b)&(te<b+10))]
                    crow['ten_second_binmean_rms_mps']=rms(binmeans);tables['clock_temporal_adequacy.csv'].append(crow)
                for i,et in enumerate(un):clocks.append(dict(window=T,support=support,velocity=velocity,epoch_ns=int(et),t_s=te[i],rows=int(counts[i]),common_A_mps=meanA[i],common_C_mps=meanC[i]))
                Xclock=np.column_stack([np.ones(len(ix)),tt]);E=np.eye(len(un))[inv]
                h=gp[ix]-gc[ix];delta_p=pr[ix]-pc[ix];delta_v=vp[ix]-vc[ix]
                hlinear=np.einsum('ij,ij->i',jc[ix,:3],delta_p)+np.einsum('ij,ij->i',jc[ix,3:6],delta_v)
                checks.append(dict(kind='RESIDUAL_C_MINUS_A_PROJECTION',key=key,max_abs_error=np.max(abs((rC-rA)-(h-project(E,h))))))
                # Nested clock restriction identity on MSE, with row weights fixed.
                checks.append(dict(kind='CLOCK_MSE_COST_IDENTITY',key=key,max_abs_error=abs((np.mean(rB*rB)-np.mean(rA*rA))-np.mean((meanA[inv]-bB)**2))))
                # No iid assumption: this is a nominal independent-row sigma diagnostic only.
                sigma_floor=np.sqrt(np.sum(sig**2*(1-1/counts[inv]))/len(ix))
                info,Sp,C,dirs=arc.information(*arc.design(jp[ix,:6],tt,T,'M2'),frame)
                support_info[key]=info
                if support=='SUPPORT_A_NATURAL':
                    old=next(z for z in oldinf if z['window']==T and z['model']=='M2' and z['velocity']==velocity and z['support']=='NATURAL_SUPPORT')
                    checks.append(dict(kind='INFORMATION_REPLAY',key=key,lambda_relative_error=abs(info['lambda_min']/old['lambda_min']-1)))
                for basis,gpoint in [('REFERENCE_COMMON_TANGENT',jp[ix]),('CV_TRAJECTORY_POINT',jc[ix])]:
                    Ap,N=arc.design(gpoint[:,:6],tt,T,'M2');ih,_,CC,dd=arc.information(Ap,N,frame)
                    phE=project(E,h);phclock=project(Xclock,h);phN=project(N,h);rh=h-phN
                    weak=dd[:,0];cw=CC@weak;projweak=cw*(cw@rh)/(cw@cw)
                    den=h@h;denrem=rh@rh
                    response=np.linalg.lstsq(CC,rh,rcond=None)[0]
                    pjr=dict(window=T,support=support,velocity=velocity,basis=basis,rows=len(ix),signature_rms_mps=rms(h),signature_abs_p95_mps=np.quantile(abs(h),.95),
                        linearization_error_rms_mps=rms(hlinear-h),linearization_error_abs_p95_mps=np.quantile(abs(hlinear-h),.95),linearization_relative_norm=np.linalg.norm(hlinear-h)/np.linalg.norm(h),
                        exact_total_caseC_minus_caseA_prediction_rms_mps=rms(rC-rA),
                        per_epoch_clock_absorbed_energy_fraction=phE@phE/den,affine_clock_absorbed_energy_fraction=phclock@phclock/den,M2_nuisance_absorbed_energy_fraction=phN@phN/den,
                        after_per_epoch_clock_signature_rms_mps=rms(h-phE),after_affine_clock_signature_rms_mps=rms(h-phclock),after_M2_nuisance_signature_rms_mps=rms(rh),
                        weak_signature_fraction_of_remaining_energy=projweak@projweak/denrem if denrem>0 else None,weak_equivalent_position_coefficient_m=(cw@rh)/(cw@cw),
                        linear_position_response_norm_m=np.linalg.norm(response),linear_position_response_x=response[0],linear_position_response_y=response[1],linear_position_response_z=response[2],
                        projection_identity='ENERGY_PROJECTIONS_NOT_CAUSAL_ATTRIBUTION; linear position response not an optimizer output')
                    tables['trajectory_signature_projection.csv'].append(pjr)
                if velocity=='PVT_PRIMARY':
                    rh=yy-gh[ix];cm=np.bincount(inv,weights=rh)/counts;resH=rh-cm[inv]
                    details[key]=dict(information=info,cv_p0=p0,cv_v=vcv,cv_qT=p0+T*vcv,
                        normalized_floor_model_sigma_mps=sigma_floor,raw_sigma_median_mps=np.median(sig),caseA_rms_over_nominal_sigma_floor=rms(rA)/sigma_floor,
                        hermite_caseA_rms_mps=rms(resH),hermite_prediction_difference_rms_mps=rms(gh[ix]-gp[ix]),hermite_after_clock_prediction_difference_rms_mps=rms(resH-rA),
                        ref_velocity_accuracy_median_mps=np.median([refrows[i]['vel_acc'] for i in ix]),
                        clock_A=meanA,clock_B=bcoef,clock_C=meanC,clock_D=dcoef)
                    for loc,i in enumerate(ix):
                        oracle_rows.append(dict(window=T,support=support,source_index=int(i),epoch_ns=int(ee[loc]),t_s=tt[loc],satellite=int(ss[loc]),y_mps=yy[loc],sigma_y_mps=sig[loc],
                            prediction_reference_mps=gp[i],prediction_CV_mps=gc[i],prediction_Hermite_mps=gh[i],prediction_position_velocity_mps=gs[i],
                            **{f'residual_{k}_mps':case_res[k][loc] for k in 'ABCD'},**{f'clock_{k}_mps':case_cl[k][loc] for k in 'ABCD'},
                            mismatch_signature_mps=h[loc],linear_signature_mps=hlinear[loc],source_row_id=line[(int(ee[loc]),int(ss[loc]))]['row_id']))
                    # Position/motion direction alignment uses the same support-specific reference information axis.
                    prof=profiles['PVT_PRIMARY'];vel=prof['v'];acc=prof['a'];normal=prof['normal'];weak=np.array(info['weak_ecef'])
                    for kind,array in [('VELOCITY',vel),('ACCELERATION',acc),('CURVATURE_NORMAL',normal),('POSITION_MINUS_CV',prof['p']-(p0+prof['t'][:,None]*vcv)),('VELOCITY_MINUS_CV',vel-vcv)]:
                        axes=np.array([cosine(v,weak,True) for v in array]);signed=np.array([cosine(v,weak,False) for v in array]);good=np.isfinite(axes)
                        tables['weak_direction_motion_alignment.csv'].append(dict(window=T,support=support,kind=kind,n=int(good.sum()),axis_angle_median_deg=np.nanmedian(axes),axis_angle_p95_deg=np.nanquantile(axes,.95),signed_angle_median_deg=np.nanmedian(signed),
                            weak_ecef=info['weak_ecef'],weak_enu=info['weak_enu'],identity='AXIS_ANGLE_0_TO_90; signed angle also retained; no causal inference'))
                    for fit in [f for f in fits if f['model']=='M2' and f['window']==T and f['direction']=='LEGACY_D0' and f['radius_m'] in [0,arc.R616]]:
                        error=np.array([fit['error_'+c] for c in 'xyz']);frac=float((error@weak)**2/(error@error))
                        tables['weak_direction_motion_alignment.csv'].append(dict(window=T,support=support,kind=f"SAVED_M2_ERROR_{fit['fit_id']}",n=1,axis_angle_median_deg=cosine(error,weak,True),axis_angle_p95_deg=cosine(error,weak,True),signed_angle_median_deg=cosine(error,weak),error_squared_weak_fraction=frac,weak_ecef=info['weak_ecef'],weak_enu=info['weak_enu'],identity='FIXED_SAVED_ENDPOINT_ERROR_NO_REFIT'))
                        if support=='SUPPORT_A_NATURAL':checks.append(dict(kind='SAVED_ERROR_ALIGNMENT',key=fit['fit_id'],abs_difference=abs(frac-fit['error_weak_fraction'])))
                        state=np.r_[[fit['p0_'+c] for c in 'xyz'],[fit['velocity_'+c] for c in 'xyz'],fit['b0_mps'],fit['bdot_mps2']]
                        pred,_,_=w.predict('M2',state,[obs[i] for i in ix],ORIGIN,False);rf=yy-pred
                        qactual=state[:3]+T*state[3:6];qoracle=p0+T*vcv
                        cr=dict(window=T,support=support,fit_id=fit['fit_id'],radius_m=fit['radius_m'],fit_source=fit['source'],fit_support='SAVED_NATURAL_SUPPORT; evaluated on stated subset without refit',rows=len(ix),
                            actual_residual_rms_mps=rms(rf),oracle_D_residual_rms_mps=rms(rD),residual_M2_minus_D_mps=rms(rf)-rms(rD),
                            actual_endpoint_error_m=fit['endpoint_error_m'],oracle_CV_endpoint_error_m=np.linalg.norm(qoracle-vec(references['endpoints'][str(T)],'primary_p')),
                            endpoint_actual_minus_oracle_distance_m=np.linalg.norm(qactual-qoracle),p0_parameter_difference_m=np.linalg.norm(state[:3]-p0),velocity_parameter_difference_mps=np.linalg.norm(state[3:6]-vcv),
                            actual_b0_mps=state[6],oracle_D_b0_mps=dcoef[0],b0_difference_mps=state[6]-dcoef[0],actual_bdot_mps2=state[7],oracle_D_bdot_mps2=dcoef[1],bdot_difference_mps2=state[7]-dcoef[1],
                            full_converged=fit['full_converged'],comparison_identity='TRUTH_ASSOCIATED_CV_ORACLE_VS_SAVED_DOPPLER_ESTIMATE')
                        tables['m2_oracle_comparison.csv'].append(cr)
                        if support=='SUPPORT_A_NATURAL':checks.append(dict(kind='SAVED_M2_PREDICTION_REPLAY',key=fit['fit_id'],rms_abs_error=abs(rms(rf)-fit['full_residual_rmse_mps'])))
                # Save complete aligned sensitivity decomposition, never hide its missing rows.
                details[key+'_sensitivity_counts']=dict(rows=len(ix),natural_support_rows=len(idx),dropped_nonfinite_reference_velocity=len(idx)-len(ix))
        print(f'WINDOW_DONE=W{T}',flush=True)
    # Support controls are conditional comparisons of identical remaining rows, not causal percentages.
    for T in WINDOWS:
        for support in ['SUPPORT_A_NATURAL','SUPPORT_B_W30_SATELLITES']+(['SUPPORT_C_W120_WITHOUT_PRN15'] if T==120 else []):
            key=f'{T}_{support}_PVT_PRIMARY';i=support_info[key];base=support_info['30_SUPPORT_B_W30_SATELLITES_PVT_PRIMARY'];n=support_info[f'{T}_SUPPORT_A_NATURAL_PVT_PRIMARY']
            rec=dict(window=T,support=support,rows=i['rows'],lambda_min=i['lambda_min'],condition=i['position_condition'],weak_proxy_m=i['weak_position_std_proxy_m'],weak_ecef=i['weak_ecef'],weak_enu=i['weak_enu'],
                lambda_ratio_to_W30_same6=i['lambda_min']/base['lambda_min'],natural_lambda_div_this=n['lambda_min']/i['lambda_min'],information_basis='COMMON_REFERENCE_TRAJECTORY_TANGENT; NATIVE_UNIT_ROWS',row_identity='C_EQUALS_B' if support.startswith('SUPPORT_C') else 'FIXED_PREDECLARED')
            for c in 'ABCD':rec[f'case_{c}_rms_mps']=decomps[key]['metric'][c]['rms']
            tables['satellite_support_effect.csv'].append(rec)
    for T in [30,120]:
        for rad in [0.,arc.R616]:
            for a,b in [('M1','M2'),('M1','M3'),('M0','M4')]:
                aa=next(f for f in fits if f['model']==a and f['window']==T and f['direction']=='LEGACY_D0' and abs(f['radius_m']-rad)<1e-6)
                bb=next(f for f in fits if f['model']==b and f['window']==T and f['direction']=='LEGACY_D0' and abs(f['radius_m']-rad)<1e-6)
                dr=bb['full_residual_rmse_mps']-aa['full_residual_rmse_mps'];de=bb['endpoint_error_m']-aa['endpoint_error_m'];dv=bb['heldout_trimmed_rmse_mps']-aa['heldout_trimmed_rmse_mps']
                tables['residual_blindness_interpretation.csv'].append(dict(window=T,radius_m=rad,pair=a+'__'+b,fit_A=aa['fit_id'],fit_B=bb['fit_id'],delta_residual_mps=dr,delta_validation_mps=dv,delta_endpoint_error_m=de,residual_position_reversal=dr*de<0,validation_position_reversal=dv*de<0,full_A_converged=aa['full_converged'],full_B_converged=bb['full_converged'],
                    state_change='VELOCITY_AND_CLOCK_SLOPE' if b=='M2' else 'VELOCITY_ONLY_MATCHED_CONSTANT_BIAS' if a=='M1' else 'VELOCITY_ONLY_BOTH_MISSING_RECEIVER_CLOCK',
                    interpretation_scope='CONDITION_SPECIFIC_SAVED_PAIR; WEAK_INFORMATION_AND_MISMATCH_NOT_INDEPENDENTLY_CAUSAL'))
    # Budget is carried alongside each case, with identical source support/denominator.
    budgetmap={(r['window'],r['support'],r['velocity']):r for r in budget}
    for row in tables['oracle_residual_decomposition.csv']:
        row.update({k:v for k,v in budgetmap[(row['window'],row['support'],row['velocity'])].items() if k not in ['window','support','velocity']})
    (O/'raw').mkdir(exist_ok=True)
    pd.DataFrame(oracle_rows).to_parquet(O/'raw/oracle_rows.parquet',index=False)
    pd.DataFrame(clocks).to_parquet(O/'raw/common_clock_epochs.parquet',index=False)
    pd.DataFrame(per_sat).to_parquet(O/'raw/per_satellite_residuals.parquet',index=False)
    pd.DataFrame(temporal).to_parquet(O/'raw/temporal_residual_bins.parquet',index=False)
    save(O/'analysis_tables.json',tables);save(O/'diagnostic_details.json',details);save(O/'local_information.json',support_info);save(O/'physics_metadata.json',physics_meta);save(O/'numerical_checks.json',checks)
    save(O/'ANALYSIS_COMPLETE.json',dict(utc=now(),run_id=protocol['run_id'],new_candidate_fits=0,EGSHS_runs=0,source_wrappers_modified=False,rows_saved=len(oracle_rows),source_files_changed=[x['path'] for x in protocol['source_bindings'] if sha(x['path'])!=x['sha256']],loaded_scientific_runtime_modules=[k for k in sys.modules if k.startswith('leo_positioning')]))
    print('ANALYSIS_COMPLETE; NEW_CANDIDATE_FITS=0',flush=True)

if __name__=='__main__':main()
