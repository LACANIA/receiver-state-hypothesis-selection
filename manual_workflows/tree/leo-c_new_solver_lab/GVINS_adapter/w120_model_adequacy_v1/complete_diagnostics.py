from common import *
import pandas as pd

def main():
    assert not (O/'final_tables.json').exists()
    # This is an explicit local linear response calculation, not a nonlinear
    # candidate solve. The source A-D residuals and comparison scope are fixed.
    save(O/'LOCAL_RESPONSE_DIAGNOSTIC_LOCK.json',dict(utc=now(),definition='At the saved oracle CV trajectory, K=pinv(P_N_perp Jp) P_N_perp. Apply separately to rA, bA-bB, (I-P_affine)h and rD. Signed vectors add; norms do not. Compare sum to saved M2-minus-oracle q(T). No optimizer call.',purpose='Separate contributions to the remaining W120 endpoint displacement without claiming causal identification',
        same_row_sensitivity='Recompute PVT A-D on the exact rows where frozen position-derived velocity exists; never compare differing denominators as a pure velocity effect',new_candidate_fits=0))
    a=read(O/'analysis_tables.json');dd=read(O/'diagnostic_details.json');z=pd.read_parquet(O/'raw/oracle_rows.parquet');obsall=read(S/'candidate_observations.json');fit=read(S/'fit_tables.json')['initialization_fit_results.csv'];responses=[];same=[];tests=[]
    for (T,support),g in z.groupby(['window','support'],sort=False):
        key=f'{T}_{support}_PVT_PRIMARY';t=g.t_s.to_numpy();p0=np.array(dd[key]['cv_p0']);v=np.array(dd[key]['cv_v'])
        records=[r for r in obsall if r['receive_gpst_ns']<ORIGIN+int(T)*10**9];ix=g.source_index.to_numpy(int);obs=[w.observation_from_mapping(records[i]) for i in ix]
        state=np.r_[p0,v,dd[key]['clock_D'][:2]]
        predicted,J,_=w.predict('M2',state,obs,ORIGIN,True)
        Ap=J[:,:3];N=J[:,3:].copy();N[:,:3]-=float(T)*Ap
        PP=Ap-project(N,Ap)
        # Thin SVD pseudoinverse acts only on residualized position; no normal inverse.
        u,s,vt=np.linalg.svd(PP,full_matrices=False);K=(vt.T/s)@u.T
        h=g.mismatch_signature_mps.to_numpy();X=np.column_stack([np.ones(len(t)),t]);motion=h-project(X,h)
        components={'MEASUREMENT_REFERENCE_FLOOR':g.residual_A_mps.to_numpy(),'COMMON_CLOCK_AFFINE_REMAINDER':(g.clock_A_mps-g.clock_B_mps).to_numpy(),'CV_TRAJECTORY_SIGNATURE_AFTER_AFFINE':motion,'COMBINED_D':g.residual_D_mps.to_numpy()}
        closure=components['MEASUREMENT_REFERENCE_FLOOR']+components['COMMON_CLOCK_AFFINE_REMAINDER']+components['CV_TRAJECTORY_SIGNATURE_AFTER_AFFINE']-components['COMBINED_D']
        assert np.max(abs(closure))<1e-10
        weak=arc.signfix(vt[-1]);vv={}
        for name,res in components.items():
            q=K@res;vv[name]=q
            responses.append(dict(window=int(T),support=support,component=name,linear_position_response_ecef_m=q,linear_position_response_norm_m=np.linalg.norm(q),signed_weak_response_m=q@weak,
                response_identity='LOCAL_UNCALIBRATED_LINEAR_RESPONSE_NOT_CAUSAL_ERROR_ALLOCATION',residual_component_rms_mps=rms(res)))
        actual=next((f for f in fit if f['model']=='M2' and f['window']==T and f['radius_m']==0 and f['direction']=='LEGACY_D0'),None)
        if actual is not None:
            aq=np.array([actual['qT_'+c] for c in 'xyz'])-(p0+float(T)*v)
            tests.append(dict(window=int(T),support=support,residual_closure_max_mps=np.max(abs(closure)),linear_response_sum_error_m=np.linalg.norm(vv['COMBINED_D']-sum(vv[k] for k in components if k!='COMBINED_D')),
                actual_minus_oracle_q_ecef_m=aq,actual_minus_oracle_distance_m=np.linalg.norm(aq),linear_predicted_delta_q_m=vv['COMBINED_D'],linear_vs_actual_delta_q_difference_m=np.linalg.norm(vv['COMBINED_D']-aq),
                identity='Saved M2 is NATURAL fit even when evaluated on subset; only natural comparison is its own linear optimum comparison'))
        valid=np.isfinite(g.prediction_position_velocity_mps.to_numpy());q=g.loc[valid];tt=q.t_s.to_numpy();_,inv,count=np.unique(q.epoch_ns.to_numpy(),return_inverse=True,return_counts=True)
        for velocity,col in [('PVT_PRIMARY','prediction_reference_mps'),('POSITION_DERIVED_SENSITIVITY','prediction_position_velocity_mps')]:
            yy=q.y_mps.to_numpy();pr=q[col].to_numpy();pc=q.prediction_CV_mps.to_numpy();clockA=np.bincount(inv,weights=yy-pr)/count;clockC=np.bincount(inv,weights=yy-pc)/count
            clB,b=row_clock(tt,yy-pr,1);clD,d=row_clock(tt,yy-pc,1)
            rr={'A':yy-pr-clockA[inv],'B':yy-pr-clB,'C':yy-pc-clockC[inv],'D':yy-pc-clD}
            same.append(dict(window=int(T),support=support,velocity=velocity,rows=len(q),**{k+'_rms_mps':rms(x) for k,x in rr.items()},clock_cost_mse=np.mean(rr['B']**2)-np.mean(rr['A']**2),motion_cost_mse=np.mean(rr['C']**2)-np.mean(rr['A']**2)))
    interpretation={
        (30,'M1__M2'):'Velocity plus clock-slope release: lower residual with far worse endpoint. W30 conditional information is very weak; oracle-D and actual-M2 residuals differ only 0.00117 m/s despite 37.27 km position separation. The source M1 static/constant-clock restriction and actual temporal common-component remainder both contribute.',
        (120,'M1__M2'):'Both residual and endpoint improve. Conditional information is enhanced by time and PRN15; CV+affine expresses more actual motion/clock variation than static+constant clock, although both mismatch costs remain important and the M2 endpoint is still 3.34 km away.',
        (30,'M1__M3'):'Clean velocity release with constant common bias produces a very weak position direction. The clock remains temporally restricted; the saved M3 fit uses a very large velocity and distant position. Do not interpret this as a correctly specified-model noise comparison.',
        (120,'M1__M3'):'Lower residual still accompanies larger endpoint error at R616, with both full fits converged. Constant-clock dynamic M3 lacks the affine component retained by M2; weak information and time-model mismatch remain coupled.',
        (30,'M0__M4'):'Both hypotheses omit receiver common clock. Velocity release absorbs part of the approximately 80 m/s common component through geometry, producing low residual but extreme velocity and position. Missing-clock misspecification dominates the fairness of this pair.',
        (120,'M0__M4'):'Both residual and endpoint improve, but M4 still has a very large error and speed. Both omit the receiver clock, so this is a domain/state-misspecified comparison, not evidence of an adequate clock-free GNSS positioning model.'}
    for r in a['residual_blindness_interpretation.csv']:
        r['mechanism_interpretation']=interpretation[(r['window'],r['pair'])]
    save(O/'linear_position_response_budget.json',responses);save(O/'linear_position_response_checks.json',tests);save(O/'same_row_reference_sensitivity.json',same)
    save(O/'final_tables.json',a)
    print('POSTHOC_DIAGNOSTICS_COMPLETE; NEW_CANDIDATE_FITS=0')

if __name__=='__main__':main()
