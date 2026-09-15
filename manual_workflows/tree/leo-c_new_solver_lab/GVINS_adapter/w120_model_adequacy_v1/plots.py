from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':200,'axes.grid':True,'grid.alpha':.18})
a=read(O/'final_tables.json')
def primary(name):return [r for r in a[name] if r.get('support','SUPPORT_A_NATURAL')=='SUPPORT_A_NATURAL' and r.get('velocity','PVT_PRIMARY')=='PVT_PRIMARY']
rr=primary('oracle_residual_decomposition.csv');colors={'A':'#008C95','B':'#E69F00','C':'#0072B2','D':'#CC3311'}
labels={'A':'A: reference + epoch clock','B':'B: reference + affine clock','C':'C: reference CV + epoch clock','D':'D: reference CV + affine clock'}
fig,ax=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for c in 'ABCD':
    s=[r for r in rr if r['case']==c];ax[0].plot(WINDOWS,[r['residual_rms_mps'] for r in s],'-o',color=colors[c],label=labels[c],lw=2)
m=primary('m2_oracle_comparison.csv');m=[r for r in m if r['radius_m']==0];ax[0].scatter([r['window'] for r in m],[r['actual_residual_rms_mps'] for r in m],s=90,facecolors='none',edgecolors='black',marker='s',label='Saved M2; no W60 fit',zorder=5)
ax[0].set(xlabel='Window length (s)',ylabel='Residual RMS (m/s)',title='a  Fixed-trajectory oracle comparisons');ax[0].set_xticks(WINDOWS);ax[0].set_ylim(bottom=0);ax[0].legend(fontsize=8,loc='upper left')
x=np.arange(3);base=[r for r in rr if r['case']=='A']
for j,(key,label,c) in enumerate([('mse_clock_form_cost','Clock restriction',colors['B']),('mse_motion_model_cost','CV restriction',colors['C']),('mse_interaction','Interaction','#882255')]):
    ax[1].bar(x+(j-1)*.23,[r[key] for r in base],width=.22,label=label,color=c)
ax[1].axhline(0,color='black',lw=.7);ax[1].set_xticks(x,[f'W{T}' for T in WINDOWS]);ax[1].set(ylabel='Difference in mean squared residual ((m/s)²)',title='b  Empirical budget, including interaction');ax[1].legend(fontsize=8,loc='upper left')
fig.suptitle('Posthoc model adequacy on natural GPS L1 support',fontsize=13)
fig.savefig(O/'residual_decomposition.png');plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained');mr=primary('reference_motion_model_adequacy.csv')
ax[0].plot(WINDOWS,[r['position_CV_deviation_rms_m'] for r in mr],'-o',label='3D position deviation RMS',color='#0072B2',lw=2)
ax[0].plot(WINDOWS,[r['position_CV_deviation_p95_m'] for r in mr],'--s',label='3D position deviation p95',color='#CC3311',lw=2)
ax[0].set(title='a  Deviation from position-fitted CV',ylabel='Position deviation (m)',xlabel='Window length (s)');ax[0].set_ylim(bottom=0);ax[0].legend(fontsize=9)
for v,label,c,marker in [('PVT_PRIMARY','Device PVT velocity','#0072B2','o'),('POSITION_DERIVED_SENSITIVITY','Position-derived sensitivity','#E69F00','s')]:
    r=[r for r in a['reference_motion_model_adequacy.csv'] if r['velocity']==v]
    ax[1].plot(WINDOWS,[r['velocity_CV_deviation_rms_mps'] for r in r],'-'+marker,color=c,label=label,lw=2)
ax[1].set(title='b  Velocity deviation from the same CV',ylabel='Velocity deviation RMS (m/s)',xlabel='Window length (s)');ax[1].set_ylim(bottom=0);ax[1].legend(fontsize=9)
for q in ax:q.set_xticks(WINDOWS)
fig.suptitle('Reference trajectory characterization only; no Doppler positioning fit',fontsize=12)
fig.savefig(O/'motion_deviation_vs_window.png');plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained');su=a['satellite_support_effect.csv']
for supp,label,c,mk in [('SUPPORT_A_NATURAL','Natural support (6 / 6 / 7 PRNs)','#0072B2','o'),('SUPPORT_B_W30_SATELLITES','Original six PRN identities only','#CC3311','s')]:
    rows=[r for r in su if r['support']==supp]
    ax[0].plot(WINDOWS,[r['lambda_min'] for r in rows],'-'+mk,label=label,color=c,lw=2,fillstyle='none')
    ax[1].plot(WINDOWS,[r['case_D_rms_mps'] for r in rows],'-'+mk,label=label,color=c,lw=2,fillstyle='none')
ax[0].set(yscale='log',xlabel='Window length (s)',ylabel='M2 conditional position minimum eigenvalue (s⁻²)',title='a  Information with matched satellite identities');ax[0].legend(fontsize=8,loc='upper left')
ax[1].plot(WINDOWS,[r['residual_rms_mps'] for r in rr if r['case']=='A'],'--^',label='Reference + epoch clock, natural',color='#008C95')
ax[1].set(xlabel='Window length (s)',ylabel='Residual RMS (m/s)',title='b  CV + affine-clock mismatch persists');ax[1].legend(fontsize=8,loc='upper left');ax[1].set_ylim(bottom=0)
for q in ax:q.set_xticks(WINDOWS)
fig.suptitle('Information gain and model mismatch coexist; no window selection',fontsize=12)
fig.savefig(O/'information_vs_model_mismatch.png');plt.close(fig)
print('THREE_AUTHORIZED_INTERNAL_FIGURES_SAVED')
