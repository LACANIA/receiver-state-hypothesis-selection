from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':220,'axes.grid':True,'grid.alpha':.18})
colors={'M0':'#777777','M1':'#0072B2','M2':'#D55E00','M3':'#009E73','M4':'#CC79A7'}
inf=read(O/'information_full.json');rows=read(O/'fit_tables.json')['initialization_fit_results.csv']

fig,axs=plt.subplots(1,2,figsize=(11.8,4.6),layout='constrained')
for model in MODELS:
    sub=sorted([r for r in inf if r['model']==model and r['velocity']=='PVT_PRIMARY' and r['support']=='NATURAL_SUPPORT'],key=lambda r:r['window'])
    for ax,key in zip(axs,['lambda_min','weak_position_std_proxy_m']):
        ax.plot([r['window'] for r in sub],[r[key] for r in sub],marker='o',color=colors[model],label=model,lw=2)
        ax.set_yscale('log');ax.set_xticks(WINDOWS);ax.set_xlabel('Window length (s)')
axs[0].set_ylabel(r'Minimum eigenvalue of $S_p$ ($s^{-2}$)');axs[0].set_title('a  Conditional position information',loc='left')
axs[1].set_ylabel('Weak-direction proxy (m), unit 1 m/s scale');axs[1].set_title('b  Uncalibrated local noise response',loc='left')
axs[0].legend(ncol=3,fontsize=9)
fig.suptitle('Reference-trajectory tangent analysis | natural GPS L1 support: 6 / 6 / 7 / 8 satellites\nCommon support unavailable: only 3 satellites persist over every W300 raw epoch',fontsize=11)
fig.savefig(O/'information_vs_window.png');plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(11.8,7.2),layout='constrained')
for i,model in enumerate(['M1','M2']):
    for T,c in [(30,'#0072B2'),(120,'#D55E00')]:
        sub=sorted([r for r in rows if r['model']==model and r['window']==T and r['direction']=='LEGACY_D0'],key=lambda r:r['radius_m'])
        xx=[r['radius_m']/1000 for r in sub]
        axs[i,0].plot(xx,[r['endpoint_error_m']/1000 for r in sub],'-o',color=c,label=f'W{T}',lw=2)
        axs[i,1].plot(xx,[r['full_residual_rmse_mps'] for r in sub],'-o',color=c,label=f'W{T}',lw=2)
    axs[i,0].set_ylabel('Endpoint position error (km)');axs[i,1].set_ylabel('Full-record residual RMSE (m/s)')
    for j in range(2):
        axs[i,j].set_xlabel('Initial position offset radius (km)');axs[i,j].set_title(f'{chr(97+i*2+j)}  {model}',loc='left');axs[i,j].legend();axs[i,j].set_ylim(bottom=0)
fig.suptitle('Legacy ECEF direction: normalized [500, -300, 200]\nRTK-associated origin initialization; zero initial velocity and clock; no selector',fontsize=12)
fig.savefig(O/'initialization_basin_M1_M2.png');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11.8,4.7),layout='constrained')
for ax,T in zip(axs,[30,120]):
    for lab,color,marker in [('WEAK','#CC3311','o'),('STRONG','#0077BB','s'),('LEGACY_D0','#009988','^')]:
        sub=sorted([r for r in rows if r['model']=='M2' and r['window']==T and r['direction']==lab and r['radius_m']>0],key=lambda r:r['radius_m'])
        ax.plot([r['radius_m']/1000 for r in sub],[r['endpoint_error_m']/1000 for r in sub],marker=marker,color=color,label=lab,linewidth=1.7,markersize=7,fillstyle='none')
    ymax=max(r['endpoint_error_m']/1000 for r in rows if r['model']=='M2' and r['window']==T)
    ax.set_xlabel('Initial position offset radius (km)');ax.set_ylabel('Endpoint position error (km)');ax.set_title(f'W{T}: fixed reference-information directions',loc='left');ax.set_ylim(0,ymax*1.15);ax.legend(fontsize=9,loc='lower right')
fig.suptitle('M2 directional initialization diagnostic\nDirections fixed from each window before fitting; coincident curves indicate matching endpoints',fontsize=12)
fig.savefig(O/'weak_vs_strong_direction_M2.png');plt.close(fig)
print('THREE_INTERNAL_DIAGNOSTIC_FIGURES_SAVED')
