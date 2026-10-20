#!/usr/bin/env python3
"""Aggregate five pre-registered stochastic ambient seeds. True vs ARX vs proposed only."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_5seed_summary'
OUT.mkdir(exist_ok=True)
seeds=[20261020,20261021,20261022,20261023,20261024]
fr=[];dr=[]
for s in seeds:
 p=ROOT/f'results_seed_{s}'
 a=pd.read_csv(p/'stage_true_baseline_proposed_errors.csv');a['seed']=s;fr.append(a)
 d=pd.read_csv(p/'device_true_baseline_proposed_diagnostics.csv');d['seed']=s;dr.append(d)
df=pd.concat(fr,ignore_index=True)
dv=pd.concat(dr,ignore_index=True)
truth='true_Hobs_s';columns={'ARX baseline':'ARX_baseline_Hobs_s','Proposed':'WMU_multitaper_Hobs_s'}
summary=[]
for method,col in columns.items():
 err=df[col]-df[truth]
 summary.append({'method':method,'n_samples':len(df),'RMSE_s':np.sqrt(np.mean(err**2)),'MAE_s':np.mean(abs(err)),
  'bias_s':np.mean(err),'max_abs_error_s':max(abs(err)),
  'stage4_false_inertia_mean_abs_s':np.mean(abs(err[df.stage==4])),
  'rmse_nonzero_stages_s':np.sqrt(np.mean(err[df.stage<4]**2))})
agg=pd.DataFrame(summary)
reduction=100*(1-agg.loc[agg.method=='Proposed','RMSE_s'].iloc[0]/agg.loc[agg.method=='ARX baseline','RMSE_s'].iloc[0])
print(agg.to_string(index=False), 'RMSE reduction (%)', reduction)
groups=df.groupby('stage',as_index=False).agg({truth:'mean',
 'ARX_baseline_Hobs_s':['mean','std'],'WMU_multitaper_Hobs_s':['mean','std'],
 'uncertainty_model_heuristic_lo_s':'mean','uncertainty_model_heuristic_hi_s':'mean',
 'heuristic_band_covers_true':'mean'})
groups.columns=['stage','true_Hobs_s','ARX_mean','ARX_std','Proposed_mean','Proposed_std','model_band_lo_mean','model_band_hi_mean','heuristic_coverage_rate']
agg.to_csv(OUT/'five_seed_summary_metrics.csv',index=False)
df.to_csv(OUT/'five_seed_all_stages_true_ARX_proposed.csv',index=False)
dv.to_csv(OUT/'five_seed_device_diagnostics.csv',index=False)
groups.to_csv(OUT/'five_seed_stage_means.csv',index=False)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
BLACK='#20262E';ORANGE='#D58A3B'; BLUE='#2377B7'

def export(name):plt.tight_layout();plt.savefig(OUT/name,dpi=200,bbox_inches='tight');plt.close()

fig,axs=plt.subplots(2,1,figsize=(11,8),sharex=True,height_ratios=[1.6,1.0]);xx=groups.stage.to_numpy()
axs[0].plot(xx,groups.true_Hobs_s,'k--o',lw=2.5,ms=8,label='True observed inertia')
axs[0].errorbar(xx-.045,groups.ARX_mean,yerr=groups.ARX_std,fmt='s-',color=ORANGE,lw=1.8,capsize=3,label='ARX baseline (same WMU)')
axs[0].errorbar(xx+.045,groups.Proposed_mean,yerr=groups.Proposed_std,fmt='o-',color=BLUE,lw=1.8,capsize=3,label='Proposed WMU multitaper')
axs[0].fill_between(xx,groups.model_band_lo_mean,groups.model_band_hi_mean,color=BLUE,alpha=.13,
 label='Proposed model mismatch envelope (heuristic)')
for j,v in enumerate(groups.true_Hobs_s):
 axs[0].annotate(f'{v:.3f}s',(xx[j],v),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
axs[0].set(title='Fig A | GFM observed inertia under five ambient seeds',ylabel='Hobs [s, system base]')
axs[0].legend(fontsize=8.5,ncol=2)
axs[1].axhline(0,color='k',lw=1)
axs[1].plot(xx,groups.ARX_mean-groups.true_Hobs_s,'-s',lw=1.8,color=ORANGE,label='ARX error')
axs[1].plot(xx,groups.Proposed_mean-groups.true_Hobs_s,'-o',lw=1.8,color=BLUE,label='Proposed error')
axs[1].set(xlabel='Stage (Stage 4: zero GFM inertia)',ylabel='Mean error (estimated - true) [s]',xticks=xx)
axs[1].legend(fontsize=9,ncol=2)
export('FigA_five_seed_true_baseline_proposed.png')

fig,ax=plt.subplots(figsize=(9.7,4.7))
labels=['Standard ARX\n(same WMU)','Proposed DPSS Multitaper\n(WMU)']; values=agg.RMSE_s.values
bars=ax.barh(labels,values,color=[ORANGE,BLUE],height=.48)
for z,bar in zip(values,bars):
 ax.text(z+max(values)*.025,bar.get_y()+bar.get_height()/2,f'{z:.4f} s',va='center',fontsize=12)
ax.invert_yaxis()
ax.set_xlim(0,max(values)*1.3)
ax.set(xlabel='RMSE against TRUE GFM Hobs [s]',title=f'Fig B | Error reduction vs ARX baseline: {reduction:.1f}% (5 seeds x 4 stages)')
export('FigB_RMSE_reduction.png')

fig,axs=plt.subplots(1,2,figsize=(12.4,4.8),sharey=True)
for ax,device in zip(axs,['ESS-VSG','PV-VSG']):
 subset=dv[dv.device==device]
 g=subset.groupby('stage',as_index=False).agg(true=('true_Hobs_s','mean'),
   arx=('ARX_Hobs_s','mean'),arx_sd=('ARX_Hobs_s','std'),prop=('Proposed_Hobs_s','mean'),prop_sd=('Proposed_Hobs_s','std'))
 ax.plot(g.stage,g.true,'k--o',label='True',lw=2.2)
 ax.errorbar(g.stage-.035,g.arx,yerr=g.arx_sd,color=ORANGE,fmt='s-',label='ARX baseline',lw=1.5)
 ax.errorbar(g.stage+.035,g.prop,yerr=g.prop_sd,color=BLUE,fmt='o-',label='Proposed WMU',lw=1.5)
 ax.set(title=device,xlabel='Stage',xticks=xx)
axs[0].set_ylabel('Hobs [s, system base]');axs[1].legend(fontsize=9)
fig.suptitle('Fig C | Device-level inertia decomposition, five ambient seeds')
export('FigC_ESS_PV_true_baseline_proposed.png')

fig,ax=plt.subplots(figsize=(11,4.7))
truth_sys=2.8+groups.true_Hobs_s
ax.plot(xx,truth_sys,'k--o',label='True SG + GFM effective inertia',lw=2.2)
ax.plot(xx,2.8+groups.ARX_mean,'-s',color=ORANGE,label='SG anchor + ARX GFM',lw=1.8)
ax.plot(xx,2.8+groups.Proposed_mean,'-o',color=BLUE,label='SG anchor + Proposed GFM',lw=1.8)
ax.set(title='Fig D | System-wide effective inertia with known physical SG anchor',xlabel='Stage',ylabel='System-base inertia [s]',xticks=xx)
ax.legend(fontsize=9)
export('FigD_system_total_true_baseline_proposed.png')

metadata={'seeds':seeds,'n_stages_each_seed':4,'n_total':len(df),'RMSE_reduction_pct':reduction,
 'baseline':'Conventional ARX estimator from same WMU 1-cycle 240 Hz feature representation',
 'proposed':'WMU-derived 1-cycle features and DPSS multitaper, physics-guided nonnegative multi-time-scale dictionary with CV and model mismatch uncertainty',
 'caveat':'Both ARX and proposed use same WMU 240Hz data; isolates algorithm comparison but does NOT prove WMU superiority over compliant PMU.',
 'CI':'Heuristic model discrepancy envelope is NOT a calibrated confidence interval; bootstrap-only intervals were overconfident and omitted from main plots.'}
(OUT/'study_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
print('SAVED',OUT)
