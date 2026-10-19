#!/usr/bin/env python3
"""Synthetic WMU/PMU-equivalent SG + GFM inertia decomposition validation.

Run: python wmu_multires_validation.py
Dependencies: numpy scipy pandas matplotlib

Not IEEE 39-bus; not an EMT converter model; no internal VSG controller
is provided to the estimators. Closed-loop aggregate swing plus two
measured black-box terminal response branches. Frequency and terminal
powers are constructed into 3-phase 15.36-kHz waveforms and recovered
using signal processing. Two spectral estimation strategies are compared:
  (1) apparent H from imaginary cross-spectrum (non-parametric),
  (2) parsimonious external transfer model fitting (inertia + lagged droop).
The latter ASSUMES an external model form; it is not model-free.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, optimize

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
OUT.mkdir(parents=True, exist_ok=True)
SEED=20261019
rng=np.random.default_rng(SEED)
F0=60.0; SBASE=500.0; H_SG=2.8   # SG: 350 MVA, H=4 s => 2.8 s system base
FS_STATE=240; FS_RAW=15360; FS_PMU=60
assert FS_RAW % FS_STATE == 0 and FS_STATE % FS_PMU == 0
T_EPOCH=120.0; EPOCHS=3; DURATION=T_EPOCH*EPOCHS
DT=1/FS_STATE
T=np.arange(0,DURATION,DT); N=len(T)
ESS_RATING=100.0; PV_RATING=50.0
M_SG=2*SBASE*H_SG/F0  # MW s/Hz
# Transfer response kernel: -D/(1+s*tau_d) - M*s/(1+s*tau_i)
D_ESS,D_PV=19.0,11.0  # MW/Hz: external equivalent droop strengths
D_SG=27.0 # other SG/load damping (MW/Hz), assumed constant
TAU_I_ESS,TAU_I_PV=0.045,0.075 # external inertial response time constants
TAU_D_ESS,TAU_D_PV=0.095,0.115 # droop frequency filtering time constants
# H_i in local rating base (seconds), changed between test epochs.
H_ESS_SET=np.array([3.0,5.0,3.0]); H_PV_SET=np.array([2.2,2.2,0.8])
H_ESS_SYS=H_ESS_SET*ESS_RATING/SBASE
H_PV_SYS=H_PV_SET*PV_RATING/SBASE
TRUE_H_GFM=H_ESS_SYS+H_PV_SYS
TRUE_H_TOTAL=H_SG+TRUE_H_GFM

# ambient load perturbation: 0.15-4 Hz, NOT a generator trip or an injected probe
load_noise=signal.sosfiltfilt(signal.butter(4,[0.15,4.0],btype='bandpass',fs=FS_STATE,output='sos'), rng.normal(size=N))
load_noise*=2.2/np.std(load_noise)
LOAD=load_noise + 0.5*np.sin(2*np.pi*.47*T+0.2)+0.4*np.sin(2*np.pi*1.4*T)
# Model state: f offset Hz, 2 filtered RoCoF states, 2 filtered freq states
F=np.zeros(N); R_ESS=np.zeros(N);R_PV=np.zeros(N); Z_ESS=np.zeros(N); Z_PV=np.zeros(N)
P_ESS=np.zeros(N);P_PV=np.zeros(N); ROCOF=np.zeros(N)
for k in range(N-1):
    ep=min(int(T[k]//T_EPOCH),EPOCHS-1)
    me=2*SBASE*H_ESS_SYS[ep]/F0
    mp=2*SBASE*H_PV_SYS[ep]/F0
    P_ESS[k]=-D_ESS*Z_ESS[k]-me*R_ESS[k]
    P_PV[k]=-D_PV*Z_PV[k]-mp*R_PV[k]
    ROCOF[k]=(-LOAD[k]-D_SG*F[k]+P_ESS[k]+P_PV[k])/M_SG
    F[k+1]=F[k]+DT*ROCOF[k]
    R_ESS[k+1]=R_ESS[k]+DT*(ROCOF[k]-R_ESS[k])/TAU_I_ESS
    R_PV[k+1]=R_PV[k]+DT*(ROCOF[k]-R_PV[k])/TAU_I_PV
    Z_ESS[k+1]=Z_ESS[k]+DT*(F[k]-Z_ESS[k])/TAU_D_ESS
    Z_PV[k+1]=Z_PV[k]+DT*(F[k]-Z_PV[k])/TAU_D_PV
P_ESS[-1]=P_ESS[-2]; P_PV[-1]=P_PV[-2];ROCOF[-1]=ROCOF[-2]

# 15,360 sample/s waveforms, processed in small contiguous 5-second blocks.
# All input P are P0 + delta; only the deltas enter estimators.
FS_HI=FS_RAW; DEC=FS_RAW//FS_STATE
P0_ESS,P0_PV,P0_LOAD=55.0,35.0,90.0
VLL=13.8e3; VPK=np.sqrt(2)*VLL/np.sqrt(3)
# One-cycle filtered WMU features and six-cycle PMU-equivalent features.
WINDOW_WMU=FS_RAW//60; WINDOW_PMU=6*WINDOW_WMU

def lfilter_state(window):
    return np.zeros(window-1)

si=np.ones(WINDOW_WMU)/WINDOW_WMU
sp=np.ones(WINDOW_PMU)/WINDOW_PMU
zi_w={key:lfilter_state(WINDOW_WMU) for key in ('f','ess','pv','load')}
zi_p={key:lfilter_state(WINDOW_PMU) for key in ('f','ess','pv','load')}
features_w={key:np.zeros(N) for key in zi_w}
features_p={key:np.zeros(N) for key in zi_p}
preview=None
phase_end=0.0
chunk_secs=5
for start in range(0,N,int(chunk_secs*FS_STATE)):
    stop=min(start+chunk_secs*FS_STATE,N)
    # FS_RAW/FS_STATE = 64 integer, state samples held/interpolated between ticks.
    nt=(stop-start)*DEC
    ti=(start*DEC+np.arange(nt))/FS_RAW
    fhi=np.interp(ti,T[start:stop],F[start:stop],left=F[start],right=F[stop-1])
    pdelta_ess=np.interp(ti,T[start:stop],P_ESS[start:stop])
    pdelta_pv=np.interp(ti,T[start:stop],P_PV[start:stop])
    pdelta_load=np.interp(ti,T[start:stop],LOAD[start:stop])
    phase=phase_end+2*np.pi*np.cumsum(F0+fhi)/FS_RAW
    phase_end=float(phase[-1])
    angles=phase[:,None] - 2*np.pi/3*np.arange(3)[None,:]
    vabc=VPK*(np.cos(angles)+0.0015*np.cos(5*angles))
    vabc+=VPK*rng.normal(0,2e-5,vabc.shape)
    # Clarke voltage angle -> instantaneous frequency from actual 3-phase voltage.
    alpha=(2*vabc[:,0]-vabc[:,1]-vabc[:,2])/3
    beta=(vabc[:,1]-vabc[:,2])/np.sqrt(3)
    ang=np.unwrap(np.arctan2(beta,alpha))
    f_inst=np.gradient(ang,1/FS_RAW)/(2*np.pi)  # absolute frequency in Hz
    streams={'f': f_inst}
    for key,base,dp in [('ess',P0_ESS,pdelta_ess),('pv',P0_PV,pdelta_pv),('load',P0_LOAD,pdelta_load)]:
        target=base+dp
        current_amp=2*(target*1e6)/(3*VPK)
        iabc=current_amp[:,None]*(np.cos(angles)+.002*np.cos(5*angles+.23))
        iabc+=current_amp[:,None]*rng.normal(0,3e-4,vabc.shape)
        streams[key]=np.sum(vabc*iabc,axis=1)/1e6
        if key=='ess' and start==0:
            preview=(ti[:int(.12*FS_RAW)].copy(),vabc[:int(.12*FS_RAW)].copy(),iabc[:int(.12*FS_RAW)].copy())
    for key,raw in streams.items():
        yw,zi_w[key]=signal.lfilter(si,[1.0],raw,zi=zi_w[key])
        yp,zi_p[key]=signal.lfilter(sp,[1.0],raw,zi=zi_p[key])
        features_w[key][start:stop]=yw[DEC-1::DEC][:stop-start]
        features_p[key][start:stop]=yp[DEC-1::DEC][:stop-start]

# Synthetic independent instrumentation error on extracted channels
# (frequency error is deliberately small; timestamp offset not simulated).
for spec in (features_w,features_p):
    spec['f']+=rng.normal(0, 0.00011 if spec is features_w else 0.00007, N)
    for key in ('ess','pv','load'):
        spec[key]+=rng.normal(0,.010 if spec is features_w else .006,N)

# Spectral transfer: G(f)= delta P / delta frequency; CSD(f,p) is conj(F)*P.
# Coherence < 0.75 removed. Spectral parameter fitting is to a REDUCED
# EXTERNAL input-output model, not a proprietary internal VSG controller.
def spectral(f,p,fs):
    f=signal.detrend(f,type='constant');p=signal.detrend(p,type='constant')
    nperseg=int(round(fs*17.066667))
    opts=dict(fs=fs,nperseg=nperseg,noverlap=nperseg//2,window='hann')
    hz,sff=signal.welch(f,**opts)
    _,sfp=signal.csd(f,p,**opts)
    _,coh=signal.coherence(f,p,**opts)
    return hz,sfp/np.maximum(sff,1e-15),coh

BAND=(.34,3.05)
def external_model(hz, H_sys,D,tau_i,tau_d):
    s=2j*np.pi*hz
    M=2*SBASE*H_sys/F0
    return -(D/(1+s*tau_d)+M*s/(1+s*tau_i))

def fit_gfm(hz,G,coh):
    mask=(hz>=BAND[0]) & (hz<=BAND[1]) & (coh>.72)
    if mask.sum()<15: return np.nan,None,mask,np.nan
    fr=hz[mask];obs=G[mask];w=np.sqrt(coh[mask])
    def fn(x):
        err=(external_model(fr,*x)-obs)*w
        return np.r_[err.real,err.imag]
    candidates=[]
    for ini in [(0.5,14,.05,.1),(1.0,14,.08,.06),(0.3,25,.02,.2)]:
        fit=optimize.least_squares(fn,ini,bounds=([0,0,.01,.01],[3,65,.35,.35]),max_nfev=200,loss='soft_l1',f_scale=.25)
        candidates.append(fit)
    sol=min(candidates,key=lambda fit:np.sum(fn(fit.x)**2))
    return float(sol.x[0]),sol.x,mask,float(np.sqrt(np.mean(fn(sol.x)**2)))

def h_app(hz,G):
    with np.errstate(divide='ignore',invalid='ignore'):
        return -F0*G.imag/(2*SBASE*2*np.pi*hz)

def weighted_band(hz,H,coh):
    mask=(hz>=BAND[0]) & (hz<=BAND[1]) & (coh>.72) & np.isfinite(H)
    if mask.sum()<10:return np.nan
    return float(np.average(H[mask],weights=coh[mask]**2))

def true_hobs_curve(hz,stage):
    # Inertial component ONLY, without phase artifact from delayed droop.
    w=2*np.pi*hz
    return H_ESS_SYS[stage]/(1+(w*TAU_I_ESS)**2)+H_PV_SYS[stage]/(1+(w*TAU_I_PV)**2)

# We evaluate at the center of three fixed estimation windows, skipping
# 12 seconds after each controlled H change to avoid mixing operating points.
rows=[]; curves={}; details=[]
for ep in range(EPOCHS):
    t0=ep*T_EPOCH+12;t1=(ep+1)*T_EPOCH-3
    mask=(T>=t0)&(T<t1)
    period={}
    for resname,raw,decimate,fs in [('WMU',features_w,1,FS_STATE),('PMU-like',features_p,4,FS_PMU)]:
        f=raw['f'][mask][::decimate]
        vals={k:raw[k][mask][::decimate] for k in ('ess','pv','load')}
        spectra={}
        for key in ('ess','pv','load'):
            h,g,c=spectral(f,vals[key],fs)
            spectra[key]=(h,g,c)
        params={}
        for key in ('ess','pv'):
            hz,g,coh=spectra[key]
            est,x,use,fit_err=fit_gfm(hz,g,coh)
            params[key]=dict(H_fit=est,coef=x,fit_residual=fit_err,H_app=weighted_band(hz,h_app(hz,g),coh))
        hz,g,c=spectra['load']
        # swing identity: -PL/F = j*w*M_s + D_s - sum(P_gfm/F).
        # Here -load/f => Hsys via imag; delayed droop contaminates 'inertia'.
        # Since h_app(G) = -Im(G) f0 / (2S w), and G_load=PL/F ≈ -(D+jwM),
        # h_app(G_load) is POSITIVE total H (no extra minus sign).
        Hsys_spectrum=h_app(hz,g)
        Hsys=weighted_band(hz,Hsys_spectrum,c)
        period[resname]=dict(ess=params['ess'],pv=params['pv'],Hsys=Hsys,
                              fit_gfm=params['ess']['H_fit']+params['pv']['H_fit'],
                              app_gfm=params['ess']['H_app']+params['pv']['H_app'],
                              spectra=spectra)
    hz=period['WMU']['spectra']['ess'][0]
    truth_curve=true_hobs_curve(hz,ep)
    truth_band=float(np.mean(truth_curve[(hz>=BAND[0])&(hz<=BAND[1])]))
    true_total=TRUE_H_TOTAL[ep]
    true_gfm=TRUE_H_GFM[ep]
    w=period['WMU'];p=period['PMU-like']
    row={
        'epoch':ep+1,'window_start_s':t0,'window_end_s':t1,
        'true_H_SG_s':H_SG,'true_H_GFM_set_s':true_gfm,'true_H_system_ideal_s':true_total,
        'true_H_GFM_observed_band_s':truth_band,
        'WMU_system_apparent_s':w['Hsys'],
        'PMU_system_apparent_s':p['Hsys'],
        'naive_WMU_minus_PMU_s':w['Hsys']-p['Hsys'],
        'known_SG_anchored_WMU_s':w['Hsys']-H_SG,
        'WMU_GFM_spectral_apparent_s':w['app_gfm'],
        'WMU_GFM_blackbox_fit_s':w['fit_gfm'],
        'PMU_GFM_blackbox_fit_s':p['fit_gfm'],
        'WMU_ESS_fit_s':w['ess']['H_fit'],
        'WMU_PV_fit_s':w['pv']['H_fit'],
        'PMU_ESS_fit_s':p['ess']['H_fit'],
        'PMU_PV_fit_s':p['pv']['H_fit'],
    }
    for est in ('naive_WMU_minus_PMU_s','known_SG_anchored_WMU_s','WMU_GFM_spectral_apparent_s','WMU_GFM_blackbox_fit_s','PMU_GFM_blackbox_fit_s'):
        row[est+'_err_vs_Hset_s']=row[est]-true_gfm
    rows.append(row)
    curves[ep]=(hz,truth_curve,period)
    details.append(period)

df=pd.DataFrame(rows)
df.to_csv(OUT/'metrics_true_vs_estimates.csv',index=False,float_format='%.6f')
# Source-specific comparison: local Hs needs conversion to system base
sgdf=pd.DataFrame({'epoch':np.arange(1,EPOCHS+1),
                  'true_H_ESS_s':H_ESS_SYS,'true_H_PV_s':H_PV_SYS,
                  'WMU_ESS_fit_s':[details[i]['WMU']['ess']['H_fit'] for i in range(EPOCHS)],
                  'WMU_PV_fit_s':[details[i]['WMU']['pv']['H_fit'] for i in range(EPOCHS)],
                  'PMU_ESS_fit_s':[details[i]['PMU-like']['ess']['H_fit'] for i in range(EPOCHS)],
                  'PMU_PV_fit_s':[details[i]['PMU-like']['pv']['H_fit'] for i in range(EPOCHS)]})
sgdf.to_csv(OUT/'component_metrics.csv',index=False,float_format='%.6f')
# short 3-phase WMU raw waveform preview + recovered features
if preview is not None:
    rawt,rawv,rawi=preview
    pd.DataFrame({'time_s':rawt,'v_a_V':rawv[:,0],'v_b_V':rawv[:,1],'v_c_V':rawv[:,2],
                  'i_a_A':rawi[:,0],'i_b_A':rawi[:,1],'i_c_A':rawi[:,2]}).to_csv(OUT/'wmU_abc_120ms.csv',index=False)
feat=pd.DataFrame({'time_s':T,'f_true_Hz':F0+F,'wmU_f_Hz':features_w['f'],
                   'pmu_like_f_Hz':features_p['f'], 'load_delta_true_MW':LOAD,
                   'wmU_ess_P_MW':features_w['ess'],'wmU_pv_P_MW':features_w['pv'],
                   'wmU_load_P_MW':features_w['load'],
                   'true_ess_deltaP_MW':P_ESS,'true_pv_deltaP_MW':P_PV,
                   'H_gfm_set_true_s':TRUE_H_GFM[np.minimum((T//T_EPOCH).astype(int),EPOCHS-1)]})
feat.iloc[::4,:].to_csv(OUT/'features_60fps.csv',index=False,float_format='%.8f')

# ===== Figures ======
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
labels=['Stage 1','Stage 2','Stage 3']
color=plt.rcParams['axes.prop_cycle'].by_key()['color']
# Fig 1: same-input waveform and feature, ground truth traces with errors
fig,axs=plt.subplots(2,1,figsize=(11,7.0),layout='constrained')
si_plot=(T>=32)&(T<44)
axs[0].plot(T[si_plot],F[si_plot]*1000,label='True Δf',color='black',lw=2)
axs[0].plot(T[si_plot],(features_w['f'][si_plot]-F0)*1000,label='WMU 240 fps',alpha=.75)
axs[0].plot(T[si_plot][::4],(features_p['f'][si_plot][::4]-F0)*1000,label='PMU-like 60 fps',lw=1.7,alpha=.9)
axs[0].set(ylabel='Frequency deviation (mHz)',title='Fig. 1 | Closed-loop ambient frequency: truth vs. extracted features')
axs[0].legend(ncol=3);axs[0].grid(alpha=.2)
si_plot=(T>=90)&(T<92)
axs[1].plot(T[si_plot],P_ESS[si_plot],color='black',label='True ESS ΔP',lw=2)
axs[1].plot(T[si_plot],features_w['ess'][si_plot]-P0_ESS,label='WMU-derived ΔP',alpha=.75)
axs[1].plot(T[si_plot][::4],features_p['ess'][si_plot][::4]-P0_ESS,label='PMU-like ΔP',alpha=.85)
axs[1].set(xlabel='Time (s)',ylabel='ESS power variation (MW)')
axs[1].grid(alpha=.2);axs[1].legend(ncol=3)
fig.savefig(OUT/'Fig1_ambient_true_vs_extracted.png',dpi=180);plt.close(fig)

# Fig2: central logical failure: PMU ≠ SG-only; subtraction fails
x=np.arange(EPOCHS)
fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
axs[0].plot(x,df['true_H_system_ideal_s'],marker='o',color='black',lw=2.5,label='True SG + GFM H-set (ideal)')
axs[0].plot(x,df['WMU_system_apparent_s'],marker='s',lw=1.7,label='WMU overall apparent H')
axs[0].plot(x,df['PMU_system_apparent_s'],marker='^',lw=1.7,label='PMU-like overall apparent H')
axs[0].axhline(H_SG,color='gray',ls='--',label='Known true SG H')
axs[0].set(ylabel='System-base inertia (s)',title='Fig. 2 | PMU also observes GFM-induced frequency dynamics')
axs[0].legend(ncol=2);axs[0].grid(alpha=.2)
axs[1].plot(x,df['true_H_GFM_set_s'],marker='o',color='black',lw=2.5,label='True GFM H-set (reference)')
axs[1].plot(x,df['naive_WMU_minus_PMU_s'],marker='x',label='Naive: WMU − PMU')
axs[1].plot(x,df['known_SG_anchored_WMU_s'],marker='s',label='SG anchor: WMU Hsys − known SG')
axs[1].plot(x,df['WMU_GFM_blackbox_fit_s'],marker='D',label='WMU: fitted GFM terminal response')
axs[1].set(xlabel='Controlled synthetic stages',ylabel='GFM contribution (s)',xticks=x,xticklabels=labels)
axs[1].legend(ncol=2);axs[1].grid(alpha=.2)
fig.savefig(OUT/'Fig2_true_and_decomposition.png',dpi=180);plt.close(fig)

# Fig3: each component with truth
fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for i,(key,basename) in enumerate([('ess','ESS-VSG'),('pv','PV-VSG')]):
    true=sgdf['true_H_ESS_s' if key=='ess' else 'true_H_PV_s'].values
    axs[i].plot(x,true,'ko-',lw=2.3,label='True H-set on system base')
    axs[i].plot(x,sgdf['WMU_ESS_fit_s' if key=='ess' else 'WMU_PV_fit_s'],marker='s',label='WMU fit')
    axs[i].plot(x,sgdf['PMU_ESS_fit_s' if key=='ess' else 'PMU_PV_fit_s'],marker='^',label='PMU-like fit')
    axs[i].set(xlabel='Stage',xticks=x,xticklabels=labels,title=basename,ylabel='H contribution (s)')
    axs[i].grid(alpha=.2);axs[i].legend(fontsize=8)
fig.suptitle('Fig. 3 | Individual GFM inertia: fit vs. known synthetic parameters',fontsize=12)
fig.savefig(OUT/'Fig3_ESS_PV_true_vs_fit.png',dpi=180);plt.close(fig)

# Fig4: spectral H_app and true dynamic Hobs (inertial response only)
fig,axs=plt.subplots(1,EPOCHS,figsize=(13,4.2),sharey=True,layout='constrained')
for ep,ax in enumerate(axs):
    hz,htrue,period=curves[ep]
    ghz,gess,cess=period['WMU']['spectra']['ess']
    _,gpv,cpv=period['WMU']['spectra']['pv']
    ax.plot(hz,htrue,'k-',lw=2.2,label='True inertial Hobs(f)')
    ax.plot(hz,h_app(hz,gess)+h_app(hz,gpv),label='Raw WMU apparent H(f)',alpha=.8)
    x1=period['WMU']['ess']['coef'];x2=period['WMU']['pv']['coef']
    if x1 is not None and x2 is not None:
        h_fit=h_app(hz,external_model(hz,*x1)+external_model(hz,*x2))
        ax.plot(hz,h_fit,'--',label='Fitted full-response apparent H(f)')
    ax.set(xlim=(.3,3.2),xlabel='Ambient modulation frequency (Hz)',title=labels[ep])
    ax.grid(alpha=.2)
axs[0].set(ylabel='H on system base (s)')
axs[-1].legend(fontsize=7,loc='best')
fig.suptitle('Fig. 4 | Frequency-dependent observed inertia vs. raw spectral apparent inertia',fontsize=12)
fig.savefig(OUT/'Fig4_true_Hobs_vs_apparent_spectrum.png',dpi=180);plt.close(fig)

# Fig5 errors ALWAYS against known true set parameters
algos=[('naive_WMU_minus_PMU_s','Naive\nWMU − PMU'),('known_SG_anchored_WMU_s','WMU Hsys\n− known SG'),
       ('WMU_GFM_spectral_apparent_s','WMU\nraw GFM spectrum'),('PMU_GFM_blackbox_fit_s','PMU-like\nterminal fit'),('WMU_GFM_blackbox_fit_s','WMU\nterminal fit')]
fig,axs=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
width=.15
for j in range(EPOCHS):
    xs=np.arange(len(algos))+(j-1)*width
    yy=[df.loc[j,algo] for algo,_ in algos]
    axs[0].bar(xs,yy,width,label=f'Stage {j+1}')
axs[0].axhline(0,color='gray',linewidth=.6)
axs[0].set(xticks=np.arange(len(algos)),xticklabels=[label for _,label in algos],ylabel='Estimated GFM H (s)',title='Estimated GFM inertia (truth varies by stage)')
axs[0].legend(fontsize=8)
for j in range(EPOCHS):
    xs=np.arange(len(algos))+(j-1)*width
    errs=[df.loc[j,algo]-df.loc[j,'true_H_GFM_set_s'] for algo,_ in algos]
    axs[1].bar(xs,errs,width,label=f'Stage {j+1}')
axs[1].axhline(0,color='black',lw=1)
axs[1].set(xticks=np.arange(len(algos)),xticklabels=[label for _,label in algos],ylabel='Estimate minus true H-set (s)',title='Absolute signed error vs. true H-set')
fig.suptitle('Fig. 5 | Every comparison is evaluated against the same ground truth',fontsize=12)
fig.savefig(OUT/'Fig5_method_errors.png',dpi=180);plt.close(fig)

# Fig6 raw high-res short waveform
if preview is not None:
    tp,vp,ip=preview
    fig,axs=plt.subplots(2,1,figsize=(10.5,5.8),sharex=True,layout='constrained')
    for ii,label in enumerate('abc'):
        axs[0].plot(tp*1000,vp[:,ii]/1000,lw=.95,label=f'V{label}')
        axs[1].plot(tp*1000,ip[:,ii]/1000,lw=.95,label=f'I{label}')
    axs[0].set(ylabel='Phase voltage (kV)',title='Fig. 6 | Synthesized three-phase WMU waveform: 15,360 samples/s')
    axs[1].set(xlabel='Time (ms)',ylabel='Phase current (kA)')
    for ax in axs:ax.legend(ncol=3);ax.grid(alpha=.2)
    fig.savefig(OUT/'Fig6_raw_abc_waveform.png',dpi=180);plt.close(fig)

# Stats (truth H_set reference)
metrics=[]
for name,label in algos:
    err=df[name].values-df['true_H_GFM_set_s'].values
    metrics.append({'method':label.replace('\n',' '),'MAE_s':float(np.mean(np.abs(err))),
                    'RMSE_s':float(np.sqrt(np.mean(err**2))), 'bias_s':float(np.mean(err)),
                    'MaxAbsError_s':float(np.max(np.abs(err)))})
pd.DataFrame(metrics).to_csv(OUT/'summary_method_errors.csv',index=False,float_format='%.6f')
# Compare both simulated measurement pipelines with the same underlying ground truth.
valid=T>10
quality=[]
for method,d in [('WMU (one-cycle feature)',features_w),('PMU-like (six-cycle feature)',features_p)]:
    ff=d['f'][valid]-F0-F[valid]
    eps=d['ess'][valid]-P0_ESS-P_ESS[valid]
    quality.append({'method':method,'frequency_RMSE_mHz':float(np.sqrt(np.mean(ff**2))*1e3),
                    'ESS_power_RMSE_MW':float(np.sqrt(np.mean(eps**2)))})
pd.DataFrame(quality).to_csv(OUT/'signal_quality_vs_truth.csv',index=False,float_format='%.6f')
print('\nSYSTEM PARAMETERS: SG true H=',H_SG,'s, stages GFM H=',TRUE_H_GFM)
print('OPEN LOOP? NO: frequency integrates SG swing + instantaneous GFM feedback and ambient load fluctuations.')
print('\nKEY INERTIA RESULTS:')
print(df[['epoch','true_H_GFM_set_s','WMU_system_apparent_s','PMU_system_apparent_s','naive_WMU_minus_PMU_s','known_SG_anchored_WMU_s','WMU_GFM_blackbox_fit_s','PMU_GFM_blackbox_fit_s']].to_string(index=False,float_format=lambda a: f'{a:.4f}'))
print('\nERROR SUMMARY AGAINST TRUE GFM H-set:')
print(pd.DataFrame(metrics).to_string(index=False,float_format=lambda a: f'{a:.4f}'))
print('\nFILES:',OUT)
