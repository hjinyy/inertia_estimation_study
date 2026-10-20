#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible synthetic SG / ESS-VSG / PV-VSG multiscale inertia study.

Run: python run_multiscale_validation.py
Packages: numpy scipy pandas matplotlib

IMPORTANT: Not a real IEEE 39-bus or proprietary GFM controller. Generator and
estimator use DIFFERENT dynamic model classes. The estimator uses NO true
inertia, internal filter, or reference controller telemetry; those variables
are reserved for numerical ground truth. Nonetheless the proposed method is
PHYSICS-GUIDED (frequency-proportional and derivative-proportional paths), NOT
fully assumption-free, nor a claim of structural identifiability.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import os
from scipy import signal, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / ('results_seed_' + os.environ['WMU_SEED'] if 'WMU_SEED' in os.environ else 'results')
OUT.mkdir(exist_ok=True, parents=True)
SEED = int(os.environ.get('WMU_SEED','20261019'))
rng = np.random.default_rng(SEED)
FS_STATE, FS_RAW, FS_PMU = 240, 15360, 60
RATE_RATIO = FS_RAW // FS_STATE
F0, SBASE, HSG = 60., 500., 2.8
D_SG = 27.0
STAGE_DURATION = 120.0
H_ESS = np.array([.60, 1.00, .60, .00])  # 500MVA system base
H_PV = np.array([.22, .22, .08, .00])
N_STAGE = len(H_ESS)
N = round(N_STAGE*STAGE_DURATION*FS_STATE)
T = np.arange(N)/FS_STATE
dt = 1/FS_STATE
assert FS_RAW%FS_STATE==0 and FS_STATE%FS_PMU==0
# DEVICE DATA BELOW EXISTS ONLY INSIDE SYNTHETIC DATA GENERATOR / ORACLE.
# Unknown and unavailable to estimation routines.
DEV = [
 {'id':'ESS-VSG','D':19., 'tau_i':[.019,.075], 'weight_i':[.57,.43],
  'tau_d':[.070,.240], 'weight_d':[.60,.40], 'delay':.012, 'rating':100.},
 {'id':'PV-VSG','D':11., 'tau_i':[.028,.110], 'weight_i':[.65,.35],
  'tau_d':[.080,.290], 'weight_d':[.65,.35], 'delay':.021, 'rating':50.},
]

# Random ambient load (not disturbance probing), with informative 0.15-9.5Hz content.
# CAUTION: this intentionally broadband stimulus is favorable for identification;
# actual ambient power spectra may offer insufficient excitation above ~2 Hz.
load = np.zeros(N)
for band, rms in [((.13,1.5),1.3),((1.5,9.5),3.2)]:
    bpf=signal.butter(3,band,fs=FS_STATE,btype='bandpass',output='sos')
    x=signal.sosfiltfilt(bpf,rng.normal(size=N))
    load+=rms*x/np.std(x)
load+=.2*np.sin(2*np.pi*.47*T)

# CLOSED-LOOP SG frequency: SG rotor inertia + droop feedback from two VSGs.
# Plant: mixtures of TWO (unknown to estimator) inertial/droop time constants,
# with actual input delay. No use of estimator's transfer-dictionary for synthesis.
f=np.zeros(N); rocof=np.zeros(N)
p_in=np.zeros((2,N)); p_dr=np.zeros((2,N))
xi=np.zeros((2,2)); zd=np.zeros((2,2))
msg=2*SBASE*HSG/F0
for k in range(N-1):
    st=min(int(T[k]//STAGE_DURATION), N_STAGE-1)
    for j,dev in enumerate(DEV):
        mj=2*SBASE*([H_ESS,H_PV][j][st])/F0
        p_in[j,k] = -mj * np.dot(dev['weight_i'],xi[j])
        p_dr[j,k] = -dev['D'] * np.dot(dev['weight_d'],zd[j])
    rocof[k]=(-load[k]-D_SG*f[k]+np.sum(p_in[:,k]+p_dr[:,k]))/msg
    f[k+1]=f[k]+dt*rocof[k]
    for j,dev in enumerate(DEV):
        kd=max(0,k-round(dev['delay']*FS_STATE))
        xi[j]+=dt*(rocof[kd]-xi[j])/np.array(dev['tau_i'])
        zd[j]+=dt*(f[kd]-zd[j])/np.array(dev['tau_d'])
p_in[:,-1]=p_in[:,-2];p_dr[:,-1]=p_dr[:,-2];rocof[-1]=rocof[-2]
p_terminal=p_in+p_dr

# GENERATE RAW 3-PHASE VOLTAGE / CURRENT and extract frequency/active power from
# samples. Never feed the ground-truth P or f into estimators.
VLL=13.8e3;VPK=np.sqrt(2)*VLL/np.sqrt(3)
PBASE=(55.,35.)
W_WMU=round(FS_RAW/60)  # 1-cycle measurement filter
W_PMU=6*W_WMU          # slow 'PMU-like' benchmark, not IEEE-certified
W_PMU_FAST=2*W_WMU     # 2-cycle comparator; still not IEEE C37.118 qualified
box_w=np.ones(W_WMU)/W_WMU
box_p=np.ones(W_PMU)/W_PMU
box_pfast=np.ones(W_PMU_FAST)/W_PMU_FAST
keys=['f','ess','pv']
ziw={k:np.zeros(W_WMU-1) for k in keys}
zipmu={k:np.zeros(W_PMU-1) for k in keys}
zipfast={k:np.zeros(W_PMU_FAST-1) for k in keys}
feat_w={k:np.zeros(N) for k in keys}
feat_p={k:np.zeros(N) for k in keys}
feat_pfast={k:np.zeros(N) for k in keys}
raw_preview=None; phase_offset=0.
chunk=2*FS_STATE
for start in range(0,N,chunk):
    stop=min(start+chunk,N)
    ti=(start*RATE_RATIO+np.arange((stop-start)*RATE_RATIO))/FS_RAW
    freq_hi=np.interp(ti,T[start:stop],f[start:stop])
    phase=phase_offset+2*np.pi*np.cumsum(F0+freq_hi)/FS_RAW
    phase_offset=float(phase[-1])
    ang=phase[:,None]-np.arange(3)[None,:]*2*np.pi/3
    vabc=VPK*(np.cos(ang)+.0012*np.cos(5*ang))
    vabc+=VPK*rng.normal(0,1.5e-5,vabc.shape)
    alpha=(2*vabc[:,0]-vabc[:,1]-vabc[:,2])/3
    beta=(vabc[:,1]-vabc[:,2])/np.sqrt(3)
    theta=np.unwrap(np.arctan2(beta,alpha))
    inst_freq=np.gradient(theta,1/FS_RAW)/(2*np.pi)
    streams={'f':inst_freq}
    for j,key in enumerate(['ess','pv']):
        dp=np.interp(ti,T[start:stop],p_terminal[j,start:stop])
        pabs=PBASE[j]+dp
        peak_current=2*pabs*1e6/(3*VPK)
        iabc=peak_current[:,None]*(np.cos(ang)+.0018*np.cos(5*ang+.1))
        iabc+=peak_current[:,None]*rng.normal(0,1.3e-4,ang.shape)
        streams[key]=(vabc*iabc).sum(axis=1)/1e6
        if start==0 and j==0:
            n_preview=min(len(ti),round(.06*FS_RAW))
            raw_preview=pd.DataFrame({'time_s':ti[:n_preview],
                **{f'v_{q}_V':vabc[:n_preview,h] for h,q in enumerate('abc')},
                **{f'i_{q}_A':iabc[:n_preview,h] for h,q in enumerate('abc')}})
    for key,stream in streams.items():
        a,ziw[key]=signal.lfilter(box_w,[1.],stream,zi=ziw[key])
        b,zipmu[key]=signal.lfilter(box_p,[1.],stream,zi=zipmu[key])
        bf,zipfast[key]=signal.lfilter(box_pfast,[1.],stream,zi=zipfast[key])
        feat_pfast[key][start:stop]=bf[RATE_RATIO-1::RATE_RATIO][:stop-start]
        feat_w[key][start:stop]=a[RATE_RATIO-1::RATE_RATIO][:stop-start]
        feat_p[key][start:stop]=b[RATE_RATIO-1::RATE_RATIO][:stop-start]
# Sensor error (input noise independent of synthetic plant)
for dat,nf,npower in [(feat_w,.00003,.005),(feat_p,.00002,.003),(feat_pfast,.00002,.003)]:
    dat['f']+=rng.normal(0,nf,N)
    for key in ['ess','pv']:
        dat[key]+=rng.normal(0,npower,N)
raw_preview.to_csv(OUT/'wmU_abc_60ms.csv',index=False)


# ========== Improved DPSS estimator and conventional ARX baseline ==========

# No true parameters from DEV/H_ESS/H_PV enter either estimator below.
# Both baseline and proposed see the SAME 1-cycle WMU features at 240 Hz.
# PMU-like features are generated only for potential future sensor ablations.
from scipy import linalg
import json, warnings, time

FREQ_RANGE = (0.4, 7.0)            # same Hobs evaluation band across methods
FIT_RANGE = (0.25, 9.0)
SEG_SEC = 8.5333333                # 2048 samples at 240fps / 512 at 60fps
DPSS_NW, DPSS_K = 3.0, 4
# A generic distributed-filter vocabulary. NOT the actual 2-pole generator model.
# Normalized shape constraints: droop gain >=0, inertia gain >=0.
TAU_D = np.array([.035,.075,.14,.28,.48])
TAU_I = np.array([.007,.016,.035,.07,.145])
DELAYS = np.arange(0.,.046,.005)
REG = [0.006,0.035]                # predeclared search; CV on disjoint time-blocks
N_BOOT = 18


def segment_periodograms(u,y,fs,nseg):
    """Independent non-overlapping DPSS segments, one output spectral tuple per block.
    u(t) = measured frequency deviation, y(t) = -(measured P minus Pbar).
    No imaginary reference excitation or simulator-truth leakage.
    """
    nblock=len(u)//nseg
    tapers=signal.windows.dpss(nseg,DPSS_NW,Kmax=DPSS_K,sym=False)
    out=[]
    for bi in range(nblock):
        sl=slice(bi*nseg,(bi+1)*nseg)
        xu=signal.detrend(u[sl],type='linear')
        xy=signal.detrend(y[sl],type='linear')
        uf=np.fft.rfft(tapers*xu[None,:],axis=1)
        yf=np.fft.rfft(tapers*xy[None,:],axis=1)
        ff=np.mean(np.conj(uf)*uf,axis=0).real
        fp=np.mean(np.conj(uf)*yf,axis=0)
        pp=np.mean(np.conj(yf)*yf,axis=0).real
        out.append(np.stack([ff,fp,pp],axis=0))
    return np.fft.rfftfreq(nseg,1/fs),np.stack(out,axis=0)


def frf_from_parts(part):
    # Averaging spectra before the ratio reduces high-frequency variance.
    ff=np.mean(part[:,0],axis=0).real
    fp=np.mean(part[:,1],axis=0)
    pp=np.mean(part[:,2],axis=0).real
    H=fp /np.maximum(ff,1e-12)
    coh=(abs(fp)**2)/np.maximum(ff*pp,1e-15)
    # Small ambient excitation near-DC is downweighted, not given infinite authority.
    return H,np.clip(coh,0,1),ff


def estimate_arx(f,p,fs,hz_eval):
    """Standard linear ARX input-output identification baseline.
    y=-P, u=f, model: y[k]=sum(a_i*y[k-i])+sum(b_j*u[k-j])+e[k].
    Order and weak ridge chosen with a *chronological* 80/20 split.
    Converting ARX transfer response to apparent H follows the same spectral
    definition as the Proposed method, NOT paper-specific ARMAX inertia formula.
    """
    u=np.asarray(f,float)-np.mean(f)
    y=-(np.asarray(p,float)-np.mean(p))
    choices=[(8,16),(16,32),(24,48)]
    selected=None
    for na,nb in choices:
        n=len(u); offset=max(na,nb-1)
        yy=y[offset:]
        X=np.column_stack(
            [y[offset-i:n-i] for i in range(1,na+1)] +
            [u[offset-j:n-j] if j else u[offset:] for j in range(nb)])
        cp=int(.80*len(yy))
        colnorm=np.sqrt(np.mean(X[:cp]**2,axis=0))+1e-10
        Z=X/colnorm
        A=Z[:cp].T@Z[:cp]
        ridge=0.001*np.trace(A)/len(A)
        beta=linalg.solve(A+ridge*np.eye(A.shape[0]),Z[:cp].T@yy[:cp],assume_a='pos')/colnorm
        validation_error=float(np.mean((Z[cp:]@(beta*colnorm)-yy[cp:])**2))
        if selected is None or validation_error<selected['val']:
            selected=dict(val=validation_error,na=na,nb=nb,beta=beta)
    na=selected['na'];nb=selected['nb'];beta=selected['beta']
    w=2*np.pi*np.asarray(hz_eval)/fs
    _,H=signal.freqz(beta[na:],np.r_[1.,-beta[:na]],worN=w)
    hi=(F0/(2*SBASE))*np.imag(H)/(2*np.pi*hz_eval)
    return float(np.mean(hi)),hi,f'ARX({na},{nb})'


def make_dictionary(hz,delay):
    w=2*np.pi*np.asarray(hz)
    s=1j*w[:,None]
    ed=np.exp(-1j*w*delay)[:,None]
    # Droop columns are physical MW/Hz, inertia columns physical system-base sec.
    d=ed/(1+s*TAU_D[None,:])
    i=ed*(2*SBASE/F0)*s/(1+s*TAU_I[None,:])
    return np.column_stack((d,i))


def train_nnls(hz,frf,coh,ff,delay,lam=0.035,iterative=True):
    # Sff is used to reject nearly unexcited frequencies; coherence downweights noise.
    use=(hz>=FIT_RANGE[0])&(hz<=FIT_RANGE[1])&(coh>.18)&np.isfinite(frf)
    if use.sum()<25:return None
    zhz=hz[use];obs=frf[use]
    dict0=make_dictionary(zhz,delay)
    scale=np.sqrt(np.mean(abs(dict0)**2,axis=0))
    Acomplex=dict0/scale
    snr=np.sqrt(np.maximum(coh[use],.05))
    # De-emphasize bands whose frequency excursion is scarcely excited.
    energy_weight=np.sqrt(np.clip(ff[use]/np.percentile(ff[use],70),.1,2.0))
    ww=snr*energy_weight
    X=np.vstack([Acomplex.real*ww[:,None],Acomplex.imag*ww[:,None]])
    y=np.r_[obs.real*ww,obs.imag*ww]
    # normalized regularization is relative to total excitation.
    penalty=lam*np.maximum(np.linalg.norm(X,axis=0),.01)**2
    coeffnorm=np.zeros(X.shape[1]);
    for it in range(3 if iterative else 1):
        # Stabilize lightly populated basis poles via iterative reweighted ridge;
        # not a proof of sparsity optimality. Kept strictly nonnegative.
        if iterative and it>0:
            penalty2=penalty * np.sqrt((np.median(coeffnorm[coeffnorm>1e-4])+.02)/(coeffnorm+.02))
        else: penalty2=penalty
        Z=np.vstack([X,np.diag(np.sqrt(penalty2))])
        t=np.r_[y,np.zeros(X.shape[1])]
        coeffnorm,_=optimize.nnls(Z,t,maxiter=400)
    coeff=coeffnorm/scale
    return coeff


def train_and_select(hz,periodograms):
    nblk=len(periodograms)
    tr=periodograms[np.arange(nblk)%3!=2]
    vl=periodograms[np.arange(nblk)%3==2]
    Ht,ct,ff=frf_from_parts(tr)
    Hv,cv,ffv=frf_from_parts(vl)
    valid=(hz>=FIT_RANGE[0])&(hz<=FIT_RANGE[1])&(cv>.12)
    # Prediction loss measured on *held-out temporal blocks*, not alternating bins.
    chosen=None
    for delay in DELAYS:
        for lam in REG:
            coeff=train_nnls(hz,Ht,ct,ff,delay,lam,iterative=True)
            if coeff is None:continue
            pred=make_dictionary(hz[valid],delay)@coeff
            scalev=np.maximum(np.mean(abs(Hv[valid])**2),1e-6)
            loss=float(np.mean(abs(pred-Hv[valid])**2)/scalev)
            # Extra small preference for simpler models when loss is tied.
            if chosen is None or loss<chosen['validation_loss']:
                chosen=dict(delay=delay,reg=lam,validation_loss=loss)
    if chosen is None:return None
    H,coh,power=frf_from_parts(periodograms)
    coeff=train_nnls(hz,H,coh,power,chosen['delay'],chosen['reg'],iterative=True)
    chosen['coeff']=coeff
    chosen['coh_mean']=float(np.mean(coh[(hz>=FIT_RANGE[0])&(hz<=FIT_RANGE[1])]))
    return chosen


def inertia_curve(hz,par):
    # IMPORTANT: only the fitted inertia-path spectral imaginary contribution.
    w=2*np.pi*hz
    d=par['delay'];c=par['coeff'][len(TAU_D):]
    basis=(np.exp(-1j*w*d)[:,None]*(1j*w[:,None])/(1+1j*w[:,None]*TAU_I))
    z=basis@c
    curve=np.imag(z)/w
    return curve


def model_total_response(hz,par):
    return make_dictionary(hz,par['delay'])@par['coeff']


def benchmark_true_curve(hz,stage,device):
    # ORACLE SCORING ONLY. NEVER IN THE ESTIMATOR.
    dev=DEV[device]; w=2*np.pi*np.asarray(hz)
    roots=(1j*w[:,None])/(1+1j*w[:,None]*np.asarray(dev['tau_i']))
    roots*=np.exp(-1j*w[:,None]*dev['delay'])
    return ([H_ESS,H_PV][device][stage])*np.imag(roots@np.asarray(dev['weight_i']))/w


def block_bootstrap(hz,parts,par,rng,nboot=N_BOOT):
    """Resample non-overlapping DPSS blocks; model choice frozen at CV selection.
    Gives conditional measurement variability, not full uncertainty for wrong
    topology / controller identification. CI may under-cover biased estimates.
    """
    vals=[]
    for _ in range(nboot):
        idx=rng.integers(0,len(parts),len(parts))
        H,coh,power=frf_from_parts(parts[idx])
        coeff=train_nnls(hz,H,coh,power,par['delay'],par['reg'],iterative=True)
        if coeff is None: continue
        variant={**par,'coeff':coeff}
        grid=np.linspace(*FREQ_RANGE,101)
        vals.append(np.mean(inertia_curve(grid,variant)))
    return np.asarray(vals)


def reliability_flag(value,samples,coh_mean,validation,model_low,model_high):
    # Conservative diagnostic only: not a validated probability of correctness.
    if len(samples)<8:return 'INSUFFICIENT_BOOTSTRAP'
    if coh_mean<.30:return 'LOW_COHERENCE'
    if validation>.15:return 'WEAK_HELDOUT_FIT'
    if model_low<=0<=model_high:return 'ZERO_COMPATIBLE / DO_NOT_CLAIM_INERTIA'
    if (model_high-model_low)>max(.12,.6*abs(value)):
        return 'MODEL_AMBIGUITY'
    return 'ESTIMABLE_PROVISIONAL_NOT_CALIBRATED'

print('>>> Identifying conventional ARX and proposed multitaper estimators...',flush=True)
MAINHZ=np.linspace(*FREQ_RANGE,101)
rows=[];spec_results=[];device_results=[]
starttime=time.time()
for st in range(N_STAGE):
    a,b=st*STAGE_DURATION+15,(st+1)*STAGE_DURATION-8
    region=(T>=a)&(T<b)
    strow=dict(stage=st+1,window_s=f'{a:.0f}-{b:.0f}',true_H_SG_s=HSG,
               true_H_GFM_set_s=float(H_ESS[st]+H_PV[st]))
    totals=dict(arx=0.,proposed=0.,true=0.)
    bootstrap_comps=[]
    for j,key in enumerate(['ess','pv']):
        u=feat_w['f'][region]-F0
        y=feat_w[key][region]
        hz,parts=segment_periodograms(u,-y,FS_STATE,nseg=2048)
        par=train_and_select(hz,parts)
        if par is None: raise RuntimeError(f'Identification failed stage {st+1} {key}')
        proposal_curve=inertia_curve(MAINHZ,par)
        proposed=float(np.mean(proposal_curve))
        truths=benchmark_true_curve(MAINHZ,st,j)
        actual=float(np.mean(truths))
        # Fair benchmark: ARX and proposed see identical WMU 240-Hz terminal features.
        f_bas=feat_w['f'][region]
        p_bas=feat_w[key][region]
        baseline, arxcurve, order=estimate_arx(f_bas,p_bas,FS_STATE,MAINHZ)
        bs=block_bootstrap(hz,parts,par,rng)
        bootstrap_comps.append(bs)
        totals['true']+=actual;totals['arx']+=baseline;totals['proposed']+=proposed
        device_results.append(dict(stage=st+1,device=DEV[j]['id'],true_Hobs_s=actual,
            ARX_Hobs_s=baseline,Proposed_Hobs_s=proposed,
            ARX_error_s=baseline-actual,Proposed_error_s=proposed-actual,
            true_Hset_s=float([H_ESS,H_PV][j][st]),
            proposed_Hset_proxy_s=float(par['coeff'][len(TAU_D):].sum()),
            true_D_MW_per_Hz=DEV[j]['D'],proposed_D_MW_per_Hz=float(par['coeff'][:len(TAU_D)].sum()),
            true_delay_ms=DEV[j]['delay']*1000,proposed_delay_ms=par['delay']*1000,
            spectral_cv_loss=par['validation_loss'],mean_coherence=par['coh_mean'],
            ARX_order=order))
        if st in [0,3]:spec_results.append(dict(stage=st+1,device=DEV[j]['id'],freq=MAINHZ,
              truth=truths,baseline=arxcurve,proposed=proposal_curve))
    # Bootstrap sums paired by run number. Approx assumption: component uncertainty
    # is independent conditional on common frequency; full physical covariance omitted.
    sumboot=np.sum(np.vstack(bootstrap_comps),axis=0)
    ci_lo,ci_hi=np.percentile(sumboot,[2.5,97.5])
    dsub=[v for v in device_results if v['stage']==st+1]
    # Stationary block-bootstrap quantifies only conditional spectral noise.
    # Model-form error is not captured by resampling. Add an explicitly heuristic
    # cross-validated spectral mismatch sensitivity margin (not a 95% CI).
    model_mismatch=float(sum(2*abs(q['Proposed_Hobs_s'])*np.sqrt(max(q['spectral_cv_loss'],0))
                             for q in dsub))
    model_mismatch=max(.007,model_mismatch)
    model_low,model_high=ci_lo-model_mismatch,ci_hi+model_mismatch
    flag=reliability_flag(totals['proposed'],sumboot,
        min(q['mean_coherence'] for q in dsub),max(q['spectral_cv_loss'] for q in dsub),
        model_low,model_high)
    strow.update(true_Hobs_s=totals['true'],ARX_baseline_Hobs_s=totals['arx'],
        WMU_multitaper_Hobs_s=totals['proposed'],
        ARX_error_s=totals['arx']-totals['true'],
        Proposed_error_s=totals['proposed']-totals['true'],
        proposed_bootstrap_95_lo_s=ci_lo,proposed_bootstrap_95_hi_s=ci_hi,
        bootstrap_ci_covers_true=(ci_lo<=totals['true']<=ci_hi),
        model_mismatch_margin_s=model_mismatch,
        uncertainty_model_heuristic_lo_s=model_low,uncertainty_model_heuristic_hi_s=model_high,
        heuristic_band_covers_true=(model_low<=totals['true']<=model_high),
        reliability=flag)
    rows.append(strow)
    print(f"Stage {st+1}: True={totals['true']:.5f} ARX={totals['arx']:.5f} "
          f"Proposed={totals['proposed']:.5f} CI[{ci_lo:.4f},{ci_hi:.4f}] {flag}",flush=True)

stage_df=pd.DataFrame(rows)
device_df=pd.DataFrame(device_results)
stage_df.to_csv(OUT/'stage_true_baseline_proposed_errors.csv',index=False)
device_df.to_csv(OUT/'device_true_baseline_proposed_diagnostics.csv',index=False)
rmse=lambda a,b:float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
mae=lambda a,b:float(np.mean(abs(np.asarray(a)-np.asarray(b))))
ytrue=stage_df.true_Hobs_s.values
metric=[]
for label,col in [('ARX Baseline (same WMU)','ARX_baseline_Hobs_s'),('Proposed WMU Multitaper','WMU_multitaper_Hobs_s')]:
    z=stage_df[col].values
    metric.append(dict(method=label,RMSE_s=rmse(z,ytrue),MAE_s=mae(z,ytrue),
        bias_s=float(np.mean(z-ytrue)),max_abs_error_s=float(np.max(abs(z-ytrue))),
        negative_control_abs_s=float(abs(z[-1]-ytrue[-1]))))
metric_df=pd.DataFrame(metric)
metric_df.to_csv(OUT/'summary_accuracy.csv',index=False)
print('\nSUMMARY\n',metric_df.to_string(index=False))

# ========= FIGURES: only TRUE / ARX BASELINE / PROPOSED =========
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','figure.dpi':115,
 'axes.spines.top':False,'axes.spines.right':False})
BLACK='#242b34'; ORANGE='#D08732'; BLUE='#2277B2'
xx=stage_df.stage.values

def exportfig(name):
    plt.savefig(OUT/name,dpi=200,bbox_inches='tight');plt.close()

fig,(ax,ae)=plt.subplots(2,1,figsize=(10,7.5),sharex=True,gridspec_kw={'height_ratios':[1.5,1]})
ax.plot(xx,ytrue,'k--o',lw=2.4,ms=8,label='True GFM Hobs (oracle scoring)')
ax.plot(xx,stage_df.ARX_baseline_Hobs_s,'-s',lw=2,color=ORANGE,ms=7,label='Baseline: ARX / same WMU')
ax.plot(xx,stage_df.WMU_multitaper_Hobs_s,'-o',lw=2,color=BLUE,ms=7,label='Proposed: WMU DPSS + constrained multiscale')
lo=stage_df.uncertainty_model_heuristic_lo_s.values;hi=stage_df.uncertainty_model_heuristic_hi_s.values
ax.fill_between(xx,lo,hi,color=BLUE,alpha=.15,label='Model-mismatch sensitivity band (heuristic)')
for k,tv in enumerate(ytrue):
    ax.annotate(f'{tv:.3f}s',(xx[k],tv),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
ax.set(title='Fig 1 | GFM observed inertia: true vs ARX baseline vs proposed',ylabel='System-base Hobs [s]')
ax.legend(fontsize=9,ncol=2)
ae.axhline(0,c=BLACK,lw=1)
ae.plot(xx,stage_df.ARX_error_s,'-s',color=ORANGE,label='ARX error')
ae.plot(xx,stage_df.Proposed_error_s,'-o',color=BLUE,label='Proposed error')
ae.set(ylabel='Estimated minus true [s]',xlabel='Stage (4: true inertia = 0)',xticks=xx)
ae.legend(ncol=2,fontsize=9)
fig.tight_layout();exportfig('Fig1_stage_true_baseline_proposed.png')

fig,ax=plt.subplots(figsize=(9,4.8))
names=metric_df.method.tolist(); yy=np.arange(len(names))
bars=ax.barh(yy,metric_df.RMSE_s,color=[ORANGE,BLUE],height=.46)
for z,bar in zip(metric_df.RMSE_s,bars):
    ax.text(float(z)+max(metric_df.RMSE_s)*.02,bar.get_y()+bar.get_height()/2,f'{z:.4f} s',va='center',fontsize=11)
ax.set(yticks=yy,yticklabels=names,xlabel='RMSE vs true Hobs [s]',title='Fig 2 | Quantitative error (lower is better)',xlim=(0,max(metric_df.RMSE_s)*1.32))
ax.invert_yaxis();fig.tight_layout();exportfig('Fig2_RMSE_baseline_proposed.png')

fig,axs=plt.subplots(1,2,figsize=(12.3,4.5),sharey=True)
for ax,st in zip(axs,[1,4]):
    c=[s for s in spec_results if s['stage']==st and s['device']=='ESS-VSG'][0]
    ax.plot(c['freq'],c['truth'],'k--',lw=2.2,label='TRUE inertial Hobs')
    ax.plot(c['freq'],c['baseline'],color=ORANGE,lw=1.6,label='ARX / same WMU')
    ax.plot(c['freq'],c['proposed'],color=BLUE,lw=1.8,label='Proposed / WMU')
    ax.set(title=f'Stage {st}: ESS-VSG'+(' (zero inertia)' if st==4 else ''),xlabel='Ambient fluctuation frequency [Hz]')
    ax.grid(alpha=.15)
axs[0].set_ylabel('Frequency-resolved Hobs [s]')
axs[1].legend(fontsize=8)
fig.suptitle('Fig 3 | Do both estimators separate droop delay from inertia?')
fig.tight_layout();exportfig('Fig3_true_frequency_resolved_response.png')

fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),sharey=True)
for ax,dev in zip(axes,DEV):
    subset=device_df[device_df.device==dev['id']]
    ax.plot(subset.stage,subset.true_Hobs_s,'k--o',lw=2.1,label='True')
    ax.plot(subset.stage,subset.ARX_Hobs_s,'-s',lw=1.8,color=ORANGE,label='ARX / PMU')
    ax.plot(subset.stage,subset.Proposed_Hobs_s,'-o',lw=1.8,color=BLUE,label='Proposed / WMU')
    ax.set(title=dev['id'],xlabel='Stage',xticks=xx)
axes[0].set_ylabel('System-base Hobs [s]');axes[1].legend(fontsize=9)
fig.suptitle('Fig 4 | ESS and PV device-level contribution versus true Hobs')
fig.tight_layout();exportfig('Fig4_ESS_PV_component_true_comparison.png')

# Bootstrap coverage on held-out generated truth is a *diagnostic*, not a
# guaranteed interval; report both the successes and misses.
summarymeta={
 'experiment':'Closed-loop SG+ESS/PV black-box ambient synthetic waveform, 4 x 120-s stages',
 'sampling':{'raw_abc_Hz':FS_RAW,'WMU_features_Hz':FS_STATE,'PMU_like_fps':FS_PMU},
 'estimators':{
  'baseline':'SAME WMU 1-cycle 240-Hz features as proposed, ARX input/output fit selected by chronological hold-out, spectral apparent-inertia conversion',
  'proposed':'WMU 1-cycle features, DPSS multitaper, non-overlap 8.53s blocks, temporal CV, constrained nonnegative response dictionary, iteratively reweighted ridge, block bootstrap'},
 'not_claimed':['No proof of universal separation of droop and inertia','No IEEE 39-bus EMT system or real grid measurements',
 'No exact reimplementation of published Tuttelberg ARMAX inertia estimator',
 'Synthetic excitation intentionally broadband up to 9.5Hz; a real power-grid ambient spectrum can be much narrower',
 'Stationary block bootstrap underestimates total error; displayed band adds heuristic heldout model-mismatch margin, NOT a calibrated statistical confidence interval'],
 'parameters':{'seed':SEED,'dpss_NW':DPSS_NW,'dpss_K':DPSS_K,'bootstraps':N_BOOT,
 'estimated_band_Hz':list(FREQ_RANGE),'training_band_Hz':list(FIT_RANGE),
 'identification_seconds_per_stage':STAGE_DURATION-23},
 'RMSE_improvement_pct':float(100*(1-metric_df.iloc[1].RMSE_s/metric_df.iloc[0].RMSE_s))
}
(OUT/'experiment_metadata.json').write_text(json.dumps(summarymeta,ensure_ascii=False,indent=2))
print('ALL RESULTS',OUT, 'seconds',round(time.time()-starttime,1))
