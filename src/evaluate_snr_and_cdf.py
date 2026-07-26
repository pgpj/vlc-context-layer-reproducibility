#!/usr/bin/env python3
from pathlib import Path
import json,sys
import numpy as np,pandas as pd,torch
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from phase3_worker import P, METHODS, build_bundle, forward, ROOM

torch.set_num_threads(2)
seeds=list(P.seeds); snrs=[0,5,10,15,20,25,30,35,40]
methods=list(METHODS)

def sigma_for_snr(snr):
    floor=1.0; dynamic=np.sqrt(P.aoa_std_deg_at_20db**2-floor**2)
    return float(np.sqrt(floor**2+(dynamic*10**((20.0-snr)/28.2))**2))

def errors(model,x,y):
    model.eval();out=[];scale=torch.tensor(ROOM)
    with torch.no_grad():
        for i in range(0,len(x),1024):
            pred,_=forward(model,x[i:i+1024]);out.append(torch.linalg.vector_norm((pred-y[i:i+1024])*scale,dim=1).numpy())
    return np.concatenate(out)
rows=[];cdf_rows=[]
for seed in seeds:
    loaded={}
    for method in methods:
        model=METHODS[method]['builder']();state=torch.load(ROOT/'checkpoints'/f'{method}_seed_{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(state);loaded[method]=model
    for snr in snrs:
        bundle=build_bundle(seed,P.test_trajectories,3,'mixed',snr_db=snr,aoa_std_deg=sigma_for_snr(snr))
        for method,model in loaded.items():
            x=bundle[1] if METHODS[method]['single'] else bundle[0];e=errors(model,x,bundle[2])
            rows.append({'seed':seed,'snr_db':snr,'aoa_std_deg':sigma_for_snr(snr),'method':method,'label':METHODS[method]['label'],'mae_m':e.mean(),'rmse_m':np.sqrt(np.mean(e**2)),'p90_m':np.quantile(e,.9),'p95_m':np.quantile(e,.95)})
            if snr==20:
                for value in e: cdf_rows.append({'seed':seed,'method':method,'error_m':float(value)})
    print('completed seed',seed,flush=True)
df=pd.DataFrame(rows);df.to_csv(ROOT/'results'/'snr_sweep_per_seed.csv',index=False)
agg=df.groupby(['snr_db','method','label'],as_index=False).agg(mae_m_mean=('mae_m','mean'),mae_m_std=('mae_m','std'),rmse_m_mean=('rmse_m','mean'),p95_m_mean=('p95_m','mean'),aoa_std_deg=('aoa_std_deg','first'))
agg.to_csv(ROOT/'results'/'snr_sweep_aggregate.csv',index=False)
cdf=pd.DataFrame(cdf_rows);cdf.to_csv(ROOT/'results'/'cdf_error_samples_20db.csv',index=False)
# SNR plot, core methods only
core=['adr_tcn_lstm_federated','fl_static_no_densification','adr_tcn_lstm_central','adr_tcn_central','adr_lstm_central']
fig,ax=plt.subplots(figsize=(7.2,4.1))
for method in core:
 g=agg[agg.method==method].sort_values('snr_db');x=g.snr_db.to_numpy();y=100*g.mae_m_mean.to_numpy();s=100*g.mae_m_std.to_numpy();ax.plot(x,y,marker='o',label=METHODS[method]['label']);ax.fill_between(x,y-s,y+s,alpha=.12)
ax.set_xlabel('Optical SNR (dB)');ax.set_ylabel('Mean absolute error (cm)');ax.set_title('Repeated-run SNR sweep (mean ± SD, five seeds)');ax.grid(True,alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'figures'/'phase3_snr_sweep_repeated.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures'/'phase3_snr_sweep_repeated.png',dpi=300,bbox_inches='tight');plt.close(fig)
# CDF at 20 dB
fig,ax=plt.subplots(figsize=(7.0,4.0))
for method in core:
 e=np.sort(cdf[cdf.method==method].error_m.to_numpy());prob=np.arange(1,len(e)+1)/len(e);ax.plot(100*e,prob,label=METHODS[method]['label'])
ax.set_xlabel('Localization error (cm)');ax.set_ylabel('Empirical CDF');ax.set_xlim(left=0);ax.set_ylim(0,1);ax.grid(True,alpha=.3);ax.legend(fontsize=7);ax.set_title('Aggregated error distribution at 20 dB (13,800 windows/method)');fig.tight_layout();fig.savefig(ROOT/'figures'/'phase3_cdf_20db_repeated.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures'/'phase3_cdf_20db_repeated.png',dpi=300,bbox_inches='tight');plt.close(fig)
print(agg[agg.snr_db==20][['method','mae_m_mean','mae_m_std','p95_m_mean']].sort_values('mae_m_mean').to_string(index=False))
