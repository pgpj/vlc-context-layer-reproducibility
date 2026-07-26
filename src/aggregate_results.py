#!/usr/bin/env python3
from pathlib import Path
import csv,json,glob,math
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for f in sorted((ROOT/'results'/'workers').glob('*_seed_*.json')):
    if f.name.endswith('_history.json'): continue
    rows.extend(json.loads(f.read_text()))
df=pd.DataFrame(rows)
df.to_csv(ROOT/'results'/'per_seed_all_conditions.csv',index=False)
metric_cols=['mae_m','rmse_m','p90_m','p95_m','max_m']
agg=[]
for (method,label,condition),g in df.groupby(['method','label','test_condition'],sort=False):
    r={'method':method,'label':label,'test_condition':condition,'seeds':g.seed.nunique(),
       'parameters':int(g.parameters.iloc[0]),'model_size_kib_fp32':g.model_size_kib_fp32.iloc[0],
       'update_size_kib_fp32':g.update_size_kib_fp32.iloc[0],
       'latency_batch1_ms_cpu_mean':g.latency_batch1_ms_cpu.mean(),
       'latency_batch1_ms_cpu_std':g.latency_batch1_ms_cpu.std(ddof=1)}
    for m in metric_cols:
        r[m+'_mean']=g[m].mean();r[m+'_std']=g[m].std(ddof=1)
    agg.append(r)
adf=pd.DataFrame(agg)
adf.to_csv(ROOT/'results'/'aggregate_all_conditions.csv',index=False)
mixed=adf[adf.test_condition=='mixed'].copy().sort_values('mae_m_mean')
mixed.to_csv(ROOT/'results'/'aggregate_ablation_mixed.csv',index=False)
# Paired comparisons using identical seeds.
comparisons=[
 ('adr_tcn_lstm_central','adr_tcn_central','Hybrid central vs TCN-only'),
 ('adr_tcn_lstm_central','adr_lstm_central','Hybrid central vs LSTM-only'),
 ('adr_tcn_lstm_federated','fl_static_no_densification','Federated hybrid vs federated static/no densification'),
 ('adr_tcn_lstm_federated','adr_tcn_lstm_central','Federated hybrid vs centralized hybrid'),
 ('adr_tcn_lstm_federated','single_pd_lstm_central','ADR federated hybrid vs single-PD LSTM'),
]
comp=[]
mdf=df[df.test_condition=='mixed']
for a,b,label in comparisons:
    ga=mdf[mdf.method==a].set_index('seed').sort_index();gb=mdf[mdf.method==b].set_index('seed').sort_index();common=ga.index.intersection(gb.index)
    va=ga.loc[common,'mae_m'].to_numpy();vb=gb.loc[common,'mae_m'].to_numpy();diff=va-vb
    t=stats.ttest_rel(va,vb)
    try: w=stats.wilcoxon(va,vb,alternative='two-sided')
    except Exception: w=type('W',(),{'statistic':np.nan,'pvalue':np.nan})()
    comp.append({'comparison':label,'method_a':a,'method_b':b,'n':len(common),'a_mae_cm_mean':100*va.mean(),'b_mae_cm_mean':100*vb.mean(),
                 'difference_a_minus_b_cm_mean':100*diff.mean(),'relative_change_a_vs_b_percent':100*(va.mean()-vb.mean())/vb.mean(),
                 'paired_t_stat':t.statistic,'paired_t_p':t.pvalue,'wilcoxon_stat':w.statistic,'wilcoxon_p':w.pvalue})
pd.DataFrame(comp).to_csv(ROOT/'results'/'paired_comparisons.csv',index=False)
# Publication-ready summary in cm.
summary=[]
for _,r in mixed.iterrows():
    summary.append({'Method':r.label,'MAE (cm)':f"{100*r.mae_m_mean:.2f} ± {100*r.mae_m_std:.2f}",
                    'RMSE (cm)':f"{100*r.rmse_m_mean:.2f} ± {100*r.rmse_m_std:.2f}",
                    'P95 (cm)':f"{100*r.p95_m_mean:.2f} ± {100*r.p95_m_std:.2f}",
                    'Parameters':int(r.parameters),'Model size (KiB)':f"{r.model_size_kib_fp32:.1f}",
                    'Update (KiB)':f"{r.update_size_kib_fp32:.1f}",
                    'CPU latency (ms)':f"{r.latency_batch1_ms_cpu_mean:.3f}"})
pd.DataFrame(summary).to_csv(ROOT/'results'/'manuscript_summary_table.csv',index=False)
print(mixed[['method','mae_m_mean','mae_m_std','rmse_m_mean','p95_m_mean','parameters']].to_string(index=False))
print('\nComparisons')
print(pd.DataFrame(comp).to_string(index=False))
