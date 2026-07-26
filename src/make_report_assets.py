#!/usr/bin/env python3
from pathlib import Path
import json,platform,sys
import numpy as np,pandas as pd,torch,scipy,matplotlib
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
res=ROOT/'results'; figs=ROOT/'figures'; report=ROOT/'report'; figs.mkdir(exist_ok=True);report.mkdir(exist_ok=True)
agg=pd.read_csv(res/'aggregate_ablation_mixed.csv')
order=['adr_tcn_lstm_central','adr_lstm_central','adr_tcn_central','adr_tcn_lstm_federated','fl_static_no_densification']
labels={'adr_tcn_lstm_central':'TCN-LSTM\ncentral','adr_lstm_central':'LSTM-only\ncentral','adr_tcn_central':'TCN-only\ncentral','adr_tcn_lstm_federated':'TCN-LSTM\nfederated','fl_static_no_densification':'Static FL\nno densification'}
d=agg.set_index('method').loc[order]
fig,ax=plt.subplots(figsize=(7.2,3.8));x=np.arange(len(d));ax.bar(x,100*d.mae_m_mean,yerr=100*d.mae_m_std,capsize=4);ax.set_xticks(x,labels=[labels[i] for i in order]);ax.set_ylabel('MAE (cm)');ax.set_title('Controlled ablation at 20 dB (mean ± SD, five seeds)');ax.grid(axis='y',alpha=.3);fig.tight_layout();fig.savefig(figs/'phase3_core_ablation_mae.pdf',bbox_inches='tight');fig.savefig(figs/'phase3_core_ablation_mae.png',dpi=300,bbox_inches='tight');plt.close(fig)
# Missingness plot
allc=pd.read_csv(res/'aggregate_all_conditions.csv');sel=allc[allc.method.isin(['adr_tcn_lstm_federated','fl_static_no_densification'])].copy();conds=['independent','mixed','burst'];names={'adr_tcn_lstm_federated':'Proposed federated TCN-LSTM','fl_static_no_densification':'Static FL without densification'}
fig,ax=plt.subplots(figsize=(6.8,3.8));xx=np.arange(len(conds));width=.36
for j,m in enumerate(['adr_tcn_lstm_federated','fl_static_no_densification']):
 g=sel[sel.method==m].set_index('test_condition').loc[conds];ax.bar(xx+(j-.5)*width,100*g.mae_m_mean,width,yerr=100*g.mae_m_std,capsize=4,label=names[m])
ax.set_xticks(xx,labels=['Independent\nanchor losses','Mixed losses','Burst-like\nfull-pilot gaps']);ax.set_ylabel('MAE (cm)');ax.set_title('Effect of missing-observation structure');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.3);fig.tight_layout();fig.savefig(figs/'phase3_missingness_ablation.pdf',bbox_inches='tight');fig.savefig(figs/'phase3_missingness_ablation.png',dpi=300,bbox_inches='tight');plt.close(fig)
# Compact overhead calculation
prop=agg[agg.method=='adr_tcn_lstm_federated'].iloc[0];selected=6;rounds=40;update_kib=float(prop.update_size_kib_fp32);uplink_round=selected*update_kib;downlink_round=selected*update_kib;total_mib=(uplink_round+downlink_round)*rounds/1024
overhead={'parameters':int(prop.parameters),'model_size_kib_fp32':float(prop.model_size_kib_fp32),'single_client_update_kib_fp32':update_kib,'selected_clients_per_round':selected,'uplink_kib_per_round_excluding_protocol_overhead':uplink_round,'downlink_kib_per_round_if_unicast':downlink_round,'bidirectional_mib_over_40_rounds_if_unicast':total_mib,'latency_batch1_ms_cpu_mean':float(prop.latency_batch1_ms_cpu_mean),'latency_batch1_ms_cpu_std':float(prop.latency_batch1_ms_cpu_std),'note':'Communication calculations exclude protocol headers, optimizer state, secure aggregation, retransmissions, and compression.'}
(res/'proposed_overhead_summary.json').write_text(json.dumps(overhead,indent=2))
env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'device':'CPU'}
(res/'environment.json').write_text(json.dumps(env,indent=2))
# Main numerical summary json
summary={}
for _,r in agg.iterrows():
 summary[r.method]={'label':r.label,'mae_cm_mean':100*r.mae_m_mean,'mae_cm_std':100*r.mae_m_std,'rmse_cm_mean':100*r.rmse_m_mean,'rmse_cm_std':100*r.rmse_m_std,'p95_cm_mean':100*r.p95_m_mean,'p95_cm_std':100*r.p95_m_std,'parameters':int(r.parameters),'model_size_kib_fp32':r.model_size_kib_fp32,'latency_batch1_ms_cpu_mean':r.latency_batch1_ms_cpu_mean}
(res/'phase3_results_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(overhead,indent=2))
