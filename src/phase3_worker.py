#!/usr/bin/env python3
"""Final controlled R2 ablation worker.

The submitted R1 archive contained no original code/data/weights/seeds. This worker
implements a frozen, auditable simulation constrained by the manuscript geometry and
published 20-dB performance. Every result is labelled as a new R2 controlled ablation,
not as recovered original data.
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
AUDITED=ROOT.parent/'05_REPRODUCIBILITY_RECONSTRUCTION'/'src'
import sys
sys.path.insert(0,str(AUDITED))
from optical_simulator import OpticalConfig, los_gains, apply_gain_mismatch, calibrate_relative_gains, add_awgn_for_snr, generate_smooth_trajectory

LEDS=np.array([[1,1,3],[4,1,3],[1,3,3],[4,3,3],[1,5,3],[4,5,3]],dtype=np.float64)
ROOM=np.array([5.,6.],dtype=np.float32)

@dataclass(frozen=True)
class Protocol:
    seeds: Tuple[int,...]=(11,23,37,53,71)
    train_trajectories:int=160
    validation_trajectories:int=40
    test_trajectories:int=40
    steps:int=80
    sampling_hz:int=10
    window:int=12
    snr_db:float=20.0
    aoa_std_deg_at_20db:float=3.0
    independent_anchor_loss_probability:float=0.07
    global_gap_frame_fraction:float=0.03
    global_gap_burst_min:int=2
    global_gap_burst_max:int=4
    gain_mismatch_log_std:float=0.08
    clients:int=8
    participation:float=0.75
    central_epochs:int=30
    federated_rounds:int=40
    local_epochs:int=1
    batch_size:int=256
    lr:float=1e-3
    weight_decay:float=1e-5
    reconstruction_weight:float=0.2
    tcn_channels:Tuple[int,int]=(32,32)
    dilations:Tuple[int,int]=(1,2)
    kernel:int=3
    hidden:int=48
    dropout:float=0.1

P=Protocol()

class Chomp1d(nn.Module):
    def __init__(self,n:int): super().__init__(); self.n=n
    def forward(self,x): return x[:,:,:-self.n] if self.n else x
class TCNBlock(nn.Module):
    def __init__(self,cin,cout,kernel,dilation,dropout):
        super().__init__(); pad=(kernel-1)*dilation
        self.net=nn.Sequential(nn.Conv1d(cin,cout,kernel,padding=pad,dilation=dilation),Chomp1d(pad),nn.ReLU(),nn.Dropout(dropout),nn.Conv1d(cout,cout,kernel,padding=pad,dilation=dilation),Chomp1d(pad),nn.ReLU(),nn.Dropout(dropout))
        self.res=nn.Identity() if cin==cout else nn.Conv1d(cin,cout,1)
    def forward(self,x): return torch.relu(self.net(x)+self.res(x))
class ResidualTCNLSTM(nn.Module):
    def __init__(self,input_dim=32,signal_dim=24):
        super().__init__(); blocks=[]; cin=input_dim
        for cout,d in zip(P.tcn_channels,P.dilations): blocks.append(TCNBlock(cin,cout,P.kernel,d,P.dropout)); cin=cout
        self.tcn=nn.Sequential(*blocks); self.lstm=nn.LSTM(cin,P.hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(P.hidden,32),nn.ReLU(),nn.Linear(32,2)); self.recon=nn.Linear(P.hidden,signal_dim)
    def forward(self,x):
        base=x[:,-1,-2:]; z=self.tcn(x.transpose(1,2)).transpose(1,2); z,_=self.lstm(z)
        candidate=torch.sigmoid(self.head(z[:,-1])); gate=(x[:,-1,-8:-2].sum(dim=1,keepdim=True)<0.5).to(x.dtype); return base*(1-gate)+candidate*gate,self.recon(z)
class ResidualTCN(nn.Module):
    def __init__(self,input_dim=32):
        super().__init__(); blocks=[]; cin=input_dim
        for cout,d in zip(P.tcn_channels,P.dilations): blocks.append(TCNBlock(cin,cout,P.kernel,d,P.dropout)); cin=cout
        self.tcn=nn.Sequential(*blocks); self.head=nn.Sequential(nn.Linear(cin,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x):
        base=x[:,-1,-2:]; candidate=torch.sigmoid(self.head(self.tcn(x.transpose(1,2))[:,:,-1])); gate=(x[:,-1,-8:-2].sum(dim=1,keepdim=True)<0.5).to(x.dtype); return base*(1-gate)+candidate*gate
class ResidualLSTM(nn.Module):
    def __init__(self,input_dim=32):
        super().__init__(); self.lstm=nn.LSTM(input_dim,P.hidden,batch_first=True); self.head=nn.Sequential(nn.Linear(P.hidden,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x):
        z,_=self.lstm(x); base=x[:,-1,-2:]; candidate=torch.sigmoid(self.head(z[:,-1])); gate=(x[:,-1,-8:-2].sum(dim=1,keepdim=True)<0.5).to(x.dtype); return base*(1-gate)+candidate*gate
class StaticDNN(nn.Module):
    def __init__(self,input_dim=32): super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x):
        current=x[:,-1]; base=current[:,-2:]; candidate=torch.sigmoid(self.net(current)); gate=(current[:,-8:-2].sum(dim=1,keepdim=True)<0.5).to(x.dtype); return base*(1-gate)+candidate*gate
class SinglePDLSTM(nn.Module):
    def __init__(self,input_dim=14):
        super().__init__(); self.lstm=nn.LSTM(input_dim,P.hidden,batch_first=True); self.head=nn.Sequential(nn.Linear(P.hidden,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x):
        z,_=self.lstm(x); base=x[:,-1,-2:]; candidate=torch.sigmoid(self.head(z[:,-1])); gate=(x[:,-1,-8:-2].sum(dim=1,keepdim=True)<0.5).to(x.dtype); return base*(1-gate)+candidate*gate

def count_params(m): return int(sum(p.numel() for p in m.parameters() if p.requires_grad))
def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def angular_noise(true_u,rng,std_deg):
    tangent=rng.normal(size=true_u.shape); tangent-=np.sum(tangent*true_u,axis=-1,keepdims=True)*true_u
    tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-12)
    angle=rng.normal(0,np.deg2rad(std_deg),size=true_u.shape[:-1])[...,None]
    noisy=true_u*np.cos(angle)+tangent*np.sin(angle)
    return noisy/np.maximum(np.linalg.norm(noisy,axis=-1,keepdims=True),1e-12)

def mask_for_mode(rng,mode):
    mask=np.ones((P.steps,6),dtype=np.float32)
    if mode in ('mixed','independent'):
        prob=P.independent_anchor_loss_probability if mode=='mixed' else P.independent_anchor_loss_probability+P.global_gap_frame_fraction
        mask[rng.random(mask.shape)<prob]=0
    if mode in ('mixed','burst'):
        fraction=P.global_gap_frame_fraction if mode=='mixed' else P.independent_anchor_loss_probability+P.global_gap_frame_fraction
        target=max(1,int(round(P.steps*fraction))); marked=0; guard=0
        while marked<target and guard<1000:
            guard+=1; start=int(rng.integers(P.steps)); length=int(rng.integers(P.global_gap_burst_min,P.global_gap_burst_max+1)); end=min(P.steps,start+length)
            newly=int(np.sum(~np.all(mask[start:end]==0,axis=1))); mask[start:end,:]=0; marked+=max(newly,end-start)
    return mask

def robust_seed_position(candidates,reliability,mask):
    score=reliability*mask; idx=np.argsort(score,axis=1)[:,-2:]
    cand=np.take_along_axis(candidates,idx[:,:,None],axis=1); w=np.take_along_axis(score,idx,axis=1)
    den=w.sum(axis=1,keepdims=True); valid=den[:,0]>1e-12; w=w/np.maximum(den,1e-12)
    base=(cand*w[:,:,None]).sum(axis=1); base[~valid]=0; base=np.clip(base,[0,0],ROOM)
    return base.astype(np.float32)

def build_bundle(seed,ntraj,split,mode='mixed',snr_db=None,aoa_std_deg=None):
    snr_value=P.snr_db if snr_db is None else float(snr_db); aoa_std=P.aoa_std_deg_at_20db if aoa_std_deg is None else float(aoa_std_deg)
    cfg=OpticalConfig(LEDS,receiver_height=.85,optical_power_w=5.,lambertian_order=1.,pd_area_m2=1e-4,responsivity=.53,fov_deg=85.,refractive_index=1.5,filter_gain=1.)
    xa=[]; xs=[]; y=[]; full=[]; missing=[]; clients=[]
    for tr in range(ntraj):
        base_seed=seed*100000+split*10000+tr
        rtraj=np.random.default_rng(base_seed); rnoise=np.random.default_rng(base_seed+1000); rmask=np.random.default_rng(base_seed+2000+{'mixed':1,'independent':2,'burst':3}[mode])
        xy=generate_smooth_trajectory(rtraj,P.steps,dt=1/P.sampling_hz,room_xy=(5.,6.),speed_range=(.2,1.2))
        rx=np.column_stack([xy,np.full(P.steps,.85)]); vec=LEDS[None,:,:]-rx[:,None,:]; true_u=vec/np.linalg.norm(vec,axis=-1,keepdims=True)
        noisy_u=angular_noise(true_u,rnoise,aoa_std)
        # PD front-end is still simulated to define relative reliability and calibration effects.
        gains=los_gains(xy,cfg); mismatched,factors=apply_gain_mismatch(gains,rnoise,std=P.gain_mismatch_log_std); calibrated=calibrate_relative_gains(mismatched,factors); noisy_gains=add_awgn_for_snr(calibrated,snr_value,rnoise)
        power=np.linalg.norm(noisy_gains,axis=-1); power/=np.maximum(power.sum(axis=1,keepdims=True),1e-12)
        reliability=power.astype(np.float32)
        dz=3.-.85; cand=LEDS[None,:,:2]-dz*noisy_u[:,:,:2]/np.clip(noisy_u[:,:,2:3],.2,None); cand[:,:,0]=np.clip(cand[:,:,0],0,5); cand[:,:,1]=np.clip(cand[:,:,1],0,6)
        mask=mask_for_mode(rmask,mode); base=robust_seed_position(cand,reliability,mask)
        # ADR signal: unit AOA vector plus relative reliability for each of six anchors.
        sig=np.concatenate([noisy_u,reliability[:,:,None]],axis=-1).astype(np.float32) # T,6,4
        sparse=(sig*mask[:,:,None]).reshape(P.steps,24); adr_in=np.concatenate([sparse,mask,base/ROOM],axis=1).astype(np.float32)
        miss=np.repeat(1-mask,4,axis=1).astype(np.float32); sigflat=sig.reshape(P.steps,24)
        # Single-PD/RSS comparator: six relative powers + masks + weighted LED centroid seed.
        rss=(power*mask).astype(np.float32); den=rss.sum(axis=1,keepdims=True); centroid=(rss@LEDS[:,:2])/np.maximum(den,1e-12); centroid[den[:,0]<=1e-12]=0; centroid=np.clip(centroid,[0,0],ROOM)
        single_in=np.concatenate([rss,mask,centroid/ROOM],axis=1).astype(np.float32)
        client=tr%P.clients
        for i in range(P.window-1,P.steps):
            sl=slice(i-P.window+1,i+1); xa.append(adr_in[sl]); xs.append(single_in[sl]); y.append((xy[i]/ROOM).astype(np.float32)); full.append(sigflat[sl]); missing.append(miss[sl]); clients.append(client)
    return tuple(torch.tensor(v) for v in (np.stack(xa),np.stack(xs),np.stack(y),np.stack(full),np.stack(missing),np.asarray(clients,dtype=np.int64)))

def forward(model,x):
    out=model(x); return out if isinstance(out,tuple) else (out,None)
def loss_fn(model,x,y,full,missing,reconstruction):
    pred,recon=forward(model,x); per=nn.functional.smooth_l1_loss(pred,y,reduction='none').mean(dim=1); gap=(x[:,-1,-8:-2].sum(dim=1)<0.5).to(x.dtype); weights=1.0+9.0*gap; loss=(per*weights).sum()/weights.sum()
    if reconstruction:
        denom=missing.sum().clamp_min(1); loss=loss+P.reconstruction_weight*(((recon-full)**2*missing).sum()/denom)
    return loss

def evaluate(model,x,y):
    model.eval(); errors=[]; scale=torch.tensor(ROOM)
    with torch.no_grad():
        for i in range(0,len(x),1024):
            pred,_=forward(model,x[i:i+1024]); errors.append(torch.linalg.vector_norm((pred-y[i:i+1024])*scale,dim=1).numpy())
    e=np.concatenate(errors); return {'mae_m':float(e.mean()),'rmse_m':float(np.sqrt(np.mean(e**2))),'p90_m':float(np.quantile(e,.9)),'p95_m':float(np.quantile(e,.95)),'max_m':float(e.max()),'n_test_windows':int(len(e))}

def central_train(model,tr,va,use_single,reconstruction,seed):
    seed_all(seed); x=tr[1] if use_single else tr[0]; vx=va[1] if use_single else va[0]; y,full,missing=tr[2],tr[3],tr[4]; vy=va[2]
    gen=torch.Generator().manual_seed(seed); dl=DataLoader(TensorDataset(x,y,full,missing),batch_size=P.batch_size,shuffle=True,generator=gen); opt=torch.optim.Adam(model.parameters(),lr=P.lr,weight_decay=P.weight_decay); best=1e9; state=copy.deepcopy(model.state_dict()); hist=[]
    for ep in range(1,P.central_epochs+1):
        model.train(); total=0; n=0
        for xb,yb,fb,mb in dl:
            opt.zero_grad(set_to_none=True); l=loss_fn(model,xb,yb,fb,mb,reconstruction); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=float(l.detach().item())*len(xb); n+=len(xb)
        vm=evaluate(model,vx,vy)['mae_m']; hist.append({'step':ep,'train_loss':total/n,'val_mae_m':vm})
        if vm<best: best=vm; state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state); return model,hist

def fed_train(model,tr,va,use_single,reconstruction,seed):
    seed_all(seed); x=tr[1] if use_single else tr[0]; vx=va[1] if use_single else va[0]; y,full,missing,client=tr[2],tr[3],tr[4],tr[5].numpy(); vy=va[2]
    rng=np.random.default_rng(seed); global_state=copy.deepcopy(model.state_dict()); best=1e9; best_state=copy.deepcopy(global_state); hist=[]; selected_n=int(round(P.clients*P.participation))
    for rr in range(1,P.federated_rounds+1):
        selected=np.sort(rng.choice(np.arange(P.clients),selected_n,replace=False)); states=[]; weights=[]
        for c in selected:
            local=copy.deepcopy(model); local.load_state_dict(global_state); idx=np.flatnonzero(client==c); gen=torch.Generator().manual_seed(seed*100000+rr*100+int(c)); dl=DataLoader(TensorDataset(x[idx],y[idx],full[idx],missing[idx]),batch_size=P.batch_size,shuffle=True,generator=gen); opt=torch.optim.Adam(local.parameters(),lr=P.lr,weight_decay=P.weight_decay); local.train()
            for _ in range(P.local_epochs):
                for xb,yb,fb,mb in dl:
                    opt.zero_grad(set_to_none=True); l=loss_fn(local,xb,yb,fb,mb,reconstruction); l.backward(); torch.nn.utils.clip_grad_norm_(local.parameters(),5); opt.step()
            states.append(local.state_dict()); weights.append(len(idx))
        total=float(sum(weights)); global_state={k:sum(s[k]*(w/total) for s,w in zip(states,weights)) for k in global_state}; model.load_state_dict(global_state); vm=evaluate(model,vx,vy)['mae_m']; hist.append({'step':rr,'val_mae_m':vm,'selected_clients':' '.join(map(str,selected))})
        if vm<best: best=vm; best_state=copy.deepcopy(global_state)
    model.load_state_dict(best_state); return model,hist

def latency(model,input_dim):
    model.eval(); x=torch.zeros(1,P.window,input_dim); n=100
    with torch.no_grad():
        for _ in range(10): forward(model,x)
        t=time.perf_counter()
        for _ in range(n): forward(model,x)
    return (time.perf_counter()-t)*1000/n

METHODS={
 'fl_static_no_densification':dict(builder=lambda:StaticDNN(32),single=False,fed=True,recon=False,label='Federated static DNN without temporal densification'),
 'single_pd_lstm_central':dict(builder=lambda:SinglePDLSTM(14),single=True,fed=False,recon=False,label='Single-PD/RSS LSTM (centralized)'),
 'adr_tcn_central':dict(builder=lambda:ResidualTCN(32),single=False,fed=False,recon=False,label='ADR TCN-only (centralized)'),
 'adr_lstm_central':dict(builder=lambda:ResidualLSTM(32),single=False,fed=False,recon=False,label='ADR LSTM-only (centralized)'),
 'adr_tcn_lstm_central':dict(builder=lambda:ResidualTCNLSTM(32,24),single=False,fed=False,recon=True,label='ADR TCN-LSTM (centralized)'),
 'adr_tcn_lstm_federated':dict(builder=lambda:ResidualTCNLSTM(32,24),single=False,fed=True,recon=True,label='Proposed ADR TCN-LSTM (federated)'),
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--method',choices=METHODS,required=True); args=ap.parse_args(); torch.set_num_threads(2)
    spec=METHODS[args.method]; method_seed=args.seed*100+list(METHODS).index(args.method)+1; seed_all(method_seed)
    print(f'Generating common datasets: seed={args.seed}',flush=True); tr=build_bundle(args.seed,P.train_trajectories,1,'mixed'); va=build_bundle(args.seed,P.validation_trajectories,2,'mixed'); tests={m:build_bundle(args.seed,P.test_trajectories,3,m) for m in ('mixed','independent','burst')}
    model=spec['builder'](); start=time.perf_counter(); print(f'Training {args.method}',flush=True)
    if spec['fed']: model,hist=fed_train(model,tr,va,spec['single'],spec['recon'],method_seed)
    else: model,hist=central_train(model,tr,va,spec['single'],spec['recon'],method_seed)
    seconds=time.perf_counter()-start; params=count_params(model); rows=[]
    for condition,b in tests.items():
        x=b[1] if spec['single'] else b[0]; metrics=evaluate(model,x,b[2]); rows.append({'seed':args.seed,'method':args.method,'label':spec['label'],'test_condition':condition,**metrics,'parameters':params,'model_size_kib_fp32':params*4/1024,'update_size_kib_fp32':params*4/1024 if spec['fed'] else 0.,'latency_batch1_ms_cpu':latency(model,14 if spec['single'] else 32),'training_seconds_cpu':seconds,'effective_sample_passes':P.central_epochs if not spec['fed'] else P.federated_rounds*P.participation*P.local_epochs,'train_windows':len(tr[2]),'validation_windows':len(va[2]),'test_windows':len(b[2])})
    out=ROOT/'results'/'workers'; out.mkdir(parents=True,exist_ok=True); (out/f'{args.method}_seed_{args.seed}.json').write_text(json.dumps(rows,indent=2)); (out/f'{args.method}_seed_{args.seed}_history.json').write_text(json.dumps(hist,indent=2)); torch.save(model.state_dict(),ROOT/'checkpoints'/f'{args.method}_seed_{args.seed}.pt'); print(json.dumps(rows[0],indent=2),flush=True)
if __name__=='__main__': main()
