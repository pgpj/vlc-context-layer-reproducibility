#!/usr/bin/env python3
from __future__ import annotations
import concurrent.futures, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable
WORKER=ROOT/'src'/'phase3_worker.py'
seeds=[11,23,37,53,71]
methods=['fl_static_no_densification','single_pd_lstm_central','adr_tcn_central','adr_lstm_central','adr_tcn_lstm_central','adr_tcn_lstm_federated']

def run(job):
 seed,method=job
 out=ROOT/'results'/'workers'/f'{method}_seed_{seed}.json'
 if out.exists(): return f'SKIP {seed} {method}'
 log=ROOT/'logs'/f'{method}_seed_{seed}.log'
 env=os.environ.copy();env['TERM']='xterm';env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
 t=time.time()
 try:
  p=subprocess.run([PYTHON,str(WORKER),'--seed',str(seed),'--method',method],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
  log.write_text(p.stdout,encoding='utf-8')
  if p.returncode!=0: return f'FAIL {seed} {method} rc={p.returncode}'
  return f'OK {seed} {method} {time.time()-t:.1f}s'
 except subprocess.TimeoutExpired as e:
  log.write_text((e.stdout or '') if isinstance(e.stdout,str) else str(e.stdout),encoding='utf-8')
  return f'TIMEOUT {seed} {method}'

jobs=[(s,m) for s in seeds for m in methods]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for result in ex.map(run,jobs):
  print(result,flush=True)
