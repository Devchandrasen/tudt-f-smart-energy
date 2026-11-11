"""Secondary one-step hindsight diagnostic; never an input to control."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from numba import njit
from data_pipeline import ROOT,prepare,observe
from physics import network,physics_tables

@njit(cache=True)
def regret(output,pf,price,pmax,capacity,ref):
    total=0.;eligible=0;previous=.55
    for i in range(len(output)):
        current=output[i,11]
        if output[i,8]>.5:previous=current;continue
        best=1e30
        for k in range(17):
            u=(k-8)*pmax/8
            de=-u/.94 if u>=0 else -u*.94
            energy=previous*capacity+de
            if energy<.12*capacity-1e-10 or energy>.92*capacity+1e-10 or pf[i,k,0]<.5:continue
            grid=pf[i,k,6];cost=1000*price[i]*(max(grid,0)-.35*max(-grid,0))-1000*de*ref
            best=min(best,cost)
        if best<1e29:
            chosen=output[i,10]-(current-previous)*capacity*1000*ref
            if chosen-best < -1e-6:raise ValueError('negative regret under common feasible action set')
            total+=max(0.,chosen-best);eligible+=1
        previous=current
    return total,eligible

def job(path):
    a=np.load(path);meta=json.loads(str(a['metadata']));out=a['outputs'];region=meta[0]['region'];origin=meta[0]['window'];d=prepare(region);net=network();params=json.loads((ROOT/'results/frozen_parameters.json').read_text());cache={};rows=[]
    for m,o in zip(meta,out):
        key=(m['scenario'],m['seed'])
        if key not in cache:
            obs=observe(d,origin,168,*key);idx=obs['indices'];cache[key]=(physics_tables(obs['truth'][idx+1],net),obs['price'][idx])
        pf,price=cache[key];total,n=regret(o,pf,price,.26,.8,params['price_ref'])
        rows.append({'one_step_regret_eur':total/n if n else np.nan,'regret_eligible_hours':n,'one_step_regret_total_eur':total})
    p=path.with_name(path.name.replace('_traces.npz','_summary.csv'));df=pd.read_csv(p)
    assert len(df)==len(rows)
    for col in rows[0]:df[col]=[x[col] for x in rows]
    df.to_csv(p,index=False);return {'file':path.name,'n':len(rows)}

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=6) as p:
        for r in p.map(job,sorted((ROOT/'results').glob('primary_*_traces.npz'))):print(r,flush=True)
