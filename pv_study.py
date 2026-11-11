"""Solar-specific task transfer; separate from the mixed-renewable pilot.

The gate parameters remain frozen. Forecasters use the same model family on
load and solar observations. Physical PV capacity and loading are fixed before
these exploratory task runs; all capacity cells are retained.
"""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import joblib
from data_pipeline import ROOT,CONFIG,prepare,observe,evidence
from physics import network,physics_tables
from run_study import model_for,configurations,evaluate

def solar_data():
    p=ROOT/'data/DE_LU_solar_causal.csv'
    if p.exists():return pd.read_csv(p,parse_dates=['timestamp']).set_index('timestamp')
    d=prepare('DE_LU');raw=pd.read_csv(ROOT/'data/opsd_2020-10-06.csv',usecols=['utc_timestamp','DE_LU_solar_generation_actual'],parse_dates=['utc_timestamp']).set_index('utc_timestamp')
    solar=raw.iloc[:,0].ffill().reindex(d.index)
    scale=float(solar.loc[:'2019-09-30'].quantile(.95));d=d.copy();d['ren']=400*solar/scale
    d.to_csv(p,index_label='timestamp');(ROOT/'data/solar_scaling.json').write_text(json.dumps({'training_q95_mw':scale,'fit_end':'2019-09-30','representation':'400 kW scaled actual OPSD solar; independent of wind generation'},indent=2))
    return d

def pv_job(origin):
    d=solar_data();model=model_for(d,'DE_LU_solar',origin);params=json.loads((ROOT/'results/frozen_parameters.json').read_text());learned=joblib.load(ROOT/'data/learned_gate.joblib')
    cfgs=[x for x in configurations(params,False) if x[0] in ['ungated','TUDT_F','always_half','interval_gate']]
    rows=[];traces=[];metadata=[]
    nets={(cap,rating):network('ieee33',rating,.6,cap) for cap in [3.6,7.2] for rating in CONFIG['line_rating_factors']}
    for vol in CONFIG['renewable_volatility_factors']:
        for scenario in CONFIG['stress_scenarios']:
            # These four conditions are deterministic. Five paired seed labels
            # are retained without claiming five distinct physical trajectories.
            e=evidence(observe(d,origin,672,scenario,11,volatility=vol),model)
            for cap in [3.6,7.2]:
                for rating in CONFIG['line_rating_factors']:
                    net=nets[cap,rating];pf=(physics_tables(e['pred'],net,'pv'),physics_tables(e['truth_next'],net,'pv'))
                    for cfg in cfgs:
                        out,s,score=evaluate(e,net,cfg,learned,params['error_cutoff_kw'],params['price_ref'],'pv',pf)
                        for seed in CONFIG['seeds']:
                            rows.append({'region':'DE_LU','window':origin,'scenario':scenario,'seed':seed,'volatility':vol,'rating':rating,'pv_capacity_mw':cap,'load_multiplier':.6,'task':'solar_pv','method':cfg[0],'interval':'split',**s})
                        if rating==1 and vol==1:traces.append(out);metadata.append({'window':origin,'scenario':scenario,'pv_capacity_mw':cap,'method':cfg[0]})
    p=ROOT/'results'/f'pv_{origin}';pd.DataFrame(rows).to_csv(str(p)+'_summary.csv',index=False)
    np.savez_compressed(str(p)+'_traces.npz',outputs=np.array(traces),metadata=json.dumps(metadata))
    (ROOT/'results'/f'pv_{origin}_manifest.json').write_text(json.dumps({'origin':origin,'task':'solar-only PV curtailment','runs':len(rows),'hours':672,'gate_parameters':'unchanged frozen DE_LU development gate','analysis_status':'exploratory task extension after mixed-profile pilot; all predefined solar-specific cells retained'},indent=2))
    return {'origin':origin,'runs':len(rows)}

if __name__=='__main__':
    amendment={'date':'2026-11-05','reason':'mixed solar/wind profiles are insufficient to label the second task specifically as PV curtailment','design':'actual OPSD solar only; fixed 0.6 load multiplier; all 3.6/7.2-MW installed-PV capacity cells; 3x3 rating/volatility grid; nine 28-day windows; original four stress conditions; five paired seed labels','gate':'unchanged; no outcome-based tuning','status':'exploratory; earlier mixed-renewable curtailment results retained as pilot diagnostics'}
    (ROOT/'pv_amendment.json').write_text(json.dumps(amendment,indent=2));solar_data()
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(pv_job,x) for x in CONFIG['windows'] if not (ROOT/'results'/f'pv_{x}_manifest.json').exists()]
        for f in as_completed(jobs):print(f.result(),flush=True)
