"""Diagnostics that do not change the frozen gate or primary hypotheses."""
import json,time,platform,sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from data_pipeline import ROOT,CONFIG,prepare,observe,evidence
from run_study import model_for

def block_job(origin):
    d=prepare('DE_LU');m=model_for(d,'DE_LU',origin);rows=[]
    for scenario in CONFIG['scenarios']:
        for seed in CONFIG['seeds']:
            obs=observe(d,origin,168,scenario,seed)
            for method in CONFIG['interval_methods']:
                e=evidence(obs,m,method=method);hit=(e['truth_next']>=e['lo'])&(e['truth_next']<=e['hi'])
                day=hit.reshape(7,24,2).all(axis=1).mean(axis=0)
                rows.append({'window':origin,'scenario':scenario,'seed':seed,'method':method,'load_daily_simultaneous_coverage':day[0],'renewable_daily_simultaneous_coverage':day[1]})
    return rows

def main():
    with ProcessPoolExecutor(max_workers=6) as p:rows=[r for x in p.map(block_job,CONFIG['windows']) for r in x]
    pd.DataFrame(rows).to_csv(ROOT/'results/block_coverage.csv',index=False)
    d=prepare('DE_LU');m=model_for(d,'DE_LU','2020-01-01');o=observe(d,'2020-01-01',168,'delayed',11)
    runs=[]
    for _ in range(10):
        st=time.perf_counter();e=evidence(o,m);runs.append((time.perf_counter()-st)*1000/168)
    x=e['scores'];w=np.array(CONFIG['hand_weights']);scoretimes=[]
    for _ in range(1000):
        st=time.perf_counter_ns();_s=x@w;scoretimes.append((time.perf_counter_ns()-st)/168/1000)
    result={'hardware':platform.processor(),'cpu_count':__import__('os').cpu_count(),'python':sys.version,'batch_hours':168,'forecast_evidence_batch_mean_ms_per_hour':float(np.mean(runs)),'forecast_evidence_batch_p95_ms_per_hour':float(np.quantile(runs,.95)),'fusion_batch_mean_microseconds_per_hour':float(np.mean(scoretimes)),'interpretation':'Batch amortised measurements, not online worst-case latency; AC tables and training excluded.'}
    (ROOT/'results/runtime.json').write_text(json.dumps(result,indent=2))
    versions={}
    for module in ['numpy','pandas','scipy','sklearn','numba','pandapower','matplotlib','joblib']:
        mod=__import__(module);versions[module]=mod.__version__
    (ROOT/'results/environment.json').write_text(json.dumps(versions,indent=2));print(result)

if __name__=='__main__':main()
