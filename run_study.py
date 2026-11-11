"""Reconstructed revision study. Development is run and frozen before test jobs.

python isj_revision/run_study.py develop
python isj_revision/run_study.py primary --workers 6
python isj_revision/run_study.py stress --workers 6
"""
from pathlib import Path
import argparse,itertools,json,time,os,platform,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from data_pipeline import ROOT,CONFIG,prepare,train,observe,evidence,digest
from physics import network,physics_tables
from control import simulate,summarize,COLUMNS

RESULT=ROOT/'results';RESULT.mkdir(exist_ok=True)
WEIGHTS=np.array(CONFIG['hand_weights'])
SUBSETS=[tuple(i for i in range(4) if mask&(1<<i)) for mask in range(1,16)]

def model_for(df,region,origin):
    p=ROOT/'data'/f'model_{region}_{origin}.joblib'
    if not p.exists():
        temporary=p.with_name(p.name+f'.{os.getpid()}.tmp')
        joblib.dump(train(df,origin),temporary)
        os.replace(temporary,p)
    return joblib.load(p)

def score_for(e,configuration,learned):
    name,weights,low,high,mode=configuration
    if name=='learned_gate':s=1-learned.predict_proba(e['scores'])[:,1]
    elif name=='interval_gate':s=-e['width']
    else:s=e['scores']@np.array(weights)
    return s,float(low),float(high),int(mode)

def evaluate(e,net,configuration,learned,cutoff,price_ref,task='battery',pf=None):
    p,a=pf if pf is not None else (physics_tables(e['pred'],net,task),physics_tables(e['truth_next'],net,task))
    s,low,high,mode=score_for(e,configuration,learned)
    output=simulate(e['pred'],e['truth_next'],e['price'],s,low,high,mode,p,a,net['pmax'],net['capacity'],price_ref,cutoff,int(task=='pv'),float(net['gen'].sum()))
    summary=summarize(output,net['capacity'],price_ref)
    summary['full_below_threshold']=float(((output[:,2]==2)&(s<high)).mean())
    summary['mean_score']=float(s.mean())
    summary['candidate_correctable_hours']=int(((a[:,:,0].max(axis=1)>.5)&(output[:,8]>.5)).sum())
    summary['no_tested_physical_alternative_hours']=int((a[:,:,0].max(axis=1)<.5).sum())
    summary['low_threshold']=low;summary['high_threshold']=high
    return output,summary,s

def pareto_flags(values):
    good=np.ones(len(values),bool)
    for i in range(len(values)):
        good[i]=not np.any(np.all(values<=values[i],axis=1)&np.any(values<values[i],axis=1))
    return good

def develop():
    d=prepare('DE_LU');packs=[];clean_errors=[];price_refs=[];net=network()
    for origin in CONFIG['development_windows']:
        model=model_for(d,'DE_LU',origin);price_refs.append(model.price_ref)
        for scenario in CONFIG['scenarios']:
            for seed in CONFIG['seeds']:
                e=evidence(observe(d,origin,168,scenario,seed),model)
                e['pf']=(physics_tables(e['pred'],net),physics_tables(e['truth_next'],net))
                e['origin']=origin;e['scenario']=scenario;e['seed']=seed
                packs.append(e)
                if scenario=='clean' and seed==CONFIG['seeds'][0]:
                    clean_errors.extend(abs((e['truth_next'][:,0]-e['truth_next'][:,1])-(e['pred'][:,0]-e['pred'][:,1])))
    cutoff=float(np.quantile(clean_errors,.9));price_ref=float(np.mean(price_refs))
    X=np.concatenate([e['scores'] for e in packs]);err=np.concatenate([abs((e['truth_next'][:,0]-e['truth_next'][:,1])-(e['pred'][:,0]-e['pred'][:,1])) for e in packs])
    learned=LogisticRegression(C=1.,max_iter=2000,random_state=22355).fit(X,err>cutoff)
    joblib.dump(learned,ROOT/'data/learned_gate.joblib')
    thresholds={}
    for subset in SUBSETS:
        w=np.zeros(4);w[list(subset)]=WEIGHTS[list(subset)];w/=w.sum()
        s=X@w;thresholds['subset_'+''.join(str(i+1) for i in subset)]=[float(np.quantile(s,.25)),float(np.quantile(s,.5))]
    thresholds['interval_gate']=[float(np.quantile(np.concatenate([-e['width'] for e in packs]),.25))]*2
    ls=1-learned.predict_proba(X)[:,1];thresholds['learned_gate']=[float(np.quantile(ls,.25)),float(np.quantile(ls,.5))]
    rng=np.random.default_rng(CONFIG['search_seed']);candidates=[(WEIGHTS,.65,.85)]
    for _ in range(CONFIG['pareto_candidates']-1):
        w=rng.dirichlet(np.ones(4)*3);lo=rng.uniform(.45,.9);hi=rng.uniform(max(.7,lo+.025),.99);candidates.append((w,lo,hi))
    rows=[]
    for i,(w,lo,hi) in enumerate(candidates):
        results=[]
        for e in packs:
            _,r,_=evaluate(e,net,('search',w,lo,hi,1),learned,cutoff,price_ref,pf=e['pf']);results.append(r)
        a=pd.DataFrame(results)
        row={'candidate':i,'low':lo,'high':hi,**{f'w{j+1}':w[j] for j in range(4)}}
        row.update(a[['adjusted_cost_eur','high_error_automation','false_abstention','abstention']].mean().to_dict());rows.append(row)
        if i%25==0:print('development search',i,flush=True)
    table=pd.DataFrame(rows);objectives=table[['adjusted_cost_eur','high_error_automation','false_abstention']].to_numpy()
    good=pareto_flags(objectives);table['pareto']=good
    norm=(objectives-objectives.min(axis=0))/np.maximum(np.ptp(objectives,axis=0),1e-12)
    distances=np.sqrt((norm**2).sum(axis=1));distances[~good]=np.inf
    selected=int(np.argmin(distances));table['selected']=table.candidate==selected;table.to_csv(RESULT/'development_pareto.csv',index=False)
    w,lo,hi=candidates[selected]
    params={'protocol_sha256':digest(ROOT/'protocol.json'),'selection':'minimum equal-weight normalized Euclidean distance to development ideal among nondominated points',
            'candidate':selected,'weights':w.tolist(),'low':lo,'high':hi,'error_cutoff_kw':cutoff,'price_ref':price_ref,'thresholds':thresholds,
            'learned_coefficients':learned.coef_.tolist(),'learned_intercept':learned.intercept_.tolist(),
            'calibration_regions':['DE_LU'],'development_steps':len(X),'development_windows':CONFIG['development_windows']}
    (RESULT/'frozen_parameters.json').write_text(json.dumps(params,indent=2))
    # Sensitivity uses development data only and is never selected on test results.
    sensitivity=[]
    for decay in CONFIG['freshness_grid_hours']:
        for history in CONFIG['history_grid_hours']:
            for origin in CONFIG['development_windows']:
                model=model_for(d,'DE_LU',origin)
                for scenario in ['clean','delayed','slow_ramp']:
                    e=evidence(observe(d,origin,168,scenario,11),model,decay=decay,lookback=history)
                    _,r,_=evaluate(e,net,('TUDT_F',w,lo,hi,1),learned,cutoff,price_ref)
                    sensitivity.append({'decay':decay,'history':history,'window':origin,'scenario':scenario,**r})
    pd.DataFrame(sensitivity).to_csv(RESULT/'development_freshness.csv',index=False)
    print('FROZEN',json.dumps({k:params[k] for k in ['candidate','weights','low','high','error_cutoff_kw']}),flush=True)

def configurations(params,subsets=True):
    w=params['weights'];out=[('ungated',w,0,0,0),('always_half',w,0,0,2),('always_defer',w,0,0,5),('constrained',w,0,0,3),
                         ('TUDT_fixed',WEIGHTS.tolist(),.65,.85,1),('TUDT_F',w,params['low'],params['high'],1)]
    for n in ['interval_gate','learned_gate']:
        lo,hi=params['thresholds'][n];out.append((n,w,lo,hi,4 if n=='interval_gate' else 1))
    if subsets:
        for subset in SUBSETS:
            name='subset_'+''.join(str(i+1) for i in subset);sw=np.zeros(4);sw[list(subset)]=WEIGHTS[list(subset)];sw/=sw.sum()
            lo,hi=params['thresholds'][name];out.append((name,sw.tolist(),lo,hi,1))
    return out

def detector_rows(e,meta):
    rows=[]
    for j,name in enumerate(['timestamp_lag','roughness','jump','completed_mismatch']):
        pred=e['detectors'][:,j];true=e['fault'];tp=int((pred&true).sum());fp=int((pred&~true).sum());fn=int((~pred&true).sum());tn=int((~pred&~true).sum())
        delays=[];starts=np.where(true&~np.r_[False,true[:-1]])[0]
        for st in starts:
            end=st
            while end<len(true) and true[end]:end+=1
            hits=np.where(pred[st:end])[0]
            if len(hits):delays.append(int(hits[0]))
        rows.append({**meta,'detector':name,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'detected_episodes':len(delays),'episodes':len(starts),'delay_sum':sum(delays)})
    return rows

def job(region,origin,kind,regime='original'):
    started=time.time();params=json.loads((RESULT/'frozen_parameters.json').read_text())
    assert params['protocol_sha256']==digest(ROOT/'protocol.json'),'protocol changed after calibration'
    learned=joblib.load(ROOT/'data/learned_gate.joblib');d=prepare(region);model=model_for(d,region,origin)
    hours=CONFIG['primary_hours'] if kind=='primary' else CONFIG['stress_hours']
    cfgs=configurations(params,subsets=(kind=='primary' and region=='DE_LU'))
    records=[];detectors=[];coverage=[];ranking=[];correlations=[];traces=[];trace_meta=[]
    vols=[1.] if kind=='primary' else CONFIG['renewable_volatility_factors']
    ratings=[1.] if kind=='primary' else CONFIG['line_rating_factors']
    load_multiplier,pv_capacity={'original':(1.,1.8),'lower_load':(.6,1.8),'higher_pv':(1.,3.6),'lower_load_higher_pv':(.6,3.6)}[regime]
    nets={rating:network('two_bus' if kind=='primary' else 'ieee33',rating,load_multiplier,pv_capacity) for rating in ratings}
    scenarios=CONFIG['scenarios'] if kind=='primary' else CONFIG['stress_scenarios']
    for vol in vols:
        for scenario in scenarios:
            deterministic=scenario not in ['missing','noisy','fdi_like']
            cache=None
            for seed in CONFIG['seeds']:
                if cache is not None and deterministic:e=cache
                else:
                    e=evidence(observe(d,origin,hours,scenario,seed,volatility=vol),model)
                    cache=e
                basic={'region':region,'window':origin,'scenario':scenario,'seed':seed,'volatility':vol,'kind':kind,'operating_regime':regime}
                detectors.extend(detector_rows(e,basic))
                for a in range(4):
                    for b in range(a+1,4):
                        x=e['scores'][:,a];y=e['scores'][:,b]
                        correlations.append({**basic,'a':a+1,'b':b+1,'correlation':float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else np.nan})
                for rating in ratings:
                    net=nets[rating]
                    pf=(physics_tables(e['pred'],net),physics_tables(e['truth_next'],net))
                    for cfg in cfgs:
                        out,s,sc=evaluate(e,net,cfg,learned,params['error_cutoff_kw'],params['price_ref'],pf=pf)
                        meta={**basic,'rating':rating,'task':'battery','method':cfg[0],'interval':'split'}
                        records.append({**meta,**s})
                        if kind=='primary' or (seed==11 and rating==1 and vol==1):traces.append(out);trace_meta.append(meta)
                        if cfg[0]=='ungated' and kind=='primary':
                            for rankname in ['TUDT_F','interval_gate','learned_gate']:
                                rcfg=next(x for x in cfgs if x[0]==rankname);rs,_,_,_=score_for(e,rcfg,learned)
                                for cov in CONFIG['matched_coverage_targets']:
                                    k=int(round(len(rs)*cov));ix=np.argsort(-rs,kind='stable')[:k]
                                    ranking.append({**basic,'score':rankname,'target_coverage':cov,'selected':k,'selected_high_error':int(out[ix,5].sum()),'selected_candidate_violation':int((1-out[ix,17]).sum()),'mean_abs_error':float(out[ix,19].mean())})
                    if kind=='stress':
                        pvpf=(physics_tables(e['pred'],net,'pv'),physics_tables(e['truth_next'],net,'pv'))
                        for name in ['ungated','TUDT_F','always_half','interval_gate']:
                            cfg=next(x for x in cfgs if x[0]==name)
                            out,s,sc=evaluate(e,net,cfg,learned,params['error_cutoff_kw'],params['price_ref'],'pv',pvpf)
                            records.append({**basic,'rating':rating,'task':'pv','method':name,'interval':'split',**s})
                            if seed==11 and rating==1 and vol==1:traces.append(out);trace_meta.append({**basic,'rating':rating,'task':'pv','method':name,'interval':'split'})
                if kind=='primary':
                    for interval in CONFIG['interval_methods']:
                        z=e if interval=='split' else evidence(observe(d,origin,hours,scenario,seed),model,method=interval)
                        hit=(z['truth_next']>=z['lo'])&(z['truth_next']<=z['hi'])
                        cov={'region':region,'window':origin,'scenario':scenario,'seed':seed,'interval':interval,'load_coverage':float(hit[:,0].mean()),'renewable_coverage':float(hit[:,1].mean()),'width_kw':float(z['width'].mean())}
                        # First six and last six hours of a fault block describe adaptation, not a formal recovery guarantee.
                        active=np.where(z['fault'])[0]
                        cov['early_coverage']=float(hit[active[:6]].mean()) if len(active) else np.nan
                        cov['late_coverage']=float(hit[active[-6:]].mean()) if len(active) else np.nan
                        coverage.append(cov)
                        if interval!='split':
                            cfg=next(x for x in cfgs if x[0]=='TUDT_F');out,s,_=evaluate(z,network(),cfg,learned,params['error_cutoff_kw'],params['price_ref'])
                            meta={**basic,'rating':1.,'task':'battery','method':'TUDT_F','interval':interval};records.append({**meta,**s});traces.append(out);trace_meta.append(meta)
    stem=f'{kind}_{region}_{origin}'+('' if regime=='original' else '_'+regime)
    pd.DataFrame(records).to_csv(RESULT/f'{stem}_summary.csv',index=False)
    pd.DataFrame(detectors).to_csv(RESULT/f'{stem}_detectors.csv',index=False)
    pd.DataFrame(coverage).to_csv(RESULT/f'{stem}_coverage.csv',index=False)
    pd.DataFrame(ranking).to_csv(RESULT/f'{stem}_ranking.csv',index=False)
    pd.DataFrame(correlations).to_csv(RESULT/f'{stem}_correlations.csv',index=False)
    np.savez_compressed(RESULT/f'{stem}_traces.npz',outputs=np.array(traces),metadata=json.dumps(trace_meta),columns=np.array(COLUMNS))
    manifest={'region':region,'origin':origin,'kind':kind,'operating_regime':regime,'records':len(records),'hours':hours,'elapsed_seconds':time.time()-started,'model_fit_end':model.fit_end,'last_calibration_target':model.calibration_end,'protocol_sha256':digest(ROOT/'protocol.json'),'parameters_sha256':digest(RESULT/'frozen_parameters.json'),'trace_policy':'all primary runs; stress seed 11, nominal rating and volatility; all stress run summaries retained and reproducible'}
    (RESULT/f'{stem}_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','primary','stress']);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--quick',action='store_true');ap.add_argument('--extend-stress',action='store_true');args=ap.parse_args()
    if args.stage=='develop':develop();return
    assert (RESULT/'frozen_parameters.json').exists(),'run development first'
    windows=CONFIG['windows'][:1] if args.quick else CONFIG['windows']
    regimes=['lower_load','higher_pv','lower_load_higher_pv'] if args.extend_stress else ['original']
    jobs=[(region,origin,args.stage,regime) for region in (CONFIG['regions'] if args.stage=='primary' else ['DE_LU']) for origin in windows for regime in regimes]
    jobs=[x for x in jobs if not (RESULT/(f'{x[2]}_{x[0]}_{x[1]}'+('' if x[3]=='original' else '_'+x[3])+'_manifest.json')).exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs=[pool.submit(job,*x) for x in jobs]
        for fut in as_completed(futs):print(json.dumps(fut.result()),flush=True)

if __name__=='__main__':main()
