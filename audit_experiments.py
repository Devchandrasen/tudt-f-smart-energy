"""Executed post-audit amendment. Original results are never overwritten.

Stages: develop, primary, physical, sensitivity. All amended output lives in
results/audit_v2. See audit_amendment.json for the design and inferential scope.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import argparse, json, time, hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from data_pipeline import ROOT,CONFIG,prepare,observe,evidence,features,rf_predict,digest
from run_study import model_for,pareto_flags
from physics import network,physics_tables
from control_v2 import simulate_v2,summarize,COLUMNS
from pv_study import solar_data

A=ROOT/'results/audit_v2'; A.mkdir(exist_ok=True)
P=json.loads((ROOT/'results/frozen_parameters.json').read_text())
CORE=['missing','delayed','noisy','fdi_like']
SCORES=['composite','interval','learned']
POLICIES=['binary','three_regime']

def get_scores(e):
    # The trained logistic coefficients are part of the frozen JSON record.
    # This is equivalent to sklearn predict_proba and avoids requiring an
    # omitted pickle cache in a standalone reproduction archive.
    linear=e['scores']@np.asarray(P['learned_coefficients'][0])+P['learned_intercept'][0]
    return {'composite':e['scores']@np.array(P['weights']),
            'interval':-e['width'],'learned':1-expit(linear)}

def run(e,pf,net,score=None,lo=0.,hi=0.,mode=0,**kwargs):
    if score is None:score=np.ones(len(e['pred']))
    out=simulate_v2(e['pred'],e['truth_next'],e['price'],score,float(lo),float(hi),mode,
        pf[0],pf[1],net['pmax'],net['capacity'],P['price_ref'],P['error_cutoff_kw'],
        pv_capacity=float(net['gen'].sum()),**kwargs)
    return out,summarize(out,net['capacity'],P['price_ref'])

def pack(region,origin,hours,scenario,seed=11,vol=1.,solar=False,**kw):
    d=solar_data() if solar else prepare(region)
    m=model_for(d,'DE_LU_solar' if solar else region,origin)
    return evidence(observe(d,origin,hours,scenario,seed,volatility=vol),m,**kw)

def dev_packs():
    cache=A/'development_packs.joblib'
    if cache.exists():return joblib.load(cache)
    packs=[];net=network()
    for origin in CONFIG['development_windows']:
        for scenario in CONFIG['scenarios']:
            previous=None
            for seed in CONFIG['seeds']:
                if previous is not None and scenario not in ['missing','noisy','fdi_like']:e=previous
                else:e=pack('DE_LU',origin,168,scenario,seed);previous=e
                pf=(physics_tables(e['pred'],net),physics_tables(e['truth_next'],net))
                out,_=run(e,pf,net)
                packs.append({'e':e,'pf':pf,'origin':origin,'scenario':scenario,'seed':seed,
                              'scores':get_scores(e),'common':out})
    joblib.dump(packs,cache);return packs

def target_thresholds(packs,score,target,policy):
    values=np.concatenate([x['scores'][score][x['common'][:,22]>.5] for x in packs])
    low=float(np.quantile(values,1-target)) if target<1 else float(values.min()-1e-9)
    high=low if policy=='binary' else float(np.quantile(values,1-target*2/3))
    return low,high

def develop():
    started=time.time();packs=dev_packs();net=network();thresholds={};frontier=[]
    for score in SCORES:
        for policy in POLICIES:
            for target in np.linspace(.1,1.,11):
                lo,hi=target_thresholds(packs,score,target,policy);summaries=[]
                for x in packs:
                    _,s=run(x['e'],x['pf'],net,x['scores'][score],lo,hi,4 if policy=='binary' else 1)
                    summaries.append(s)
                t=pd.DataFrame(summaries);frontier.append({'score':score,'policy':policy,'target':float(target),'low':lo,'high':hi,
                    **t[['high_error_automation','active_false_abstention','false_abstention','adjusted_cost_eur','activity','review_requested','active_false_rate']].mean().to_dict()})
            for target in [.5,.75,.9]:
                thresholds[f'{score}:{policy}:{target}']=list(target_thresholds(packs,score,target,policy))
    f=pd.DataFrame(frontier);f['pareto']=False
    for _,ids in f.groupby(['score','policy']).groups.items():
        ids=list(ids);f.loc[ids,'pareto']=pareto_flags(f.loc[ids,['high_error_automation','active_false_abstention','adjusted_cost_eur']].to_numpy())
    f.to_csv(A/'development_frontier.csv',index=False)
    old=pd.read_csv(ROOT/'results/development_pareto.csv');rows=[]
    for _,c in old.iterrows():
        w=np.array([c[f'w{j}'] for j in range(1,5)]);sums=[]
        for x in packs:
            _,s=run(x['e'],x['pf'],net,x['e']['scores']@w,c.low,c.high,1);sums.append(s)
        z=pd.DataFrame(sums)
        rows.append({'candidate':int(c.candidate),'low':c.low,'high':c.high,**{f'w{j}':c[f'w{j}'] for j in range(1,5)},
            **z[['adjusted_cost_eur','high_error_automation','active_false_abstention','false_abstention','zero_candidate_review','activity']].mean().to_dict()})
    t=pd.DataFrame(rows);values=t[['adjusted_cost_eur','high_error_automation','active_false_abstention']].to_numpy()
    t['pareto_active']=pareto_flags(values)
    distance=np.linalg.norm((values-values.min(0))/np.maximum(np.ptp(values,axis=0),1e-12),axis=1)
    distance[~t.pareto_active]=np.inf;k=int(np.argmin(distance));t['selected_active']=t.index==k
    t.to_csv(A/'development_objective_sensitivity.csv',index=False)
    chosen=t.iloc[k]
    frozen={'analysis_status':'post-audit exploratory, development-only calibration; original test results already inspected',
            'amendment_sha256':digest(ROOT/'audit_amendment.json'),'parent_parameters_sha256':digest(ROOT/'results/frozen_parameters.json'),
            'thresholds':thresholds,'active_objective_candidate':int(chosen.candidate),
            'active_weights':[float(chosen[f'w{j}']) for j in range(1,5)],'active_low':float(chosen.low),'active_high':float(chosen.high),
            'unchanged_original_candidate':P['candidate'],'elapsed_seconds':time.time()-started}
    (A/'frozen_amended_parameters.json').write_text(json.dumps(frozen,indent=2))
    print('DEVELOPMENT COMPLETE',frozen['active_objective_candidate'],round(time.time()-started,1),flush=True)

def mode_rows(out,meta):
    rows=[]
    for regime in range(3):
        mask=out[:,2]==regime;x=out[mask]
        rows.append({**meta,'regime':regime,'hours':len(x),'active_candidates':int(x[:,22].sum()),
           'issued_active':int(x[:,4].sum()),'high_error_actions':int(x[:,6].sum()),
           'violations':int(x[:,8].sum()),'fallback_overrides':int(x[:,33].sum())})
    return rows

def primary_job(region,origin):
    frozen=json.loads((A/'frozen_amended_parameters.json').read_text());assert frozen['amendment_sha256']==digest(ROOT/'audit_amendment.json')
    net=network();rows=[];ranks=[];modes=[];traces=[];metadata=[];gate_rows=[]
    for scenario in CONFIG['scenarios']:
        previous=None
        for seed in CONFIG['seeds']:
            if previous is not None and scenario not in ['missing','noisy','fdi_like']:e=previous
            else:e=pack(region,origin,168,scenario,seed);previous=e
            pf=(physics_tables(e['pred'],net),physics_tables(e['truth_next'],net));scores=get_scores(e)
            base={'region':region,'window':origin,'scenario':scenario,'seed':seed}
            common,_=run(e,pf,net);eligible=np.where(common[:,22]>.5)[0]
            for score in SCORES:
                for target in [.5,.75,.9]:
                    n=int(round(len(eligible)*target));ix=eligible[np.argsort(-scores[score][eligible],kind='stable')[:n]]
                    good=common[:,23]>.5
                    ranks.append({**base,'score':score,'target':target,'eligible':len(eligible),'selected':n,
                        'selected_high_error':int(common[ix,5].sum()),'selected_candidate_violations':int((1-common[ix,21]).sum()),
                        'good_active_opportunities':int(good.sum()),'deferred_good_active':int(good.sum()-good[ix].sum())})
                for policy in POLICIES:
                    for target in [.5,.75,.9]:
                        lo,hi=frozen['thresholds'][f'{score}:{policy}:{target}'];out,s=run(e,pf,net,scores[score],lo,hi,4 if policy=='binary' else 1)
                        meta={**base,'method':f'{score}_{policy}_{target}','score':score,'policy':policy,'target':target}
                        rows.append({**meta,**s});modes.extend(mode_rows(out,meta))
            extras=[('ungated',None,0,0,0,17),('constrained17',None,0,0,3,17),('constrained9',None,0,0,3,9),
                    ('original_composite',scores['composite'],P['low'],P['high'],1,17),
                    ('original_interval',scores['interval'],*P['thresholds']['interval_gate'],4,17),
                    ('original_learned',scores['learned'],*P['thresholds']['learned_gate'],1,17),
                    ('active_objective',e['scores']@np.array(frozen['active_weights']),frozen['active_low'],frozen['active_high'],1,17)]
            for name,score,lo,hi,mode,levels in extras:
                out,s=run(e,pf,net,score,lo,hi,mode,constrained_levels=levels)
                meta={**base,'method':name,'score':'reference','policy':'reference','target':np.nan};rows.append({**meta,**s});modes.extend(mode_rows(out,meta))
                if name in ['ungated','original_composite','constrained17']:
                    traces.append(out);metadata.append(meta)
            for name in ['bias','slow_ramp','clean','delayed']:
                if scenario!=name:continue
                sc=scores['composite']
                for active_label,mask in [('all',np.ones(len(sc),bool)),('corrupted',e['fault'])]:
                    if mask.any():gate_rows.append({**base,'population':active_label,'n':int(mask.sum()),'mean_score':float(sc[mask].mean()),'q10':float(np.quantile(sc[mask],.1)),'q90':float(np.quantile(sc[mask],.9)),
                        'low_fraction':float((sc[mask]<P['low']).mean()),'mid_fraction':float(((sc[mask]>=P['low'])&(sc[mask]<P['high'])).mean()),'full_fraction':float((sc[mask]>=P['high']).mean())})
    stem=A/f'primary_{region}_{origin}'
    pd.DataFrame(rows).to_csv(str(stem)+'_summary.csv',index=False);pd.DataFrame(ranks).to_csv(str(stem)+'_ranking.csv',index=False)
    pd.DataFrame(modes).to_csv(str(stem)+'_modes.csv',index=False);pd.DataFrame(gate_rows).to_csv(str(stem)+'_gate_response.csv',index=False)
    np.savez_compressed(str(stem)+'_traces.npz',outputs=np.array(traces),metadata=json.dumps(metadata),columns=np.array(COLUMNS))
    return {'stage':'primary','region':region,'window':origin,'runs':len(rows)}

def scaled_pf(table,rating):
    x=table.copy();x[:,:,3]/=rating
    x[:,:,0]=((x[:,:,5]>.5)&(x[:,:,1]>=.95-1e-7)&(x[:,:,2]<=1.05+1e-7)&(x[:,:,3]<=1+1e-7))
    x[:,:,4]=np.maximum(0,.95-x[:,:,1])+np.maximum(0,x[:,:,2]-1.05)+np.maximum(0,x[:,:,3]-1)
    x[:,:,4]=np.where(x[:,:,5]>.5,x[:,:,4],np.maximum(1,x[:,:,4]))
    return x

def physical_job(origin):
    rows=[];modes=[];traces=[];metadata=[]
    for task in ['battery','solar']:
        cells=[('original',1.,1.8),('lower_load',.6,1.8),('higher_pv',1.,3.6),('lower_load_higher_pv',.6,3.6)] if task=='battery' else [('solar3.6',.6,3.6),('solar7.2',.6,7.2)]
        for vol in CONFIG['renewable_volatility_factors']:
            for scenario in CONFIG['stress_scenarios']:
                e=pack('DE_LU',origin,672,scenario,11,vol,solar=task=='solar');scores=get_scores(e)
                for cell,load,cap in cells:
                    net=network('ieee33',1.,load,cap)
                    nominal=(physics_tables(e['pred'],net,'pv' if task=='solar' else 'battery'),physics_tables(e['truth_next'],net,'pv' if task=='solar' else 'battery'))
                    for rating in CONFIG['line_rating_factors']:
                        pf=tuple(scaled_pf(t,rating) for t in nominal)
                        base={'window':origin,'scenario':scenario,'seed':11,'identical_seed_labels':5,'volatility':vol,'rating':rating,'task':task,'cell':cell,'load_multiplier':load,'capacity_mw':cap}
                        cfg=[('ungated',None,0,0,0,17,0),('original_composite',scores['composite'],P['low'],P['high'],1,17,0)]
                        if task=='battery':cfg += [('constrained17',None,0,0,3,17,0),('constrained9',None,0,0,3,9,0),('always_half',None,0,0,2,17,0)]
                        else:cfg += [('ungated_correction',None,0,0,0,17,1),('composite_correction',scores['composite'],P['low'],P['high'],1,17,1),('interval_correction',scores['interval'],*P['thresholds']['interval_gate'],4,17,1)]
                        for name,score,lo,hi,mode,levels,fallback in cfg:
                            out,s=run(e,pf,net,score,lo,hi,mode,task=int(task=='solar'),constrained_levels=levels,pv_fallback=fallback)
                            meta={**base,'method':name};rows.append({**meta,**s});modes.extend(mode_rows(out,meta))
                            if rating==1 and vol==1 and name in ['ungated','original_composite','composite_correction']:
                                traces.append(out);metadata.append(meta)
    stem=A/f'physical_{origin}'
    pd.DataFrame(rows).to_csv(str(stem)+'_summary.csv',index=False);pd.DataFrame(modes).to_csv(str(stem)+'_modes.csv',index=False)
    np.savez_compressed(str(stem)+'_traces.npz',outputs=np.array(traces),metadata=json.dumps(metadata),columns=np.array(COLUMNS))
    return {'stage':'physical','window':origin,'runs':len(rows)}

def sensitivity():
    specs=[('default',{},{}),('width375',{'width_scale':375.},{}),('width1500',{'width_scale':1500.},{}),
      ('mix25',{'mix':.25},{}),('mix75',{'mix':.75},{}),('smooth3',{'smooth_window':3},{}),('smooth12',{'smooth_window':12},{}),
      ('cautious25',{}, {'cautious_factor':.25}),('cautious75',{}, {'cautious_factor':.75}),
      ('price_narrow',{}, {'high_price_factor':1.05,'low_price_factor':.90}),('price_wide',{}, {'high_price_factor':1.15,'low_price_factor':.80}),
      ('jump95',{'jump_q':.95},{}),('jump995',{'jump_q':.995},{}),('rough90',{'rough_q':.90},{}),('rough99',{'rough_q':.99},{}),('mismatch90',{'error_q':.90},{}),('mismatch99',{'error_q':.99},{})]
    rows=[];net=network();d=prepare('DE_LU')
    for origin in CONFIG['development_windows']:
        model=model_for(d,'DE_LU',origin)
        hist=d.loc[:pd.Timestamp(model.fit_end),['load','ren']].to_numpy();j=np.abs(np.diff(hist,axis=0)).sum(1);rough=pd.Series(j).rolling(6,min_periods=1).mean().to_numpy()
        origin_ts=pd.Timestamp(origin,tz='UTC');cal=(d.index>=origin_ts-pd.Timedelta(hours=672))&(d.index+pd.Timedelta(hours=1)<origin_ts)
        x=features(d[['load','ren']].to_numpy(),d.index,d.price.to_numpy(),d.load_forecast.to_numpy())
        pred,_,_=rf_predict(model.model,x[cal]);y=d[['load','ren']].shift(-1).to_numpy()[cal]
        er=np.abs((y[:,0]-y[:,1])-(pred[:,0]-pred[:,1]))
        for scenario in ['clean','delayed','slow_ramp']:
            obs=observe(d,origin,168,scenario,11)
            for name,ek,ck in specs:
                ek=dict(ek);m=model
                if 'jump_q' in ek:m=replace(m,jump=float(np.quantile(j,ek.pop('jump_q'))))
                if 'rough_q' in ek:m=replace(m,rough=float(np.quantile(rough,ek.pop('rough_q'))))
                if 'error_q' in ek:m=replace(m,error=float(np.quantile(er,ek.pop('error_q'))))
                e=evidence(obs,m,**ek);pf=(physics_tables(e['pred'],net),physics_tables(e['truth_next'],net))
                _,s=run(e,pf,net,e['scores']@np.array(P['weights']),P['low'],P['high'],1,**ck)
                rows.append({'setting':name,'window':origin,'scenario':scenario,**s})
    pd.DataFrame(rows).to_csv(A/'development_inner_sensitivity.csv',index=False)
    print('SENSITIVITY COMPLETE',len(rows),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','primary','physical','sensitivity']);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--rerun',action='store_true',help='Recompute amended runs even when their completion manifests exist');args=ap.parse_args()
    if args.stage=='develop':develop();return
    if args.stage=='sensitivity':sensitivity();return
    assert (A/'frozen_amended_parameters.json').exists()
    if args.stage=='primary':jobs=[(r,o) for r in CONFIG['regions'] for o in CONFIG['windows']];fn=primary_job
    else:jobs=[(o,) for o in CONFIG['windows']];fn=physical_job
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[]
        for job in jobs:
            stem=f'primary_{job[0]}_{job[1]}' if args.stage=='primary' else f'physical_{job[0]}'
            marker=A/f'{stem}_manifest.json'
            if marker.exists() and not args.rerun:continue
            futures.append((pool.submit(fn,*job),marker))
        mapping={f:p for f,p in futures}
        for fut in as_completed(mapping):
            result=fut.result();result['amendment_sha256']=digest(ROOT/'audit_amendment.json');mapping[fut].write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
