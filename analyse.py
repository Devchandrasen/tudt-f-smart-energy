from pathlib import Path
import json,warnings,re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_pipeline import ROOT,CONFIG
from run_study import pareto_flags

R=ROOT/'results';F=R/'figures';F.mkdir(exist_ok=True)
CORE=['missing','delayed','noisy','fdi_like']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})

def read_all(pattern):
    frames=[]
    for p in sorted(R.glob(pattern)):
        if pattern=='primary_*_summary.csv' and not re.match(r'primary_(DE_LU|DK_1)_2020-\d\d-\d\d_summary\.csv$',p.name):continue
        try:frames.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:pass
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def intervals(d,kind):
    rows=[];rng=np.random.default_rng(22355)
    d=d.query("interval=='split' and task=='battery'")
    if kind=='stress' and 'operating_regime' in d:d=d[d.operating_regime.fillna('original')=='original']
    for region,regiondata in d.groupby('region'):
        if kind=='primary':regiondata=regiondata[regiondata.scenario.isin(CORE)]
        monthly=regiondata.groupby(['window','method']).mean(numeric_only=True)
        for outcome in ['high_error_automation','false_abstention','physical_violation','adjusted_cost_eur']:
            wide=monthly[outcome].unstack('method')
            family=[]
            for comp in CONFIG['primary_comparators']:
                diff=(wide[comp]-wide['TUDT_F']).dropna().to_numpy();n=len(diff)
                boot=diff[rng.integers(0,n,size=(10000,n))].mean(axis=1)
                p=float(wilcoxon(diff,alternative='two-sided',method='auto').pvalue) if np.any(diff) else 1.
                family.append({'kind':kind,'region':region,'outcome':outcome,'comparator':comp,'n_months':n,'mean_comparator_minus_gate':float(diff.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'p_raw':p})
            order=np.argsort([x['p_raw'] for x in family]);last=0.
            for rank,i in enumerate(order):last=max(last,(len(order)-rank)*family[i]['p_raw']);family[i]['p_holm']=min(1,last)
            rows.extend(family)
    return pd.DataFrame(rows)

def save(fig,name):
    fig.savefig(F/f'{name}.pdf');fig.savefig(F/f'{name}.png',dpi=180);plt.close(fig)

def main():
    p=read_all('primary_*_summary.csv');s=read_all('stress_*_summary.csv')
    if p.window.nunique()!=9 or s.window.nunique()!=9 or len(p)!=12600 or len(s)!=77760:raise ValueError('complete, nonduplicated study required')
    keys=['region','window','scenario','seed','volatility','rating','task','method','interval']
    assert not p.duplicated(keys).any()
    p.to_csv(R/'primary_all.csv',index=False);s.to_csv(R/'stress_all.csv',index=False)
    primary=p.query("interval=='split' and region=='DE_LU'");core=primary[primary.scenario.isin(CORE)]
    if 'operating_regime' in s:s['operating_regime']=s.operating_regime.fillna('original')
    main=core.groupby('method').mean(numeric_only=True)
    for name,g in core.groupby('method'):
        main.loc[name,'selective_error_macro']=main.loc[name,'selective_error']
        main.loc[name,'selective_error']=float((g.high_error_automation*g.steps).sum()/(g.activity*g.steps).sum()) if (g.activity*g.steps).sum() else np.nan
        main.loc[name,'high_error_automation_count']=int(round((g.high_error_automation*g.steps).sum()))
        main.loc[name,'active_hours_count']=int(round((g.activity*g.steps).sum()))
        main.loc[name,'total_hours']=int(g.steps.sum())
    main.to_csv(R/'primary_core_summary.csv')
    external=p.query("interval=='split' and region=='DK_1'").groupby('method').mean(numeric_only=True)
    for name,g in p.query("interval=='split' and region=='DK_1'").groupby('method'):
        external.loc[name,'selective_error']=float((g.high_error_automation*g.steps).sum()/(g.activity*g.steps).sum()) if (g.activity*g.steps).sum() else np.nan
    external.to_csv(R/'external_summary.csv')
    stress=s.groupby(['task','method']).mean(numeric_only=True);stress.to_csv(R/'stress_overall.csv')
    stressgrid=s.groupby(['task','rating','volatility','method']).mean(numeric_only=True);stressgrid.to_csv(R/'stress_grid.csv')
    s.groupby(['operating_regime','task','method']).mean(numeric_only=True).to_csv(R/'stress_by_regime.csv')
    inference=pd.concat([intervals(p,'primary'),intervals(s,'stress')],ignore_index=True);inference.to_csv(R/'paired_inference.csv',index=False)
    detectors=read_all('primary_*_detectors.csv')
    sums=detectors.groupby(['region','scenario','detector'])[['tp','fp','fn','tn','detected_episodes','episodes','delay_sum']].sum()
    for name,num,den in [('fpr','fp',('fp','tn')),('fnr','fn',('fn','tp')),('precision','tp',('tp','fp')),('recall','tp',('tp','fn'))]:
        sums[name]=sums[num]/sums[list(den)].sum(axis=1).replace(0,np.nan)
    sums['mean_detected_delay_h']=sums.delay_sum/sums.detected_episodes.replace(0,np.nan);sums.to_csv(R/'detector_summary.csv')
    correlations=read_all('primary_*_correlations.csv');corr=correlations.query("region=='DE_LU'").groupby(['a','b']).correlation.mean();corr.to_csv(R/'channel_correlations.csv')
    cov=read_all('primary_*_coverage.csv');cov.groupby(['region','scenario','interval']).mean(numeric_only=True).to_csv(R/'coverage_summary.csv')
    rank=read_all('primary_*_ranking.csv');rank['error_rate']=rank.selected_high_error/rank.selected;rank['candidate_violation_rate']=rank.selected_candidate_violation/rank.selected
    rank.groupby(['region','score','target_coverage']).mean(numeric_only=True).to_csv(R/'matched_coverage_summary.csv')
    names=['ungated','always_half','constrained','interval_gate','learned_gate','TUDT_fixed','TUDT_F'];a=main.loc[names]
    fig,axs=plt.subplots(1,3,figsize=(10,3.3))
    for ax,col,title in zip(axs,['high_error_automation','false_abstention','adjusted_cost_eur'],['High-error automation / all hours','False abstention / all hours','Energy-adjusted cost (EUR/week)']):
        ax.barh(names,a[col],color=['#9ba5b1']*5+['#cf8d32','#167d8d']);ax.set_title(title,fontsize=9);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.tight_layout();save(fig,'primary_outcomes')
    dev=pd.read_csv(R/'development_pareto.csv');fig,ax=plt.subplots(figsize=(3.8,3.0))
    sc=ax.scatter(dev.high_error_automation,dev.false_abstention,c=dev.adjusted_cost_eur,s=19,cmap='viridis');z=dev[dev.pareto];ax.scatter(z.high_error_automation,z.false_abstention,facecolors='none',edgecolors='black',s=40,label='Nondominated')
    chosen=dev[dev.selected];ax.scatter(chosen.high_error_automation,chosen.false_abstention,marker='*',s=140,c='#d85b31',label='Frozen choice');ax.set(xlabel='High-error automation / all hours',ylabel='False abstention / all hours');fig.colorbar(sc,ax=ax,label='Adjusted cost (EUR/week)');ax.legend(fontsize=10);save(fig,'development_pareto')
    fig,ax=plt.subplots(figsize=(3.8,3.0))
    for name,g in rank.query("region=='DE_LU'").groupby('score'):
        z=g.groupby('target_coverage').error_rate.mean();ax.plot(z.index,z.values,marker='o',label={'TUDT_F':'TUDT-F','interval_gate':'Interval','learned_gate':'Learned'}[name])
    ax.set(xlabel='Matched selected fraction\nof common proposals',ylabel='High-error state fraction');ax.legend(fontsize=10);save(fig,'matched_coverage')
    matrix=np.eye(4)
    for (i,j),v in corr.items():matrix[int(i)-1,int(j)-1]=matrix[int(j)-1,int(i)-1]=v
    fig,ax=plt.subplots(figsize=(3.8,3.4));im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(4),['TSS','DRS','MUS','SAS']);ax.set_yticks(range(4),['TSS','DRS','MUS','SAS'])
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Mean within-window correlation');save(fig,'channel_correlations')
    subsets=main.loc[[x for x in main.index if x.startswith('subset_')]]
    interactions=[]
    for a1 in range(1,5):
        for b1 in range(a1+1,5):
            pair=main.loc[f'subset_{a1}{b1}'];single=min(main.loc[f'subset_{a1}','high_error_automation'],main.loc[f'subset_{b1}','high_error_automation'])
            interactions.append({'pair':f'{a1}{b1}','pair_high_error_automation':pair.high_error_automation,'best_single_high_error_automation':single,'pair_minus_best_single':pair.high_error_automation-single,'interpretation':'descriptive nonadditivity at development-matched coverage; not an independent causal channel effect'})
    pd.DataFrame(interactions).to_csv(R/'pair_interactions.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,3));ax.bar(np.arange(len(subsets)),subsets.high_error_automation,color='#167d8d');ax.set_xticks(np.arange(len(subsets)),[x.replace('subset_','') for x in subsets.index],rotation=45);ax.set(xlabel='Retained channels (1=TSS, 2=DRS, 3=MUS, 4=SAS)',ylabel='High-error automation');save(fig,'all_subsets')
    fig,axs=plt.subplots(1,2,figsize=(8,3))
    for task,ax in zip(['battery','pv'],axs):
        z=stress.loc[task].loc[[x for x in names if x in stress.loc[task].index]];ax.barh(z.index,z.physical_violation,color='#167d8d');ax.set(title=task.upper(),xlabel='Realised physical violation / all hours');ax.invert_yaxis()
    fig.tight_layout();save(fig,'stress_outcomes')
    fig,axs=plt.subplots(1,2,figsize=(8,3))
    z=cov.query("region=='DE_LU'").groupby(['scenario','interval']).mean(numeric_only=True)
    for m in ['split','block_max','adaptive']:
        g=z.xs(m,level='interval');axs[0].plot(g.index,g.load_coverage,marker='o',label=m);axs[1].plot(g.index,g.width_kw,marker='o',label=m)
    for ax in axs:ax.tick_params(axis='x',rotation=60);ax.legend(fontsize=10)
    axs[0].axhline(.9,color='black',ls='--',lw=.8);axs[0].set_title('Load coverage',fontsize=12);axs[1].set_title('Interval width (kW)',fontsize=12);fig.tight_layout();save(fig,'conformal_comparison')
    facts={'n_primary_runs':len(p),'n_stress_runs':len(s),'primary_controller_hours':int(p.steps.sum()),'stress_controller_hours':int(s.steps.sum()),'months':9,'primary_main':json.loads(a.reset_index().to_json(orient='records')),'stress_main':json.loads(stress.reset_index().to_json(orient='records')),'primary_inference':json.loads(inference.query("kind=='primary' and region=='DE_LU'").to_json(orient='records')),'maximum_ac_validation_error':float(pd.read_csv(R/'ac_validation.csv').max_error.max())}
    (R/'facts.json').write_text(json.dumps(facts,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in facts.items() if not isinstance(v,list)},indent=2));print(a[['high_error_automation','false_abstention','physical_violation','adjusted_cost_eur']]);print(stress[['physical_violation','adjusted_cost_eur','curtailed_kwh']])

if __name__=='__main__':main()
