"""Summaries, paired uncertainty and publication displays for the amendment."""
import json,itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_pipeline import ROOT,CONFIG
from audit_experiments import A,CORE,SCORES,POLICIES

F=A/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})

def read(pattern):return pd.concat([pd.read_csv(p) for p in sorted(A.glob(pattern))],ignore_index=True)

def aggregate(df,keys):
    records=[]
    for key,g in df.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(keys,key));row.update(g.mean(numeric_only=True).to_dict())
        steps=g.steps.sum();active=(g.activity*g.steps).sum();good=g.good_active_count.sum()
        row.update({'total_hours':int(steps),'runs':len(g),'high_error_count':int(round((g.high_error_automation*g.steps).sum())),
          'active_hours':int(round(active)),'selective_error':float((g.high_error_automation*g.steps).sum()/active) if active else np.nan,
          'good_active_count':int(good),'active_false_count':int(g.active_false_count.sum()),
          'active_false_rate':float(g.active_false_count.sum()/good) if good else np.nan,
          'zero_review_count':int(g.zero_review_count.sum()),'regret_eur':float(g.regret_sum_eur.sum()/g.regret_count.sum()) if g.regret_count.sum() else np.nan})
        records.append(row)
    return pd.DataFrame(records)

def paired(diff):
    diff=np.asarray(diff,float);diff=diff[np.isfinite(diff)];n=len(diff)
    rng=np.random.default_rng(22355);boot=diff[rng.integers(0,n,size=(10000,n))].mean(1)
    return {'n_months':n,'difference':float(diff.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),
      'p_raw':float(wilcoxon(diff).pvalue) if np.any(abs(diff)>1e-14) else 1.}

def holm(group):
    group=group.copy();order=np.argsort(group.p_raw.to_numpy());p=np.zeros(len(group));last=0.
    for rank,i in enumerate(order):last=max(last,(len(group)-rank)*group.iloc[i].p_raw);p[i]=min(1,last)
    group['p_holm']=p;return group

def wilson(x,n):
    z=1.959963984540054;p=x/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(center-half),float(center+half)

def main():
    p=read('primary_*_summary.csv');physical=pd.concat([read('physical_*_summary.csv'),read('followup_*_summary.csv')],ignore_index=True)
    assert len(p)==18000 and len(physical)==11016
    assert p.groupby('region').window.nunique().eq(9).all() and physical.window.nunique()==9
    p.to_csv(A/'primary_all.csv',index=False);physical.to_csv(A/'physical_all.csv',index=False)
    core=p[p.scenario.isin(CORE)];summary=aggregate(core,['region','method','score','policy','target']);summary.to_csv(A/'primary_core.csv',index=False)
    aggregate(p,['region','method','score','policy','target']).to_csv(A/'primary_all_conditions.csv',index=False)
    ps=aggregate(physical,['task','cell','method']);ps.to_csv(A/'physical_summary.csv',index=False)
    rank=read('primary_*_ranking.csv');gr=rank.groupby(['region','score','target']).sum(numeric_only=True).reset_index()
    gr['selected_error']=gr.selected_high_error/gr.selected;gr['active_false_rate']=gr.deferred_good_active/gr.good_active_opportunities
    gr.to_csv(A/'common_active_ranking.csv',index=False)
    ranking_ci=[]
    for region in CONFIG['regions']:
        for target in [.5,.75,.9]:
            g=rank[(rank.region==region)&(rank.target==target)].groupby(['window','score']).sum(numeric_only=True)
            for first,second in itertools.combinations(SCORES,2):
                x=g.loc[(slice(None),first),:].droplevel('score');y=g.loc[(slice(None),second),:].droplevel('score')
                ranking_ci.append({'region':region,'target':target,'score_a':first,'score_b':second,**paired(x.selected_high_error/x.selected-y.selected_high_error/y.selected)})
    pd.DataFrame(ranking_ci).to_csv(A/'common_active_ranking_inference.csv',index=False)
    mode=read('primary_*_modes.csv');mode=mode[mode.scenario.isin(CORE)]
    mode.groupby(['region','method','regime']).sum(numeric_only=True).reset_index().to_csv(A/'primary_modes.csv',index=False)
    pd.concat([read('physical_*_modes.csv'),read('followup_*_modes.csv')]).groupby(['task','cell','method','regime']).sum(numeric_only=True).reset_index().to_csv(A/'physical_modes.csv',index=False)
    gate=read('primary_*_gate_response.csv');gate.groupby(['region','scenario','population']).mean(numeric_only=True).reset_index().to_csv(A/'gate_response.csv',index=False)
    rows=[]
    for region in CONFIG['regions']:
        monthly=core[core.region==region].groupby(['window','method']).mean(numeric_only=True)
        pooled=core[core.region==region].groupby(['window','method'])[['regret_sum_eur','regret_count']].sum()
        monthly['regret_eur']=pooled.regret_sum_eur/pooled.regret_count
        for policy in POLICIES:
            for outcome in ['high_error_automation','active_false_abstention','adjusted_cost_eur','regret_eur']:
                for a,b in itertools.combinations(SCORES,2):
                    x=monthly.loc[(slice(None),f'{a}_{policy}_0.75'),outcome].droplevel('method');y=monthly.loc[(slice(None),f'{b}_{policy}_0.75'),outcome].droplevel('method')
                    rows.append({'region':region,'policy':policy,'outcome':outcome,'score_a':a,'score_b':b,**paired(x-y)})
    inf=pd.DataFrame(rows);inf=pd.concat([holm(g) for _,g in inf.groupby(['region','policy','outcome'])],ignore_index=True);inf.to_csv(A/'factorial_inference.csv',index=False)
    rows=[]
    for (task,cell),g in physical.groupby(['task','cell']):
        base='ungated';gate_method='original_composite'
        comparisons=[(gate_method,base)]
        if task=='solar':comparisons += [('composite_correction','ungated_correction'),('composite_correction','original_composite'),('composite_feasible_recovery','ungated_feasible_recovery'),('composite_feasible_recovery','original_composite')]
        else:comparisons += [('constrained17','constrained9')]
        monthly=g.groupby(['window','method']).mean(numeric_only=True)
        for method,reference in comparisons:
            for outcome in ['physical_violation','physical_severity','max_violation_run_hours','cost_eur','energy_correctable_violation']:
                x=monthly.loc[(slice(None),method),outcome].droplevel('method');y=monthly.loc[(slice(None),reference),outcome].droplevel('method')
                rows.append({'task':task,'cell':cell,'method':method,'reference':reference,'outcome':outcome,**paired(x-y)})
    pi=pd.DataFrame(rows);pi=pd.concat([holm(g) for _,g in pi.groupby(['task','outcome'])],ignore_index=True);pi.to_csv(A/'physical_inference.csv',index=False)
    # Monthly cluster intervals for detector rates; injected labels are evaluator-only.
    d=pd.concat([pd.read_csv(q) for q in (ROOT/'results').glob('primary_DE_LU_*_detectors.csv')],ignore_index=True)
    rows=[];rng=np.random.default_rng(22355)
    for (scenario,det),g in d.groupby(['scenario','detector']):
        totals=g[['tp','fp','fn','tn','detected_episodes','episodes','delay_sum']].sum();row={'scenario':scenario,'detector':det,**totals.to_dict()}
        monthly=g.groupby('window')[['tp','fp','fn','tn']].sum().to_numpy();draw=monthly[rng.integers(0,len(monthly),size=(10000,len(monthly)))].sum(1)
        for name,num,den in [('recall',0,(0,2)),('fpr',1,(1,3)),('precision',0,(0,1)),('fnr',2,(0,2))]:
            denominator=totals.iloc[list(den)].sum();row[name]=float(totals.iloc[num]/denominator) if denominator else np.nan
            bot=draw[:,list(den)].sum(1);values=np.divide(draw[:,num],bot,out=np.full(10000,np.nan),where=bot>0)
            row[name+'_low']=float(np.nanquantile(values,.025)) if np.isfinite(values).any() else np.nan
            row[name+'_high']=float(np.nanquantile(values,.975)) if np.isfinite(values).any() else np.nan
        row['detected_delay_h']=float(totals.delay_sum/totals.detected_episodes) if totals.detected_episodes else np.nan;rows.append(row)
    pd.DataFrame(rows).to_csv(A/'detector_intervals.csv',index=False)
    real=pd.read_csv(ROOT/'results/real_event_records.csv').query("split=='test'");rr=[]
    for detector,g in real.groupby('detector'):
        truth=g.fault.astype(bool);alarm=g.alarm.astype(bool);tp=int((truth&alarm).sum());fp=int((~truth&alarm).sum());n1=int(truth.sum());n0=int((~truth).sum())
        low,high=wilson(tp,n1);fl,fh=wilson(fp,n0)
        rr.append({'detector':detector,'tp':tp,'fn':n1-tp,'fp':fp,'tn':n0-fp,'recall':tp/n1,'recall_low':low,'recall_high':high,'fpr':fp/n0,'fpr_low':fl,'fpr_high':fh,
          'detected_events':int(g.loc[truth,'detection_delay_s'].notna().sum()),'median_detected_delay_ms':float(1000*g.loc[truth,'detection_delay_s'].median())})
    pd.DataFrame(rr).to_csv(A/'natural_detector_intervals.csv',index=False)
    # Existing interval variants and development freshness are surfaced, not silently rerun/renamed.
    primary=pd.read_csv(ROOT/'results/primary_all.csv');cov=pd.read_csv(ROOT/'results/coverage_summary.csv');block=pd.read_csv(ROOT/'results/block_coverage.csv')
    intervals=primary.query("region=='DE_LU' and method=='TUDT_F'").groupby(['scenario','interval']).mean(numeric_only=True).reset_index()
    ci=cov.query("region=='DE_LU'").merge(intervals[['scenario','interval','full_rate','cautious_rate','abstention','activity','high_error_automation']],on=['scenario','interval'])
    bc=block.groupby(['scenario','method']).mean(numeric_only=True).reset_index().rename(columns={'method':'interval'})
    ci=ci.merge(bc[['scenario','interval','load_daily_simultaneous_coverage','renewable_daily_simultaneous_coverage']],on=['scenario','interval']);ci.to_csv(A/'interval_authorization.csv',index=False)
    fresh=pd.read_csv(ROOT/'results/development_freshness.csv');fresh.groupby(['scenario','decay','history']).mean(numeric_only=True).reset_index().to_csv(A/'freshness_summary.csv',index=False)
    inner=pd.read_csv(A/'development_inner_sensitivity.csv');inner.groupby(['setting','scenario']).mean(numeric_only=True).reset_index().to_csv(A/'inner_sensitivity_summary.csv',index=False)
    # Subset test coverage alongside the originally reported outcomes.
    subsets=primary[primary.method.str.startswith('subset_') & primary.scenario.isin(CORE)]
    subsets.groupby('method').mean(numeric_only=True).reset_index().to_csv(A/'subset_with_coverage.csv',index=False)
    # Recount v1 denominators without changing the preserved v1 traces.
    legacy=[]
    for path in sorted((ROOT/'results').glob('primary_DE_LU_*_traces.npz')):
        z=np.load(path);metadata=json.loads(str(z['metadata']))
        for out,meta in zip(z['outputs'],metadata):
            if meta['scenario'] not in CORE or meta['interval']!='split':continue
            active=out[:,0]!=8;good=active&(out[:,5]<.5)&(out[:,17]>.5);abstain=out[:,3]>.5
            legacy.append({**meta,'active_false_count':int((good&abstain).sum()),'good_active_count':int(good.sum()),'zero_review_count':int(((~active)&(out[:,7]>.5)).sum()),'hours':len(out)})
    l=pd.DataFrame(legacy).groupby('method')[['active_false_count','good_active_count','zero_review_count','hours']].sum().reset_index()
    l['active_false_rate']=l.active_false_count/l.good_active_count
    l.to_csv(A/'legacy_active_denominators.csv',index=False)
    figures(summary,gr)
    metadata={'primary_runs':len(p),'physical_runs':len(physical),'primary_monthly_units_per_region':9,
      'physical_unique_seed_per_deterministic_condition':1,'analysis_status':'post-audit exploratory; original frozen evidence retained',
      'effect_direction':'score_a minus score_b or method minus reference; positive error/violation/cost differences are unfavorable to first method',
      'ci':'paired nine-month cluster bootstrap, 10000 resamples; descriptive temporal approximation',
      'holm':'within region/policy/outcome across three score pairs at target .75; physical task/outcome across retained cell comparisons',
      'amendment_sha256':__import__('data_pipeline').digest(ROOT/'audit_amendment.json')}
    (A/'analysis_manifest.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)

def figures(summary,ranking):
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.8))
    f=pd.read_csv(A/'development_frontier.csv');colors={'composite':'#285c9e','interval':'#c07624','learned':'#42836b'}
    for score,g in f.groupby('score'):
        z=g[g.policy=='three_regime'];axes[0].plot(z.active_false_abstention*100,z.high_error_automation*100,'o-',label=score,color=colors[score])
    axes[0].set(xlabel='Good active-candidate deferral (% of all hours)',ylabel='High-error automation (% of all hours)',title='Development operating points')
    axes[0].legend(fontsize=8)
    for score,g in ranking.query("region=='DE_LU'").groupby('score'):
        axes[1].plot(g.target*100,g.selected_error*100,'o-',label=score,color=colors[score])
    axes[1].set(xlabel='Selected common nonzero candidates (%)',ylabel='High-error fraction among selected (%)',title='Test common-candidate ranking')
    fig.tight_layout();fig.savefig(F/'active_tradeoffs.pdf');fig.savefig(F/'active_tradeoffs.png',dpi=170);plt.close(fig)
    fig=plt.figure(figsize=(6.5,4.3));ax=fig.add_subplot(111,projection='3d')
    for score,g in f.query("policy=='three_regime'").groupby('score'):
        ax.plot(g.active_false_abstention*100,g.high_error_automation*100,g.adjusted_cost_eur,'o-',label=score,color=colors[score],markersize=4)
        nd=g[g.pareto];ax.scatter(nd.active_false_abstention*100,nd.high_error_automation*100,nd.adjusted_cost_eur,marker='*',s=60,color=colors[score])
    ax.set_xlabel('Deferral (% all hours)',labelpad=7);ax.set_ylabel('High-error auto. (%)',labelpad=7);ax.set_zlabel('Adjusted EUR/week',labelpad=7);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(F/'development_frontier_3d.pdf');plt.close(fig)

if __name__=='__main__':main()
