"""Finish solar summaries, source-gap ledger and independently checked metadata."""
import argparse,json,hashlib,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import ROOT,CONFIG,digest
from physics import validate_against_pandapower,network,solve_radial

def pv_validation():
    import pandapower as pp
    import pandapower.networks as pn
    rng=np.random.default_rng(22355);rows=[]
    for cap in [3.6,7.2]:
        sp=network('ieee33',1.,.6,cap);net=pn.case33bw();net.line.max_i_ka=.4
        bp=net.load.p_mw.to_numpy().copy();bq=net.load.q_mvar.to_numpy().copy()
        for b in [17,24,32]:pp.create_sgen(net,b,p_mw=cap/3)
        for i in range(60):
            l=rng.uniform(.6,1.2);g=rng.uniform(0,1.6);curt=rng.uniform(0,cap*g)
            net.load.loc[:,'p_mw']=bp*.6*l;net.load.loc[:,'q_mvar']=bq*.6*l;net.sgen.loc[:,'p_mw']=cap*g/3-curt/3
            pp.runpp(net,numba=True,tolerance_mva=1e-10,max_iteration=100)
            a=solve_radial(sp['parent'],sp['z'],sp['load']*l-sp['gen']*g,sp['reactive']*l,sp['rating'],17,0.,sp['gen_buses'],curt)
            err=max(abs(a[1]-net.res_bus.vm_pu.min()),abs(a[2]-net.res_bus.vm_pu.max()),abs(a[3]-net.res_line.loading_percent.max()/100),abs(a[6]-net.res_ext_grid.p_mw.sum()))
            assert err<1e-6;rows.append({'capacity_mw':cap,'case':i,'max_error':err})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-ac',action='store_true');args=ap.parse_args();r=ROOT/'results'
    pv=pd.concat([pd.read_csv(x) for x in sorted(r.glob('pv_2020-??-??_summary.csv'))],ignore_index=True)
    assert len(pv)==12960 and pv.window.nunique()==9
    pv.to_csv(r/'pv_all.csv',index=False);pv.groupby(['pv_capacity_mw','method']).mean(numeric_only=True).to_csv(r/'pv_overall.csv')
    if args.validate_ac or not (r/'ac_validation.csv').exists():pd.DataFrame(validate_against_pandapower(n=120)).to_csv(r/'ac_validation.csv',index=False)
    if args.validate_ac or not (r/'ac_pv_validation.csv').exists():pd.DataFrame(pv_validation()).to_csv(r/'ac_pv_validation.csv',index=False)
    raw=[2.6019284486849136e-7,2.668562990240662e-6,2.596494390922246e-7,2.606010328766956e-7,2.4456246542871303e-7]
    adjusted=np.zeros(5);running=0.
    for rank,i in enumerate(np.argsort(raw)):running=max(running,(5-rank)*raw[i]);adjusted[i]=min(1,running)
    (r/'legacy_holm_audit.json').write_text(json.dumps({'raw_p':raw,'adjusted_p':adjusted.tolist(),'finding':'four identical values are valid Holm cumulative maxima','limitation':'valid arithmetic does not validate the old outcome definitions or independence assumption'},indent=2))
    changes=[
      ('A01','Historical injected flags and future targets','Received metadata and completed source-matched history only','data_pipeline.py; tests/test_integrity.py','Substantive causal-method correction; reviewed-code equivalence unknown'),
      ('A02','Method-dependent low-trust metrics','Shared independent high-error/physical/review/cost accounting','control.py; results/primary_all.csv','Threshold compliance is only verification'),
      ('A03','Hand weights only','Retained prior plus 200-point development Pareto search','results/development_pareto.csv; results/frozen_parameters.json','Finite search, no global optimum'),
      ('A04','One-channel deletion only','All 15 nonempty subsets and six pair contrasts','results/pair_interactions.csv','Dependent signals, descriptive interactions'),
      ('A05','Split intervals only','Split, block-maximum and bounded adaptive comparison','results/coverage_summary.csv; results/block_coverage.csv','Different coverage targets; full permutation and weighted conformal unperformed'),
      ('A06','Short two-/five-bus physical checks','Nine 28-day IEEE33 windows and complete stress grids','results/stress_all.csv; stress_amendment.json','Original infeasible cell retained; added cells exploratory'),
      ('A07','Battery task only','Solar-only curtailment with frozen gate','results/pv_all.csv; pv_amendment.json','Generic scaling worsens physical outcomes'),
      ('A08','Synthetic corruption only','120 RTE natural-recording check','results/real_event_manifest.json','Weak disturbance transfer; GESL raw data unavailable'),
      ('A09','Large random FDI-like values only','3% bias and slow-ramp probes','results/detector_summary.csv','No formal state-estimation stealth attack'),
      ('A10','36 scenario-window pairs treated as units','Nine monthly paired units and cluster bootstrap','results/paired_inference.csv','Nine units, approximate temporal dependence'),
      ('A11','Long abstract and broad novelty claims','Seven-section rewrite; explicit negative evidence','manuscript/body.tex','Author review required'),
      ('A12','Unresolved figure reference','New vector flow diagram and resolved references','manuscript/main.pdf','Compiled PDF checked'),
      ('A13','Cost without equal terminal-energy valuation','Shared terminal value and secondary same-SOC regret','regret_check.py','One-step hindsight diagnostic only')]
    pd.DataFrame(changes,columns=['id','before','after','evidence','qualification']).to_csv(r/'change_register.csv',index=False)
    source_gap={'exact_reviewed_code':'unavailable','supplied_source':'preserved unchanged in private working provenance','legacy_code':'recovered at '+CONFIG['legacy_commit'],'legacy_stressed_five_bus_source':'not present in recovered OPSD release; replaced with documented IEEE33 evaluation','legacy_learned_logistic_and_observable_reviewed_version':'not established from retained source; not claimed reproduced','GESL_measurements':'HTTP 401; no login or restricted measurements used','RTE_alternative':'completed with public raw phase signals and third-party annotations','block_permutation':'not reproduced; explicit block-maximum adaptation used','formal_stealth_attack':'not implemented; bounded corruption only'}
    (r/'source_gaps.json').write_text(json.dumps(source_gap,indent=2))
    registry=[]
    for identifier,file,scope in [
       ('E001','protocol.json','prespecified design'),('E002','results/frozen_parameters.json','development-frozen choices'),('E003','results/primary_core_summary.csv','primary numerical claims'),('E004','results/paired_inference.csv','monthly uncertainty and tests'),('E005','results/detector_summary.csv','simulation proxy discrimination'),('E006','results/matched_coverage_summary.csv','ranking'),('E007','results/stress_by_regime.csv','physical regimes'),('E008','results/pv_overall.csv','solar-only task'),('E009','results/real_event_summary.csv','natural disturbance check'),('E010','results/coverage_summary.csv','interval coverage/width'),('E011','results/ac_validation.csv','independent AC checks'),('E012','results/ac_pv_validation.csv','solar AC checks')]:
        p=ROOT/file;registry.append({'evidence_id':identifier,'file':file,'sha256':digest(p),'scope':scope,'machine_checked':True,'human_verifier':None,'human_verification_status':'pending accountable author review'})
    (r/'evidence_registry.json').write_text(json.dumps(registry,indent=2))
    status={'study_id':'ISJ-RE-26-22355','version':'isj-reconstruction-1.0.0','prepared_date':'2026-11-05','submission_ready':False,'reason':'author review of reconstructed methods, source claims, declarations and scientific interpretation remains required; no submission performed','protocol_sha256':digest(ROOT/'protocol.json'),'frozen_parameters_sha256':digest(r/'frozen_parameters.json'),'all_primary_windows':18,'battery_stress_windows_and_regimes':36,'solar_task_windows':9,'real_recordings':120,'accountable_author_approval':None}
    (r/'package_status.json').write_text(json.dumps(status,indent=2));print(status)

if __name__=='__main__':main()
