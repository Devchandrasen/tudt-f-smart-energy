import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_pipeline import ROOT,CONFIG,digest,prepare,train,observe,evidence,features
from physics import network,physics_tables
from control import simulate

def test_protocol_and_outer_parameters_frozen():
    a=json.loads((ROOT/'results/frozen_parameters.json').read_text())
    assert a['protocol_sha256']==digest(ROOT/'protocol.json')
    assert len(pd.read_csv(ROOT/'results/development_pareto.csv'))==200

def test_future_truth_and_inputs_do_not_change_earlier_commands():
    d=prepare('DE_LU');m=train(d,'2019-11-01');o=observe(d,'2019-11-01',168,'clean',11);a=evidence(o,m)
    cut=int(o['indices'][80]);o['truth'][cut+1:]*=1.7;o['received'][cut+1:]*=1.7
    o['x']=features(o['received'],o['times'],o['price'],d.loc[o['times'],'load_forecast'].to_numpy());b=evidence(o,m)
    net=network();params=json.loads((ROOT/'results/frozen_parameters.json').read_text());w=np.array(params['weights'])
    def run(e):return simulate(e['pred'],e['truth_next'],e['price'],e['scores']@w,params['low'],params['high'],1,physics_tables(e['pred'],net),physics_tables(e['truth_next'],net),.26,.8,params['price_ref'],params['error_cutoff_kw'])
    np.testing.assert_array_equal(run(a)[:81,1],run(b)[:81,1])

def test_primary_trace_rates_and_cost_match_shared_summaries():
    for path in sorted((ROOT/'results').glob('primary_*_traces.npz')):
        a=np.load(path);out=a['outputs'];s=pd.read_csv(path.with_name(path.name.replace('_traces.npz','_summary.csv')))
        assert len(s)==len(out)
        for name,index in [('activity',4),('high_error_automation',6),('false_abstention',7),('physical_violation',8)]:
            np.testing.assert_allclose(out[:,:,index].mean(axis=1),s[name],atol=1e-12)
        np.testing.assert_allclose(out[:,:,10].sum(axis=1),s.cost_eur,atol=1e-8)

def test_seed_repeats_are_not_inference_units():
    x=pd.read_csv(ROOT/'results/paired_inference.csv')
    assert (x.n_months==9).all()

def test_natural_record_selection_has_no_overlap():
    s=json.loads((ROOT/'data/rte/selection.json').read_text())['records'];d={x['event'] for x in s if x['split']=='development'};t={x['event'] for x in s if x['split']=='test'}
    assert len(d)==40 and len(t)==80 and not d&t
