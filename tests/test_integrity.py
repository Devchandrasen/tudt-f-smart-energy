import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from data_pipeline import prepare,train,observe,evidence
from physics import network,physics_tables,validate_against_pandapower
from control import simulate

def test_future_measurements_do_not_change_past_evidence():
    d=prepare('DE_LU');m=train(d,'2019-11-01');o=observe(d,'2019-11-01',168,'clean',11)
    a=evidence(o,m);cut=int(o['indices'][80]);changed={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in o.items()}
    changed['received'][cut+1:]*=1.8;changed['truth'][cut+1:]*=1.8
    from data_pipeline import features
    changed['x']=features(changed['received'],changed['times'],changed['price'],d.loc[changed['times'],'load_forecast'].to_numpy())
    b=evidence(changed,m)
    np.testing.assert_allclose(a['scores'][:81],b['scores'][:81]);np.testing.assert_allclose(a['pred'][:81],b['pred'][:81])

def test_evaluator_fault_labels_are_ignored():
    d=prepare('DE_LU');m=train(d,'2019-11-01');o=observe(d,'2019-11-01',168,'noisy',11)
    a=evidence(o,m);o['fault']=~o['fault'];o['event']=~o['event'];b=evidence(o,m)
    np.testing.assert_array_equal(a['scores'],b['scores'])

def test_training_targets_precede_origin():
    d=prepare('DE_LU');m=train(d,'2020-01-01')
    import pandas as pd
    assert pd.Timestamp(m.calibration_end)<pd.Timestamp('2020-01-01',tz='UTC')

def test_exact_radial_ac_matches_pandapower():
    rows=validate_against_pandapower(n=8)
    assert max(x['max_error'] for x in rows)<1e-6

def test_same_policy_receives_identical_outcomes():
    d=prepare('DE_LU');m=train(d,'2019-11-01');e=evidence(observe(d,'2019-11-01',168,'clean',11),m)
    net=network();p=physics_tables(e['pred'],net);a=physics_tables(e['truth_next'],net)
    args=(e['pred'],e['truth_next'],e['price'],np.ones(168),0.,0.,0,p,a,.26,.8,m.price_ref,30.)
    x=simulate(*args);y=simulate(*args)
    np.testing.assert_array_equal(x,y);assert (x[:,11]>=.12-1e-8).all();assert (x[:,11]<=.92+1e-8).all()

def test_holds_are_in_physical_denominator():
    n=10;v=np.ones((n,2))*600;pf=np.zeros((n,17,7));pf[:,:,0]=1;pf[:,:,5]=1;actual=pf.copy();actual[:,:,0]=0;actual[:,:,4]=.1
    x=simulate(v,v,np.ones(n)*.1,np.zeros(n),.65,.85,1,pf,actual,.26,.8,.1,30)
    assert x[:,3].sum()==n;assert x[:,8].sum()==n
