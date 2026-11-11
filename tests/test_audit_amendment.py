import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from control import simulate
from control_v2 import simulate_v2
from audit_experiments import scaled_pf
from physics import network,physics_tables

def fixture(n=8):
    pred=np.ones((n,2))*100;price=np.ones(n)*.2
    pf=np.zeros((n,17,7));pf[:,:,0]=1;pf[:,:,1:3]=1;pf[:,:,5]=1
    pf[:,:,6]=np.arange(17)[None,:]*.01
    return pred,price,pf

def test_zero_candidate_review_is_separate_from_foregone_active_dispatch():
    pred,price,pf=fixture();price[:]=.1
    x=simulate_v2(pred,pred,price,np.zeros(len(pred)),.6,.8,1,pf,pf,.26,.8,.1,30.)
    assert x[:,27].sum()==len(pred)
    assert x[:,24].sum()==0
    assert x[:,23].sum()==0

def test_constrained_candidate_is_recorded_separately():
    pred,price,pf=fixture();price[:]=.1
    x=simulate_v2(pred,pred,price,np.ones(len(pred)),0.,0.,3,pf,pf,.26,.8,.1,30.)
    assert (x[:,0]==8).all()
    np.testing.assert_array_equal(x[:,20],x[:,1])
    assert (x[:,20]!=x[:,0]).any()

def test_pv_fallback_uses_forecast_not_future_truth():
    pred,price,pf=fixture();pf[:,:,0]=0;pf[:,10:,0]=1
    truth=pred.copy();actual=pf.copy()
    args=(pred,truth,price,np.zeros(len(pred)),.6,.8,1,pf,actual,.8,3.2,.1,30.)
    x=simulate_v2(*args,task=1,pv_fallback=1)
    changed=list(args);changed[1]=truth*10;changed[8]=np.zeros_like(actual)
    y=simulate_v2(*changed,task=1,pv_fallback=1)
    assert (x[:,1]==10).all();assert (x[:,33]==1).all()
    np.testing.assert_array_equal(x[:,1:3],y[:,1:3])

def test_amended_default_preserves_original_gate_commands():
    pred,price,pf=fixture();pred[:,0]=500;pred[:,1]=100
    score=np.linspace(.3,1,len(pred))
    args=(pred,pred,price,score,.6,.8,1,pf,pf,.26,.8,.1,30.)
    old=simulate(*args);new=simulate_v2(*args)
    np.testing.assert_array_equal(old[:,1:7],new[:,1:7])
    np.testing.assert_allclose(old[:,8:15],new[:,8:15])

def test_rating_rescaling_equals_independent_ac_table():
    values=np.array([[500.,200.],[600.,400.],[700.,300.]])
    nominal=physics_tables(values,network('ieee33'))
    direct=physics_tables(values,network('ieee33',.6))
    np.testing.assert_allclose(scaled_pf(nominal,.6),direct,atol=1e-10)

def test_feasible_only_recovery_does_not_invent_a_feasible_command():
    pred,price,pf=fixture();pf[:,:,0]=0
    x=simulate_v2(pred,pred,price,np.zeros(len(pred)),.6,.8,1,pf,pf,.8,3.2,.1,30.,task=1,pv_fallback=2)
    assert (x[:,1]==0).all();assert (x[:,16]==0).all()
