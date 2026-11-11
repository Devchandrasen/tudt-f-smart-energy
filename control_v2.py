"""Post-audit command accounting. V1 code and outputs remain unchanged.

The first twenty columns keep the v1 meanings, including the shared rule
proposal in column zero. A separate field records the comparator's own
pre-filter candidate. Command selection never reads truth or actual_pf.
"""
import numpy as np
from numba import njit
from control import COLUMNS as V1_COLUMNS, summarize as summarize_v1

EXTRA_COLUMNS = ['controller_candidate_index','controller_candidate_realised_feasible',
 'active_candidate','good_active_candidate','active_false_abstention',
 'review_requested','gate_defer','zero_candidate_review','power_correctable_violation',
 'energy_correctable_violation','voltage_violation','thermal_violation','solver_failure',
 'fallback_override','regret_eur','regret_eligible','physical_alternative_any',
 'energy_alternative_any','pre_soc','controller_candidate_forecast_feasible']
COLUMNS = V1_COLUMNS + EXTRA_COLUMNS

@njit(cache=True)
def simulate_v2(pred,truth,price,score,low,high,mode,forecast_pf,actual_pf,
                pmax,capacity,price_ref,error_cutoff,task=0,pv_capacity=1.8,
                constrained_levels=17,cautious_factor=.5,pv_fallback=0,
                high_price_factor=1.1,low_price_factor=.85):
    n=len(pred);out=np.zeros((n,40));soc=.55;eta=.94
    for i in range(n):
        energy=soc*capacity;pre_soc=soc
        energy_ok=np.ones(17,np.bool_)
        for k in range(17):
            u=(k-8)*pmax/8
            en=energy-u/eta if u>=0 else energy-u*eta
            energy_ok[k]=en>=.12*capacity-1e-10 and en<=.92*capacity+1e-10
        proposal=8
        if task==0:
            if price[i]>price_ref*high_price_factor and (pred[i,0]-pred[i,1])/600>.3:proposal=16
            elif price[i]<price_ref*low_price_factor:proposal=0
            while proposal!=8 and not energy_ok[proposal]:proposal+=1 if proposal<8 else -1
        else:
            proposal=16
            for k in range(17):
                if forecast_pf[i,k,0]>.5:proposal=k;break
        shared_proposal=proposal;command=proposal;regime=2;gate_defer=0
        if mode==3:
            best=1e30;command=8 if task==0 else 0
            for k in range(17):
                if task==0 and ((constrained_levels==9 and k%2!=0) or not energy_ok[k]):continue
                if forecast_pf[i,k,0]<.5:continue
                u=(k-8)*pmax/8 if task==0 else 0.
                grid=forecast_pf[i,k,6]
                cost=price[i]*(max(grid,0)-.35*max(-grid,0))
                if task==0:cost+=price_ref*(u/eta if u>=0 else u*eta)
                if cost<best:best=cost;command=k
            proposal=command
        elif mode==2:
            command=int(round((proposal-8)*cautious_factor))+8 if task==0 else int(round(proposal*cautious_factor));regime=1
        elif mode==5:
            command=8 if task==0 else 0;regime=0;gate_defer=1
        elif mode==1 or mode==4:
            if score[i]<low:command=8 if task==0 else 0;regime=0;gate_defer=1
            elif mode==1 and score[i]<high:
                command=int(round((proposal-8)*cautious_factor))+8 if task==0 else int(round(proposal*cautious_factor));regime=1
        rejected=int(forecast_pf[i,command,0]<.5)
        review=int(gate_defer==1 or rejected==1)
        override=0
        if task==1 and pv_fallback>0 and rejected==1:
            # Forecast-only correction preserving emergency fallback. It may
            # override a requested deferral, which is separately logged.
            command=16 if pv_fallback==1 else 0
            for k in range(17):
                if forecast_pf[i,k,0]>.5:command=k;break
            override=1
            if pv_fallback==2 and command==0:regime=0;override=0
        elif rejected==1:
            command=8 if task==0 else 0;regime=0
        if task==0 and not energy_ok[command]:command=8;regime=0;review=1
        u=(command-8)*pmax/8 if task==0 else 0.
        if task==0:
            energy-=u/eta if u>=0 else u*eta;soc=energy/capacity
        activity=int(abs(u)>1e-9) if task==0 else int(command>0)
        abstain=int(review==1 and activity==0)
        actual=actual_pf[i,command];shared_actual=actual_pf[i,shared_proposal];cand_actual=actual_pf[i,proposal]
        error=abs((truth[i,0]-truth[i,1])-(pred[i,0]-pred[i,1]));high_error=error>error_cutoff
        candidate_active=int(proposal!=8) if task==0 else int(proposal>0)
        good_active=int(candidate_active==1 and not high_error and cand_actual[0]>.5)
        zero_review=int(candidate_active==0 and review==1 and not high_error and cand_actual[0]>.5)
        active_false=int(good_active==1 and abstain==1)
        power_any=0;energy_any=0;best_actual=1e30
        for k in range(17):
            if actual_pf[i,k,0]>.5:
                power_any=1
                if task==1 or energy_ok[k]:
                    energy_any=1
                    grid=actual_pf[i,k,6]
                    c=1000*price[i]*(max(grid,0)-.35*max(-grid,0))
                    if task==0:
                        ku=(k-8)*pmax/8
                        c+=1000*price_ref*(ku/eta if ku>=0 else ku*eta)
                    best_actual=min(best_actual,c)
        grid=actual[6];cost=1000*price[i]*(max(grid,0)-.35*max(-grid,0))
        adj_one=cost
        if task==0:adj_one+=1000*price_ref*(u/eta if u>=0 else u*eta)
        eligible=int(actual[0]>.5 and energy_any==1)
        regret=max(0.,adj_one-best_actual) if eligible==1 else 0.
        violation=1-actual[0]
        curt=min(command*pv_capacity/16,pv_capacity*truth[i,1]/400) if task==1 else 0.
        out[i,:20]=np.array([1.*shared_proposal,1.*command,1.*regime,1.*abstain,1.*activity,1.*high_error,1.*(high_error and activity),1.*(abstain and not high_error and shared_actual[0]>.5),violation,actual[4],cost,soc,actual[1],actual[2],actual[3],1.*rejected,forecast_pf[i,command,0],shared_actual[0],curt*1000,error])
        out[i,20:]=np.array([1.*proposal,cand_actual[0],1.*candidate_active,1.*good_active,1.*active_false,1.*review,1.*gate_defer,1.*zero_review,violation*power_any,violation*energy_any,1.*(actual[1]<.95-1e-7 or actual[2]>1.05+1e-7),1.*(actual[3]>1+1e-7),1.*(actual[5]<.5),1.*override,regret,1.*eligible,1.*power_any,1.*energy_any,pre_soc,forecast_pf[i,proposal,0]])
    return out

def summarize(out,capacity,price_ref):
    d=summarize_v1(out[:,:20],capacity,price_ref)
    d.update({k:float(out[:,20+j].mean()) for j,k in enumerate(EXTRA_COLUMNS)})
    d['active_false_count']=int(out[:,24].sum());d['good_active_count']=int(out[:,23].sum())
    d['active_false_rate']=float(out[:,24].sum()/out[:,23].sum()) if out[:,23].sum() else np.nan
    d['zero_review_count']=int(out[:,27].sum());d['review_count']=int(out[:,25].sum())
    d['regret_eur']=float(out[:,34].sum()/out[:,35].sum()) if out[:,35].sum() else np.nan
    d['regret_count']=int(out[:,35].sum());d['regret_sum_eur']=float(out[:,34].sum())
    return d
