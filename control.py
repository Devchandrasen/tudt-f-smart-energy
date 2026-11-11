"""Shared command and outcome accounting for every comparator."""
import numpy as np
from numba import njit

@njit(cache=True)
def simulate(pred,truth,price,score,low,high,mode,forecast_pf,actual_pf,pmax,capacity,price_ref,error_cutoff,task=0,pv_capacity=1.8):
    n=len(pred);out=np.zeros((n,20));soc=.55;eta=.94
    # columns documented in COLUMNS. Mode 0 ungated, 1 three-band, 2 half,
    # 3 constrained finite-grid, 4 binary selective, 5 defer.
    for i in range(n):
        energy=soc*capacity
        feasible_energy=np.ones(17,np.bool_)
        for k in range(17):
            u=(k-8)*pmax/8
            next_energy=energy-u/eta if u>=0 else energy-u*eta
            feasible_energy[k]=next_energy>=.12*capacity-1e-10 and next_energy<=.92*capacity+1e-10
        net=(pred[i,0]-pred[i,1])/600
        # Candidate policy is fixed before the gate and shared at equal SOC.
        proposal=8
        if task==0:
            if price[i]>price_ref*1.1 and net>.3:proposal=16
            elif price[i]<price_ref*.85:proposal=0
            if not feasible_energy[proposal]:
                while proposal!=8 and not feasible_energy[proposal]:proposal+=1 if proposal<8 else -1
        else:
            proposal=0
            # Choose minimum forecast-feasible curtailment; if none, maximal
            # tested curtailment is a disclosed fallback, not a safety certificate.
            for k in range(17):
                if forecast_pf[i,k,0]>.5:proposal=k;break
                proposal=16
        base_proposal=proposal;regime=2;abstain=0
        command=proposal
        if mode==2:command=int(round((proposal-8)*.5))+8 if task==0 else int(round(proposal*.5));regime=1
        elif mode==5:command=8 if task==0 else 0;regime=0;abstain=1
        elif mode==1 or mode==4:
            if score[i]<low:command=8 if task==0 else 0;regime=0;abstain=1
            elif score[i]<high and mode==1:
                command=int(round((proposal-8)*.5))+8 if task==0 else int(round(proposal*.5));regime=1
        elif mode==3:
            best=1e30;command=8 if task==0 else 0
            for k in range(17):
                if task==0 and (k%2!=0 or not feasible_energy[k]):continue
                if forecast_pf[i,k,0]<.5:continue
                u=(k-8)*pmax/8 if task==0 else -k*pv_capacity/16
                grid=forecast_pf[i,k,6]
                cost=price[i]*(max(grid,0)-.35*max(-grid,0))
                if task==0:cost+=price_ref*(u/eta if u>=0 else u*eta)
                if cost<best:best=cost;command=k
        # Every method receives the identical forecast-command physical filter.
        rejected=0
        if forecast_pf[i,command,0]<.5:
            command=8 if task==0 else 0;regime=0;abstain=1;rejected=1
        if task==0 and not feasible_energy[command]:command=8;regime=0;abstain=1
        u=(command-8)*pmax/8 if task==0 else 0.
        if task==0:
            energy-=u/eta if u>=0 else u*eta;soc=energy/capacity
        actual=actual_pf[i,command];proposal_actual=actual_pf[i,base_proposal]
        error=abs((truth[i,0]-truth[i,1])-(pred[i,0]-pred[i,1]))
        high_error=error>error_cutoff
        activity=abs(u)>1e-9 if task==0 else command>0
        independent_good=(not high_error) and proposal_actual[0]>.5
        grid=actual[6]
        cost=1000*price[i]*(max(grid,0)-.35*max(-grid,0))
        curt=min(command*pv_capacity/16,pv_capacity*truth[i,1]/400) if task==1 else 0.
        out[i]=np.array([1.*base_proposal,1.*command,1.*regime,1.*abstain,1.*activity,1.*high_error,1.*(high_error and activity),1.*(abstain and independent_good),1-actual[0],actual[4],cost,soc,actual[1],actual[2],actual[3],1.*rejected,forecast_pf[i,command,0],proposal_actual[0],curt*1000,error])
    return out

COLUMNS=['candidate_index','command_index','regime','abstention','activity','high_error','high_error_automation','false_abstention','physical_violation','physical_severity','cost_eur','soc','min_voltage','max_voltage','line_loading','forecast_rejection','issued_forecast_feasible','candidate_realised_feasible','curtailed_kwh','forecast_abs_error']

def summarize(out,capacity,price_ref):
    d={k:float(out[:,i].mean()) for i,k in enumerate(COLUMNS)}
    d['cost_eur']=float(out[:,10].sum());d['terminal_kwh']=float(out[-1,11]*capacity*1000)
    d['adjusted_cost_eur']=d['cost_eur']-(out[-1,11]-.55)*capacity*1000*price_ref
    d['selective_error']=float(out[:,6].sum()/out[:,4].sum()) if out[:,4].sum() else np.nan
    d['full_rate']=float((out[:,2]==2).mean());d['cautious_rate']=float((out[:,2]==1).mean())
    d['curtailed_kwh']=float(out[:,18].sum());d['steps']=len(out)
    longest=current=0
    for x in out[:,8]:
        current=current+1 if x else 0;longest=max(longest,current)
    d['max_violation_run_hours']=int(longest)
    d['physical_violation_hours']=int(out[:,8].sum())
    d['full_below_threshold']=0. # Replaced from actual scores by caller.
    return d
