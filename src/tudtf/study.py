"""Causal dispatch study. All network checks are performed before command selection.

No test targets or fault labels enter the trust gate. Independent plant outcomes
are evaluated separately. Run with --quick for a short execution check.
"""
from pathlib import Path
import argparse, json, time, platform, copy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import simbench as sb
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = None
STEPS = np.arange(-1, 1.01, .25)
MODES = ["Ungated", "Metadata gate", "Agreement gate", "Trust gate"]
SEEDS = [11, 29, 47]
SCENARIOS = ["Clean", "Missing", "Delayed", "Noisy", "Bias", "Disturbance"]

def profiles():
    net = sb.get_simbench_net("1-MV-rural--0-sw")
    load = net.profiles["load"]
    ren = net.profiles["renewables"]
    # Aggregate benchmark profiles using installed power as the weight.
    lp = sum(float(r.p_mw)*load[str(r.profile)+"_pload"].to_numpy()
             for r in net.load.itertuples()) / net.load.p_mw.sum()
    gp = sum(float(r.p_mw)*ren[str(r.profile)].to_numpy()
             for r in net.sgen.itertuples()) / net.sgen.p_mw.sum()
    lp = lp.reshape(-1,4).mean(axis=1)
    gp = gp.reshape(-1,4).mean(axis=1)
    dates = pd.date_range("2016-01-01",periods=len(lp),freq="h")
    if OUT is not None:
        pd.DataFrame({"time":dates,"load_factor":lp,"generation_factor":gp}).to_csv(OUT/"hourly_profiles.csv",index=False)
    return dates, np.c_[lp,gp], net

def features(x, dates, indices):
    indices=np.asarray(indices)
    hour=dates.hour.to_numpy()[indices]
    doy=dates.dayofyear.to_numpy()[indices]
    return np.c_[x[indices],x[indices-1],x[indices-24],
                 np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                 np.sin(2*np.pi*doy/366),np.cos(2*np.pi*doy/366)]

class Grid:
    def __init__(self,name,sbnet):
        self.name=name
        if name=="IEEE 33":
            self.net=pn.case33bw()
            # Source test case supplies no usable thermal ratings.
            self.net.line.max_i_ka=.4
            for bus in [17,24,32]:
                pp.create_sgen(self.net,bus,p_mw=.6,q_mvar=0)
            self.battery_bus=17
            self.pmax=.8; self.capacity=3.2
        else:
            self.net=copy.deepcopy(sbnet)
            pp.runpp(self.net,numba=False)
            self.battery_bus=int(self.net.res_bus.vm_pu.idxmax())
            self.pmax=2.; self.capacity=8.
        self.battery=pp.create_sgen(self.net,self.battery_bus,p_mw=0.,q_mvar=0.,name="Study battery")
        self.load_p=self.net.load.p_mw.copy()
        self.load_q=self.net.load.q_mvar.copy()
        self.gen_p=self.net.sgen.p_mw.copy(); self.gen_p.loc[self.battery]=0
        self.cache={}
        self._initial=True
        self.set_state(.6,.4,0)
        base=self.solve(raw=True)
        sensitivities=[]
        for dl,dg,du in [(1e-3,0,0),(0,1e-3,0),(0,0,1e-3)]:
            self.set_state(.6+dl,.4+dg,du)
            sensitivities.append((self.solve(raw=True)-base)/1e-3)
        self.linear_base=base
        self.sens=np.array(sensitivities)
        self.nb=len(self.net.bus)
        self.nl=len(self.net.line)
        self.nt=len(self.net.trafo)

    def set_state(self,l,g,u):
        self.net.load.loc[:,"p_mw"]=self.load_p*l
        self.net.load.loc[:,"q_mvar"]=self.load_q*l
        self.net.sgen.loc[:,"p_mw"]=self.gen_p*g
        self.net.sgen.at[self.battery,"p_mw"]=u

    def solve(self,raw=False):
        try:
            pp.runpp(self.net,numba=False,init="auto" if self._initial else "results",
                     recycle=None if self._initial else {"bus_pq":True,"trafo":False,"gen":True},max_iteration=30)
            self._initial=False
            a=np.r_[self.net.res_bus.vm_pu.to_numpy(),
                    self.net.res_line.loading_percent.to_numpy()/100,
                    self.net.res_trafo.loading_percent.to_numpy()/100,
                    self.net.res_ext_grid.p_mw.sum(),self.net.res_line.pl_mw.sum()+self.net.res_trafo.pl_mw.sum()]
            return a if raw else self.metrics(a)
        except pp.LoadflowNotConverged:
            return np.full(len(self.net.bus)+len(self.net.line)+len(self.net.trafo)+2,np.nan) if raw else (0,2,np.nan,np.nan,np.nan,np.nan,np.nan)

    def metrics(self,a):
        if not np.isfinite(a).all(): return (0,2,np.nan,np.nan,np.nan,np.nan,np.nan)
        v=a[:self.nb]; thermal=a[self.nb:-2]
        severity=max(0,.95-v.min())+max(0,v.max()-1.05)+max(0,thermal.max()-1)
        feasible=int(v.min()>=.95-1e-7 and v.max()<=1.05+1e-7 and thermal.max()<=1+1e-7)
        return feasible,severity,float(a[-2]),float(a[-1]),float(v.min()),float(v.max()),float(thermal.max())

    def evaluate(self,l,g,u):
        key=(round(float(l),10),round(float(g),10),round(float(u),10))
        if key not in self.cache:
            self.set_state(l,g,u); self.cache[key]=self.solve()
        return self.cache[key]

    def approximate(self,l,g,u):
        return self.metrics(self.linear_base+np.array([l-.6,g-.4,u])@self.sens)

def next_soc(soc,u,cap):
    return soc-u/(.94*cap) if u>=0 else soc-.94*u/cap

def energy_cost(gridpower,u,price):
    # EUR, MW, MWh and one-hour control interval. Stored energy is valued at
    # EUR 130/MWh for the terminal correction; throughput costs EUR 10/MWh.
    ds= -u/.94 if u>=0 else -.94*u
    return 1000*(price*max(gridpower,0)-.04*max(-gridpower,0)+.01*abs(u)-.13*ds)

def price_at(hour):
    return .07 if hour<7 else (.24 if 17<=hour<22 else .15)

def candidate(grid,pred,soc,price):
    choices=[]
    for ratio in STEPS:
        u=ratio*grid.pmax
        if .15-1e-10<=next_soc(soc,u,grid.capacity)<=.9+1e-10:
            m=grid.approximate(*pred,u)
            # Soft-constraint ranking only; acceptance uses the AC equations.
            j=energy_cost(m[2],u,price)+1e5*m[1]
            choices.append((j,u))
    choices.sort()
    # Every accepted candidate has an AC check before the command is issued.
    for _,u in choices:
        if grid.evaluate(*pred,u)[0]: return u,True
    # A hold command is issued when no feasible action exists in the grid.
    # Its feasibility flag is retained; hold is never called a safe action.
    return 0.,bool(grid.evaluate(*pred,0.)[0])

def corrupt(true,scenario,seed,start):
    rng=np.random.default_rng(seed+start)
    obs=true.copy(); present=np.ones(len(true),dtype=bool); age=np.zeros(len(true))
    ref=np.clip(true+rng.normal(0,.01,size=true.shape),0,None)
    if scenario=="Missing":
        present=rng.random(len(true))>=.12
    elif scenario=="Delayed":
        obs[2:]=true[:-2]; age[:]=2
    elif scenario=="Noisy":
        obs=np.clip(true+rng.normal(0,.08,size=true.shape),0,None)
    elif scenario=="Bias":
        # Two six-hour bursts in each 72-hour evaluation block.
        for offset in [12,48]:
            a=24+offset; b=min(a+6,len(true)); obs[a:b,0]*=1.5; obs[a:b,1]*=.5
    if scenario=="Missing":
        for i in range(1,len(obs)):
            if not present[i]: obs[i]=obs[i-1]; age[i]=age[i-1]+1
    return obs,ref,present,age


def chronological_partitions(dates):
    """Keep both observation and next-hour target inside their partition."""
    indices = np.arange(len(dates))
    train_end = np.flatnonzero(dates >= pd.Timestamp("2016-09-01"))[0]
    cal_end = np.flatnonzero(dates >= pd.Timestamp("2016-11-01"))[0]
    train = indices[(indices >= 24) & (indices + 1 < train_end)]
    cal = indices[(indices >= train_end) & (indices + 1 < cal_end)]
    return train, cal


def evidence_channels(obs, ref, present, age, i, recent, q, cal_mae):
    """Use current meter readings and completed past forecast residuals only."""
    if recent:
        errs = np.array([abs(a - b) for a, b in recent])
        cover = np.mean(np.all(errs <= q, axis=1))
        mae = float(np.mean(errs))
    else:
        cover = .9
        mae = cal_mae
    maturity = .5 * min(1, cover / .9) + .5 * np.exp(-mae / (3 * cal_mae))
    freshness = np.exp(-age[i] / 2)
    completeness = float(np.mean(present[max(0, i - 5):i + 1]))
    agreement = np.exp(-float(np.mean(abs(obs[i] - ref[i]))) / .10)
    return np.array([freshness, completeness, maturity, agreement])


def gate_regime(trust, low=.65, high=.85):
    return "full" if trust >= high else ("cautious" if trust >= low else "hold")

def run_block(grid,dates,data,model,q,cal_mae,start,length,scenario,seed,weights,low,high):
    sl=slice(start-24,start+length+1)
    true=data[sl].copy()
    if scenario=="Disturbance":
        # Actual operating-condition change, also visible to both meters.
        true[24+24:24+48,0]*=1.28; true[24+24:24+48,1]*=.55
    obs,ref,present,age=corrupt(true,scenario,seed,start)
    localdates=dates[sl]
    predictions=model.predict(features(obs,localdates,np.arange(24,24+length)))
    predictions=np.clip(predictions,0,None)
    scores=[]; rows=[]; replay=[]
    soc={m:.55 for m in MODES}
    history=[]
    for k in range(length):
        i=24+k; pred=predictions[k]
        # Only earlier predictions and their reference observations are used.
        if k>0: history.append((predictions[k-1],ref[i]))
        recent=history[-24:]
        parts=evidence_channels(obs,ref,present,age,i,recent,q,cal_mae)
        freshness,completeness,maturity,agreement=parts
        trust=float(parts@weights)
        hour=int(localdates[i+1].hour); price=price_at(hour)
        for mode in MODES:
            prior=soc[mode]
            u,model_ok=candidate(grid,pred,prior,price)
            if mode=="Ungated": regime="full"
            elif mode=="Metadata gate": regime="full" if age[i]==0 and present[i] else "hold"
            elif mode=="Agreement gate": regime="full" if agreement>=.65 else "hold"
            else: regime=gate_regime(trust,low,high)
            command=u if regime=="full" else (.5*u if regime=="cautious" else 0.)
            # Scaling can remove useful support. The actual command is checked.
            check=grid.evaluate(*pred,command)
            if command!=0 and not check[0]: command=0.; check=grid.evaluate(*pred,0.)
            actual=grid.evaluate(*true[i+1],command)
            cost=energy_cost(actual[2],command,price)
            soc[mode]=next_soc(prior,command,grid.capacity)
            eligible=[r*grid.pmax for r in STEPS if .15-1e-10<=next_soc(prior,r*grid.pmax,grid.capacity)<=.9+1e-10]
            oracle=[(energy_cost(grid.evaluate(*true[i+1],a)[2],a,price),a) for a in eligible if grid.evaluate(*true[i+1],a)[0]]
            regret=max(0,cost-min(c for c,a in oracle)) if actual[0] and oracle else np.nan
            rows.append(dict(network=grid.name,block=str(dates[start]),scenario=scenario,seed=seed,hour=k,
                             controller=mode,trust=trust,freshness=freshness,completeness=completeness,
                             maturity=maturity,agreement=agreement,regime=regime,proposal_mw=u,
                             observed_load_factor=obs[i,0],observed_generation_factor=obs[i,1],
                             reference_load_factor=ref[i,0],reference_generation_factor=ref[i,1],
                             forecast_load_factor=pred[0],forecast_generation_factor=pred[1],
                             plant_load_factor=true[i+1,0],plant_generation_factor=true[i+1,1],
                             command_mw=command,soc=prior,next_soc=soc[mode],model_feasible=check[0],
                             plant_feasible=actual[0],severity=actual[1],grid_mw=actual[2],loss_mw=actual[3],
                             vmin=actual[4],vmax=actual[5],max_thermal_utilisation=actual[6],cost_eur=cost,regret_eur=regret))
        # Selective-risk analysis uses one common SOC and the same proposal.
        u,ok=candidate(grid,pred,.55,price)
        actual=grid.evaluate(*true[i+1],u)
        replay.append(dict(network=grid.name,block=str(dates[start]),scenario=scenario,seed=seed,hour=k,
                           trust=trust,agreement=agreement,freshness=freshness,completeness=completeness,
                           maturity=maturity,proposal_mw=u,model_feasible=int(ok),
                           plant_feasible=actual[0],severity=actual[1]))
    return rows,replay

def summarize(rows):
    d=pd.DataFrame(rows)
    d["violation"]=1-d.plant_feasible
    d["nonzero"]=abs(d.command_mw)>1e-9
    d["authorised"]=d.regime!="hold"
    block=d.groupby(["network","block","scenario","seed","controller"]).agg(
        violation_rate=("violation","mean"),nonzero_rate=("nonzero","mean"),
        authorised_rate=("authorised","mean"),severity=("severity","mean"),
        cost_eur=("cost_eur","sum"),regret_eur=("regret_eur","mean"),loss_mwh=("loss_mw","sum"),
        model_feasible_rate=("model_feasible","mean")).reset_index()
    block.to_csv(OUT/"block_results.csv",index=False)
    summary=block.groupby(["network","scenario","controller"]).mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    d.to_csv(OUT/"decisions.csv",index=False)
    return d,block,summary

def figures(d,block,replay):
    plt.rcParams.update({"font.family":"serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    palette=["#506779","#ac7849","#78965b","#234f96"]
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout="constrained")
    for ax,network in zip(axes,block.network.unique()):
        t=block[block.network==network]
        for j,mode in enumerate(MODES):
            vals=[100*t[(t.scenario==s)&(t.controller==mode)].violation_rate.mean() for s in SCENARIOS]
            ax.bar(np.arange(6)+(j-1.5)*.19,vals,.18,color=palette[j],label=mode)
        ax.set_xticks(range(6),SCENARIOS,rotation=25,ha="right"); ax.set_ylabel("Plant constraint violations (%)"); ax.set_title(network)
    axes[0].legend(fontsize=8,ncol=2)
    fig.savefig(OUT/"constraint_outcomes.pdf"); fig.savefig(OUT/"constraint_outcomes.png",dpi=220);plt.close(fig)
    r=pd.DataFrame(replay); curve=[]
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout="constrained")
    for ax,network in zip(axes,r.network.unique()):
        t=r[(r.network==network)&(r.scenario!="Clean")]
        for channel,color in [("trust","#234f96"),("agreement","#78965b")]:
            xx=[];yy=[]
            for threshold in np.linspace(0,.99,100):
                chosen=t[channel]>=threshold
                risk=100*(1-t.loc[chosen,"plant_feasible"]).mean() if chosen.any() else np.nan
                cov=100*chosen.mean();xx.append(cov);yy.append(risk)
                curve.append(dict(network=network,channel=channel,threshold=threshold,coverage_pct=cov,risk_pct=risk))
            ax.plot(xx,yy,color=color,label=channel.capitalize())
        ax.set_xlabel("Authorised proposals (%)");ax.set_ylabel("Conditional constraint violations (%)");ax.set_title(network);ax.legend(fontsize=9)
    fig.savefig(OUT/"risk_coverage.pdf");fig.savefig(OUT/"risk_coverage.png",dpi=220);plt.close(fig)
    pd.DataFrame(curve).to_csv(OUT/"risk_coverage.csv",index=False)
    r.to_csv(OUT/"proposal_replay.csv",index=False)

def main(argv=None):
    global OUT
    ap=argparse.ArgumentParser(description="Run the causal distribution-network benchmark.")
    ap.add_argument("--quick",action="store_true",help="Run 384 controller records as an execution check.")
    ap.add_argument("--output",type=Path,help="Output directory (default: results/full or results/quick).")
    args=ap.parse_args(argv)
    OUT=(args.output or Path("results")/("quick" if args.quick else "full")).resolve()
    from .verify import protect_reference
    protect_reference(OUT)
    OUT.mkdir(parents=True,exist_ok=True)
    dates,data,sbnet=profiles()
    train,cal=chronological_partitions(dates)
    model=RandomForestRegressor(n_estimators=60,min_samples_leaf=8,max_features=.75,random_state=42,n_jobs=2)
    model.fit(features(data,dates,train),data[train+1])
    cp=model.predict(features(data,dates,cal)); err=abs(cp-data[cal+1]);
    # A joint split-calibrated box. Scale factors are fitted on training data.
    # Serial dependence prevents an unconditional coverage guarantee.
    training_error=abs(model.predict(features(data,dates,train))-data[train+1])
    scales=np.maximum(np.quantile(training_error,.9,axis=0),1e-6)
    nonconformity=np.max(err/scales,axis=1)
    rank=min(len(cal),int(np.ceil((len(cal)+1)*.9)))
    q=scales*np.sort(nonconformity)[rank-1];cal_mae=float(err.mean())
    pred=model.predict(features(data,dates,np.arange(24,len(data)-1)))
    forecast={"calibration_mae_factor":cal_mae,"interval_halfwidth_factor":q.tolist(),"training_residual_scales":scales.tolist(),
              "calibration_joint_coverage":float(np.mean(np.all(err<=q,axis=1))),
              "test_mae_factor":float(abs(pred[np.arange(24,len(data)-1)>=7320]-data[25:][np.arange(24,len(data)-1)>=7320]).mean())}
    starts=[pd.Timestamp(f"2016-{month:02d}-{day:02d}") for month in [11,12] for day in [1,11,21]]
    starts=[int(np.flatnonzero(dates==t)[0]) for t in starts]
    if args.quick: starts=starts[:1]
    length=24 if args.quick else 72
    seeds=SEEDS[:1] if args.quick else SEEDS
    scenarios=SCENARIOS if not args.quick else ["Clean","Bias"]
    rows=[]; replay=[]; metadata=[]; begin=time.perf_counter()
    for name in ["IEEE 33","SimBench MV"]:
        grid=Grid(name,sbnet)
        metadata.append({"network":name,"buses":len(grid.net.bus),"lines":len(grid.net.line),"transformers":len(grid.net.trafo),
                         "battery_bus_index":grid.battery_bus,"pmax_mw":grid.pmax,"capacity_mwh":grid.capacity,
                         "load_mw":float(grid.load_p.sum()),"generation_mw":float(grid.gen_p.sum()),
                         "slack_voltage_pu":grid.net.ext_grid.vm_pu.tolist()})
        for start in starts:
            for scenario in scenarios:
                for seed in seeds:
                    rr,rp=run_block(grid,dates,data,model,q,cal_mae,start,length,scenario,seed,np.array([.30,.25,.25,.20]),.65,.85)
                    rows.extend(rr);replay.extend(rp)
                print(f"{name} {dates[start].date()} {scenario}: {len(rows)} decisions, {len(grid.cache)} AC states, {time.perf_counter()-begin:.1f}s",flush=True)
        pd.DataFrame(rows).to_csv(OUT/"decisions_checkpoint.csv",index=False)
    d,blocks,summary=summarize(rows);figures(d,blocks,replay)
    meta={"created_utc":pd.Timestamp.now(tz="UTC").isoformat(),"runtime_seconds":time.perf_counter()-begin,
          "python":platform.python_version(),"pandapower":pp.__version__,"simbench":sb.__version__,
          "networks":metadata,"forecast":forecast,"seeds":seeds,"block_hours":length,
          "blocks":[str(dates[i]) for i in starts],"scenarios":scenarios,"decision_count":len(d),
          "weights":[.30,.25,.25,.20],"thresholds":[.65,.85],"quick":args.quick}
    (OUT/"metadata.json").write_text(json.dumps(meta,indent=2))
    print(summary.to_string(index=False),flush=True)

if __name__=="__main__":main()
