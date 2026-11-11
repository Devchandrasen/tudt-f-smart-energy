"""Causal OPSD preparation, observable streams and chronological forecasting."""
from pathlib import Path
from dataclasses import dataclass
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'protocol.json').read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(region):
    dest = ROOT / 'data' / f'{region}_causal.csv'
    if dest.exists():
        return pd.read_csv(dest, parse_dates=['timestamp']).set_index('timestamp')
    prefix = region + '_'
    names = ['load_actual_entsoe_transparency','load_forecast_entsoe_transparency',
             'price_day_ahead','solar_generation_actual','wind_generation_actual']
    cols = ['utc_timestamp'] + [prefix+x for x in names]
    raw = pd.read_csv(ROOT/'data/opsd_2020-10-06.csv',usecols=cols,parse_dates=['utc_timestamp'])
    raw.columns=['timestamp','load','load_forecast','price','solar','wind']
    raw=raw.set_index('timestamp').sort_index()
    raw=raw.loc['2018-10-01':'2020-09-30'].copy()
    raw['raw_available']=raw[['load','solar','wind']].notna().all(axis=1).astype(int)
    # No backfill; no interpolation that sees a future sample. Initial unknown rows excluded.
    raw[['load','load_forecast','price','solar','wind']]=raw[['load','load_forecast','price','solar','wind']].ffill()
    raw=raw.dropna()
    fit=raw.loc[:'2019-09-30']
    ls=float(fit.load.median()); rs=float((fit.solar+fit.wind).quantile(.95))
    out=pd.DataFrame(index=raw.index)
    out['load']=600*raw.load/ls
    out['ren']=260*raw.solar/rs+140*raw.wind/rs
    out['price']=raw.price/1000
    out['load_forecast']=600*raw.load_forecast/ls
    out['raw_available']=raw.raw_available
    out.to_csv(dest,index_label='timestamp')
    (ROOT/'data'/f'{region}_scaling.json').write_text(json.dumps({'load_mw_median':ls,'renewable_mw_q95':rs,'fit_end':'2019-09-30','input_sha256':digest(ROOT/'data/opsd_2020-10-06.csv')},indent=2))
    return out

def features(values, times, price, lf):
    v=pd.DataFrame(values,columns=['l','r'],index=times)
    a=[v.to_numpy(),np.c_[price,lf]]
    for lag in [1,24,168]:a.append(v.shift(lag).to_numpy())
    a.append(v.rolling(24,min_periods=1).mean().to_numpy())
    h=times.hour.to_numpy();d=times.dayofyear.to_numpy()
    a.append(np.c_[np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),np.sin(2*np.pi*d/366),np.cos(2*np.pi*d/366)])
    return np.nan_to_num(np.concatenate(a,axis=1))

def conformal_q(x,alpha=.1):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not len(x):return 0.
    level=min(1,np.ceil((len(x)+1)*(1-alpha))/len(x))
    return float(np.quantile(x,level,method='higher'))

@dataclass
class Forecast:
    model: object
    q: np.ndarray
    qb: np.ndarray
    residuals: np.ndarray
    jump: float
    rough: float
    error: float
    price_ref: float
    fit_end: str
    calibration_end: str

def train(df,origin):
    origin=pd.Timestamp(origin,tz='UTC')
    x=features(df[['load','ren']].to_numpy(),df.index,df.price.to_numpy(),df.load_forecast.to_numpy())
    y=df[['load','ren']].shift(-1).to_numpy()
    # A training row is allowed only if its target has been revealed before the origin.
    eligible=(df.index+pd.Timedelta(hours=1)<origin)&(df.index>=pd.Timestamp('2018-10-08',tz='UTC'))
    split=origin-pd.Timedelta(hours=672)
    fit=eligible&(df.index<split);cal=eligible&(df.index>=split)
    if fit.sum()<500 or cal.sum()<100:raise ValueError('insufficient chronological history')
    rf=RandomForestRegressor(n_estimators=60,min_samples_leaf=8,max_features=.75,random_state=42,n_jobs=2)
    rf.fit(x[fit],y[fit]); pred,lo,hi=rf_predict(rf,x[cal])
    excess=np.maximum(np.maximum(y[cal]-hi,lo-y[cal]),0)
    q=np.array([conformal_q(excess[:,j]) for j in range(2)])
    # Non-overlapping daily maxima: conservative simultaneous-block comparator.
    k=len(excess)//24;mx=excess[:k*24].reshape(k,24,2).max(axis=1)
    qb=np.array([conformal_q(mx[:,j]) for j in range(2)])
    hist=df.loc[fit,['load','ren']].to_numpy();delta=np.diff(hist,axis=0)
    jump=np.abs(delta).sum(axis=1)
    rough=pd.Series(jump).rolling(6,min_periods=1).mean().to_numpy()
    err=np.abs((y[cal,0]-y[cal,1])-(pred[:,0]-pred[:,1]))
    return Forecast(rf,q,qb,excess,np.quantile(jump,.99),np.quantile(rough,.95),np.quantile(err,.95),float(df.loc[fit,'price'].median()),str(df.index[fit][-1]),str(df.index[cal][-1]+pd.Timedelta(hours=1)))

def rf_predict(rf,x):
    trees=np.array([t.predict(x) for t in rf.estimators_])
    return trees.mean(axis=0),np.quantile(trees,.05,axis=0),np.quantile(trees,.95,axis=0)

def observe(df,origin,hours,scenario,seed,volatility=1.):
    origin=pd.Timestamp(origin,tz='UTC')
    idx=df.index.get_indexer(pd.date_range(origin,periods=hours+1,freq='h'))
    if (idx<0).any():raise ValueError('missing window timestamps')
    start=idx[0];stop=idx[-1]+1
    first=max(0,start-192)
    base=df.iloc[first:stop].copy(); truth=base[['load','ren']].to_numpy().copy()
    if volatility!=1:
        smooth=pd.Series(truth[:,1]).rolling(24,min_periods=1).mean().to_numpy()
        truth[:,1]=np.maximum(0,smooth+volatility*(truth[:,1]-smooth))
    loc=start-first;n=len(base); eval_idx=np.arange(loc,loc+hours)
    rng=np.random.default_rng(seed)
    fault=np.zeros(n,bool); event=np.zeros(n,bool); age=np.zeros(n); available=base.raw_available.to_numpy().astype(bool)
    src=np.arange(n); received=truth.copy()
    active=np.arange(loc,loc+hours)
    if scenario=='missing':
        bad=rng.random(hours)<.12; fault[active[bad]]=True;available[active[bad]]=False
        received[~available]=np.nan
    elif scenario=='delayed':
        src[active]=active-2;age[active]=2;received[active]=truth[src[active]];fault[active]=True
    elif scenario=='noisy':
        sigma=np.std(truth[:loc],axis=0)*[.08,.12]
        received[active]=np.maximum(0,truth[active]+rng.normal(0,sigma,(hours,2)));fault[active]=True
    elif scenario=='disturbance':
        block=active[int(.45*hours):int(.45*hours)+24]
        truth[block]*=[1.28,.55];received[block]=truth[block];event[block]=True
    elif scenario=='fdi_like':
        bad=active[rng.random(hours)<.06];received[bad]*=rng.choice([.55,1.55],size=(len(bad),2));fault[bad]=True
    elif scenario in ['bias','slow_ramp']:
        bad=active[int(.25*hours):int(.75*hours)]
        amp=np.ones(len(bad)) if scenario=='bias' else np.linspace(0,1,len(bad))
        received[bad,0]*=1+.03*amp;received[bad,1]*=1-.03*amp;fault[bad]=amp>0
    elif scenario!='clean':raise ValueError(scenario)
    # Derived lags are rebuilt from what was actually received.
    received=pd.DataFrame(received).ffill().to_numpy()
    for i in range(n):
        if not available[i] and i>0:src[i]=src[i-1];age[i]=i-src[i]
    x=features(received,base.index,base.price.to_numpy(),base.load_forecast.to_numpy())
    return {'x':x,'truth':truth,'received':received,'age':age,'available':available,'source':src,'fault':fault,'event':event,'indices':eval_idx,'times':base.index,'price':base.price.to_numpy(),'scenario':scenario}

def evidence(obs,forecast,method='split',decay=2.,lookback=24,width_scale=750.,mix=.5,smooth_window=6):
    pred,lo,hi=rf_predict(forecast.model,obs['x'])
    n=len(pred); scores=np.zeros((n,4));det=np.zeros((n,4),bool)
    widths=np.zeros((n,2)); intervals_lo=np.zeros((n,2));intervals_hi=np.zeros((n,2))
    q=forecast.q.copy();alphas=np.array([.1,.1]);completed=[];last_source=-1
    jumps=np.r_[0,np.abs(np.diff(obs['received'],axis=0)).sum(axis=1)]
    for i in range(n):
        s=int(obs['source'][i])
        if obs['available'][i] and s>last_source and s>0:
            # Forecast targeting s was issued at s-1. Never evaluate prediction i with target i+1 here.
            value=obs['received'][i]
            err=abs((value[0]-value[1])-(pred[s-1,0]-pred[s-1,1]))
            hit=(value>=intervals_lo[s-1])&(value<=intervals_hi[s-1])
            excess=np.maximum(np.maximum(value-hi[s-1],lo[s-1]-value),0)
            completed.append((err,float(hit.mean()),excess));last_source=s
            if method=='adaptive':
                alphas=np.clip(alphas+.01*(.1-(~hit).astype(float)),.01,.5)
                pool=np.r_[forecast.residuals[-672:],np.array([z[2] for z in completed[-168:]])]
                q=np.array([conformal_q(pool[:,j],alphas[j]) for j in range(2)])
        if method=='block_max':q=forecast.qb
        low=np.maximum(0,lo[i]-q);high=hi[i]+q
        intervals_lo[i]=low;intervals_hi[i]=high;widths[i]=high-low
        recent=completed[-lookback:]
        err=np.mean([z[0] for z in recent]) if recent else forecast.error
        coverage=np.mean([z[1] for z in recent]) if recent else .9
        fresh=np.exp(-obs['age'][i]/decay)
        mismatch=np.exp(-err/max(forecast.error,1e-6))
        rough=jumps[max(0,i-smooth_window+1):i+1].mean()
        rough_score=np.exp(-rough/max(forecast.rough,1e-6))
        jump_score=np.exp(-jumps[i]/max(forecast.jump,1e-6))
        complete=obs['available'][max(0,i-5):i+1].mean()
        valid=float(np.isfinite(obs['received'][i]).all() and (obs['received'][i]>=0).all())
        sharp=np.exp(-widths[i].sum()/width_scale)
        scores[i]=[mix*fresh+(1-mix)*mismatch,.4*complete+.3*rough_score+.3*valid,mix*sharp+(1-mix)*coverage,.8*mix*jump_score+.8*(1-mix)*mismatch+.2*valid]
        det[i]=[obs['age'][i]>1,rough>forecast.rough,jumps[i]>forecast.jump,err>forecast.error]
    idx=obs['indices']
    return {'pred':pred[idx],'lo':intervals_lo[idx],'hi':intervals_hi[idx],'scores':scores[idx],'detectors':det[idx],
            'truth_next':obs['truth'][idx+1], 'truth_now':obs['truth'][idx], 'fault':obs['fault'][idx], 'event':obs['event'][idx],
            'price':obs['price'][idx], 'times':obs['times'][idx].astype(str).to_numpy(),'age':obs['age'][idx],
            'available':obs['available'][idx],'source':obs['source'][idx], 'width':widths[idx].sum(axis=1)}
