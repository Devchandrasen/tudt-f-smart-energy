"""Bounded external disturbance check on publicly shared RTE waveforms.

This is NOT a replacement for a PMU sensor-quality benchmark. Only jump and
roughness proxies have a defensible mapping. No timestamps/forecast labels are
invented. The third-party 300-record annotation sample is itself selected.
"""
from pathlib import Path
import json,hashlib,concurrent.futures,requests
import numpy as np
import pandas as pd
from data_pipeline import ROOT

SOURCE='https://raw.githubusercontent.com/pashaeimehran/fault-inception-detection/main/'
MEDIA='https://media.githubusercontent.com/media/pashaeimehran/fault-inception-detection/main/verification_data/'
DATA=ROOT/'data/rte';DATA.mkdir(exist_ok=True)

def download_record(item):
    event=item['event'];p=DATA/f'event_{event:05d}.npz';meta=DATA/f'event_{event:05d}.json'
    if p.exists() and meta.exists():return json.loads(meta.read_text())
    url=MEDIA+f'event_{event:05d}.csv';r=requests.get(url,timeout=(30,120));r.raise_for_status()
    import io
    frame=pd.read_csv(io.BytesIO(r.content),usecols=['t','Ia','Ib','Ic','Va','Vb','Vc'])
    # Ignore all third-party derived features and their model's predicted onset.
    arrays=frame[['t','Ia','Ib','Ic','Va','Vb','Vc']].to_numpy()
    np.savez_compressed(p,observations=arrays)
    result={**item,'url':url,'raw_csv_sha256':hashlib.sha256(r.content).hexdigest(),'rows':len(frame)}
    meta.write_text(json.dumps(result,indent=2));return result

def proxies(event):
    a=np.load(DATA/f'event_{event:05d}.npz')['observations'];v=a[:,1:]
    # Disjoint 20-ms RMS cycles, computable only after a cycle has ended.
    n=len(v)//128;v=v[:n*128].reshape(n,128,6)
    rms=np.sqrt((v*v).mean(axis=1));base=np.maximum(np.median(rms[:4],axis=0),1.)
    level=rms/base
    jump=np.r_[0.,np.abs(np.diff(level,axis=0)).mean(axis=1)]
    rough=pd.Series(jump).rolling(6,min_periods=1).mean().to_numpy()
    times=a[127:n*128:128,0]
    return times,jump,rough

def main():
    annotation=ROOT/'data/rte_manual_annotations.json'
    if not annotation.exists():
        r=requests.get(SOURCE+'data/processed/labelstudio_export.json',timeout=60);r.raise_for_status();annotation.write_text(r.text)
    source=json.loads(annotation.read_text());eligible=[]
    for task in source:
        results=task['annotations'][0]['result'];choices=[x['value']['choices'][0] for x in results if x['type']=='choices']
        onset=[x['value']['start'] for x in results if 'true_fault_start' in x.get('value',{}).get('timeserieslabels',[])]
        fault=bool(onset);event=int(task['data']['event_idx'])
        if choices and choices[0]=='no_fault':fault=False;onset=[]
        elif not onset:continue
        if fault and onset[0]<.1:continue
        eligible.append({'event':event,'fault':fault,'onset_s':onset[0] if onset else None,'annotation_choice':choices[0] if choices else 'unrecorded'})
    # Prespecified bounded selection: 20 development and 40 test recordings per
    # class, ordered by a seeded hash of event identifier, independent of proxies.
    chosen=[]
    for cls in [False,True]:
        candidates=sorted([x for x in eligible if x['fault']==cls],key=lambda x:hashlib.sha256(f"22355:{x['event']}".encode()).hexdigest())
        if len(candidates)<60:raise ValueError('fewer than 60 eligible records per class')
        for i,item in enumerate(candidates[:60]):chosen.append({**item,'split':'development' if i<20 else 'test'})
    selection={'source_annotation_url':SOURCE+'data/processed/labelstudio_export.json','annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),'selection_rule':'20 development + 40 test per class by SHA256(22355:event), exclude positive onset <0.1s; frozen before computing proxies','records':chosen,'scope':'naturally occurring physical disturbances, not sensor-fault or smart-energy dispatch validation'}
    (DATA/'selection.json').write_text(json.dumps(selection,indent=2))
    print('Downloading',len(chosen),'RTE records',flush=True)
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(download_record,chosen):
            completed.append(result)
            if len(completed)%20==0:print('RTE downloaded',len(completed),flush=True)
    thresholds=[]
    for j,name in enumerate(['jump','roughness']):
        clean=np.concatenate([proxies(x['event'])[j+1][5:] for x in chosen if x['split']=='development' and not x['fault']])
        thresholds.append(float(np.quantile(clean,.99)))
    results=[]
    for item in chosen:
        times,jump,rough=proxies(item['event'])
        for j,(name,values) in enumerate([('jump',jump),('roughness',rough)]):
            alarm=values>thresholds[j];alarm[:5]=False
            if item['fault']:
                hit=np.where(alarm&(times>=item['onset_s']))[0]
                # Event-level classification uses any alarm over the same full
                # recording duration in both classes; onset delay is separate.
                delay=float(times[hit[0]]-item['onset_s']) if len(hit) else None
            else:delay=None
            results.append({**item,'detector':name,'threshold':thresholds[j],'alarm':bool(alarm.any()),'detection_delay_s':delay,'max_proxy':float(values.max()),'cycles':len(values)})
    out=ROOT/'results';pd.DataFrame(results).to_csv(out/'real_event_records.csv',index=False)
    rows=[]
    for name,g in pd.DataFrame(results).query("split=='test'").groupby('detector'):
        true=g.fault.to_numpy();alarm=g.alarm.to_numpy()
        rows.append({'detector':name,'tp':int((true&alarm).sum()),'fp':int((~true&alarm).sum()),'fn':int((true&~alarm).sum()),'tn':int((~true&~alarm).sum()),'median_detection_delay_s':g.loc[g.fault,'detection_delay_s'].median()})
    pd.DataFrame(rows).to_csv(out/'real_event_summary.csv',index=False)
    (out/'real_event_manifest.json').write_text(json.dumps({'status':'completed external disturbance transfer check','n_development':40,'n_test':80,'threshold_fit':'development no-fault recordings only','thresholds':thresholds,'missing_capabilities':['sensor-fault ground truth','timestamp age evidence','forecast/state mismatch evidence','battery/PV dispatch outcomes'],'records':completed},indent=2))
    for url,name in [('https://raw.githubusercontent.com/rte-france/digital-fault-recording-database/main/LICENCE.md','RTE_ODbL.txt'),(SOURCE+'LICENSE','RTE_annotation_BSD.txt')]:
        r=requests.get(url,timeout=30);r.raise_for_status();(ROOT/'provenance'/name).write_text(r.text,encoding='utf-8')
    print(rows,flush=True)

if __name__=='__main__':main()
