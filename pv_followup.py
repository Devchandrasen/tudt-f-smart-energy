import json,hashlib,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from data_pipeline import ROOT,CONFIG,digest
from audit_experiments import A,P,pack,get_scores,run,scaled_pf,mode_rows
from physics import network,physics_tables

def job(origin):
    rows=[];modes=[];checked_states=0;checked_pairs=0;command_hash=hashlib.sha256()
    for vol in CONFIG['renewable_volatility_factors']:
        for scenario in CONFIG['stress_scenarios']:
            e=pack('DE_LU',origin,672,scenario,11,vol,solar=True);scores=get_scores(e)
            for cap in [3.6,7.2]:
                net=network('ieee33',1.,.6,cap);nominal=(physics_tables(e['pred'],net,'pv'),physics_tables(e['truth_next'],net,'pv'))
                for rating in CONFIG['line_rating_factors']:
                    pf=tuple(scaled_pf(x,rating) for x in nominal)
                    paired_outputs=[]
                    for method,mode in [('composite_feasible_recovery',1),('ungated_feasible_recovery',0)]:
                        out,s=run(e,pf,net,scores['composite'],P['low'],P['high'],mode,task=1,pv_fallback=2)
                        meta={'window':origin,'scenario':scenario,'seed':11,'identical_seed_labels':5,'volatility':vol,'rating':rating,'task':'solar','cell':f'solar{cap}','load_multiplier':.6,'capacity_mw':cap,'method':method}
                        rows.append({**meta,**s});modes.extend(mode_rows(out,meta))
                        paired_outputs.append(out)
                    np.testing.assert_array_equal(paired_outputs[0][:,1],paired_outputs[1][:,1])
                    np.testing.assert_allclose(paired_outputs[0][:,8:15],paired_outputs[1][:,8:15],rtol=0,atol=0)
                    command_hash.update(paired_outputs[0][:,1].astype(np.int16).tobytes());checked_pairs+=1;checked_states+=len(e['pred'])
    stem=A/f'followup_{origin}';pd.DataFrame(rows).to_csv(str(stem)+'_summary.csv',index=False);pd.DataFrame(modes).to_csv(str(stem)+'_modes.csv',index=False)
    result={'window':origin,'runs':len(rows),'amendment_sha256':digest(ROOT/'pv_followup_amendment.json'),'command_equality_pairs':checked_pairs,'command_equality_states':checked_states,'matched_commands_sha256':command_hash.hexdigest()}
    (A/f'followup_{origin}_manifest.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rerun',action='store_true');args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(job,o) for o in CONFIG['windows'] if args.rerun or not (A/f'followup_{o}_manifest.json').exists() or 'command_equality_states' not in json.loads((A/f'followup_{o}_manifest.json').read_text())]
        for fut in as_completed(futures):print(json.dumps(fut.result()),flush=True)
