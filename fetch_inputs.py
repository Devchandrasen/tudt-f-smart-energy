"""Optional raw OPSD acquisition; the numerical archive includes causal extracts."""
import hashlib,json,os
import requests
from data_pipeline import ROOT

def main():
    path=ROOT/'data/opsd_2020-10-06.csv';path.parent.mkdir(exist_ok=True)
    expected=json.loads((ROOT/'data/DE_LU_scaling.json').read_text())['input_sha256']
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('existing raw file differs from archived hash; retained unchanged')
        print('Verified existing OPSD release');return
    temporary=path.with_suffix('.download')
    url='https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv'
    response=requests.get(url,stream=True,timeout=(30,120));response.raise_for_status();h=hashlib.sha256()
    with temporary.open('wb') as f:
        for chunk in response.iter_content(4*1024*1024):f.write(chunk);h.update(chunk)
    if h.hexdigest()!=expected:raise ValueError('download hash differs; unverified download retained separately')
    os.replace(temporary,path);print('Downloaded and verified OPSD 2020-10-06')

if __name__=='__main__':main()
