"""Align recorded detector flags, factors and measured values in the COCOON PHIL profiles."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

TOLERANCE = pd.Timedelta(milliseconds=10)


def read_test(archive, fdd_path, factor_path):
    raw_alerts=archive.read(fdd_path)
    raw_factors=archive.read(factor_path)
    alerts=json.loads(raw_alerts)['test_details']
    factors=json.loads(raw_factors)['test_details']
    folder=fdd_path.rsplit('/',1)[0]
    assert alerts.keys()==factors.keys()
    results=[]
    for test, entry in alerts.items():
        measure_path=f'{folder}/Tabla_DB_MEAS/DB_INPUT_{test}.csv'
        raw_meas=archive.read(measure_path)
        data=pd.read_csv(io.BytesIO(raw_meas),sep=None,engine='python')
        data.index=pd.DatetimeIndex(pd.to_datetime(data.Tag0_Value,utc=True,format='mixed'))
        data=data.sort_index()
        assert data.index.is_unique
        ft=factors[test]
        ts=pd.DatetimeIndex(pd.to_datetime(entry['time'],utc=True,format='mixed'))
        tsf=pd.DatetimeIndex(pd.to_datetime(ft['time'],utc=True,format='mixed'))
        assert ts.is_unique and ts.is_monotonic_increasing and ts.equals(tsf)
        assert len(ts)==len(ft['Factors'])==len(entry['FDD'])
        idx=data.index.get_indexer(ts,method='nearest',tolerance=TOLERANCE)
        matched=idx>=0
        assert len(np.unique(idx[matched]))==matched.sum()
        src=np.zeros(len(ts),dtype=bool)
        unknown_flags=set()
        for i,items in enumerate(entry['FDD']):
            for item in items:
                if isinstance(item,list) and len(item)==3 and str(item[0]).upper() in {'V','P','Q','I'}:
                    src[i]=True
                else:unknown_flags.add(str(item)[:60])
        if unknown_flags:raise ValueError(f'Unexpected source flag encoding: {unknown_flags}')
        changes=0
        factor_deltas=0
        value_deltas=0
        prior_positive=0
        valid_transitions=0
        unaligned=0
        no_comparable_fields=0
        max_clock_error_ms=abs(data.index[idx[matched]]-ts[matched]).max().total_seconds()*1000 if matched.any() else None
        for i in range(1,len(ts)):
            if not (matched[i] and matched[i-1]):
                unaligned+=1
                continue
            x,y=ft['Factors'][i-1:i+1]
            tags=[k for k in x.keys() & y.keys() if k in data.columns]
            if not tags:
                no_comparable_fields+=1
                continue
            a=data.iloc[idx[i-1]][tags].astype(float).to_numpy()
            b=data.iloc[idx[i]][tags].astype(float).to_numpy()
            old=np.array([x[k] for k in tags],dtype=float)
            new=np.array([y[k] for k in tags],dtype=float)
            valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(old)&np.isfinite(new)
            if not valid.any():
                no_comparable_fields+=1
                continue
            valid_transitions+=1
            if src[i-1]:
                prior_positive+=1
                f=valid & (np.abs(new-old)>1e-9)
                v=valid & (np.abs(b-a)>1e-9)
                factor_deltas+=int(f.any())
                value_deltas+=int(v.any())
                changes+=int((f&v).any())
        assert valid_transitions+unaligned+no_comparable_fields==len(ts)-1
        results.append(dict(source_files=[fdd_path,factor_path,measure_path],
            source_sha256=[hashlib.sha256(x).hexdigest() for x in (raw_alerts,raw_factors,raw_meas)],
            samples=len(ts),measurements_matched=int(matched.sum()),source_flagged=int(src.sum()),valid_transitions=valid_transitions,
            prior_flagged=prior_positive, next_factor_changed=factor_deltas, next_value_changed=value_deltas,
            same_channel_both_changed=changes, unaligned_adjacent_steps=unaligned, no_common_finite_fields=no_comparable_fields,
            max_clock_error_ms=max_clock_error_ms))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Energy_Community.zip from Zenodo record 21476677")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.archive.open("rb") as f:
        if hashlib.file_digest(f, "md5").hexdigest() != "add013334e22a484f3835d834e39cd34":
            raise ValueError("Input differs from Zenodo record 21476677")
    records = []
    with ZipFile(args.archive) as archive:
        paths = archive.namelist()
        alerts = sorted(p for p in paths if "/PHIL/Profile_Test/Smart/" in p and p.endswith("_FDD.json"))
        for path in alerts:
            directory = path.rsplit("/", 1)[0] + "/"
            factor, = [p for p in paths if p.startswith(directory) and p.endswith("_Factor_smart.json")]
            records.extend(read_test(archive, path, factor))
    if len(records) != 6:
        raise ValueError("Expected six profile runs")
    counts = pd.DataFrame(records).drop(columns=["max_clock_error_ms","source_files","source_sha256"])
    result = {
        "archive_doi": "10.5281/zenodo.21476677", "clock_tolerance_ms": 10,
        "absolute_change_threshold": 1e-9, "profile_runs": len(records),
        "runs_with_postflag_signal_and_factor_changes": sum(r["same_channel_both_changed"] > 0 for r in records),
        "maximum_matched_clock_difference_ms": max(r["max_clock_error_ms"] for r in records),
        "sums": counts.sum().astype(int).to_dict(), "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
