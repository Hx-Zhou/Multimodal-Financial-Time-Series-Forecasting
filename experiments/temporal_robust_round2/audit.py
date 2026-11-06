"""Audit irreversible clipping in the inherited financial caches; no test scoring."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();s=np.load(args.cache/'train_starts.npy');d=np.load(args.cache/'train_dates.npy')
    calendar=np.load(Path(__file__).with_name('calendar.npy'));ends=calendar[s+119]
    result={}
    for filename in ('reports_lag180.npy','reports_ratios_lag180.npy'):
        a=np.load(args.cache/filename,mmap_mode='r');counts={}
        for scope,begin,end in [('rolling_A',20230103,20230228),('rolling_B',20230703,20230831)]:
            for subset,mask in [('train',ends<begin),('val',(d>=begin)&(d<=end))]:
                values=a[s[mask]+95,:24]
                counts[scope+'_'+subset]={'clipped_fields':int((np.abs(values)>=9.9999).sum()),'fields':int(values.size)}
        result[filename]=counts
    record={'counts':result,'limitation':'Inherited z-score caches clip at +/-10 before inverse transform. Any clipped financial values retain dependence on the original official-training calibration; affected rolling experiments are approximate diagnostics, not a fresh blind validation. Full original official training remains strictly earlier than public development.',
            'policy':'Disclose this limitation; do not change current configuration selection or test timing after seeing test scores.'}
    args.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
