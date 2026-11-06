"""Build the price-row calendar used to purge complete rolling-training targets."""
import argparse,csv,hashlib
from pathlib import Path
import numpy as np
from hs300_dataset import _code

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('calendar.npy'));args=p.parse_args()
    with np.load(args.package/'data/train.npz') as values:codes=values['stock_codes'].tolist()
    files={_code(str(f)):f for f in (args.package/'data/numerical/HS300_time_series').rglob('*.csv')}
    calendars=[]
    for code in codes:
        with files[code].open(encoding='utf-8-sig',newline='') as stream:
            days=[int(r['Date'][:10].replace('-','')) for r in csv.DictReader(stream)]
        calendars.append(np.asarray([d for d in days if 20200331<=d<=20250331],np.int32))
    np.save(args.output,np.concatenate(calendars))
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())

if __name__=='__main__':main()
