import argparse,csv,json
from pathlib import Path
from datetime import date
import numpy as np
from hs300_dataset import _code
from experiments.dlinear_modal_common.data import FIELDS,consolidate,ordinal,normalize_reports

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    args=p.parse_args(); meta=json.loads((args.cache/'READY.json').read_text()); codes=meta['codes']
    calendars={}
    for file in (args.package/'data/numerical/HS300_time_series').rglob('*.csv'):
        with file.open(encoding='utf-8-sig',newline='') as stream:
            ds=[r['Date'][:10] for r in csv.DictReader(stream)]
        calendars[_code(str(file))]=np.array([ordinal(d) for d in ds if '2020-03-31'<=d<='2025-03-31'],np.int32)
    offsets=np.r_[0,np.cumsum([len(calendars[c]) for c in codes])];total=int(offsets[-1]);sources={}
    for file in (args.package/'data/tabular/hs300_tabular').rglob('*.jsonl'):
        if file.stem not in FIELDS:continue
        rows=[]
        for line in file.read_text().splitlines():
            v=json.loads(line);rows.extend(v if isinstance(v,list) else [v])
        sources.setdefault(_code(str(file)),{})[file.stem]=rows
    fitted=None
    for lag in (180,270):
        values=np.zeros((total,24),np.float32);valid=np.zeros((total,24),bool)
        for stock,code in enumerate(codes):
            block=slice(offsets[stock],offsets[stock+1]);days=calendars[code]
            for kind in ('balance_sheet','income'):
                records,_=consolidate(sources.get(code,{}).get(kind,[]),FIELDS[kind],lag)
                slots=[0,1,2,3] if kind=='balance_sheet' else [4,5]
                for record in records:
                    rows=np.flatnonzero(days>=record['available'])+offsets[stock]
                    if not len(rows):continue
                    for slot in slots:values[rows,slot]=0;valid[rows,slot]=False
                    v=record['values']
                    if kind=='balance_sheet':
                        den=v.get('total_assets');nums=[v.get('total_liab'),v.get('total_hldr_eqy_exc_min_int'),v.get('money_cap')]
                        if den is not None and den>0:
                            for slot,num in zip(slots[:3],nums):
                                if num is not None:values[rows,slot]=num/den;valid[rows,slot]=True
                            values[rows,3]=np.log1p(den/1e8);valid[rows,3]=True
                    else:
                        den=v.get('total_revenue')
                        if den is not None and den>0:
                            for slot,key in zip(slots,['n_income_attr_p','operate_profit']):
                                if v.get(key) is not None:values[rows,slot]=v[key]/den;valid[rows,slot]=True
        output, fitted=normalize_reports(values,valid,np.load(args.cache/'train_starts.npy')+95,fitted)
        old=np.load(args.cache/f'reports_lag{lag}.npy',mmap_mode='r')
        result=np.concatenate((output,valid.astype(np.float32),old[:,48:]),axis=1)
        assert result.shape==(total,54) and np.isfinite(result).all()
        np.save(args.cache/f'reports_ratios_lag{lag}.npy',result)
    (args.cache/'ratios_protocol.json').write_text(json.dumps({'names':['liabilities_assets','equity_assets','cash_assets','log_assets','net_margin','operating_margin'],'stats':fitted,'train_only':True,'same_period_same_statement':True},indent=2))
    print('Ratio features ready',flush=True)

if __name__=='__main__':main()
