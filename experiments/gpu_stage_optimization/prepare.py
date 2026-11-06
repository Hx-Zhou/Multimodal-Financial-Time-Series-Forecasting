import argparse
from pathlib import Path
import json
import time
import numpy as np
from experiments.e1_dlinear.e6_plus.data import load_store
from experiments.dlinear_modal_common.data import add_reports

def main():
    p=argparse.ArgumentParser(); p.add_argument('--package',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); start=time.time()
    print('Preparing identical official-reader data and training-only transforms',flush=True)
    store=load_store(args.package,args.output,with_news=True)
    add_reports(store,args.package,args.output)
    arrays={'raw':store.features['raw'],'stationary':store.features['stationary'],'close':store.close,
            'news_ids':store.news_ids,'news_counts':store.news_counts}
    for name,samples in [('train',store.train),('dev',store.dev)]:
        for key in ('starts','y','stocks','dates'): arrays[name+'_'+key]=getattr(samples,key)
    for key,value in arrays.items(): np.save(args.output/(key+'.npy'),value)
    meta={'zero_return':float(store.zero_return),'preparation_seconds':time.time()-start,
          'train_samples':len(store.train.y),'dev_samples':len(store.dev.y),'codes':store.codes,
          'holdout_evaluated':False,'source_protocol':store.metadata,'report_protocol':store.report_metadata}
    (args.output/'READY.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('codes','source_protocol','report_protocol')}),flush=True)

if __name__=='__main__': main()
