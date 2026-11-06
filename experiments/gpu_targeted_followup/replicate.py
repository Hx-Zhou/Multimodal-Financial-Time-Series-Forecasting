import argparse,json
from pathlib import Path
import torch
from experiments.gpu_stage_optimization import train as core

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first-output',type=Path,required=True);p.add_argument('--screen',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch',type=int,default=512)
    args=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.24)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    screen=json.loads(args.screen.read_text());selected=screen['selected']
    if not selected:raise RuntimeError('No ensemble passed preregistered screen')
    data=core.Data(args.cache,'cuda',True)
    def f(spec):
        spec=dict(spec);spec['replication_plan_sha256']=core.digest(HERE/'ENSEMBLE_PLAN.json');spec['replication_source_sha256']=core.digest(Path(__file__))
        return core.fit(spec,data,args)
    parent=f({'id':'E6_stationary','kind':'market','representation':'stationary'})
    members=screen['candidates'][selected]['members']
    for member in members:
        record=json.loads((args.first_output/member/'seed42/result.json').read_text());spec=dict(record['spec']);spec['market_checkpoint']=parent['checkpoint']
        f(spec)
    print('REPLICATION COMPLETE '+str(args.seed),flush=True)

if __name__=='__main__':main()
