import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.gpu_stage_optimization import train as core
from experiments.gpu_stage_optimization.models import build as original_build

HERE=Path(__file__).resolve().parent

class Scaled(nn.Module):
    def __init__(self,base,factor):super().__init__();self.base=base;self.factor=factor
    def forward(self,b):
        value=self.base(b);anchor=b['anchor'];return anchor+self.factor*(value-anchor)

def factory(spec):
    base=original_build(spec)
    return Scaled(base,spec['residual_scale']) if 'residual_scale' in spec else base

class Data(core.Data):
    def __init__(self,folder,device,modal=False,ratios=False):
        super().__init__(folder,device,modal)
        if ratios:
            for lag in (180,270):self.arrays['reports_lag'+str(lag)]=torch.from_numpy(np.load(self.folder/f'reports_ratios_lag{lag}.npy')).to(device)
    def batch(self,split,rows,spec,flags=None,shuffle=None,lag=180):
        b,y=super().batch(split,rows,spec,flags,shuffle,lag)
        if spec.get('historical_patch_rule') and split=='train':b['alpha']=1.
        if spec.get('recent_news_days'):
            cutoff=96-spec['recent_news_days'];b['has_news'][:,:cutoff]=0;b['news_count'][:,:cutoff]=0
        return b,y

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first-output',type=Path,required=True);p.add_argument('--lane',choices=['numerical','news','reports'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch',type=int,default=512)
    args=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.24)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    core.build=factory
    data=Data(args.cache,'cuda',args.lane!='numerical',args.lane=='reports')
    def f(**spec):
        spec.update(followup_plan_sha256=core.digest(HERE/'PLAN.json'),followup_source_sha256=core.digest(Path(__file__)))
        if spec.get('ratios'):spec.update(ratios_source_sha256=core.digest(HERE/'ratios.py'),ratios_protocol_sha256=core.digest(args.cache/'ratios_protocol.json'))
        return core.fit(spec,data,args)
    def old(name):return json.loads((args.first_output/name/f'seed{args.seed}'/'result.json').read_text())
    if args.lane=='numerical':
        r=f(id='T1_E1_half',kind='close',residual_scale=.5);core.decision(args,'T1',old('E1_baseline'),r)
        raw=f(id='T2_E6_raw_half',kind='market',representation='raw',residual_scale=.5);core.decision(args,'T2',old('E6_raw'),raw)
        st=f(id='T3_E6_stationary_half',kind='market',representation='stationary',residual_scale=.5);core.decision(args,'T3',old('E6_stationary'),st)
        ref=f(id='T4_Patch_reference',kind='patch',d_ff=256,alpha=.5,historical_patch_rule=True)
        small=f(id='T5_Patch_ff128',kind='patch',d_ff=128,alpha=.5,historical_patch_rule=True);core.decision(args,'T5',ref,small)
    elif args.lane=='news':
        market=old('E6_stationary')['checkpoint']
        common={'kind':'modal','news':True,'representation':'stationary','market_checkpoint':market,'recent_news_days':20}
        r=f(id='T6_News_recent20',**common);coverage=f(id='T7_News_recent20_coverage',coverage_news=True,**common)
        # Use the original E7 stage name so the strict content checks also apply here.
        core.decision(args,'E7',old('E7_baseline'),r,[coverage])
    else:
        market=old('E6_stationary')['checkpoint']; common={'kind':'modal','reports':True,'representation':'stationary','market_checkpoint':market,'ratios':True}
        r=f(id='T8_Report_ratios',**common);coverage=f(id='T9_Report_ratio_masks',coverage_reports=True,**common)
        core.decision(args,'E8',old('E8_mlp'),r,[coverage])

if __name__=='__main__':main()
