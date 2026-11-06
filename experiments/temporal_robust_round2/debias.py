"""Registered targeted follow-up: remove unconditional forecast drift by reflection."""
import argparse
from copy import deepcopy
from pathlib import Path
import torch
from torch.nn import functional as F
from . import run


class SymmetricForecast(run.Forecast):
    def forward(self,b):
        if not self.spec.get('odd_market'): return super().forward(b)
        normalized=dict(b,anchor=torch.zeros_like(b['anchor']),close=b['close_normalized'])
        reflected=dict(normalized,close=-normalized['close'])
        market=normalized['market'].clone()
        if self.spec.get('representation')=='stationary': market[:,:,[0,2,6]]=-market[:,:,[0,2,6]]
        else:
            old=market.clone();market[:,:,0]=-old[:,:,0];market[:,:,1]=-old[:,:,2]
            market[:,:,2]=-old[:,:,1];market[:,:,6]=-old[:,:,6]
        reflected['market']=market
        # Same weights for both signs cancel unconditional intercept/drift.
        delta=.5*self.cap*(torch.tanh(self.backbone(normalized)/self.cap)-torch.tanh(self.backbone(reflected)/self.cap))
        context=torch.tanh(-b['close_normalized'][:,-20:].mean(1,keepdim=True))
        for name in ('news','reports'):
            if hasattr(self,name):
                extra=self.aux_cap*torch.tanh(getattr(self,name)(b)/self.aux_cap)*context
                if self.training:extra=F.dropout(extra,p=self.spec.get('modality_dropout',.1),training=True)
                delta=delta+extra
        return b['anchor']+b['risk']*delta


FOLLOWUPS=[
 {'id':'S_E1_Huber','backbone':'linear_close','odd_market':True},
 {'id':'S_E2_GRU16','backbone':'gru_close','hidden':16,'odd_market':True},
 {'id':'S_E3_drop03','backbone':'gru_market','hidden':16,'channel_dropout':.3,'odd_market':True},
 {'id':'S_E4_small','backbone':'patch','d_model':64,'layers':1,'d_ff':128,'odd_market':True},
 {'id':'S_E5_layers1','backbone':'itransformer','layers':1,'odd_market':True},
 {'id':'S_E6_stationary','backbone':'linear_market','representation':'stationary','odd_market':True},
]


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--calendar',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lane',choices=['sym_linear','sym_sequence','sym_patch','news','reports','joint','replicate'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--ids',nargs='*');args=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.24)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run.Forecast=SymmetricForecast
    if args.lane.startswith('sym_'):
        indices={'sym_linear':[0,5],'sym_sequence':[1,2,4],'sym_patch':[3]}[args.lane]
        specs=[FOLLOWUPS[i] for i in indices]
    elif args.lane=='replicate':
        import json
        fixed=json.loads((args.output/'FINAL_SELECTION.json').read_text())
        market=fixed['spec_market'];joint=fixed['spec_multimodal']
        selected=json.loads((args.output/'modal_selection.json').read_text())
        specs=[market,selected['news']['spec'],selected['reports']['spec'],joint]
    else:specs=run.specifications(args.lane,args.output,args.seed)
    if args.ids:specs=[s for s in specs if s['id'] in args.ids]
    modal=args.lane in ('news','reports','joint','replicate')
    data=run.Data(args.cache,'cuda',args.calendar,modal)
    for spec in specs:
        spec=deepcopy(spec)
        for scope in run.SCOPES:run.fit(spec,data,scope,args)


if __name__=='__main__':main()
