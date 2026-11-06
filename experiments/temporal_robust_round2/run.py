"""Second optimization round: purged rolling training, bounded local-scale residuals."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.gpu_stage_optimization import train as core
from experiments.gpu_stage_optimization.models import CloseLinear, MarketLinear, GRU
from experiments.e4_patchtst_capacity.model import ScaleAwarePatchTST
from experiments.e5_itransformer.model import ITransformer

HERE = Path(__file__).resolve().parent
SCOPES = ('rolling_A', 'rolling_B', 'development')


def backbone(spec):
    kind = spec['backbone']
    if kind == 'linear_close': return CloseLinear()
    if kind == 'linear_market': return MarketLinear()
    if kind.startswith('gru'):
        model = GRU(1 if kind == 'gru_close' else 7, 'last', spec.get('channel_dropout', 0))
        hidden = spec.get('hidden', 64)
        model.gru = nn.GRU(model.channels, hidden, batch_first=True)
        model.head = nn.Linear(hidden, 24)
        nn.init.zeros_(model.head.weight); nn.init.zeros_(model.head.bias)
        return model
    if kind == 'patch':
        return ScaleAwarePatchTST(d_model=spec.get('d_model', 128), layers=spec.get('layers', 2),
                                 nhead=4, d_ff=spec.get('d_ff', 256), dropout=.3)
    if kind == 'itransformer': return ITransformer(layers=spec.get('layers', 2))
    raise ValueError(kind)


class NewsBranch(nn.Module):
    def __init__(self, mode='pooled', coverage=False, modes=3):
        super().__init__(); self.mode=mode; self.coverage=coverage; self.modes=modes
        self.encoder=nn.Sequential(nn.Linear(512,8),nn.SiLU(),nn.Dropout(.3))
        size=96*10 if mode=='daily' else (10*3 if mode=='pooled' else modes*20)
        self.head=nn.Linear(size,24, bias=False)
        nn.init.zeros_(self.head.weight)
    def forward(self,b):
        z=self.encoder(F.normalize(b['news_unique'],dim=-1))[b['news_inverse']].reshape(-1,96,8)
        if self.coverage or b.get('zero_semantics'): z=torch.zeros_like(z)
        z=z*b['has_news'][...,None]
        z=torch.cat((z,b['has_news'][...,None],b['news_count'].clamp(max=5)[...,None]/5),-1)
        if self.mode=='pooled': x=z.reshape(-1,3,32,10).mean(2).flatten(1)
        elif self.mode=='daily': x=z.flatten(1)
        else:
            n=torch.fft.rfft(z,dim=1,norm='forward')[:,:self.modes]
            m=torch.fft.rfft(b['market'][...,-1:],dim=1,norm='forward')[:,:self.modes]
            cross=n*m.conj(); x=torch.cat((cross.real,cross.imag),-1).flatten(1)
        return self.head(x)*b['has_news'].any(1,keepdim=True)


class ReportBranch(nn.Module):
    def __init__(self, masks=False):
        super().__init__(); self.masks=masks; self.head=nn.Linear(54,24,bias=False)
        nn.init.zeros_(self.head.weight)
    def forward(self,b):
        x=b['reports']
        if self.masks or b.get('zero_report_values'): x=torch.cat((torch.zeros_like(x[:,:24]),x[:,24:]),-1)
        if b.get('disable_reports'): return torch.zeros((len(x),24),device=x.device)
        return self.head(x)*b['has_report']


class Forecast(nn.Module):
    def __init__(self,spec):
        super().__init__(); self.spec=spec; self.backbone=backbone(spec)
        self.cap=spec.get('cap',.5); self.aux_cap=spec.get('aux_cap',.1)
        if spec.get('news'): self.news=NewsBranch(spec.get('news_mode','pooled'),spec.get('coverage_news',False),spec.get('modes',3))
        if spec.get('reports'): self.reports=ReportBranch(spec.get('coverage_reports',False))
    def forward(self,b):
        normalized=dict(b,anchor=torch.zeros_like(b['anchor']),close=b['close_normalized'])
        delta=self.cap*torch.tanh(self.backbone(normalized)/self.cap)
        for name in ('news','reports'):
            if hasattr(self,name):
                extra=self.aux_cap*torch.tanh(getattr(self,name)(b)/self.aux_cap)
                probability=self.spec.get('modality_dropout',.1)
                if self.training: extra=F.dropout(extra,p=probability,training=True)
                delta=delta+extra
        return b['anchor']+b['risk']*delta


class Data(core.Data):
    def __init__(self,folder,device,calendar,modal=False):
        super().__init__(folder,device,modal)
        self.calendar=np.load(calendar)
        self.maps={}
        dates=self.samples['train']['dates']
        ends=self.calendar[self.samples['train']['starts']+119]
        for name,begin,end in [('rolling_A',20230103,20230228),('rolling_B',20230703,20230831)]:
            tr=np.flatnonzero(ends<begin); va=np.flatnonzero((dates>=begin)&(dates<=end))
            assert len(tr)>0 and len(va)>0 and ends[tr].max()<dates[va].min()
            self.maps[name]={'train':('train',tr),'val':('train',va)}
        self.maps['development']={'train':('train',np.arange(len(dates))),
                                  'val':('dev',np.arange(len(self.samples['dev']['y'])))}
        self.rawstats=self.meta['source_protocol']['raw_statistics']
        self.stationarystats=self.meta['source_protocol']['stationary_statistics']
        if modal:
            for lag in (180,270):
                self.arrays['ratios'+str(lag)]=torch.from_numpy(np.load(Path(folder)/f'reports_ratios_lag{lag}.npy')).to(device)
            self.reportstats=json.loads((Path(folder)/'report_protocol.json').read_text())['statistics']
            self.ratiostats=json.loads((Path(folder)/'ratios_protocol.json').read_text())['stats']
    def info(self,scope,split):
        source,rows=self.maps[scope][split]
        return {k:self.samples[source][k][rows] for k in ('starts','y','stocks','dates')}
    def get(self,scope,split,rows,spec,flags=None,lag=180):
        source,mapping=self.maps[scope][split]; selected=mapping[rows]
        original={'kind':'modal','representation':'raw','news':spec.get('news',False),'reports':spec.get('reports',False)}
        b,y=super().batch(source,selected,original,flags=flags,lag=lag)
        close=b['close']; anchor=b['anchor']; risk=close.std(1,keepdim=True,unbiased=False).clamp_min(1e-4)
        market=b['market'].clone()
        # Input-window-only transforms; supplied affine Close scaling cancels here and in training loss.
        market[:,:,:3]/=risk[:,None,:]; market[:,:,6:7]/=risk[:,None,:]
        mean,std=self.rawstats['mean'][0],self.rawstats['std'][0]
        volume=(market[:,:,3]*std+mean).clamp_min(0)
        logv=torch.log1p(volume); market[:,:,3]=logv-logv.mean(1,keepdim=True)
        days=self.samples[source]['starts_gpu'][torch.as_tensor(selected,device=self.device),None]+self.days
        stationary=self.arrays['stationary'][days]
        market[:,:,4:6]=stationary[:,:,4:6]
        if spec.get('representation')=='stationary':
            first=stationary[:,:,:4].clone()
            sm=torch.tensor(self.stationarystats['mean'],device=self.device,dtype=first.dtype)
            ss=torch.tensor(self.stationarystats['std'],device=self.device,dtype=first.dtype)
            first=first*ss+sm; first[:,0,0]=0
            first=(first-first.mean(1,keepdim=True))/first.std(1,keepdim=True,unbiased=False).clamp_min(1e-5)
            market=torch.cat((first,stationary[:,:,4:6],((close-anchor)/risk)[:,:,None]),-1)
        b.update(market=market.clamp(-5,5),risk=risk,close_normalized=(close-anchor)/risk,alpha=1.)
        if spec.get('reports'):
            if spec.get('ratios'):
                f=self.arrays['ratios'+str(lag)][days[:,-1]].clone(); stats=self.ratiostats
            else: f=b['reports'].clone(); stats=self.reportstats
            means=torch.tensor(stats['mean'],device=self.device,dtype=f.dtype)
            stds=torch.tensor(stats['std'],device=self.device,dtype=f.dtype)
            values=f[:,:24]*stds+means
            f[:,:24]=torch.tanh(values/(1. if spec.get('ratios') else 10.))*f[:,24:48]
            f[:,48:51]/=5
            b['reports']=f
        return b,y


def score(pred,target,dates,anchor):
    out=core.scores(pred,target,dates); base=core.scores(anchor,target,dates)
    blocks=[]
    for days in np.array_split(np.unique(dates),3):
        mask=np.isin(dates,days); c=core.scores(pred[mask],target[mask]); r=core.scores(anchor[mask],target[mask])
        blocks.append({'start':int(days[0]),'end':int(days[-1]),'mae':c['mae'],'mse':c['mse'],
                       'mae_ratio_e0':c['mae']/r['mae'],'mse_ratio_e0':c['mse']/r['mse']})
    out['blocks']=blocks; out['e0']=base
    out['objective']=max(b['mae_ratio_e0'] for b in blocks)+.25*max(b['mse_ratio_e0'] for b in blocks)
    return out


@torch.inference_mode()
def predict(model,data,scope,spec,batch=512,flags=None,lag=180):
    model.eval(); info=data.info(scope,'val'); pred=[]; anchors=[]
    for left in range(0,len(info['y']),batch):
        b,_=data.get(scope,'val',np.arange(left,min(left+batch,len(info['y']))),spec,flags,lag)
        pred.append(model(b).cpu().numpy()); anchors.append(b['anchor'].expand(-1,24).cpu().numpy())
    p=np.concatenate(pred); a=np.concatenate(anchors)
    if not np.isfinite(p).all():raise ValueError('nonfinite prediction')
    return p,score(p,info['y'],info['dates'],a)


def initialize(spec,seed,scope,root):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=Forecast(spec)
    if spec.get('parent'):
        path=root/spec['parent']/f'seed{seed}'/scope/'best.pt'
        saved=core.load_checkpoint(path)
        model.backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in saved['state_dict'].items() if k.startswith('backbone.')},strict=True)
        model.backbone.requires_grad_(False)
    for name in ('news','reports'):
        if spec.get('pretrained_'+name):
            path=root/spec['pretrained_'+name]/f'seed{seed}'/scope/'best.pt'
            state=core.load_checkpoint(path)['state_dict']
            getattr(model,name).load_state_dict({k.removeprefix(name+'.'):v for k,v in state.items() if k.startswith(name+'.')},strict=True)
    torch.manual_seed(seed+10000)
    return model.cuda()


def fit(spec,data,scope,args):
    out=args.output/spec['id']/f'seed{args.seed}'/scope
    if (out/'result.json').exists():
        record=json.loads((out/'result.json').read_text()); assert record['spec']==spec
        assert core.digest(out/'best.pt')==record['checkpoint_sha256']; return record
    out.mkdir(parents=True,exist_ok=True)
    protocol={'spec':spec,'seed':args.seed,'scope':scope,'source_sha256':core.digest(Path(__file__)),
              'plan_sha256':core.digest(HERE/'PLAN.json'),'holdout_evaluated':False,
              'training_samples':len(data.info(scope,'train')['y']),'validation_samples':len(data.info(scope,'val')['y']),
              'max_epochs':args.epochs,'patience':2,'batch':args.batch,'torch':str(torch.__version__),
              'gpu':torch.cuda.get_device_name(),'cuda_visible':os.environ.get('CUDA_VISIBLE_DEVICES')}
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:core.write(out/'protocol.json',protocol)
    model=initialize(spec,args.seed,scope,args.output)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=.0002,weight_decay=.05)
    generator=np.random.default_rng(args.seed); start=time.time()
    initial,best=predict(model,data,scope,spec,args.batch); best_epoch=0; stale=0
    history=[{'epoch':0,'validation':best}]
    def save():torch.save({'state_dict':model.state_dict(),'spec':spec,'seed':args.seed,'scope':scope},out/'best.pt')
    save(); n=len(data.info(scope,'train')['y'])
    for epoch in range(1,args.epochs+1):
        if stale>=2:break
        model.train(); order=generator.permutation(n); tick=time.time()
        for left in range(0,n,args.batch):
            b,y=data.get(scope,'train',order[left:left+args.batch],spec)
            err=(model(b)-y)/b['risk']
            loss=err.abs().mean() if spec.get('loss')=='mae' else F.huber_loss(err,torch.zeros_like(err),delta=1.)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward(); nn.utils.clip_grad_norm_(params,1.); optimizer.step()
        pred,current=predict(model,data,scope,spec,args.batch)
        history.append({'epoch':epoch,'validation':current,'seconds':time.time()-tick})
        if current['objective']<best['objective']-1e-5:
            best=current; best_epoch=epoch; stale=0; save()
        else:stale+=1
        core.write(out/'history.json',history)
        print(f"{spec['id']} seed{args.seed} {scope} ep{epoch} MAE={current['mae']:.6f} MSE={current['mse']:.6f} robust={current['objective']:.6f} best={best_epoch} seconds={time.time()-tick:.1f}",flush=True)
    model.load_state_dict(core.load_checkpoint(out/'best.pt')['state_dict'])
    pred,best=predict(model,data,scope,spec,args.batch); np.save(out/'predictions.npy',pred)
    diagnostics={}
    if scope=='development':
        for name,flag in [('news',{'zero_semantics':True}),('reports',{'zero_report_values':True})]:
            if spec.get(name):
                altered,m=predict(model,data,scope,spec,args.batch,flag)
                diagnostics[name]={'mae_increase':m['mae']-best['mae'],'max_prediction_change':float(np.abs(altered-pred).max())}
        if spec.get('reports'):
            _,m=predict(model,data,scope,spec,args.batch,lag=270)
            diagnostics['lag270']={'mae':m['mae'],'mse':m['mse']}
    result={'id':spec['id'],'spec':spec,'seed':args.seed,'scope':scope,'validation':best,'best_epoch':best_epoch,
            'epochs_run':len(history)-1,'seconds':time.time()-start,'checkpoint_sha256':core.digest(out/'best.pt'),
            'holdout_evaluated':False,'diagnostics':diagnostics,'parameters':sum(p.numel() for p in model.parameters())}
    core.write(out/'result.json',result)
    print('DONE '+spec['id']+' '+scope,flush=True)
    return result


def specifications(lane,root,seed):
    if lane=='linear':return [
        {'id':'E1_MAE','backbone':'linear_close','loss':'mae'},
        {'id':'E1_Huber','backbone':'linear_close'},
        {'id':'E6_raw','backbone':'linear_market','representation':'raw'},
        {'id':'E6_stationary','backbone':'linear_market','representation':'stationary'}]
    if lane=='recurrent':return [
        {'id':'E2_GRU64','backbone':'gru_close','hidden':64},
        {'id':'E2_GRU16','backbone':'gru_close','hidden':16},
        {'id':'E3_drop01','backbone':'gru_market','hidden':16,'channel_dropout':.1},
        {'id':'E3_drop03','backbone':'gru_market','hidden':16,'channel_dropout':.3},
        {'id':'E5_layers2','backbone':'itransformer','layers':2},
        {'id':'E5_layers1','backbone':'itransformer','layers':1}]
    if lane=='patch':return [
        {'id':'E4_full','backbone':'patch','d_model':128,'layers':2,'d_ff':256},
        {'id':'E4_small','backbone':'patch','d_model':64,'layers':1,'d_ff':128}]
    selection=json.loads((root/'market_selection.json').read_text())
    common=deepcopy(selection['spec']); common.pop('id'); common['parent']=selection['id']
    if lane=='news':return [dict(common,id='E7_daily',news=True,news_mode='daily'),
                            dict(common,id='E7_pooled',news=True,news_mode='pooled'),
                            dict(common,id='E7_coverage',news=True,news_mode='pooled',coverage_news=True),
                            dict(common,id='E10_modes13',news=True,news_mode='spectral',modes=13),
                            dict(common,id='E10_modes3',news=True,news_mode='spectral',modes=3)]
    if lane=='reports':return [dict(common,id='E8_values',reports=True),
                               dict(common,id='E8_ratios',reports=True,ratios=True),
                               dict(common,id='E8_masks',reports=True,ratios=True,coverage_reports=True)]
    selected=json.loads((root/'modal_selection.json').read_text())
    news=selected['news']; report=selected['reports']
    c=dict(common,news=True,reports=True,news_mode=news['spec']['news_mode'],
           ratios=report['spec'].get('ratios',False),coverage_reports=report['spec'].get('coverage_reports',False),
           pretrained_news=news['id'],pretrained_reports=report['id'])
    return [dict(c,id='E9_dropout01',modality_dropout=.1),dict(c,id='E9_dropout05',modality_dropout=.5)]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True);p.add_argument('--calendar',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lane',choices=['linear','recurrent','patch','news','reports','joint'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--ids',nargs='*');args=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.24)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    specs=specifications(args.lane,args.output,args.seed)
    if args.ids:specs=[s for s in specs if s['id'] in args.ids]
    data=Data(args.cache,'cuda',args.calendar,args.lane in ('news','reports','joint'))
    for spec in specs:
        for scope in SCOPES:fit(spec,data,scope,args)


if __name__=='__main__':main()
