"""Shared CUDA training, chronological development only, bounded stage decisions."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .models import build

HERE=Path(__file__).resolve().parent

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(path)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scores(pred,y,dates=None):
    d=pred.astype(np.float64)-y.astype(np.float64)
    result={'mae':float(np.abs(d).mean()),'mse':float(np.square(d).mean())}
    result['horizons']={str(i+1):{'mae':float(np.abs(d[:,i]).mean()),'mse':float(np.square(d[:,i]).mean())} for i in (0,5,11,23)}
    if dates is not None:
        cutoff=int(np.unique(dates)[len(np.unique(dates))//2]); result['cutoff']=cutoff
        for key,mask in [('early',dates<cutoff),('late',dates>=cutoff)]:
            result[key]={'mae':float(np.abs(d[mask]).mean()),'mse':float(np.square(d[mask]).mean())}
    return result

def gate(candidate,reference):
    c,r=candidate['development'],reference['development']
    checks={'mae_0p1_percent':c['mae']<=r['mae']*.999,'mse_no_worse':c['mse']<=r['mse']+1e-10,
            'both_periods':all(c[k]['mae']<=r[k]['mae']*1.005 for k in ('early','late'))}
    return {'passed':all(checks.values()),'checks':checks,'mae_improvement_percent':100*(1-c['mae']/r['mae'])}

class Data:
    def __init__(self,folder,device,modal=False):
        self.folder=Path(folder); self.device=device
        self.meta=json.loads((self.folder/'READY.json').read_text())
        names=['raw','stationary','close']
        if modal: names+=['pooled_news','news_ids','news_counts','reports_lag180','reports_lag270']
        self.arrays={name:torch.from_numpy(np.array(np.load(self.folder/(name+'.npy')),copy=True)).to(device) for name in names}
        self.samples={}
        for split in ('train','dev'):
            rows={key:np.load(self.folder/(split+'_'+key+'.npy')) for key in ('starts','y','stocks','dates')}
            self.samples[split]=rows
            rows['starts_gpu']=torch.as_tensor(rows['starts'],device=device)
            rows['y_gpu']=torch.as_tensor(rows['y'],device=device)
        self.days=torch.arange(96,device=device)
    def batch(self,split,rows,spec,flags=None,shuffle=None,lag=180):
        rows=torch.as_tensor(rows,device=self.device,dtype=torch.long)
        samples=self.samples[split]; days=samples['starts_gpu'][rows,None]+self.days
        close=self.arrays['close'][days]; anchor=close[:,-1:]
        rep=spec.get('representation','raw'); market=self.arrays[rep][days].clone()
        if rep=='raw':
            market[:,:,:4]-=anchor[:,None,:]
            market=market[:,:,[0,1,2,4,5,6,3]] # Close token always at index 6.
        else:
            market[:,:,6:7]-=anchor[:,None,:]; market[:,0,0]=self.meta['zero_return']
        b={'close':close,'anchor':anchor,'market':market,'alpha':spec.get('alpha',1.)}
        content_rows=rows if shuffle is None else torch.as_tensor(shuffle,device=self.device)[rows]
        cdays=samples['starts_gpu'][content_rows,None]+self.days
        if spec.get('news') or spec['kind']=='spectral':
            ids=self.arrays['news_ids'][cdays]; unique,inverse=torch.unique(ids,return_inverse=True)
            b.update(news_unique=self.arrays['pooled_news'][unique],news_inverse=inverse.reshape(-1),
                     has_news=(ids!=0).float(),news_count=torch.log1p(self.arrays['news_counts'][cdays].float()))
        if spec.get('reports'):
            features=self.arrays['reports_lag'+str(lag)][cdays[:,-1]]
            b.update(reports=features,has_report=features[:,-3:].any(1,keepdim=True).float())
        if spec.get('coverage_news'): b['zero_semantics']=True
        if spec.get('coverage_reports'): b['zero_report_values']=True
        if flags: b.update(flags)
        return b,samples['y_gpu'][rows]

@torch.inference_mode()
def predict(model,data,spec,batch_size=512,flags=None,shuffle=None,lag=180):
    model.eval(); out=[]; n=len(data.samples['dev']['y'])
    for left in range(0,n,batch_size):
        b,_=data.batch('dev',np.arange(left,min(left+batch_size,n)),spec,flags,shuffle,lag)
        out.append(model(b).cpu().numpy())
    value=np.concatenate(out)
    if not np.isfinite(value).all(): raise ValueError('nonfinite predictions')
    return value

def load_checkpoint(path): return torch.load(path,map_location='cpu',weights_only=False)

def initialize(spec,seed,device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=build(spec)
    if spec.get('market_checkpoint'):
        saved=load_checkpoint(spec['market_checkpoint'])['state_dict']
        model.market.load_state_dict({k.removeprefix('market.'):v for k,v in saved.items() if k.startswith('market.')})
    if spec.get('pretrained_news'):
        state=load_checkpoint(spec['pretrained_news'])['state_dict']
        model.news.load_state_dict({k.removeprefix('news.'):v for k,v in state.items() if k.startswith('news.')})
        model.news.mlp.requires_grad_(False)
    if spec.get('pretrained_reports'):
        state=load_checkpoint(spec['pretrained_reports'])['state_dict']
        model.reports.load_state_dict({k.removeprefix('reports.'):v for k,v in state.items() if k.startswith('reports.')})
        model.reports.encoder.requires_grad_(False)
    torch.manual_seed(seed+10000)
    return model.to(device)

def fit(spec,data,args):
    out=args.output/spec['id']/f'seed{args.seed}'; result_path=out/'result.json'
    if result_path.exists():
        record=json.loads(result_path.read_text()); assert record['spec']==spec
        assert digest(out/'best.pt')==record['checkpoint_sha256']
        print('Preserving completed '+spec['id'],flush=True); return record
    out.mkdir(parents=True,exist_ok=True)
    model=initialize(spec,args.seed,data.device)
    params=[p for p in model.parameters() if p.requires_grad]
    optim=torch.optim.AdamW([{'params':[p for p in params if p.ndim>=2],'weight_decay':.01},
                            {'params':[p for p in params if p.ndim<2],'weight_decay':0.}],lr=.0003)
    source={p.name:digest(p) for p in HERE.glob('*.py')}
    protocol={'spec':spec,'seed':args.seed,'source_hashes':source,'plan_sha256':digest(HERE/'PLAN.json'),
          'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
          'gpu_uuid_env':os.environ.get('CUDA_VISIBLE_DEVICES'),'max_epochs':args.epochs,'patience':3,
          'batch_size':args.batch,'train_samples':len(data.samples['train']['y']),'holdout_evaluated':False}
    if (out/'protocol.json').exists():
        previous=json.loads((out/'protocol.json').read_text())
        for key in ('spec','seed','source_hashes','plan_sha256','torch','max_epochs','batch_size'):
            if previous[key]!=protocol[key]: raise RuntimeError('Resume protocol changed: '+key)
    else: write(out/'protocol.json',protocol)
    frozen={k:v.detach().cpu().clone() for k,v in model.market.state_dict().items()} if spec.get('market_checkpoint') else None
    n=len(data.samples['train']['y']); target=data.samples['dev']['y']; dates=data.samples['dev']['dates']
    generator=np.random.default_rng(args.seed); start=time.time(); first_epoch=1
    if (out/'last.pt').exists():
        resume=load_checkpoint(out/'last.pt'); model.load_state_dict(resume['state_dict']); optim.load_state_dict(resume['optimizer'])
        generator.bit_generator.state=resume['numpy_rng']; torch.set_rng_state(resume['cpu_rng']); torch.cuda.set_rng_state(resume['gpu_rng'])
        history=resume['history']; best=resume['best']; best_epoch=resume['best_epoch']; stale=resume['stale']; first_epoch=resume['epoch']+1
        print('Resuming '+spec['id']+' at epoch '+str(first_epoch),flush=True)
    else:
        initial=predict(model,data,spec,args.batch); best=scores(initial,target,dates); best_epoch=0; stale=0
        torch.save({'spec':spec,'state_dict':model.state_dict(),'seed':args.seed},out/'best.pt')
        history=[{'epoch':0,'development':best}]; write(out/'history.json',history)
    for epoch in range(first_epoch,args.epochs+1):
        if stale>=3: break
        tick=time.time(); model.train(); order=generator.permutation(n); loss_sum=torch.zeros(3,device=data.device)
        for left in range(0,n,args.batch):
            rows=order[left:left+args.batch]; b,y=data.batch('train',rows,spec)
            optim.zero_grad(set_to_none=True); err=model(b)-y
            mae=err.abs().mean(); mse=err.square().mean(); loss=mae+spec.get('mse_weight',.2)*mse
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.); optim.step()
            loss_sum+=torch.stack((loss.detach(),mae.detach(),mse.detach()))*len(rows)
        value=predict(model,data,spec,args.batch); score=scores(value,target,dates)
        row={'epoch':epoch,'development':score,'train_loss_mae_mse':(loss_sum/n).cpu().tolist(),'seconds':time.time()-tick}
        history.append(row)
        if score['mae']<best['mae']-1e-5:
            best=score; best_epoch=epoch; stale=0
            torch.save({'spec':spec,'state_dict':model.state_dict(),'seed':args.seed},out/'best.pt')
        else: stale+=1
        write(out/'history.json',history)
        torch.save({'state_dict':model.state_dict(),'optimizer':optim.state_dict(),'epoch':epoch,'history':history,
                    'best':best,'best_epoch':best_epoch,'stale':stale,'numpy_rng':generator.bit_generator.state,
                    'cpu_rng':torch.get_rng_state(),'gpu_rng':torch.cuda.get_rng_state()},out/'last.tmp')
        (out/'last.tmp').replace(out/'last.pt')
        print(f"{spec['id']} seed={args.seed} epoch={epoch} mae={score['mae']:.9f} mse={score['mse']:.9f} best={best_epoch} seconds={row['seconds']:.1f}",flush=True)
        if stale>=3: break
    model.load_state_dict(load_checkpoint(out/'best.pt')['state_dict']); value=predict(model,data,spec,args.batch)
    np.save(out/'dev_predictions.npy',value); final=scores(value,target,dates)
    assert abs(final['mae']-best['mae'])<1e-8
    if frozen:
        for k,v in model.market.state_dict().items(): torch.testing.assert_close(v.cpu(),frozen[k],rtol=0,atol=0)
    result={'id':spec['id'],'seed':args.seed,'spec':spec,'development':final,'initial':history[0]['development'],
            'best_epoch':best_epoch,'epochs_run':len(history)-1,'seconds':time.time()-start,
            'parameters':sum(p.numel() for p in model.parameters()),'checkpoint':str(out/'best.pt'),
            'checkpoint_sha256':digest(out/'best.pt'),'holdout_evaluated':False,'diagnostics':{}}
    if spec.get('news') or spec.get('reports') or spec['kind']=='spectral':
        rng=np.random.default_rng(20261101); stocks=data.samples['dev']['stocks']; shuffled=np.arange(len(stocks))
        for stock in np.unique(stocks):
            idx=np.flatnonzero(stocks==stock); shuffled[idx]=rng.permutation(idx)
        tests=[]
        if spec.get('news') or spec['kind']=='spectral': tests.append(('news_semantics_zero',{'zero_semantics':True},None,180))
        if spec.get('reports'): tests.extend([('report_values_zero',{'zero_report_values':True},None,180),('report_lag270',{},None,270)])
        tests.append(('within_stock_content_shuffled',{},shuffled,180))
        for name,flags,permutation,lag in tests:
            altered=predict(model,data,spec,args.batch,flags,permutation,lag)
            sc=scores(altered,target); sc.update(mae_increase=sc['mae']-final['mae'],max_prediction_change=float(np.abs(altered-value).max()))
            result['diagnostics'][name]=sc
        if spec['kind']=='spectral':
            previous_modes=model.modes; model.modes=1
            altered=predict(model,data,spec,args.batch); model.modes=previous_modes
            sc=scores(altered,target); sc.update(mae_increase=sc['mae']-final['mae'],max_prediction_change=float(np.abs(altered-value).max()))
            result['diagnostics']['dc_only']=sc
    write(result_path,result); print('DONE '+spec['id'],flush=True)
    return result

def wait_record(args,name):
    path=args.output/name/f'seed{args.seed}'/'result.json'
    tick=time.time()
    while not path.exists():
        if time.time()-tick>7200: raise TimeoutError('Dependency '+str(path))
        time.sleep(15)
    return json.loads(path.read_text())

def decision(args,stage,baseline,candidate,additional=()):
    check=gate(candidate,baseline)
    for other in additional:
        g=gate(candidate,other); check['checks']['beats_'+other['id']]=g['passed']
    if stage in ('E7','E8','E9','E10'):
        diag=candidate['diagnostics']; keys=['within_stock_content_shuffled']
        if stage in ('E7','E9','E10'): keys.append('news_semantics_zero')
        if stage in ('E8','E9'): keys.append('report_values_zero')
        check['checks']['content_increment']=all(diag[k]['mae_increase']>=1e-5 for k in keys)
    check['passed']=all(check['checks'].values())
    chosen=candidate if check['passed'] else baseline
    value={'stage':stage,'baseline':baseline['id'],'candidate':candidate['id'],'selected':chosen['id'],**check}
    write(args.output/'decisions'/f'{stage}_seed{args.seed}.json',value)
    print('DECISION '+json.dumps(value),flush=True); return chosen

def stage_selected(args,stage):
    path=args.output/'decisions'/f'{stage}_seed{args.seed}.json'; start=time.time()
    while not path.exists():
        if time.time()-start>7200: raise TimeoutError(stage)
        time.sleep(15)
    return wait_record(args,json.loads(path.read_text())['selected'])

def run(args):
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.24)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    modal=args.lane in ('news','reports','joint'); data=Data(args.cache,'cuda',modal)
    def f(**spec): return fit(spec,data,args)
    if args.lane=='foundation':
        b=f(id='E1_baseline',kind='close'); v=f(id='E1_mse05',kind='close',mse_weight=.5); decision(args,'E1',b,v)
        b=f(id='E6_raw',kind='market',representation='raw'); v=f(id='E6_stationary',kind='market',representation='stationary'); decision(args,'E6',b,v)
    elif args.lane=='recurrent':
        b=f(id='E2_last',kind='gru',channels=1,pool='last'); v=f(id='E2_mean',kind='gru',channels=1,pool='mean'); picked=decision(args,'E2',b,v)
        pool=picked['spec']['pool']
        b=f(id='E3_baseline',kind='gru',channels=7,pool=pool,representation='raw'); v=f(id='E3_channel_drop',kind='gru',channels=7,pool=pool,representation='raw',channel_dropout=.1); decision(args,'E3',b,v)
        b=f(id='E5_layers2',kind='itransformer',layers=2,representation='raw'); v=f(id='E5_layers1',kind='itransformer',layers=1,representation='raw'); decision(args,'E5',b,v)
    elif args.lane=='patch':
        control=f(id='E4_linear_control',kind='patch_control',alpha=.5)
        b=f(id='E4_ff256',kind='patch',d_ff=256,alpha=.5); v=f(id='E4_ff128',kind='patch',d_ff=128,alpha=.5); decision(args,'E4',b,v,[control])
    elif args.lane in ('news','reports'):
        market=wait_record(args,'E6_stationary')['checkpoint']
        common={'kind':'modal','representation':'stationary','market_checkpoint':market}
        if args.lane=='news':
            b=f(id='E7_baseline',news=True,**common); v=f(id='E7_l2_normalized',news=True,normalize=True,**common)
            norm=v['development']['mae']<b['development']['mae']; full=v if norm else b
            coverage=f(id='E7_coverage',news=True,normalize=norm,coverage_news=True,**common)
            decision(args,'E7',b,v,[coverage]); write(args.output/'decisions'/f'E7_content_seed{args.seed}.json',{'full':full['id'],'coverage':coverage['id'],'gate':gate(full,coverage)})
            b=f(id='E10_modes13',kind='spectral',representation='stationary',market_checkpoint=market,modes=13)
            v=f(id='E10_modes7',kind='spectral',representation='stationary',market_checkpoint=market,modes=7)
            simple=wait_record(args,'E7_baseline'); decision(args,'E10',b,v,[simple])
        else:
            b=f(id='E8_mlp',reports=True,**common); v=f(id='E8_linear',reports=True,linear_reports=True,**common)
            coverage=f(id='E8_coverage',reports=True,linear_reports=True,coverage_reports=True,**common)
            decision(args,'E8',b,v,[coverage]); write(args.output/'decisions'/f'E8_content_seed{args.seed}.json',{'full':v['id'],'coverage':coverage['id'],'gate':gate(v,coverage)})
    elif args.lane=='joint':
        market=wait_record(args,'E6_stationary')['checkpoint']; news=stage_selected(args,'E7'); report=stage_selected(args,'E8')
        common={'kind':'modal','representation':'stationary','market_checkpoint':market,'news':True,'reports':True,
                'normalize':news['spec'].get('normalize',False),'linear_reports':report['spec'].get('linear_reports',False)}
        b=f(id='E9_joint',**common)
        v=f(id='E9_pretrained',pretrained_news=news['checkpoint'],pretrained_reports=report['checkpoint'],**common)
        decision(args,'E9',b,v,[news,report])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lane',choices=['foundation','recurrent','patch','news','reports','joint'],required=True)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=12); p.add_argument('--batch',type=int,default=512)
    args=p.parse_args(); run(args)

if __name__=='__main__': main()
