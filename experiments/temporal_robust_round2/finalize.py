"""Export and validate fixed models, then optionally run the one final test batch."""
import argparse
from datetime import datetime,timezone
import importlib.util
import inspect
import json
from pathlib import Path
import zipfile
import numpy as np
import torch

from experiments.gpu_stage_optimization import train as core
from experiments.gpu_stage_optimization import models as old
from experiments.e1_dlinear.model import DLinear
from experiments.e1_dlinear.e6_plus.model import Heads
from experiments.e4_patchtst import model as patch
from experiments.e4_patchtst_capacity.model import ScaleAwarePatchTST
from experiments.e5_itransformer.model import ITransformer
from experiments.dlinear_modal_common import data as financial
from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split
from . import run,debias

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def module_at(path):
    spec=importlib.util.spec_from_file_location('frozen_round2_'+path.parent.name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def standalone(spec,cache):
    # The prefix supplies the exact original-reader deduplicated news pooling.
    original=cache.parent/'candidate/model.py'
    if not original.exists():original=HERE.parent/'gpu_stage_optimization/candidate/model.py'
    prefix=original.read_text().split('class Heads')[0]
    prefix=prefix.replace('def prepare_batch(', 'def _original_prepare_batch(')
    definitions=[DLinear,Heads,old.CloseLinear,old.MarketLinear,old.GRU,
                 patch.history_transform,patch.ResidualForecast,patch.PatchTST,ScaleAwarePatchTST,ITransformer,
                 run.backbone,run.NewsBranch,run.ReportBranch,run.Forecast,debias.SymmetricForecast,
                 financial.ordinal,financial.finite,financial.consolidate,financial.financial_block]
    source=prefix+'\nLOOKBACK,HORIZON=96,24\n'
    for item in definitions:
        source+='\n'+inspect.getsource(item).replace('class SymmetricForecast(run.Forecast):','class SymmetricForecast(Forecast):')+'\n'
    source+='\nSPEC='+repr(spec)+'\n'
    source+='STATIONARY_STATS='+repr(read(cache/'data_protocol.json')['stationary_statistics'])+'\n'
    source+='REPORT_STATS='+repr(read(cache/'report_protocol.json')['statistics'])+'\n'
    source+='RATIO_STATS='+repr(read(cache/'ratios_protocol.json')['stats'])+'\n'
    source+=(HERE/'standalone_preprocess.py').read_text()
    source+='\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__();self.members=nn.ModuleList([SymmetricForecast(SPEC) for _ in range(3)])\n    def forward(self,batch):return torch.stack([m(batch) for m in self.members]).mean(0)\n\ndef build_model(config=None):return Model()\n'
    return source


def export(args):
    fixed=read(args.root/'FINAL_SELECTION.json');freeze_path=args.root/'TEST_FREEZE.json'
    if freeze_path.exists():raise FileExistsError('Test freeze exists; do not overwrite frozen candidates')
    data=run.Data(args.cache,'cuda',HERE/'calendar.npy',True)
    candidates={};ensembles={}
    for label,name,key in [('market',fixed['market'],'spec_market'),('multimodal',fixed['multimodal'],'spec_multimodal')]:
        spec=fixed[key];out=args.root/'candidates'/label;out.mkdir(parents=True,exist_ok=True)
        states={};records={};predictions={}
        for scope in run.SCOPES:
            values=[]
            for seed in (42,43,44):
                folder=args.root/name/f'seed{seed}'/scope
                r=read(folder/'result.json');assert r['checkpoint_sha256']==core.digest(folder/'best.pt')
                values.append(np.load(folder/'predictions.npy'))
                records[f'{scope}_seed{seed}']=r
                if scope=='development':
                    saved=core.load_checkpoint(folder/'best.pt')['state_dict']
                    states.update({f'members.{seed-42}.{k}':v.cpu() for k,v in saved.items()})
            prediction=np.mean(values,axis=0);predictions[scope]=prediction
        (out/'model.py').write_text(standalone(spec,args.cache))
        torch.save(states,out/'model.pt')
        mod=module_at(out/'model.py');model=mod.build_model().cuda().eval()
        model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True),strict=True)
        actual,score=run.predict(model,data,'development',spec)
        np.testing.assert_allclose(actual,predictions['development'],rtol=3e-5,atol=3e-6)
        dataset=HS300Dataset(args.package,'val');split=make_validation_split(dataset)
        positions=np.linspace(0,len(actual)-1,48,dtype=int)
        raw=[{k:v for k,v in dataset[int(split.dev_indices[i])].items() if k!='y'} for i in positions]
        prepared=mod.prepare_batch(raw,'cuda')
        with torch.inference_mode():raw_prediction=model(prepared).cpu().numpy()
        np.testing.assert_allclose(raw_prediction,actual[positions],rtol=4e-5,atol=4e-6)
        dependencies={}
        for title,flags in [('news_semantics',{'zero_semantics':True}),('financial_all',{'disable_reports':True})]:
            with torch.inference_mode():altered=model(dict(prepared,**flags)).cpu().numpy()
            dependencies[title+'_max_prediction_change']=float(np.abs(altered-raw_prediction).max())
        model.cpu()
        with torch.inference_mode():
            cpu=model(mod.prepare_batch(raw[:3],'cpu')).numpy()
            missing=model(mod.prepare_batch([dict(raw[0],news=None,financial={})],'cpu')).numpy()
        np.testing.assert_allclose(cpu,raw_prediction[:3],rtol=4e-5,atol=4e-6)
        assert np.isfinite(missing).all()
        for scope,pred in predictions.items():np.save(out/f'{scope}_predictions.npy',pred)
        with zipfile.ZipFile(args.root/f'{label}_candidate.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for f in ('model.py','model.pt'):archive.write(out/f,f)
        scope_scores={}
        for scope,pred in predictions.items():
            info=data.info(scope,'val');anchors=data.arrays['close'][torch.as_tensor(info['starts']+95,device='cuda')].cpu().numpy()[:,None].repeat(24,1)
            scope_scores[scope]=run.score(pred,info['y'],info['dates'],anchors)
        candidates[label]={'id':name,'spec':spec,'seeds':[42,43,44],
                           'model_py_sha256':core.digest(out/'model.py'),'model_pt_sha256':core.digest(out/'model.pt'),
                           'validation_scores':scope_scores,'raw_sample_max_difference':float(np.abs(raw_prediction-actual[positions]).max()),
                           'unlabeled_raw_samples':len(raw),'cpu_gpu_match':True,'missing_finite':True,
                           'input_dependency_48_samples':dependencies,
                           'parameters':sum(p.numel() for p in model.parameters())}
        dataset.close();del model
    # This write happens before the test action reads holdout indices/labels.
    core.write(freeze_path,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'candidates':candidates,
                           'selection_sha256':core.digest(args.root/'FINAL_SELECTION.json'),
                           'test_policy':'One final batch for these exact two candidates, fixed equal three-seed averages. No post-test selection or training.',
                           'holdout_evaluated':False})
    print('EXPORT VALIDATED AND TEST CANDIDATES FROZEN',flush=True)
    print(json.dumps({k:{s:v['mae'] for s,v in c['validation_scores'].items()} for k,c in candidates.items()},indent=2),flush=True)


def test(args):
    freeze=read(args.root/'TEST_FREEZE.json');out=args.root/'final_holdout'
    out.mkdir(exist_ok=False)
    core.write(out/'protocol.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),
                                  'freeze_sha256':core.digest(args.root/'TEST_FREEZE.json'),**freeze})
    data=run.Data(args.cache,'cuda',HERE/'calendar.npy',True)
    with np.load(args.package/'data/val.npz') as values:
        dates=values['target_start_date'];idx=np.flatnonzero(dates>=20240124)
        assert len(idx)==15390 and dates[idx].max()==20240227
        offsets=np.r_[0,np.flatnonzero(np.diff(data.calendar)<=0)+1,len(data.calendar)]
        assert len(offsets)==len(values['stock_codes'])+1
        stocks=values['stock_index'][idx];starts=offsets[stocks]+values['input_start'][idx]
        samples={'starts':starts,'y':values['y'][idx].astype(np.float32),'stocks':stocks,'dates':dates[idx]}
    samples['starts_gpu']=torch.as_tensor(starts,device='cuda');samples['y_gpu']=torch.as_tensor(samples['y'],device='cuda')
    data.samples['holdout']=samples;data.maps['holdout']={'val':('holdout',np.arange(len(idx)))}
    results={}
    for label,candidate in freeze['candidates'].items():
        folder=args.root/'candidates'/label
        assert core.digest(folder/'model.py')==candidate['model_py_sha256']
        assert core.digest(folder/'model.pt')==candidate['model_pt_sha256']
        module=module_at(folder/'model.py');model=module.build_model().cuda().eval()
        model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True)
        prediction,score=run.predict(model,data,'holdout',candidate['spec'])
        error=torch.from_numpy(prediction).double()-torch.from_numpy(samples['y']).double()
        np.testing.assert_allclose([score['mae'],score['mse']],[error.abs().mean().item(),error.square().mean().item()],rtol=0,atol=1e-12)
        np.save(out/f'{label}_predictions.npy',prediction)
        results[label]={'id':candidate['id'],'score':score,'checkpoint_unchanged':True,'independent_metric_match':True}
        print(label+' '+json.dumps(score),flush=True)
    np.save(out/'sample_indices.npy',idx)
    e0=results['market']['score']['e0']
    np.testing.assert_allclose([e0['mae'],e0['mse']],[.35428627282119307,.33033977026568145],rtol=0,atol=1e-10)
    summary={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'results':results,'samples':len(idx),
             'official_hidden_test_evaluated':False,'selection_after_test':False,'training_after_test':False,
             'test_freeze_sha256':core.digest(args.root/'TEST_FREEZE.json'),
             'interpretation':'Known internal temporal holdout, not a new blind test. All candidates fixed before this final batch.'}
    core.write(out/'summary.json',summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--action',choices=['export','test'],required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.24)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    export(args) if args.action=='export' else test(args)


if __name__=='__main__':main()
