import argparse,importlib.util,json,shutil,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from experiments.gpu_stage_optimization import train as core
from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--package',type=Path,required=True)
    args=p.parse_args();root=args.root;out=root/'candidate';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.24)
    members=['E7_baseline','E7_coverage','E8_coverage'];seeds=[42,43,44]
    y=np.load(root/'cache/dev_y.npy');dates=np.load(root/'cache/dev_dates.npy')
    predictions=[];references=[];states={};seed_scores={};index=0
    for seed in seeds:
        location=root/('results' if seed==42 else 'replications');values=[]
        for name in members:
            folder=location/name/f'seed{seed}'
            r=json.loads((folder/'result.json').read_text());assert core.digest(folder/'best.pt')==r['checkpoint_sha256']
            values.append(np.load(folder/'dev_predictions.npy'))
            saved=core.load_checkpoint(folder/'best.pt')['state_dict']
            states.update({f'members.{index}.{k}':v.cpu() for k,v in saved.items()});index+=1
        candidate=np.mean(values,axis=0);predictions.append(candidate);references.append(values[0])
        seed_scores[str(seed)]={'candidate':core.scores(candidate,y,dates),'matched_news':core.scores(values[0],y,dates)}
    prediction=np.mean(predictions,axis=0);reference=np.mean(references,axis=0)
    score=core.scores(prediction,y,dates);ref=core.scores(reference,y,dates)
    e0=json.loads((root/'results/E1_baseline/seed42/result.json').read_text())['development']
    wins=sum(r['candidate']['mae']<r['matched_news']['mae'] for r in seed_scores.values())
    checks={'mae_0p1_percent':score['mae']<=ref['mae']*.999,'mse_no_worse':score['mse']<=ref['mse'],
            'two_seed_wins':wins>=2,'both_dev_periods':all(score[k]['mae']<=e0[k]['mae']*1.005 for k in ('early','late'))}
    source=Path(__file__).with_name('submission_model.py');shutil.copy2(source,out/'model.py');torch.save(states,out/'model.pt')
    spec=importlib.util.spec_from_file_location('standalone_candidate',out/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=mod.build_model().cuda().eval();model.load_state_dict(core.load_checkpoint(out/'model.pt'),strict=True)
    data=core.Data(root/'cache','cuda',True)
    inference_spec={'kind':'modal','news':True,'reports':True,'representation':'stationary'}
    actual=core.predict(model,data,inference_spec,512)
    np.testing.assert_allclose(actual,prediction,rtol=2e-5,atol=2e-6)
    # Check actual official-reader, unlabeled samples against the fast training representation.
    dataset=HS300Dataset(args.package,'val');split=make_validation_split(dataset)
    rng=np.random.default_rng(20261101);positions=np.unique(np.r_[np.linspace(0,len(y)-1,24,dtype=int),rng.integers(len(y),size=24)])
    raw=[{k:v for k,v in dataset[int(split.dev_indices[i])].items() if k!='y'} for i in positions]
    prepared=mod.prepare_batch(raw,'cuda');expected,_=data.batch('dev',positions,inference_spec)
    for key in ('market','anchor','has_news','news_count','has_report'):
        torch.testing.assert_close(prepared[key],expected[key],rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(prepared['reports'][:,24:],expected['reports'][:,24:],rtol=0,atol=0)
    a=prepared['news_unique'][prepared['news_inverse']].reshape(len(raw),96,512)
    b=expected['news_unique'][expected['news_inverse']].reshape(len(raw),96,512)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    with torch.inference_mode():raw_prediction=model(prepared).cpu().numpy()
    np.testing.assert_allclose(raw_prediction,actual[positions],rtol=2e-5,atol=2e-6)
    model.cpu();cpu=mod.prepare_batch(raw[:3],'cpu')
    with torch.inference_mode():cpu_prediction=model(cpu).numpy()
    np.testing.assert_allclose(cpu_prediction,raw_prediction[:3],rtol=3e-5,atol=3e-6)
    missing=[dict(raw[0],news=None,financial={})]
    with torch.inference_mode():fallback=model(mod.prepare_batch(missing,'cpu'))
    assert fallback.shape==(1,24) and torch.isfinite(fallback).all()
    model.cuda()
    diagnostics={}
    for name,flags in [('news_semantics_zero',{'zero_semantics':True}),('financial_structure_disabled',{'disable_reports':True})]:
        altered=core.predict(model,data,inference_spec,512,flags=flags);s=core.scores(altered,y)
        s['mae_increase']=s['mae']-score['mae'];s['max_prediction_change']=float(np.abs(altered-prediction).max());diagnostics[name]=s
    historical={'mae':.215003544,'mse':.129781134}
    summary={'model':'B2_three_modal_equal','seeds':seeds,'members_per_seed':members,'development':score,'matched_news_average':ref,
             'single_seed_scores':seed_scores,'checks':checks,'passed':all(checks.values()),'diagnostics':diagnostics,
             'relative_to_historical_patch_percent':{k:100*(1-score[k]/historical[k]) for k in ('mae','mse')},
             'relative_to_e0_percent':{k:100*(1-score[k]/e0[k]) for k in ('mae','mse')},
             'parameters':sum(p.numel() for p in model.parameters()),'holdout_evaluated':False,
             'financial_input':'availability/missing masks and report age; numeric content unused',
             'validation':{'official_reader_unlabeled_samples':len(raw),'cpu_gpu_match':True,'missing_modalities_finite':True,
                           'full_dev_checkpoint_match':True,'max_raw_prediction_difference':float(np.abs(raw_prediction-actual[positions]).max())},
             'model_py_sha256':core.digest(out/'model.py'),'model_pt_sha256':core.digest(out/'model.pt')}
    core.write(root/'ensemble/final_summary.json',summary);np.save(root/'ensemble/dev_predictions.npy',prediction)
    with zipfile.ZipFile(root/'submission_candidate.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ('model.py','model.pt'):archive.write(out/name,name)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
