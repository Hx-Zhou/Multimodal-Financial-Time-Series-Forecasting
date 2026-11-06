"""Consolidate existing experiments and recompute saved predictions; no inference or fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
R1=ROOT/'experiments/gpu_stage_optimization'
R2=ROOT/'experiments/temporal_robust_round2/results'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(p,y):
    assert p.shape==y.shape and np.isfinite(p).all()
    d=p.astype(np.float64)-y.astype(np.float64)
    return {'mae':float(np.abs(d).mean()),'mse':float(np.square(d).mean())}
def pair(s):return {k:float(s[k]) for k in ('mae','mse')}
def compare(c,b):return {k:100*(1-c[k]/b[k]) for k in ('mae','mse')}

def main():
    package=ROOT.parents[1]/'hs300_train_val'
    calendar=np.load(ROOT/'experiments/temporal_robust_round2/calendar.npy')
    offsets=np.r_[0,np.flatnonzero(np.diff(calendar)<=0)+1,len(calendar)]
    with np.load(package/'data/val.npz') as z:
        starts=offsets[z['stock_index']]+z['input_start']
        ends=calendar[starts+119];dates=z['target_start_date'];y=z['y'].copy()
        dev=np.flatnonzero((dates<20240124)&(ends<20240124));hold=np.flatnonzero(dates>=20240124)
    assert len(dev)==42930 and len(hold)==15390
    np.testing.assert_array_equal(hold,np.load(R2/'final_holdout/sample_indices.npy'))
    prior=read(ROOT/'experiments/e1_dlinear/e6_plus/results/summary.json')
    oldtest=read(ROOT/'experiments/e1_dlinear/e6_plus/results/holdout/summary.json')
    rows=[]
    def add(key,name,development,holdout,source,averaging='three_seed_prediction_average'):
        rows.append({'id':key,'name':name,'development':pair(development),'internal_holdout':pair(holdout),
                     'source':source,'averaging':averaging})
    add('E0','最后值基线',prior['e0_development'],oldtest['scores']['e0'],
        'experiments/e1_dlinear/e6_plus/results/holdout/summary.json','no_training')
    add('E1','历史增强 DLinear',prior['previous_e1_development'],oldtest['scores']['e1'],
        'experiments/e1_dlinear/e6_plus/results/holdout/summary.json')
    for stage in ('e6','e7'):
        predictions=np.mean([np.load(ROOT/f'experiments/e1_dlinear/e6_plus/outputs/{stage}_stationary_seed{seed}/dev_predictions.npy') for seed in (42,43,44)],axis=0)
        add(stage.upper(),'历史平稳化 DLinear' if stage=='e6' else '历史 DLinear 加新闻',metric(predictions,y[dev]),oldtest['scores'][stage+'_average'],
            'experiments/e1_dlinear/e6_plus/results/holdout/summary.json')
    for stage,folder,name in [('E8','e8_dlinear_reports','历史 DLinear 加财报'),('E9','e9_dlinear_multimodal','历史新闻财报融合'),('E10','e10_dlinear_spectf','历史频域融合')]:
        folder=ROOT/'experiments'/folder
        add(stage,name,read(folder/'results/development.json')['equal_average'],read(folder/'results/holdout.json')['equal_average'],
            (folder/'results/holdout.json').relative_to(ROOT).as_posix())
    patch=ROOT/'experiments/transformer_optimization_common/results'
    add('PatchTST','历史优化 PatchTST',read(patch/'replication.json')['development'],read(patch/'holdout.json')['scores']['transformer'],
        'experiments/transformer_optimization_common/results/holdout.json')
    b2dev=read(R1/'ensemble/final_summary.json');b2test=read(R1/'holdout_20261101/summary.json')
    add('GPU1_B2','第一批 GPU 固定 B2 组合',b2dev['development'],b2test['internal_holdout'],
        'experiments/gpu_stage_optimization/holdout_20261101/summary.json','three_seeds_times_three_members_equal_average')
    freeze=read(R2/'TEST_FREEZE.json');latest=read(R2/'final_holdout/summary.json')
    for label,c in freeze['candidates'].items():
        add('GPU2_'+label,'最新小型 GRU' if label=='market' else '最新多模态实验配置',
            c['validation_scores']['development'],latest['results'][label]['score'],
            'experiments/temporal_robust_round2/results/final_holdout/summary.json')
    verification={}
    saved={
        'E0':ROOT/'experiments/e1_dlinear/e6_plus/outputs/holdout/e0.npy',
        'E1':ROOT/'experiments/e1_dlinear/e6_plus/outputs/holdout/e1.npy',
        'E6':ROOT/'experiments/e1_dlinear/e6_plus/outputs/holdout/e6_average.npy',
        'E7':ROOT/'experiments/e1_dlinear/e6_plus/outputs/holdout/e7_average.npy',
        'GPU1_B2':R1/'holdout_20261101/predictions.npy',
        'GPU2_market':R2/'final_holdout/market_predictions.npy',
        'GPU2_multimodal':R2/'final_holdout/multimodal_predictions.npy',
    }
    for key,path in saved.items():
        actual=metric(np.load(path),y[hold]);reference=next(r for r in rows if r['id']==key)['internal_holdout']
        for k in ('mae','mse'):np.testing.assert_allclose(actual[k],reference[k],rtol=0,atol=1e-12)
        verification[key]={'recomputed':actual,'prediction_sha256':digest(path),
                           'source':path.relative_to(ROOT).as_posix(),'matches_record':True}
    checkpoints={}
    for seed in (42,43,44):
        p=ROOT/f'experiments/e1_dlinear/e6_plus/outputs/e6_stationary_seed{seed}/model.pt'
        actual=digest(p);assert actual==oldtest['checkpoint_sha256'][f'e6_stationary_seed{seed}']
        checkpoints[f'historical_E6_seed{seed}']=actual
    for label,c in freeze['candidates'].items():
        for f,key in [('model.py','model_py_sha256'),('model.pt','model_pt_sha256')]:
            actual=digest(R2/'candidates'/label/f);assert actual==c[key]
            checkpoints['latest_'+label+'_'+f]=actual
    for f,key in [('model.py','model_py_sha256'),('model.pt','model_pt_sha256')]:
        actual=digest(R1/'candidate'/f);assert actual==b2test[key]
        checkpoints['first_B2_'+f]=actual
    runs=[]
    counts={}
    for folder in ('results','followup_results','replications'):
        files=sorted((R1/folder).glob('*/seed*/result.json'));counts['gpu1_'+folder]=len(files)
        for p in files:
            r=read(p);runs.append({'round':'gpu1','configuration':r['id'],'seed':r['seed'],'scope':'development',
                                  'mae':r['development']['mae'],'mse':r['development']['mse'],
                                  'best_epoch':r['best_epoch'],'epochs_run':r['epochs_run'],
                                  'checkpoint_sha256':r['checkpoint_sha256'],'source':p.relative_to(ROOT).as_posix()})
    files=sorted(R2.glob('*/seed*/*/result.json'));counts['gpu2_all_scopes']=len(files)
    for p in files:
        r=read(p);runs.append({'round':'gpu2','configuration':r['id'],'seed':r['seed'],'scope':r['scope'],
                              'mae':r['validation']['mae'],'mse':r['validation']['mse'],
                              'best_epoch':r['best_epoch'],'epochs_run':r['epochs_run'],
                              'checkpoint_sha256':r['checkpoint_sha256'],'source':p.relative_to(ROOT).as_posix()})
    assert sum(counts.values())==148
    best=min(rows,key=lambda r:r['internal_holdout']['mae']);assert best['id']=='E6'
    assert min(rows,key=lambda r:r['internal_holdout']['mse'])['id']=='E6'
    final=next(r for r in rows if r['id']=='GPU2_market')
    e0=next(r for r in rows if r['id']=='E0')
    summary={
        'compiled_date':'2026-11-02 Asia/Shanghai',
        'task':{'input_trading_days':96,'forecast_trading_days':24,'target':'standardized Close',
                'train_samples':587587,'development_samples':42930,'purged_samples':18630,'internal_holdout_samples':15390,
                'holdout_target_start_min':20240124,'holdout_target_start_max':20240227,'holdout_target_end_max':20240329},
        'counts':{**counts,'gpu1_total':40,'gpu2_total':108,'total_new_training_runs':148,
                  'note':'Only the two newly executed GPU batches; pre-existing historical experiments are not counted again.'},
        'models':rows,'historical_best_observed_research_result':best,
        'last_round_frozen_model_result':final,
        'best_vs_e0_improvement_percent':compare(best['internal_holdout'],e0['internal_holdout']),
        'latest_vs_e0_improvement_percent':compare(final['internal_holdout'],e0['internal_holdout']),
        'official_hidden_test_result':None,'official_course_grade':None,
        'interpretation':{
            'score_policy':'Report MAE and MSE separately. Historical best research score and last frozen model score are different records; do not relabel one as the other.',
            'historical_best':'Descriptive minimum among listed fixed prediction averages, not a new selection made on an unseen test.',
            'latest_multimodal':'All three news and financial output heads are identically zero; effective model equals the market GRU and is not a validated effective three-modal submission.',
            'first_b2':'Uses news semantics/coverage and financial availability/missingness/age; no demonstrated gain from financial numerical content.',
            'holdout':'Already observed internal temporal holdout from public validation; repeated research is not an independent new blind test.',
            'submission':'No newly demonstrated improvement meeting both predictive and effective three-modal requirements; official grade unavailable.'},
        'verification':{'saved_predictions_recomputed':verification,'checkpoint_hashes':checkpoints,
                        'holdout_sample_indices_sha256':digest(R2/'final_holdout/sample_indices.npy'),
                        'val_npz_sha256':digest(package/'data/val.npz'),
                        'no_training_or_new_inference_in_this_summary':True},
    }
    OUT.mkdir(exist_ok=True)
    (OUT/'scores.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'runs.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'runs':len(runs),'historical_best':best,'last_round':final,'saved_prediction_checks':len(verification)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
