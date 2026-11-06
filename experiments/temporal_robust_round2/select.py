"""Deterministic development/rolling decisions; never opens holdout data."""
import argparse
from pathlib import Path
import json
import numpy as np
from experiments.gpu_stage_optimization import train as core
from .run import SCOPES


def records(root,name,seed=42):
    return {scope:json.loads((root/name/f'seed{seed}'/scope/'result.json').read_text()) for scope in SCOPES}


def aggregate(record):
    values=[record[s]['validation'] for s in SCOPES]
    ratios=[v['mae']/v['e0']['mae'] for v in values]
    checks={'mean_mae_0p1_percent':np.mean(ratios)<=.999,
            'scope_mse_no_worse':all(v['mse']<=v['e0']['mse']+1e-10 for v in values),
            'all_block_mae':all(b['mae_ratio_e0']<=1.0025 for v in values for b in v['blocks']),
            'all_block_mse':all(b['mse_ratio_e0']<=1.005 for v in values for b in v['blocks'])}
    return {'eligible':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},
            'max_scope_objective':max(v['objective'] for v in values),'mean_mae_ratio_e0':float(np.mean(ratios)),
            'development':record['development']['validation']}


def rank(record):
    a=aggregate(record)
    return (not a['eligible'],a['max_scope_objective'],a['mean_mae_ratio_e0'])


def compare(candidate,reference):
    cs=[candidate[s]['validation'] for s in SCOPES];rs=[reference[s]['validation'] for s in SCOPES]
    checks={'mean_mae_0p1_percent':np.mean([c['mae']/r['mae'] for c,r in zip(cs,rs)])<=.999,
            'scope_mse_no_worse':all(c['mse']<=r['mse']+1e-10 for c,r in zip(cs,rs)),
            'all_block_mae':all(cb['mae']<=rb['mae']*1.0025 for c,r in zip(cs,rs) for cb,rb in zip(c['blocks'],r['blocks']))}
    return {'passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['market','modal','final'],required=True)
    args=p.parse_args();root=args.root
    decisions=[]
    def pair(stage,base,candidate,controls=(),content=None):
        b=records(root,base);c=records(root,candidate);gate=compare(c,b)
        for control in controls:gate['checks']['beats_'+control]=compare(c,records(root,control))['passed']
        if content:
            gate['checks']['content']=all(c['development']['diagnostics'].get(k,{}).get('mae_increase',0)>=1e-5 for k in content)
        gate['passed']=all(gate['checks'].values())
        choice=candidate if gate['passed'] else base
        decision={'stage':stage,'baseline':base,'candidate':candidate,'selected':choice,**gate,
                  'baseline_aggregate':aggregate(b),'candidate_aggregate':aggregate(c)}
        decisions.append(decision);return choice
    if args.phase=='market':
        names=[pair('E1','E1_MAE','E1_Huber'),pair('E2','E2_GRU64','E2_GRU16'),
               pair('E3','E3_drop01','E3_drop03'),pair('E4','E4_full','E4_small'),
               pair('E5','E5_layers2','E5_layers1'),pair('E6','E6_raw','E6_stationary')]
        from .debias import FOLLOWUPS
        for i,followup in enumerate(FOLLOWUPS):
            control=followup['id'].removeprefix('S_')
            chosen=pair('E'+str(i+1)+'_reflection',control,followup['id'])
            if chosen==followup['id']:names.append(chosen)
        name=min(names,key=lambda n:rank(records(root,n)))
        r=records(root,name);value={'id':name,'spec':r['development']['spec'],'aggregate':aggregate(r),
                                  'selection':'Registered rolling/development criteria; no holdout selection','decisions':decisions}
        core.write(root/'market_selection.json',value)
        print(json.dumps(value,indent=2),flush=True)
    elif args.phase=='modal':
        name=pair('E7','E7_daily','E7_pooled',['E7_coverage'],['news'])
        report=pair('E8','E8_values','E8_ratios',['E8_masks'],['reports'])
        spectral=pair('E10','E10_modes13','E10_modes3',[name],['news'])
        # For joint training keep a content-bearing branch even when evidence is insufficient;
        # status remains explicitly unaccepted and is not silently turned into a content claim.
        value={'news':{'id':name,'spec':records(root,name)['development']['spec']},
               'reports':{'id':report,'spec':records(root,report)['development']['spec']},
               'spectral':spectral,'decisions':decisions}
        core.write(root/'modal_selection.json',value);print(json.dumps(value,indent=2),flush=True)
    else:
        name=pair('E9','E9_dropout01','E9_dropout05',content=['news','reports'])
        market=json.loads((root/'market_selection.json').read_text())
        modal=json.loads((root/'modal_selection.json').read_text())
        joint=records(root,name)
        controls=[market['id'],modal['news']['id'],modal['reports']['id']]
        accepted=aggregate(joint)['eligible'] and all(compare(joint,records(root,n))['passed'] for n in controls)
        accepted=accepted and all(joint['development']['diagnostics'].get(k,{}).get('mae_increase',0)>=1e-5 for k in ('news','reports'))
        value={'market':market['id'],'market_eligible':market['aggregate']['eligible'],
               'multimodal':name,'multimodal_accepted':bool(accepted),'multimodal_aggregate':aggregate(joint),
               'spec_market':records(root,market['id'])['development']['spec'],
               'spec_multimodal':joint['development']['spec'],'stage_E9':decisions,
               'seeds':[42,43,44],'average':'fixed equal prediction average',
               'test_candidates':[market['id'],name],'holdout_evaluated':False,
               'status':'All stages frozen before replication and final holdout batch; no selection after test.'}
        core.write(root/'FINAL_SELECTION.json',value);print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':main()
