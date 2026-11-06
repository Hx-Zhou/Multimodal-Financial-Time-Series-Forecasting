"""Model contracts and CPU/CUDA parity, with nonzero outputs and missing modalities."""
from copy import deepcopy
import json
import torch
from .models import build

def main():
    torch.set_num_threads(2); torch.manual_seed(123)
    specs=[{'kind':'close'},{'kind':'market'},{'kind':'gru','channels':1},
           {'kind':'gru','channels':7,'pool':'mean','channel_dropout':.1},
           {'kind':'patch','d_ff':128},{'kind':'itransformer','layers':1},
           {'kind':'modal','news':True,'normalize':True},
           {'kind':'modal','reports':True,'linear_reports':True},
           {'kind':'modal','news':True,'reports':True},{'kind':'spectral','modes':7}]
    b={'close':torch.randn(3,96),'anchor':torch.randn(3,1),'market':torch.randn(3,96,7),
       'news_unique':torch.randn(12,512),'news_inverse':torch.randint(0,12,(3*96,)),
       'has_news':torch.ones(3,96),'news_count':torch.ones(3,96),'reports':torch.randn(3,54),'has_report':torch.ones(3,1),'alpha':.5}
    b['anchor']=b['close'][:,-1:]
    results=[]
    for spec in specs:
        model=build(spec).eval()
        with torch.no_grad():
            # Make all forecast heads nonzero: zero-init-only parity would be uninformative.
            for name,p in model.named_parameters():
                if p.requires_grad and ('head' in name or 'trend' in name or 'remainder' in name): p.normal_(0,.005)
        pred=model(b)
        assert pred.shape==(3,24) and torch.isfinite(pred).all()
        loss=pred.square().mean(); loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        row={'kind':spec['kind'],'cpu_finite':True}
        if torch.cuda.is_available():
            gpu=deepcopy(model).cuda().eval(); gb={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()}
            actual=gpu(gb).cpu(); torch.testing.assert_close(actual,pred,rtol=3e-4,atol=3e-5)
            row['cuda_max_difference']=float((actual-pred).abs().max())
        if spec['kind'] in ('modal','spectral'):
            missing=dict(b,has_news=torch.zeros_like(b['has_news']),has_report=torch.zeros_like(b['has_report']))
            expected=b['anchor']+model.scale*model.market(b['market'])
            torch.testing.assert_close(model(missing),expected,rtol=0,atol=0)
            row['missing_modalities_exact_fallback']=True
        results.append(row)
    print(json.dumps({'cuda':torch.cuda.is_available(),'results':results},indent=2))

if __name__=='__main__': main()
