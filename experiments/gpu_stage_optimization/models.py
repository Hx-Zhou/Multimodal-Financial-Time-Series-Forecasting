import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.e1_dlinear.model import DLinear
from experiments.e1_dlinear.e6_plus.model import Heads
from experiments.e4_patchtst_capacity.model import ScaleAwarePatchTST, ScaleAwareLinear
from experiments.e5_itransformer.model import ITransformer
from experiments.e10_dlinear_spectf.model import Model as SpectralModel

class CloseLinear(DLinear):
    def __init__(self):
        super().__init__(25,True)
        self.register_buffer('scale',torch.arange(1,25).float()[None]/24)
    def forward(self,b):
        anchor=b['close'][:,-1:]
        t,r=self.decompose(b['close']-anchor)
        return anchor+self.scale*(self.trend_head(t)+self.remainder_head(r))

class MarketLinear(nn.Module):
    def __init__(self):
        super().__init__(); self.market=Heads(7)
        self.register_buffer('scale',torch.arange(1,25).float()[None]/24)
    def forward(self,b): return b['anchor']+self.scale*self.market(b['market'])

class GRU(nn.Module):
    def __init__(self,channels=1,pool='last',channel_dropout=0.):
        super().__init__(); self.channels=channels; self.pool=pool; self.channel_dropout=channel_dropout
        self.gru=nn.GRU(channels,64,batch_first=True); self.head=nn.Linear(64,24)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.register_buffer('scale',torch.arange(1,25).float()[None]/24)
    def forward(self,b):
        anchor=b['anchor']; x=(b['close']-anchor)[:,:,None] if self.channels==1 else b['market']
        if self.training and self.channel_dropout:
            keep=(torch.rand((len(x),1,self.channels-1),device=x.device)>=self.channel_dropout).to(x.dtype)/(1-self.channel_dropout)
            x=torch.cat((x[:,:,:-1]*keep,x[:,:,-1:]),-1)
        out,_=self.gru(x); z=out[:,-1] if self.pool=='last' else out.mean(1)
        return anchor+self.scale*self.head(z)

class News(nn.Module):
    def __init__(self,normalize=False):
        super().__init__(); self.normalize=normalize
        self.mlp=nn.Sequential(nn.Linear(512,8),nn.SiLU(),nn.Linear(8,8)); self.heads=Heads(10)
        self.gate=nn.Parameter(torch.tensor(-2.))
    def forward(self,b):
        x=F.normalize(b['news_unique'],dim=-1) if self.normalize else b['news_unique']
        z=self.mlp(x)[b['news_inverse']].reshape(-1,96,8)*b['has_news'][...,None]
        if b.get('zero_semantics',False): z=torch.zeros_like(z)
        x=torch.cat((z,b['has_news'][...,None],b['news_count'][...,None]),-1)
        return self.heads(x)*torch.sigmoid(self.gate)*b['has_news'].any(1,keepdim=True)

class Reports(nn.Module):
    def __init__(self,linear=False):
        super().__init__(); self.encoder=nn.Identity() if linear else nn.Sequential(nn.Linear(54,16),nn.SiLU(),nn.Linear(16,8),nn.SiLU())
        self.head=nn.Linear(54 if linear else 8,24); self.gate=nn.Parameter(torch.tensor(-2.))
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,b):
        x=b['reports']
        if b.get('zero_report_values',False): x=torch.cat((torch.zeros_like(x[:,:24]),x[:,24:]),-1)
        return self.head(self.encoder(x))*torch.sigmoid(self.gate)*b['has_report']

class Modal(MarketLinear):
    def __init__(self,news=False,reports=False,normalize=False,linear_reports=False):
        super().__init__(); self.with_news=news; self.with_reports=reports; self.market.requires_grad_(False)
        # Separate fixed streams make same-modality initialization comparable in single/joint runs.
        with torch.random.fork_rng():
            torch.manual_seed(torch.initial_seed()+101)
            if news: self.news=News(normalize)
        with torch.random.fork_rng():
            torch.manual_seed(torch.initial_seed()+202)
            if reports: self.reports=Reports(linear_reports)
    def drop(self,x):
        if self.training: return x*(torch.rand((len(x),1),device=x.device)>=.1).to(x.dtype)/.9
        return x
    def forward(self,b):
        delta=self.market(b['market'])
        if self.with_news and not b.get('disable_news',False): delta=delta+self.drop(self.news(b))
        if self.with_reports and not b.get('disable_reports',False): delta=delta+self.drop(self.reports(b))
        return b['anchor']+self.scale*delta

def build(spec):
    kind=spec['kind']
    if kind=='close': return CloseLinear()
    if kind=='market': return MarketLinear()
    if kind=='gru': return GRU(spec.get('channels',1),spec.get('pool','last'),spec.get('channel_dropout',0.))
    if kind=='patch': return ScaleAwarePatchTST(d_model=128,layers=2,nhead=4,d_ff=spec.get('d_ff',256),dropout=.3)
    if kind=='patch_control': return ScaleAwareLinear()
    if kind=='itransformer': return ITransformer(layers=spec.get('layers',2))
    if kind=='modal': return Modal(spec.get('news',False),spec.get('reports',False),spec.get('normalize',False),spec.get('linear_reports',False))
    if kind=='spectral':
        model=SpectralModel(); model.modes=spec.get('modes',13); return model
    raise ValueError(kind)
