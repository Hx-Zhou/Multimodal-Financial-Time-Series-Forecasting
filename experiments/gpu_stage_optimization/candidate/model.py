"""Standalone HS300 candidate: fixed three-seed, three-member prediction average.

Financial input uses availability masks and report age. Financial numeric
content has not demonstrated incremental value in the development experiments.
"""
from datetime import date
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MARKET_MEAN=np.array([0.00023265135877211565,0.034267626780731766,0.0009141812592297254,16.719754731039984])
MARKET_STD=np.array([0.025932152768513905,0.021643513053313603,0.024242647191207608,1.4045918433208695])
FIELDS={
 'income':('total_revenue','n_income_attr_p','basic_eps','operate_profit'),
 'balance_sheet':('total_assets','total_liab','total_hldr_eqy_exc_min_int','money_cap'),
 'cashflow':('n_cashflow_act','n_cashflow_inv_act','n_cash_flows_fnc_act','c_cash_equ_end_period'),
}

def _ordinal(value):
    text=str(value).replace('-','')[:8]
    return date(int(text[:4]),int(text[4:6]),int(text[6:8])).toordinal()

def _finite(value):return value is not None and isinstance(value,(int,float)) and np.isfinite(value)

def _report_structure(financial,cutoff):
    features=np.zeros(54,np.float32);end_day=_ordinal(cutoff)
    for k,(kind,fields) in enumerate(FIELDS.items()):
        try:rows=financial[kind]
        except (KeyError,TypeError):rows=[]
        groups={}
        for row in rows:groups.setdefault(int(row['end_date']),[]).append(row)
        records={}
        for period,group in groups.items():
            availability=[]
            for row in group:
                actual=[_ordinal(row[key]) for key in ('ann_date','f_ann_date') if row.get(key)]
                availability.append(max(actual) if actual else _ordinal(period)+180)
            available=max(max(availability),_ordinal(period)); valid=[]
            for field in fields:
                values={float(row[field]) for row in group if _finite(row.get(field))}
                valid.append(len(values)==1 and all(_finite(row.get(field)) for row in group))
            records[period]=(available,valid)
        eligible=[period for period,(available,_) in records.items() if available<=end_day]
        if not eligible:continue
        current=max(eligible);_,valid=records[current]
        features[24+k*8:28+k*8]=valid
        prior=records.get(current-10000)
        if prior is not None and prior[0]<=end_day:
            features[28+k*8:32+k*8]=np.asarray(valid)&np.asarray(prior[1])
        features[48+k]=min((end_day-_ordinal(current))/365.25,5)
        features[51+k]=1
    return features

def prepare_batch(samples,device='cpu'):
    if not samples:raise ValueError('samples must not be empty')
    market=[];anchors=[];reports=[];news_ids=[];news_counts=[]
    pooled=[np.zeros(512,np.float32)];day_cache={};report_cache={}
    for s in samples:
        prices=np.asarray(s['numerical'],np.float32)
        if prices.shape!=(96,7) or not np.isfinite(prices).all():raise ValueError('Expected finite numerical [96,7]')
        close=prices[:,3];mean=np.float32(s['close_mean']);std=max(np.float32(s['close_std']),np.float32(1e-6))
        previous=np.r_[close[:1],close[:-1]]
        values=np.column_stack((np.log(np.maximum(close,1e-6)/np.maximum(previous,1e-6)),
             (prices[:,1]-prices[:,2])/np.maximum(close,1e-6),
             (close-prices[:,0])/np.maximum(prices[:,0],1e-6),np.log1p(np.maximum(prices[:,4],0)),
             (prices[:,5]!=0).astype(np.float32),(prices[:,6]!=0).astype(np.float32),(close-mean)/std)).astype(np.float32)
        values[:,:4]=((values[:,:4].astype(np.float64)-MARKET_MEAN)/MARKET_STD).astype(np.float32)
        anchor=values[-1,6].copy();values[:,6]-=anchor;values[0,0]=np.float32(-MARKET_MEAN[0]/MARKET_STD[0])
        market.append(values);anchors.append([anchor])
        ids=[];counts=[];news=s.get('news');stock=s.get('stock_code',id(news))
        for i,day in enumerate(s['dates']):
            key=(stock,int(day))
            if key not in day_cache:
                if news is None:emb=np.empty((0,512),np.float32)
                elif all(hasattr(news,key) for key in ('_ids','_vectors','_range')):
                    left,right=news._range(int(day));unique=np.unique(news._ids[left:right]);emb=news._vectors[unique]
                else:emb=np.unique(np.asarray(news[i],np.float32),axis=0)
                if len(emb):
                    index=len(pooled);pooled.append(emb.mean(axis=0,dtype=np.float64).astype(np.float32));day_cache[key]=(index,len(emb))
                else:day_cache[key]=(0,0)
            index,count=day_cache[key];ids.append(index);counts.append(count)
        news_ids.append(ids);news_counts.append(counts)
        rk=(stock,int(s['dates'][-1]))
        if rk not in report_cache:report_cache[rk]=_report_structure(s.get('financial',{}),int(s['dates'][-1]))
        reports.append(report_cache[rk])
    ids=np.asarray(news_ids,np.int64);financial=np.stack(reports)
    values={'market':np.stack(market),'anchor':np.asarray(anchors,np.float32),'news_unique':np.stack(pooled),
            'news_inverse':ids.reshape(-1),'has_news':(ids!=0).astype(np.float32),
            'news_count':np.log1p(np.asarray(news_counts,np.float32)),'reports':financial,
            'has_report':financial[:,-3:].any(axis=1,keepdims=True).astype(np.float32)}
    return {k:torch.as_tensor(v,device=device) for k,v in values.items()}

class Heads(nn.Module):
    def __init__(self,channels):
        super().__init__();self.trend=nn.Linear(96*channels,24);self.remainder=nn.Linear(96*channels,24)
    def forward(self,x):
        trend=F.avg_pool1d(F.pad(x.transpose(1,2),(12,12),mode='replicate'),25,stride=1).transpose(1,2)
        return self.trend(trend.flatten(1))+self.remainder((x-trend).flatten(1))

class News(nn.Module):
    def __init__(self,coverage=False):
        super().__init__();self.coverage=coverage
        self.mlp=nn.Sequential(nn.Linear(512,8),nn.SiLU(),nn.Linear(8,8));self.heads=Heads(10);self.gate=nn.Parameter(torch.tensor(-2.))
    def forward(self,b):
        if self.coverage or b.get('zero_semantics',False):z=torch.zeros((*b['has_news'].shape,8),device=b['market'].device)
        else:z=self.mlp(b['news_unique'])[b['news_inverse']].reshape(-1,96,8)*b['has_news'][...,None]
        x=torch.cat((z,b['has_news'][...,None],b['news_count'][...,None]),-1)
        return self.heads(x)*torch.sigmoid(self.gate)*b['has_news'].any(1,keepdim=True)

class ReportStructure(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Identity();self.head=nn.Linear(54,24);self.gate=nn.Parameter(torch.tensor(-2.))
    def forward(self,b):
        if b.get('disable_reports',False):return torch.zeros_like(b['anchor'])
        features=torch.cat((torch.zeros_like(b['reports'][:,:24]),b['reports'][:,24:]),-1)
        return self.head(features)*torch.sigmoid(self.gate)*b['has_report']

class Member(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.market=Heads(7);self.register_buffer('scale',torch.arange(1,25).float()[None]/24)
        if kind=='reports':self.reports=ReportStructure()
        else:self.news=News(kind=='coverage')
    def forward(self,b):
        extra=self.reports(b) if self.kind=='reports' else self.news(b)
        return b['anchor']+self.scale*(self.market(b['market'])+extra)

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.members=nn.ModuleList([Member(kind) for _ in range(3) for kind in ('news','coverage','reports')])
    def forward(self,batch):return torch.stack([m(batch) for m in self.members]).mean(0)

def build_model(config=None):return Model()
