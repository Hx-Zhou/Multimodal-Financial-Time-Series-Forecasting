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

def _original_prepare_batch(samples,device='cpu'):
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


LOOKBACK,HORIZON=96,24

class DLinear(nn.Module):
    def __init__(self, moving_average: int = 25, center_last: bool = False):
        super().__init__()
        if moving_average <= 0 or moving_average % 2 == 0 or moving_average > LOOKBACK:
            raise ValueError("moving_average must be an odd integer in [1, 96]")
        self.moving_average = moving_average
        self.center_last = center_last
        self.trend_head = nn.Linear(LOOKBACK, HORIZON)
        self.remainder_head = nn.Linear(LOOKBACK, HORIZON)
        # Begin at the E0 last-value baseline, then learn the residual correction.
        for layer in (self.trend_head, self.remainder_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = (self.moving_average - 1) // 2
        padded = F.pad(x.unsqueeze(1), (padding, padding), mode="replicate")
        trend = F.avg_pool1d(padded, self.moving_average, stride=1).squeeze(1)
        return trend, x - trend

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["close"]
        if x.ndim != 2 or x.shape[1] != LOOKBACK:
            raise ValueError("close must have shape [B, 96]")
        last = x[:, -1:]
        # Optional E1 variant: learn changes relative to the observed endpoint.
        # Both subtraction and restoration use only the historical input.
        inputs = x - last if self.center_last else x
        trend, remainder = self.decompose(inputs)
        delta = self.trend_head(trend) + self.remainder_head(remainder)
        return last + delta


class Heads(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.trend = nn.Linear(LOOKBACK * channels, HORIZON)
        self.remainder = nn.Linear(LOOKBACK * channels, HORIZON)
        for layer in (self.trend, self.remainder):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        z = x.transpose(1, 2)
        trend = F.avg_pool1d(F.pad(z, (12, 12), mode="replicate"), 25, stride=1).transpose(1, 2)
        return self.trend(trend.flatten(1)) + self.remainder((x - trend).flatten(1))


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


def history_transform(close):
    if close.ndim != 2 or close.shape[1] != LOOKBACK or not torch.isfinite(close).all():
        raise ValueError("expected finite standardized Close [B,96]")
    anchor = close[:, -1:]
    # Input-window-only scale; no fitted validation/test statistics or future y.
    sigma = (close.var(dim=1, keepdim=True, unbiased=False)+1e-5).sqrt().detach()
    return (close-anchor)/sigma, anchor, sigma


class ResidualForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("horizon_scale", torch.arange(1, 25).float()[None]/24)

    def forward(self, batch):
        x, anchor, sigma = history_transform(batch["close"])
        return anchor+sigma*self.horizon_scale*self.residual(x)


class PatchTST(ResidualForecast):
    def __init__(self, patch_len=16, stride=8, d_model=128, nhead=4, layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        if not 0 < patch_len <= LOOKBACK or stride <= 0 or d_model % nhead:
            raise ValueError("invalid patch/attention configuration")
        self.patch_len, self.stride = patch_len, stride
        self.patches = (LOOKBACK-patch_len)//stride+1
        self.projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.empty(1, self.patches, d_model))
        nn.init.normal_(self.position, std=0.01)
        block = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.patches*d_model, HORIZON)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, x):
        tokens = self.projection(x.unfold(1, self.patch_len, self.stride))+self.position
        return self.head(self.dropout(self.encoder(tokens).flatten(1)))


class ScaleAwarePatchTST(PatchTST):
    def forward(self, batch):
        x, anchor, sigma = history_transform(batch["close"])
        return anchor+sigma*self.horizon_scale*(self.residual(x)*batch.get("alpha", 1.0))


class ITransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(96, d_model)
        self.variable = nn.Parameter(torch.empty(1, 7, d_model))
        nn.init.normal_(self.variable, std=0.01)
        block = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        for layer in self.encoder.layers:
            for p in layer.parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform_(p)
        self.head = nn.Linear(d_model, 24)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("scale", torch.arange(1, 25).float()[None]/24)

    def forward(self, batch):
        tokens = self.embedding(batch["market"].transpose(1, 2))+self.variable
        # stationary feature index6 is standardized Close; inputs already centered.
        delta = self.head(self.encoder(tokens)[:, 6])*batch.get("alpha", 1.0)
        return batch["anchor"]+self.scale*delta


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


class SymmetricForecast(Forecast):
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


def ordinal(value):
    text = str(value).replace("-", "")[:8]
    return date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()


def finite(value):
    return value is not None and isinstance(value, (int, float)) and np.isfinite(value)


def consolidate(rows, fields, lag):
    """Ambiguous revised fields are masked; never choose the last JSON row."""
    groups = {}
    for row in rows:
        end = int(row["end_date"])
        ordinal(end)  # fail on corrupt dates rather than quietly realign
        groups.setdefault(end, []).append(row)
    result, conflicts, announced = [], 0, 0
    for end, group in sorted(groups.items()):
        available = []
        for row in group:
            actual = [ordinal(row[k]) for k in ("ann_date", "f_ann_date") if row.get(k)]
            announced += bool(actual)
            # Sensitivity adds 90 days even when genuine announcement dates exist.
            available.append(max(actual) + lag - PRIMARY_LAG if actual else ordinal(end) + lag)
        values = {}
        for field in fields:
            candidates = {float(row[field]) for row in group if finite(row.get(field))}
            if len(candidates) == 1 and all(finite(row.get(field)) for row in group):
                values[field] = candidates.pop()
            else:
                values[field] = None
                conflicts += len(candidates) > 1
        result.append({"end": end, "end_ordinal": ordinal(end),
                       "available": max(max(available), ordinal(end)), "values": values})
    return result, {"rows": len(rows), "periods": len(groups), "duplicate_rows": len(rows)-len(groups),
                    "conflicting_selected_fields_masked": conflicts, "rows_with_announcement_date": announced}


def financial_block(days, records, fields):
    """Latest available period and same-quarter prior-year change at each cutoff."""
    n = len(days)
    values = np.zeros((n, 8), np.float32)
    valid = np.zeros((n, 8), bool)
    ages = np.zeros(n, np.float32)
    present = np.zeros(n, np.float32)
    by_end = {r["end"]: r for r in records}
    # Sorting by period means later periods overwrite older ones only when available.
    for current in records:
        mask = days >= current["available"]
        if not mask.any():
            continue
        values[mask] = 0
        valid[mask] = False
        present[mask] = 1
        ages[mask] = np.minimum((days[mask] - current["end_ordinal"]) / 365.25, 5)
        previous = by_end.get(current["end"] - 10000)
        for j, field in enumerate(fields):
            now = current["values"][field]
            if now is None:
                continue
            unit = 1.0 if field == "basic_eps" else 1e8
            values[mask, j] = np.sign(now) * np.log1p(abs(now) / unit)
            valid[mask, j] = True
            if previous is None or previous["values"][field] is None:
                continue
            before = previous["values"][field]
            both = mask & (days >= previous["available"])
            values[both, j+4] = 2 * (now-before) / max(abs(now)+abs(before), unit*1e-6)
            valid[both, j+4] = True
    return values, valid, ages, present


SPEC={'backbone': 'gru_market', 'hidden': 16, 'channel_dropout': 0.1, 'parent': 'E3_drop01', 'news': True, 'reports': True, 'news_mode': 'daily', 'ratios': False, 'coverage_reports': False, 'pretrained_news': 'E7_daily', 'pretrained_reports': 'E8_values', 'id': 'E9_dropout01', 'modality_dropout': 0.1}
STATIONARY_STATS={'columns': [0, 1, 2, 3], 'mean': [0.00023265135877211565, 0.034267626780731766, 0.0009141812592297254, 16.719754731039984], 'std': [0.025932152768513905, 0.021643513053313603, 0.024242647191207608, 1.4045918433208695], 'unique_training_input_days': 664537}
REPORT_STATS={'mean': [4.382832898286524, 1.822791697786449, 0.33454897380508203, 2.0234193359229558, 0.10824229902777271, 0.04088116593144235, 0.0043403736805027, 0.047008731839872565, 5.9302797367784965, 5.182756681198565, 4.9141933540860325, 3.7961992368477633, 0.11579356057022851, 0.12040967828949395, 0.0990651018011715, 0.13806153133937554, 1.3834308923404277, -1.6240086419618143, 0.14603231148336984, 3.759285281976382, 0.02390576026362044, -0.08142143421431965, 0.05015286808427208, 0.135509093125454], 'std': [1.6519836667084065, 1.8923567150986604, 0.46120192993025155, 1.990849728824996, 0.35138813858372425, 0.9252222860366969, 0.9240323130758781, 0.922800405157013, 1.65580705505769, 1.972530067298223, 1.4608089117608574, 1.421566337999233, 0.20864685077200787, 0.28016330375536025, 0.319349757824053, 0.45405665690797037, 2.7785727414606263, 2.330066640384505, 2.8858641886287324, 1.5964010462158418, 1.1913146351964967, 1.251726963487872, 1.452720586785079, 0.4950455971705761], 'observed_training_counts': [586701, 586325, 586440, 586445, 585672, 585172, 584629, 585352, 585732, 586169, 585552, 566152, 581578, 582048, 581250, 562631, 587224, 587164, 582185, 587165, 586882, 586763, 576499, 586763]}
RATIO_STATS={'mean': [0.5256223344363713, 0.43955805085086336, 0.16913994318568037, 5.9302797367784965, 0.09230561959645336, 0.12803783980764502, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'std': [0.2219427156386927, 0.22741522254887117, 0.11619044382576629, 1.65580705505769, 0.6009116221165035, 0.5327166484445608, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06], 'observed_training_counts': [585552, 585552, 564641, 585732, 585546, 585666, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
"""Source fragment appended to generated standalone model.py."""

def _financial_features(financial, cutoff):
    end=ordinal(cutoff); values=np.zeros(24,np.float32);valid=np.zeros(24,bool);age=np.zeros(6,np.float32)
    all_records={}
    for k,(kind,fields) in enumerate(FIELDS.items()):
        try:rows=financial[kind]
        except (KeyError,TypeError):rows=[]
        records,_=consolidate(rows,fields,180);all_records[kind]=records
        block,mask,a,p=financial_block(np.array([end]),records,fields)
        values[k*8:k*8+8]=block[0];valid[k*8:k*8+8]=mask[0]
        age[k]=a[0];age[k+3]=p[0]
    if SPEC.get('ratios'):
        values[:]=0;valid[:]=False
        for kind in ('balance_sheet','income'):
            eligible=[r for r in all_records[kind] if r['available']<=end]
            if not eligible:continue
            v=eligible[-1]['values']
            if kind=='balance_sheet':
                den=v.get('total_assets')
                if den is not None and den>0:
                    for slot,key in enumerate(('total_liab','total_hldr_eqy_exc_min_int','money_cap')):
                        if v.get(key) is not None:values[slot]=v[key]/den;valid[slot]=True
                    values[3]=np.log1p(den/1e8);valid[3]=True
            else:
                den=v.get('total_revenue')
                if den is not None and den>0:
                    for slot,key in zip((4,5),('n_income_attr_p','operate_profit')):
                        if v.get(key) is not None:values[slot]=v[key]/den;valid[slot]=True
    stats=RATIO_STATS if SPEC.get('ratios') else REPORT_STATS
    mean=np.asarray(stats['mean']);std=np.asarray(stats['std'])
    normalized=np.clip((values.astype(np.float64)-mean)/std,-10,10).astype(np.float32)
    normalized[~valid]=0
    return np.r_[normalized,valid.astype(np.float32),age].astype(np.float32)


def prepare_batch(samples,device='cpu'):
    b=_original_prepare_batch(samples,device)
    prices=torch.as_tensor(np.stack([np.asarray(s['numerical'],np.float32) for s in samples]),device=device)
    mean=torch.as_tensor([np.float32(s['close_mean']) for s in samples],device=device)[:,None]
    std=torch.as_tensor([max(np.float32(s['close_std']),np.float32(1e-6)) for s in samples],device=device)[:,None]
    close=(prices[:,:,3]-mean)/std;anchor=close[:,-1:]
    risk=close.std(1,keepdim=True,unbiased=False).clamp_min(1e-4)
    logv=torch.log1p(prices[:,:,4].clamp_min(0))
    market=torch.stack((((prices[:,:,0]-mean)/std-anchor)/risk,
                         ((prices[:,:,1]-mean)/std-anchor)/risk,
                         ((prices[:,:,2]-mean)/std-anchor)/risk,
                         logv-logv.mean(1,keepdim=True),
                         (prices[:,:,5]!=0).float(),(prices[:,:,6]!=0).float(),(close-anchor)/risk),-1)
    if SPEC.get('representation')=='stationary':
        first=b['market'][:,:,:4].clone()
        sm=torch.as_tensor(STATIONARY_STATS['mean'],device=device,dtype=first.dtype)
        ss=torch.as_tensor(STATIONARY_STATS['std'],device=device,dtype=first.dtype)
        first=first*ss+sm;first[:,0,0]=0
        first=(first-first.mean(1,keepdim=True))/first.std(1,keepdim=True,unbiased=False).clamp_min(1e-5)
        market=torch.cat((first,market[:,:,4:]),-1)
    b.update(close=close,anchor=anchor,risk=risk,close_normalized=(close-anchor)/risk,market=market.clamp(-5,5),alpha=1.)
    if SPEC.get('reports'):
        features=np.stack([_financial_features(s.get('financial',{}),int(s['dates'][-1])) for s in samples])
        f=torch.as_tensor(features,device=device)
        stats=RATIO_STATS if SPEC.get('ratios') else REPORT_STATS
        means=torch.as_tensor(stats['mean'],device=device,dtype=f.dtype)
        stds=torch.as_tensor(stats['std'],device=device,dtype=f.dtype)
        values=f[:,:24]*stds+means
        f[:,:24]=torch.tanh(values/(1. if SPEC.get('ratios') else 10.))*f[:,24:48]
        f[:,48:51]/=5
        b.update(reports=f,has_report=f[:,-3:].any(1,keepdim=True).float())
    return b

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.members=nn.ModuleList([SymmetricForecast(SPEC) for _ in range(3)])
    def forward(self,batch):return torch.stack([m(batch) for m in self.members]).mean(0)

def build_model(config=None):return Model()
