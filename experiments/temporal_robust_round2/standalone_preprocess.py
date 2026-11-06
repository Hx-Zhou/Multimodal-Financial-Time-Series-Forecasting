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
