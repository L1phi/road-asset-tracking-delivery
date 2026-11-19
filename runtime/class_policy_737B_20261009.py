"""Class-specific experimental policies; no GT, no detector changes."""
from collections import defaultdict
import numpy as np

def metrics(a,b):
    a=np.array(a,float);b=np.array(b,float);aa=np.prod(a[2:]-a[:2]);ba=np.prod(b[2:]-b[:2]);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(inter/max(aa+ba-inter,1e-9)),float(inter/max(min(aa,ba),1e-9))

def JT_whole_suppress(pred,ioa=.95):
    result={};audit=[]
    for f,ds in pred.items():
        kept=[]
        # A complete LED screen and a message fragment are not two assets.
        for d in sorted(ds,key=lambda d:(-(d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]),-d['confidence'],d['input_index'])):
            blocker=None
            for k in kept:
                iou,overlap=metrics(d['box'],k['box']);a=np.array(d['box']);b=np.array(k['box']);sa=a[2:]-a[:2];sb=b[2:]-b[:2]
                ratio=sb.prod()/sa.prod();vertical=abs((a[1]+a[3]-b[1]-b[3])/2)/max(sa[1],sb[1],1)
                if overlap>=ioa and 1<=ratio<=4 and vertical<=.35 and sb[0]>=1.2*sa[0] and .5<=sa[1]/sb[1]<=1.15:
                    blocker={'frame':int(f),'kept_input_index':k['input_index'],'removed_input_index':d['input_index'],'iou':iou,'ioa':overlap,'area_ratio':float(ratio),'reason':'JT_same_band_contained_message_fragment_prefer_whole'};break
            if blocker:audit.append(blocker)
            else:kept.append(d)
        result[int(f)]=sorted(kept,key=lambda d:d['input_index'])
    return result,audit

def confirmed_ZH(pred,threshold=.75,consecutive=3):
    tracks=defaultdict(list)
    for f,ds in pred.items():
        for d in ds:tracks[d['id']].append((int(f),d))
    kept={int(f):[] for f in pred};audit=[]
    for pid,obs in tracks.items():
        obs.sort(key=lambda x:x[0]);run=0;prev=None;confirmed_at=None
        for f,d in obs:
            high=d['confidence']>=threshold
            run=run+1 if high and prev==f-1 else (1 if high else 0)
            if run>=consecutive and confirmed_at is None:confirmed_at=f
            prev=f
            # Causal confirmation: do not backfill early unconfirmed observations.
            if run>=consecutive:kept[f].append(d)
        audit.append({'id':pid,'observations':len(obs),'high_observations':sum(d['confidence']>=threshold for f,d in obs),'threshold':threshold,'consecutive_required':consecutive,'confirmed_at':confirmed_at,'retained_observations':sum(d['id']==pid for ds in kept.values() for d in ds),'no_backfill':True})
    return kept,audit
