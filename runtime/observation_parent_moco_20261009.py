"""Observation-first whole/part demand. Never use initial tracks for references.

Geometry chooses the region; actual deep evidence verifies it. No whole evidence
means no merge. Track IDs are read only AFTER observation relations are verified.
"""
from collections import defaultdict
import numpy as np
from strict_temporal_parent_20261008 import geometry, warp_between

RULES={'window':5,'gate':.75,'reference_support':3,'vote_support':2,
       'reference_global_similarity':.85,'reference_size_ratio':1.6,
       'reference_center_residual':.6,'ambiguity_margin':.08,
       'whole_height':40,'whole_width':15,'reference_confidence':.5}

def key(f,d):return f'{f}:{d["input_index"]}'

def projected(d,trans,g,f):
    b=np.asarray(d['box'],float);M=warp_between(trans,g,f)
    corners=np.array([[b[0],b[1],1],[b[2],b[1],1],[b[2],b[3],1],[b[0],b[3],1]])@M.T
    return np.r_[corners[:,:2].min(0),corners[:,:2].max(0)]

def size_center(b):return np.maximum(b[2:]-b[:2],1),(b[:2]+b[2:])/2

def whole(d,reference=True):
    b=d['box']
    return (d['confidence']>= (RULES['reference_confidence'] if reference else .359)
            and b[3]-b[1]>=RULES['whole_height'] and b[2]-b[0]>=RULES['whole_width']
            and min(b[:2])>8 and b[2]<1912 and b[3]<1072)

def compatible(a,b):
    sa,ca=size_center(a);sb,cb=size_center(b)
    return (np.max(np.maximum(sa/sb,sb/sa))<=RULES['reference_size_ratio']
            and np.linalg.norm(ca-cb)/max(np.linalg.norm(sa),np.linalg.norm(sb),1)<=RULES['reference_center_residual'])

def plan(raw,trans,regions):
    """Loose discovery only, independent of IDs. Strict geometry in verification."""
    raw={int(f):ds for f,ds in raw.items()};queries=[];wanted=set()
    for f,ds in sorted(raw.items()):
        refs=[(g,e,projected(e,trans,g,f)) for g in range(max(0,f-RULES['window']),f+RULES['window']+1)
              for e in raw.get(g,[]) if whole(e)]
        for d in ds:
            anchors=[(f,e,np.array(e['box'],float),'actual') for e in ds
                     if e['input_index']!=d['input_index'] and whole(e,False) and geometry(d['box'],e['box'],regions) is not None]
            # Planning allows bounded camera/object residual; never uses this
            # loose gate as identity evidence.
            s=np.array(d['box']);ss,sc=size_center(s)
            for g,e,b in refs:
                bs,bc=size_center(b)
                if g==f and e['input_index']==d['input_index']:continue
                if bs[1]>=1.4*ss[1] and .55<=ss[0]/bs[0]<=1.35 and np.linalg.norm(sc-bc)/max(np.linalg.norm(bs),1)<=.8:
                    anchors.append((g,e,b,'temporal'))
            seen=set()
            for g,e,b,source in anchors:
                ak=key(g,e)
                if ak in seen:continue
                seen.add(ak)
                nearby=[(h,q,p) for h,q,p in refs if compatible(b,p)
                        and not(h==f and q['input_index']==d['input_index'])]
                if len({h for h,q,p in nearby})<RULES['reference_support']:continue
                # Stable reference confirmation also needs these full features.
                keys={key(h,q) for h,q,p in nearby};keys.add(ak);keys.add(key(f,d));wanted.update(keys)
                queries.append({'frame':f,'query':d,'anchor_frame':g,'anchor':e,'anchor_source':source,
                                'refs':[(h,q,p.tolist()) for h,q,p in nearby]})
    return queries,wanted

def hypotheses(queries,features,regions,trans):
    candidates=[];rejections=[];requests=set();seen=set()
    for q in queries:
        f=q['frame'];d=q['query'];ak=key(q['anchor_frame'],q['anchor']);av=features.get(ak)
        if av is None:raise ValueError(f'Missing explicitly requested parent global feature {ak}')
        refs=[(g,e,np.array(b)) for g,e,b in q['refs'] if float(np.dot(av,features[key(g,e)]))>=RULES['reference_global_similarity']]
        if len({g for g,e,b in refs})<RULES['reference_support']:continue
        # Require independent stable adjacent actual observations, NOT common ID.
        stable=[(g,e,b) for g,e,b in refs if any(1<=abs(h-g)<=3 and compatible(b,c)
                and float(np.dot(features[key(g,e)],features[key(h,t)]))>=RULES['reference_global_similarity'] for h,t,c in refs)]
        if len({g for g,e,b in stable})<RULES['reference_support']:continue
        options=[]
        if q['anchor_source']=='actual':options.append(('same_frame_actual',np.asarray(q['anchor']['box']),stable))
        else:
            # Separate motion fits avoid a late acceleration corrupting an
            # earlier reference. Select by geometry, never by appearance score.
            for name,subset in [('past',[r for r in stable if r[0]<f]),('future',[r for r in stable if r[0]>f])]:
                if len({g for g,e,b in subset})<RULES['reference_support']:continue
                subset=sorted(subset,key=lambda x:abs(x[0]-f))[:RULES['reference_support']]
                times=np.array([g-f for g,e,b in subset]);design=np.c_[np.ones(len(times)),times]
                coords=np.array([np.r_[size_center(b)[1],np.log(size_center(b)[0])] for g,e,b in subset])
                coef=np.linalg.lstsq(design,coords,rcond=None)[0]
                error=float(np.max(np.linalg.norm((design@coef)[:,:2]-coords[:,:2],axis=1)/np.linalg.norm(np.exp(coords[:,2:]),axis=1)))
                if error>.25:continue
                size=np.exp(coef[0,2:]);center=coef[0,:2]
                options.append((name,np.r_[center-size/2,center+size/2],subset))
        for source,b,rs in options:
            found=geometry(d['box'],b,regions)
            if found is None:continue
            region,frac=found;signature=(key(f,d),region,tuple(sorted(key(g,e) for g,e,z in rs)))
            if signature in seen:continue
            seen.add(signature)
            ks=[f'{key(g,e)}:{region}' for g,e,z in rs];requests.update(ks)
            candidates.append({'frame':f,'query_key':key(f,d),'query_index':d['input_index'],'region':region,
                               'whole_box':b.tolist(),'source':source,'refs':[key(g,e) for g,e,z in rs],
                               'part_keys':ks,'anchor_key':ak})
    return candidates,requests,rejections

def verify(candidates,features,parts):
    accepted=[];rejected=[];byquery=defaultdict(list)
    for c in candidates:
        scores=[float(np.dot(features[c['query_key']],parts[k])) for k in c['part_keys']]
        f=c['frame'];before=[s for k,s in zip(c['part_keys'],scores) if int(k.split(':')[0])<f];after=[s for k,s in zip(c['part_keys'],scores) if int(k.split(':')[0])>f]
        score=min(float(np.median(side)) for side in (before,after) if side) if before or after else float(np.median(scores))
        row={**c,'score':score,'reference_scores':scores}
        if score>=RULES['gate']:byquery[c['query_key']].append(row)
        else:rejected.append({**row,'reason':'local_reference_disagreement'})
    for query,rows in byquery.items():
        # Equivalent hypotheses for the same physical reference are not rivals.
        rows.sort(key=lambda x:-x['score']);best=rows[0]
        rivals=[x for x in rows[1:] if not set(x['refs'])&set(best['refs']) and not compatible(np.array(x['whole_box']),np.array(best['whole_box']))]
        if rivals and best['score']-max(x['score'] for x in rivals)<RULES['ambiguity_margin']:
            rejected.append({'query_key':query,'reason':'ambiguous_parent'});continue
        accepted.append(best)
    return accepted,rejected

def apply_relations(pred,relations):
    """Resolve IDs only now; disjoint simultaneous assets veto any union."""
    lookup={key(int(f),d):d for f,ds in pred.items() for d in ds};groups=defaultdict(list);votes=defaultdict(list)
    for r in relations:
        if r['query_key'] not in lookup:continue
        child=lookup[r['query_key']]['id'];parents={lookup[k]['id'] for k in r['refs'] if k in lookup}
        for parent in parents:
            if parent!=child:votes[tuple(sorted((child,parent)))].append(r)
    for f,ds in pred.items():
        for d in ds:groups[d['id']].append((int(f),d))
    roots={pid:pid for pid in groups};audit=[];rejected=[]
    def root(p):
        while roots[p]!=p:p=roots[p]
        return p
    from all_asset_nms_20261009 import geometry as overlap
    for pair,rs in sorted(votes.items()):
        if len({r['frame'] for r in rs})<RULES['vote_support']:
            rejected.append({'pair':pair,'reason':'insufficient_distinct_query_frames','frames':sorted({r['frame'] for r in rs})});continue
        a,b=map(root,pair)
        if a==b:continue
        members_a=[p for p in groups if root(p)==a];members_b=[p for p in groups if root(p)==b]
        conflicts=[]
        for pa in members_a:
            for pb in members_b:
                for f,d in groups[pa]:
                    for g,e in groups[pb]:
                        if f!=g:continue
                        ioa=overlap(d['box'],e['box'])[-1]
                        # Partial/partial overlap is allowed only when an
                        # independently verified whole at this frame contains BOTH.
                        covers=any(r['frame']==f and min(overlap(d['box'],r['whole_box'])[-1],overlap(e['box'],r['whole_box'])[-1])>=.85 for r in rs)
                        if ioa<.85 and not covers:conflicts.append(f)
        if conflicts:
            rejected.append({'pair':pair,'reason':'simultaneous_distinct_asset_conflict','frames':sorted(set(conflicts))});continue
        winner=min((a,b),key=lambda p:min(f for pid in groups if root(pid)==p for f,d in groups[pid]));loser=b if winner==a else a
        roots[loser]=winner;audit.append({'pair':pair,'identity':winner,'votes':[{'frame':r['frame'],'score':r['score'],'query_key':r['query_key']} for r in rs]})
    out={};dropped=[]
    for f,ds in pred.items():
        byid=defaultdict(list)
        for d in ds:byid[root(d['id'])].append(d)
        out[f]=[]
        for pid,items in byid.items():
            keep=max(items,key=lambda d:np.prod(np.array(d['box'])[2:]-np.array(d['box'])[:2]))
            out[f].append({**keep,'id':pid})
            dropped.extend({'frame':int(f),'input_index':d['input_index'],'shared_identity':pid} for d in items if d is not keep)
    return out,{'accepted':audit,'rejected':rejected,'observation_relations':relations,'duplicate_display_observations':dropped}
