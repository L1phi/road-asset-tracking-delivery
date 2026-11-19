"""Experimental geometry-first shortlist, then real deep features on demand.

No fake vectors. Rule-only links explicit. Thresholds fixed across videos.
"""
from collections import defaultdict
import numpy as np
from all_asset_nms_20261009 import geometry as overlap,POLICY
RULES={'easy_residual':.30,'easy_size_ratio':1.6,'easy_conf':.5,'adjacent_margin':.10,'hard_shortlist':3}
def shortlist(pred,trans,params,hard_limit=None):
    edges=[];byframe={int(f):ds for f,ds in pred.items()};out=defaultdict(list);incoming=defaultdict(list)
    for f,ds in byframe.items():
        for a in ds:
            ab=np.array(a['box']);size=np.maximum(ab[2:]-ab[:2],1);center=(ab[:2]+ab[2:])/2;M=np.eye(3)
            for g in range(f+1,f+params['gap']+1):
                M=trans.get(g,np.eye(3))@M;warped=(M@np.r_[center,1])[:2]
                for b in byframe.get(g,[]):
                    bb=np.array(b['box']);bs=np.maximum(bb[2:]-bb[:2],1);ratio=float(np.max(np.maximum(size/bs,bs/size)))
                    if ratio>3:continue
                    residual=float(np.linalg.norm((bb[:2]+bb[2:])/2-warped)/max(np.linalg.norm(size),np.linalg.norm(bs),1))
                    if residual>params['residual_base']+.25*(g-f-1):continue
                    cost=params['geometry_weight']*residual+.2*np.log(ratio)+.025*(g-f-1)
                    if cost>=params['birth_cost']:continue
                    row={'pair':(f,a['input_index'],g,b['input_index']),'cost':float(cost),'residual':residual,'ratio':ratio,'confidence_min':min(a['confidence'],b['confidence'])}
                    edges.append(row);out[(f,a['input_index'])].append(row);incoming[(g,b['input_index'])].append(row)
    adjacent_out={k:sorted([e for e in es if e['pair'][2]-e['pair'][0]==1],key=lambda e:(e['cost'],e['pair'])) for k,es in out.items()}
    adjacent_in={k:sorted([e for e in es if e['pair'][2]-e['pair'][0]==1],key=lambda e:(e['cost'],e['pair'])) for k,es in incoming.items()}
    easy=set();allowed=set();wanted=set()
    for src,es in out.items():
        adjacent=adjacent_out[src];best=adjacent[0] if adjacent else None;good=False
        if best:
            dest=best['pair'][2:];back=adjacent_in[dest]
            source_margin=adjacent[1]['cost']-best['cost'] if len(adjacent)>1 else float('inf')
            target_margin=back[1]['cost']-back[0]['cost'] if len(back)>1 else float('inf')
            good=back[0]['pair']==best['pair'] and min(source_margin,target_margin)>=RULES['adjacent_margin'] and best['residual']<=RULES['easy_residual'] and best['ratio']<=RULES['easy_size_ratio'] and best['confidence_min']>=RULES['easy_conf']
        if good:easy.add(best['pair']);allowed.add(best['pair'])
        else:
            for e in sorted(es,key=lambda e:(e['cost'],e['pair']))[:RULES['hard_shortlist'] if hard_limit is None else hard_limit]:
                pair=e['pair'];allowed.add(pair);wanted.update((f'{pair[0]}:{pair[1]}',f'{pair[2]}:{pair[3]}'))
    return allowed,easy,wanted,{'geometry_pairs':len(edges),'allowed_pairs':len(allowed),'rule_only_pairs':len(easy),'deep_candidate_pairs':len(allowed-easy),'rules':RULES}

def bs_nms_demand(pred):
    wanted=set()
    for f,ds in pred.items():
        for a in ds:
            for b in ds:
                if a['input_index']==b['input_index']:continue
                sa,sb,inter,iou,ioa=overlap(a['box'],b['box']);ratio=np.prod(sb)/np.prod(sa)
                if not 1.05<=ratio<=POLICY['containment_max_area_ratio'] or ioa<POLICY['containment_ioa']:continue
                aa=np.array(a['box']);bb=np.array(b['box']);centered=np.linalg.norm((aa[:2]+aa[2:]-bb[:2]-bb[2:])/2)/max(np.linalg.norm(sb),1)
                if centered<=.25 and (sa[0]/sb[0]>=.8 or sa[1]/sb[1]>=.8):wanted.update((f'{f}:{a["input_index"]}',f'{f}:{b["input_index"]}'))
    return wanted

def bs_parent_demand(pred,trans,regions,parent_params,path_cover,params):
    # Geometry-first temporary tracks supply only candidate reference grouping,
    # not final identities. Entire tracks are NOT sent to the neural network.
    from strict_temporal_parent_20261008 import geometry,warp_between
    gp=dict(params,appearance_weight=0.,appearance_min=-1.)
    preliminary,_=path_cover(pred,{},trans,**gp);tracks=defaultdict(list)
    for f,ds in preliminary.items():
        for d in ds:tracks[d['id']].append((f,d))
    wanted=set();potential=0
    for child,obs in tracks.items():
        for f,d in obs:
            for parent,refs in tracks.items():
                if child==parent:continue
                qualified=[(g,e) for g,e in refs if abs(g-f)<=parent_params['window'] and e['confidence']>=.5 and e['box'][3]-e['box'][1]>=40 and e['box'][2]-e['box'][0]>=15]
                for g,e in qualified:
                    b=np.array(e['box']);M=warp_between(trans,g,f);corners=np.array([[b[0],b[1],1],[b[2],b[1],1],[b[2],b[3],1],[b[0],b[3],1]])@M.T
                    box=np.r_[corners[:,:2].min(0),corners[:,:2].max(0)]
                    if geometry(d['box'],box,regions) is None:continue
                    potential+=1;wanted.add(f'{f}:{d["input_index"]}')
                    for r,q in qualified:wanted.add(f'{r}:{q["input_index"]}')
                    break
    return wanted,{'geometry_parent_queries':potential,'reference_and_query_global_keys':len(wanted)}
