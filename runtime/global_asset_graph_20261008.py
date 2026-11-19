from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

def path_cover(pred,cache,transforms=None,gap=5,birth_cost=.7,appearance_min=.8,min_observations=1,
               geometry_weight=.50,appearance_weight=.65,residual_base=1.5,
               strong_appearance=None,strong_residual=3.,edge_protect=False,edge_margin=8,forbidden_links=None,pair_evidence=None,geometry_anchor='center',allowed_pairs=None,geometry_only_pairs=None):
    if geometry_anchor not in ('center','bottom_center'):raise ValueError(geometry_anchor)
    if geometry_only_pairs is not None and (strong_appearance is not None or edge_protect or pair_evidence is not None):raise ValueError('Rule-only links require unmodified geometric gates')
    nodes=[(int(f),d) for f,ds in sorted(pred.items(),key=lambda x:int(x[0])) for d in sorted(ds,key=lambda d:d['input_index'])]
    n=len(nodes)
    if not n:return {int(f):[] for f in pred},{'links':[],'removed_observations':[]}
    # Rows = predecessor choice; columns = successors plus private null option.
    cost=np.full((n,2*n),1e6,dtype=float);cost[np.arange(n),n+np.arange(n)]=0
    edges={};byframe=defaultdict(list)
    geometry_only=(appearance_weight==0 and appearance_min<=-1 and pair_evidence is None and strong_appearance is None and not edge_protect)
    for j,(f,d) in enumerate(nodes):byframe[f].append(j)
    last_frame=max(f for f,d in nodes)
    for i,(f,a) in enumerate(nodes):
        va=cache.get(f'{f}:{a["input_index"]}',{}).get('appearance')
        if va is None and not geometry_only and geometry_only_pairs is None:continue
        ab=np.array(a['box']);asize=np.maximum(ab[2:]-ab[:2],1);center=(ab[:2]+ab[2:])/2;M=np.eye(3)
        if geometry_anchor=='bottom_center':center[1]=ab[3]
        for targetframe in range(f+1,min(last_frame+1,f+gap+1)):
            if transforms is not None:M=transforms[targetframe]@M
            warped=(M@np.r_[center,1])[:2]
            for j in byframe.get(targetframe,[]):
                b=nodes[j][1];vb=cache.get(f'{targetframe}:{b["input_index"]}',{}).get('appearance')
                pair=(f,a['input_index'],targetframe,b['input_index'])
                if allowed_pairs is not None and pair not in allowed_pairs:continue
                rule_only=geometry_only or (geometry_only_pairs is not None and pair in geometry_only_pairs)
                if forbidden_links is not None and (f,a['input_index'],targetframe,b['input_index']) in forbidden_links:continue
                if (va is None or vb is None) and not rule_only:continue
                sim=None if rule_only else float(np.dot(va,vb))
                extra={}
                if pair_evidence is not None:
                    sim,extra=pair_evidence(f,a,targetframe,b,sim,M)
                if sim is not None and sim<appearance_min:continue
                bb=np.array(b['box']);bsize=np.maximum(bb[2:]-bb[:2],1)
                ratio=float(np.max(np.maximum(asize/bsize,bsize/asize)))
                truncated=edge_protect and (ab[0]<=edge_margin or ab[1]<=edge_margin or ab[2]>=1920-edge_margin or ab[3]>=1080-edge_margin or
                                            bb[0]<=edge_margin or bb[1]<=edge_margin or bb[2]>=1920-edge_margin or bb[3]>=1080-edge_margin)
                if ratio>3. and not truncated:continue
                target=(bb[:2]+bb[2:])/2
                if geometry_anchor=='bottom_center':target[1]=bb[3]
                residual=float(np.linalg.norm(target-warped)/max(np.linalg.norm(asize),np.linalg.norm(bsize),1))
                dt=targetframe-f
                gate=max(residual_base,strong_residual) if strong_appearance is not None and sim>=strong_appearance else residual_base
                if residual>gate+.25*(dt-1):continue
                # Truncated frames remain observations, but cannot veto identity
                # solely by width/height changes. Require stronger appearance.
                if truncated and sim<.9:continue
                size_penalty=0. if truncated or extra.get('local_verified',False) else .20*np.log(ratio)
                c=geometry_weight*residual+(0. if rule_only else appearance_weight*(1-sim))+size_penalty+.025*(dt-1)
                if c>=birth_cost:continue
                cost[i,j]=c-birth_cost
                edges[(i,j)]={'source_frame':f,'source_input_index':a['input_index'],'target_frame':targetframe,'target_input_index':b['input_index'],
                    'appearance':sim,'geometry_residual':residual,'geometry_anchor':geometry_anchor,'size_ratio':ratio,'cost':float(c),'local_evidence':extra,
                    'appearance_used':not rule_only}
    rr,cc=linear_sum_assignment(cost);parent=list(range(n));accepted=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in zip(rr,cc):
        if j>=n or cost[i,j]>=0:continue
        assert nodes[i][0]<nodes[j][0]
        parent[find(j)]=find(i);accepted.append(edges[(i,j)])
    groups=defaultdict(list)
    for i in range(n):groups[find(i)].append(i)
    for g,indices in groups.items():assert len({nodes[i][0] for i in indices})==len(indices)
    label={g:f'graph/{k+1}' for k,g in enumerate(sorted(groups,key=lambda g:min(nodes[i][0] for i in groups[g])))}
    out={int(f):[] for f in pred};removed=[]
    for i,(f,d) in enumerate(nodes):
        g=find(i)
        if len(groups[g])<min_observations:removed.append({'frame':f,'input_index':d['input_index'],'reason':'minimum_observations','group':label[g]});continue
        out[f].append({**d,'id':label[g]})
    return out,{'links':accepted,'candidate_edges':len(edges),'removed_observations':removed}
