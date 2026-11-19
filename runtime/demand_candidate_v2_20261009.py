"""Difficult associations request real MoCo; never cap them at three edges."""
from collections import defaultdict
from candidate_moco_policy_20261009 import shortlist

def association_demand(pred,trans,params):
    allowed,easy,wanted,a=shortlist(pred,trans,params,hard_limit=100000000)
    # Broad gated pool also discovers possible gap recovery at an otherwise
    # easy source. It does NOT request unrelated detections outside those gates.
    low={f:[{**d,'confidence':0.} for d in ds] for f,ds in pred.items()}
    broad,_,_,_=shortlist(low,trans,params,hard_limit=100000000)
    easy_dest={p[2:] for p in easy}
    allowed.update(p for p in broad if p[2]-p[0]>1 and p[2:] not in easy_dest)
    reasons=defaultdict(set);deep=allowed-easy
    lookup={(int(f),d['input_index']):d for f,ds in pred.items() for d in ds}
    for p in deep:
        reason='gap_recovery' if p[2]-p[0]>1 else 'ambiguous_or_unstable_association'
        for node in (p[:2],p[2:]):
            k=f'{node[0]}:{node[1]}';wanted.add(k);reasons[k].add(reason)
            if lookup[node]['confidence']<.5:reasons[k].add('low_confidence_link_verification')
    a.update(version=2,hard_shortlist_cap_removed=True,allowed_pairs=len(allowed),deep_candidate_pairs=len(deep),
             feature_request_reasons={k:sorted(v) for k,v in sorted(reasons.items())})
    return allowed,easy,wanted,a
