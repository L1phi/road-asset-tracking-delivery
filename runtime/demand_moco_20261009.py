"""Class-explicit feature demand: no guessed/dummy neural vectors."""
from all_asset_nms_20261009 import suppress_frame
EXCLUDED={7,9}
GEOMETRY_ONLY={10}
def frame_plan(items,frame):
    active=[d for d in items if d['class_id'] not in EXCLUDED]
    bs=[d for d in active if d['class_id']==0]
    other,audit=suppress_frame([d for d in active if d['class_id']!=0],frame=frame)
    items=sorted(bs+other,key=lambda d:d['input_index'])
    keys={f'{frame}:{d["input_index"]}' for d in items if d['class_id'] not in GEOMETRY_ONLY}
    return items,keys,audit
def plan(pred):
    observations={};keys=set();audit=[]
    for f,ds in pred.items():
        observations[int(f)],wanted,rows=frame_plan(ds,int(f));keys.update(wanted);audit.extend(rows)
    return observations,keys,audit
