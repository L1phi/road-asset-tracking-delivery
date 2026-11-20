"""Class-aware overlap and containment suppression."""
import numpy as np
CONTINUOUS={3,4,7,9,15}
POLICY={'discrete_iou':.5,'continuous_iou':.85,'continuous_axis_overlap':.90,
        'containment_ioa':.95,'containment_max_area_ratio':4.,'BS_appearance_min':.9,
        'JT_vertical_center':.35,'LD_bottom_delta':.12,'LD_horizontal_overlap':.8,
        'BS_iou_axis_size_ratio':.8}

def geometry(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);sa=a[2:]-a[:2];sb=b[2:]-b[:2]
    inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0)
    area=float(np.prod(inter));aa=float(np.prod(sa));ba=float(np.prod(sb))
    return sa,sb,inter,area/max(aa+ba-area,1e-9),area/max(min(aa,ba),1e-9)

def suppress_frame(items,features=None,frame=0,policy=None):
    p={**POLICY,**(policy or {})};features=features or {};ids=[d['input_index'] for d in items]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate input_index')
    for d in items:
        b=np.asarray(d['box'],float)
        if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[2:]<=b[:2]):raise ValueError('invalid box')
        if not 0<=int(d['class_id'])<=16 or not np.isfinite(d['confidence']) or not 0<=d['confidence']<=1:raise ValueError('invalid class/confidence')
    alive={d['input_index']:d for d in items};audit=[];parent={}
    def drop(small,big,reason,iou,ioa,**extra):
        alive.pop(small['input_index']);parent[small['input_index']]=big['input_index']
        audit.append({'frame':int(frame),'class_id':small['class_id'],'suppressed_input_index':small['input_index'],
                      'kept_input_index':big['input_index'],'reason':reason,'iou':float(iou),'ioa':float(ioa),**extra})
    # Containment first: otherwise confidence NMS can keep only a text fragment.
    for big in sorted(items,key=lambda d:(-np.prod(np.asarray(d['box'])[2:]-np.asarray(d['box'])[:2]),-d['confidence'],d['input_index'])):
        if big['input_index'] not in alive:continue
        c=big['class_id']
        if c not in (0,10,13):continue
        for small in list(alive.values()):
            if small['input_index']==big['input_index'] or small['class_id']!=c:continue
            sa,sb,inter,iou,ioa=geometry(small['box'],big['box']);ratio=np.prod(sb)/np.prod(sa)
            if not 1.05<=ratio<=p['containment_max_area_ratio'] or ioa<p['containment_ioa']:continue
            a=np.asarray(small['box']);b=np.asarray(big['box']);reason=None;extra={'area_ratio':float(ratio)}
            if c==13:
                vertical=abs((a[1]+a[3]-b[1]-b[3])/2)/max(sa[1],sb[1])
                if vertical<=p['JT_vertical_center'] and sb[0]>=1.2*sa[0] and .5<=sa[1]/sb[1]<=1.15:reason='JT_contained_same_band_whole'
            elif c==10:
                bottom=abs(a[3]-b[3])/sb[1];horizontal=inter[0]/max(min(sa[0],sb[0]),1e-9)
                if bottom<=p['LD_bottom_delta'] and horizontal>=p['LD_horizontal_overlap'] and abs((a[0]+a[2]-b[0]-b[2])/2)<=.2*sb[0]:
                    reason='LD_contained_same_bottom_and_axis';extra.update(bottom_delta=float(bottom),horizontal_overlap=float(horizontal))
            else:
                va=features.get(f'{frame}:{small["input_index"]}');vb=features.get(f'{frame}:{big["input_index"]}')
                sim=None if va is None or vb is None else float(np.dot(va,vb));extra['similarity']=sim
                centered=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/max(np.linalg.norm(sb),1)
                if sim is not None and sim>=p['BS_appearance_min'] and centered<=.25 and (sa[0]/sb[0]>=.8 or sa[1]/sb[1]>=.8):reason='BS_contained_aligned_with_appearance'
            if reason:drop(small,big,reason,iou,ioa,**extra)
    kept=[]
    for d in sorted(alive.values(),key=lambda d:(-d['confidence'],d['input_index'])):
        blocked=False
        for k in kept:
            if d['class_id']!=k['class_id']:continue
            sa,sb,inter,iou,ioa=geometry(d['box'],k['box']);c=d['class_id']
            if c in CONTINUOUS:
                axis=inter/np.maximum(sa,sb)
                duplicate=iou>=p['continuous_iou'] and np.min(axis)>=p['continuous_axis_overlap'];reason='continuous_nearly_identical_extent'
            else:
                duplicate=iou>=p['discrete_iou'];reason='discrete_same_class_iou'
                # A lower component and a complete multi-sign stack are not a
                # localization jitter pair. Leave this ambiguous BS relation to
                # local-content/temporal parent evidence, not confidence alone.
                if c==0 and np.min(np.minimum(sa,sb)/np.maximum(sa,sb))<p['BS_iou_axis_size_ratio']:duplicate=False
            if duplicate:drop(d,k,reason,iou,ioa);blocked=True;break
        if not blocked:kept.append(d)
    survivors={d['input_index'] for d in kept}
    for row in audit:
        winner=row['kept_input_index'];seen=set()
        while winner in parent:
            if winner in seen:raise AssertionError('suppression cycle')
            seen.add(winner);winner=parent[winner]
        row['final_kept_input_index']=winner;assert winner in survivors
    assert len(kept)+len(audit)==len(items)
    return sorted(kept,key=lambda d:d['input_index']),audit
