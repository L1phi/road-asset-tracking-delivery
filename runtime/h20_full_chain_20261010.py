"""Video decoding, detection, motion compensation and offline tracking."""
import argparse,json,time,hashlib,sys,datetime
from pathlib import Path

WEIGHT='/xusc/ZPH/road_assets_yolo/runs/yolo26_dual737_20261008T060744Z/runs/B_videoquality/weights/last.pt'
WEIGHT_SHA='0ef1e618fba78693da8a8f72879483cd437680213e3d29aef3c2e2056fbdf64e'
MOCO='/xusc/ZPH/road_assets_yolo/video_tracking/moco_input/moco-v3-r50-1000ep.pth.tar'
MOCO_SHA='5baee69ef3c0b6f153eac41543be48dbd2ca6ff8f6327bb44fc0aa61f8fb4eee'
PARAMS={'gap':5,'birth_cost':.8,'appearance_min':.5,'geometry_weight':.25,'appearance_weight':1.25,'residual_base':3.,'min_observations':2}
REGIONS=[(0.,.5),(.5,1.),(0.,.67),(.33,1.),(.2,.8)]
ABBR=['BS','ZT','JS','FXB','SPZ','FZT','SM','HL','LMJ','LH','LD','ZH','JK','JT','CS','FXW','XHD']
SMOKE=['2026_01_21_12_08_14','2026_01_21_11_43_21','2026_01_21_11_42_44']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')

def track(raw,trans,features):
    from demand_moco_20261009 import plan
    from demand_candidate_v2_20261009 import association_demand
    from candidate_moco_policy_20261009 import bs_nms_demand
    from observation_parent_moco_20261009 import plan as parent_plan,hypotheses,verify,apply_relations
    from all_asset_nms_20261009 import suppress_frame
    from class_policy_737B_20261009 import confirmed_ZH
    from global_asset_graph_20261008 import path_cover
    from display_map import per_class_display_map
    pre,allkeys,nms=plan(raw);wanted=set();plans={};audits={}
    for c in sorted({d['class_id'] for ds in pre.values() for d in ds}):
        if c==10:continue
        cp={f:[d for d in ds if d['class_id']==c] for f,ds in pre.items()}
        allow,easy,keys,a=association_demand(cp,trans,PARAMS)
        plans[c]=(allow,easy);audits[str(c)]=a;wanted.update(keys)
        if c==0:wanted.update(bs_nms_demand(cp))
    bs={f:[d for d in ds if d['class_id']==0] for f,ds in pre.items()}
    queries,keys=parent_plan(bs,trans,REGIONS);wanted.update(keys)
    assert wanted<=allkeys
    vec=features(wanted,False)
    candidates,requests,_=hypotheses(queries,vec,REGIONS,trans)
    parts=features(requests,True)
    relations,rejections=verify(candidates,vec,parts)
    obs={}
    for f,ds in pre.items():obs[f],a=suppress_frame(ds,vec,f);nms.extend(a)
    full={k:{'appearance':a} for k,a in vec.items()};pred={f:[] for f in raw};association={};parent={};zh=[]
    for c in sorted({d['class_id'] for ds in obs.values() for d in ds}):
        params=dict(PARAMS)
        if c==10:params.update(geometry_anchor='bottom_center',appearance_weight=0.,appearance_min=-1.)
        else:params.update(allowed_pairs=plans[c][0],geometry_only_pairs=plans[c][1])
        cp,a=path_cover({f:[d for d in ds if d['class_id']==c] for f,ds in obs.items()},full,trans,**params)
        association[str(c)]=a
        if c==0:cp,parent=apply_relations(cp,relations);parent['local_rejections']=rejections
        if c==11:cp,zh=confirmed_ZH(cp,.75,3)
        for f,ds in cp.items():pred[f].extend({**d,'id':f'{c}/'+d['id']} for d in ds)
    for f,ds in pred.items():
        source={d['input_index']:d for d in raw[f]}
        assert len(ds)==len({d['id'] for d in ds})
        for d in ds:assert all(d[k]==source[d['input_index']][k] for k in ('box','confidence','class_id'))
    mapping=per_class_display_map(pred)
    for cs,ids in mapping.items():
        for i,pid in enumerate(ids,1):ids[pid]=f'{ABBR[int(cs)]}{i}'
    stats={str(c):{'IDs_not_inventory':len({d['id'] for ds in pred.values() for d in ds if d['class_id']==c}),
                   'observations':sum(d['class_id']==c for ds in pred.values() for d in ds)} for c in range(17)}
    return {'predictions':pred,'display_map':mapping,'class_statistics':stats,'parent_audit':parent,
            'NMS_audit':nms,'ZH_audit':zh,'association_audit':association,'original_box_confidence_provenance':'PASS',
            'model':'737B','offline_uses_future':True,'experimental_observation_first_demand_v2':True,
            'global_demand':len(wanted),'local_demand':len(requests)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--weight',default=WEIGHT)
    ap.add_argument('--moco',default=MOCO);ap.add_argument('--out');ap.add_argument('--all29',action='store_true')
    ap.add_argument('--yolo-batch',type=int,default=1);ap.add_argument('--moco-batch',type=int,default=32)
    ap.add_argument('--video',action='append');ap.add_argument('--stage',choices=['full','detect','track'],default='full')
    ap.add_argument('--prepared');ap.add_argument('--gate');ap.add_argument('--external-resources',action='store_true')
    ap.add_argument('--gpu-profile',choices=['h20','4090'],default='h20')
    ap.add_argument('--release-warm-cache',action='store_true');ap.add_argument('--release-phase-cache',action='store_true')
    a=ap.parse_args()
    if min(a.yolo_batch,a.moco_batch)<1:raise ValueError('Positive batches required')
    import numpy as np,torch,cv2,ultralytics
    from torchvision.models import resnet50
    from ultralytics import YOLO
    from crop_helper import crop
    assert ultralytics.__version__=='8.4.146','Use the frozen detector version'
    if a.gpu_profile=='h20':assert torch.__version__=='2.6.0+cu124'
    else:
        assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))>=(2,6)
        assert '4090' in torch.cuda.get_device_name(0),'4090 profile requires an actual 4090 variant'
    assert torch.cuda.is_available(),'No CPU fallback'
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf8'))
    for name,digest in manifest['files'].items():assert sha(root/name)==digest,('Package changed',name)
    assert sha(a.weight)==WEIGHT_SHA and sha(a.moco)==MOCO_SHA
    allsources=sorted(Path(a.sources).glob('*.avi'))
    expected=json.loads((root/'source_manifest.json').read_text(encoding='utf8'))
    selected=set(a.video) if a.video else (set(expected) if a.all29 else set(SMOKE))
    assert selected<=set(expected)
    files=[p for p in allsources if p.stem in selected]
    assert len(files)==len(selected),('Missing sources',len(files))
    for p in files:assert sha(p)==expected[p.stem]['sha256'],('Video identity mismatch',p)
    out=Path(a.out) if a.out else root/('full_chain_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    from strict_timing_20261010 import stable_warmup,ProcessResourceSampler
    sampler=None if a.external_resources else ProcessResourceSampler(out/'resources.json',.1)
    print('LOAD_AND_WARM',flush=True);startup=time.perf_counter()
    params=dict(imgsz=960,device=0,conf=.01,iou=.7,nms=True,quantize=None,augment=False,
                rect=True,max_det=300,classes=None,agnostic_nms=False,save=False,verbose=False)
    synthetic=np.random.default_rng(42).integers(0,256,(1080,1920,3),dtype=np.uint8)
    warm_yolo={'stable':True,'batches':0,'not_used':True};warm_moco=dict(warm_yolo)
    if a.stage!='track':
        yolo=YOLO(a.weight)
        warm_yolo=stable_warmup(lambda:yolo.predict([synthetic]*a.yolo_batch,**params),torch.cuda.synchronize)
        assert yolo.predictor.model.fp16 is False and yolo.predictor.model.end2end is False
    if a.stage!='detect':
        moco=resnet50(weights=None);ckpt=torch.load(a.moco,map_location='cpu',weights_only=True);prefix='module.base_encoder.'
        weights={k[len(prefix):]:v for k,v in ckpt['state_dict'].items() if k.startswith(prefix) and not k.startswith(prefix+'fc.')}
        report=moco.load_state_dict(weights,strict=False)
        assert set(report.missing_keys)=={'fc.weight','fc.bias'} and not report.unexpected_keys
        moco.fc=torch.nn.Identity();moco.eval().to('cuda:0');del ckpt,weights
        warm=torch.stack([crop(synthetic,[100,100,300,400],1.)]*a.moco_batch).to('cuda:0')
        with torch.inference_mode():warm_moco=stable_warmup(lambda:moco(warm),torch.cuda.synchronize)
        del warm
    write(out/'warmup.json',{'YOLO':warm_yolo,'MoCo':warm_moco,'synthetic_seed':42,
         'YOLO_input':'1080x1920 BGR through full predict','MoCo_input':'preprocessed CUDA 224x224 tight crops',
         'excluded_from_hot_timing':True,'different_warmup_scopes_not_compared_as_inference_speed':True})
    if not(warm_yolo['stable'] and warm_moco['stable']):raise RuntimeError('Warmup not stable; hot benchmark not started. See warmup.json')
    torch.cuda.synchronize();startup=time.perf_counter()-startup
    if a.release_warm_cache:torch.cuda.empty_cache();torch.cuda.synchronize()
    # Import algorithms before opening the hot timing interval.
    import demand_moco_20261009,demand_candidate_v2_20261009,candidate_moco_policy_20261009
    import observation_parent_moco_20261009,all_asset_nms_20261009,class_policy_737B_20261009,global_asset_graph_20261008,display_map
    write(out/'protocol.json',{'mode':'serial_detection_batch_then_serial_tracking','sources':[p.name for p in files],
        'weight_sha256':WEIGHT_SHA,'moco_sha256':MOCO_SHA,'FP32':True,'TF32':False,'params':params,
        'yolo_batch':a.yolo_batch,'moco_batch':a.moco_batch,'startup_excluded_seconds':startup,
        'fresh_detection_GMC_global_local':True,'visualization':False,'intermediate_predictions_in_RAM':True,
        'feature_decode_repeated_and_counted':True,'readiness_pass_required_by_operator':True,
        'timing_version':'STRICT_WALL_NVML_PROCESS_V2','stage':a.stage,'clock':'perf_counter_ns','GPU_sync_at_both_boundaries':True,
        'NVML_sample_interval_seconds':.1,'NVML_samples_all_physical_cards':True,
        'warmup_required_stable':True,'imports_and_identity_checks_before_hot':True,
        'sampling_and_progress_print_overhead_included':True,'page_cache_not_cleared':True,
        'torch_version':torch.__version__,'ultralytics_version':ultralytics.__version__,'opencv_version':cv2.__version__,
        'python':sys.version,'warmup':{'YOLO_batches':warm_yolo['batches'],'MoCo_batches':warm_moco['batches']}})
    write(out/'memory_policy.json',{'gpu_profile':a.gpu_profile,'gpu_name':torch.cuda.get_device_name(0),
          'release_warm_cache':a.release_warm_cache,'release_phase_cache':a.release_phase_cache,
          'warm_kernel_execution_kept_but_allocator_workspace_may_be_reallocated_inside_hot':a.release_warm_cache})
    if sampler:sampler.start()
    torch.cuda.synchronize()
    write(out/'worker_warm_ready.json',{'pid':__import__('os').getpid(),'stage':a.stage,'startup_seconds':startup})
    if a.gate:
        waiting=time.monotonic()
        while not Path(a.gate).exists():
            if time.monotonic()-waiting>900:raise RuntimeError('Start gate timed out')
            time.sleep(.01)
    memory_baseline={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    hot_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat();hot_ns=time.perf_counter_ns();hot=hot_ns/1e9
    streams={};timing={}
    if a.stage=='track' and not a.prepared:raise ValueError('--prepared required for track stage')
    for path in (files if a.stage!='track' else []):
        start=time.perf_counter();cap=cv2.VideoCapture(str(path));assert cap.isOpened()
        raw={};trans={};detections={};f=0;previous=None;pending=[];indices=[]
        def flush():
            if not pending:return
            results=yolo.predict(pending,**params);assert len(results)==len(indices)
            for idx,result in zip(indices,results):
                boxes=result.boxes.data.detach().cpu().numpy();ds=[];export=[]
                for i,b in enumerate(boxes):
                    conf=float(b[4]);cid=int(b[5]);box=[float(x) for x in b[:4]]
                    export.append({'class_id':cid,'confidence':conf,'bbox_xyxy':box})
                    if conf>=.359:ds.append({'id':f'node/{idx}/{i}','input_index':i,'class_id':cid,'box':box,'confidence':conf})
                raw[idx]=ds;detections[idx]=export
            pending.clear();indices.clear()
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(1080,1920)
            gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(960,540));M=np.eye(3)
            if previous is not None:
                pts=cv2.goodFeaturesToTrack(previous,maxCorners=600,qualityLevel=.01,minDistance=8)
                if pts is not None:
                    nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,pts,None)
                    good=status.reshape(-1).astype(bool) if status is not None else np.zeros(len(pts),bool)
                    if nxt is not None and good.sum()>=12:
                        affine,_=cv2.estimateAffinePartial2D(pts[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=3)
                        if affine is not None and np.isfinite(affine).all():M[:2]=affine;M[:2,2]*=2
            previous=gray;trans[f]=M;pending.append(frame);indices.append(f);f+=1
            if len(pending)==a.yolo_batch:flush()
        flush();cap.release();assert f==expected[path.stem]['frames']
        torch.cuda.synchronize();streams[path.stem]=(raw,trans,detections);timing[path.stem]={'frames':f,'detect_decode_GMC_seconds':time.perf_counter()-start,'started_after_hot_seconds':start-hot}
        print('DETECTED',path.stem,f,flush=True)
        if a.stage=='detect':
            prepared=Path(a.prepared) if a.prepared else out
            temporary=prepared/(path.stem+'_stream.json.tmp')
            write(temporary,{'source_sha256':expected[path.stem]['sha256'],'raw':raw,
                  'trans':{str(f):M.tolist() for f,M in trans.items()},'detections':detections})
            temporary.replace(prepared/(path.stem+'_stream.json'))
    results={}
    if a.stage=='detect' and a.release_phase_cache:torch.cuda.empty_cache();torch.cuda.synchronize()
    for path in (files if a.stage!='detect' else []):
        if a.stage=='track':
            s=time.perf_counter();source=Path(a.prepared)/(path.stem+'_stream.json')
            while not source.exists():
                if time.perf_counter()-s>900:raise RuntimeError('Fresh detection stream wait timed out')
                time.sleep(.01)
            j=json.loads(source.read_text(encoding='utf8'))
            assert j['source_sha256']==expected[path.stem]['sha256']
            raw={int(f):ds for f,ds in j['raw'].items()};trans={int(f):np.array(M) for f,M in j['trans'].items()}
            streams[path.stem]=(raw,trans,j['detections'])
            timing[path.stem]={'frames':len(raw),'fresh_stage_input_wait_and_read_seconds':time.perf_counter()-s}
        start=time.perf_counter();raw,trans,detections=streams[path.stem];feature_stats=[]
        def features(keys,local):
            started=time.perf_counter();byframe={};vectors={};tensors=[];names=[]
            for key in sorted(keys):
                fields=list(map(int,key.split(':')));byframe.setdefault(fields[0],[]).append((key,fields[1],fields[2] if local else None))
            if not byframe:
                feature_stats.append({'local':local,'crops':0,'seconds':time.perf_counter()-started});return vectors
            cap=cv2.VideoCapture(str(path));assert cap.isOpened()
            def flush_features():
                if not tensors:return
                with torch.inference_mode():array=torch.nn.functional.normalize(moco(torch.stack(tensors).to('cuda:0')),dim=-1).cpu().numpy()
                assert np.isfinite(array).all() and np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-4)
                vectors.update(zip(names,array));names.clear();tensors.clear()
            try:
                for f in range(max(byframe)+1):
                    ok,frame=cap.read();assert ok
                    lookup={d['input_index']:d for d in raw[f]}
                    for key,i,region in byframe.get(f,[]):
                        x,y,e,g=lookup[i]['box'];lo,hi=(0.,1.) if region is None else REGIONS[region]
                        tensors.append(crop(frame,[x,y+(g-y)*lo,e,y+(g-y)*hi],1.));names.append(key)
                        if len(tensors)==a.moco_batch:flush_features()
                flush_features()
            finally:cap.release()
            assert set(vectors)==set(keys)
            feature_stats.append({'local':local,'crops':len(keys),'seconds':time.perf_counter()-started});return vectors
        result=track(raw,trans,features);torch.cuda.synchronize()
        timing[path.stem].update(tracking_seconds=time.perf_counter()-start,features=feature_stats,
                                ready_after_hot_seconds=time.perf_counter()-hot)
        results[path.stem]=result
        print('TRACKED',path.stem,'global',result['global_demand'],'local',result['local_demand'],flush=True)
    torch.cuda.synchronize();ready_ns=time.perf_counter_ns();ready=(ready_ns-hot_ns)/1e9
    write(out/'computation_ready.json',{'ready_monotonic_ns':ready_ns,'hot_start_monotonic_ns':hot_ns,'stage':a.stage})
    total_frames=sum(row['frames'] for row in timing.values())
    summary={'status':'COMPUTATION_COMPLETE','videos':len(files),'frames':total_frames,'source_seconds':total_frames/10,
             'batch_wall_to_all_tracks_ready_seconds':ready,'normalized_seconds_per_10s':ready/(total_frames/10)*10,
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
             'warm_memory_baseline':memory_baseline,'timings':timing,'hot_start_monotonic_ns':hot_ns,
             'tracks_ready_monotonic_ns':ready_ns,'hot_start_UTC':hot_UTC,
             'no_rendering':True,'correctness_comparison_pending':True,'not_pipeline':True}
    for v,result in results.items():
        write(out/f'{v}_tracks.json',result)
        write(out/f'{v}_detections.json',streams[v][2])
    saved_ns=time.perf_counter_ns()
    summary['batch_wall_including_final_result_save_seconds']=(saved_ns-hot_ns)/1e9
    summary['final_result_save_seconds']=(saved_ns-ready_ns)/1e9
    resources=sampler.finish(hot_ns,ready_ns) if sampler else {'complete':True,'sampling_quality_pass':True,'GPU_sampled_peak_used_bytes':{},'delegated_to_coordinator':True}
    summary['resource_sampling_complete']=resources['complete']
    summary['resource_sampling_quality_pass']=resources['sampling_quality_pass']
    summary['GPU_sampled_peak_used_bytes']=resources['GPU_sampled_peak_used_bytes']
    summary['timing_record_write_and_sampler_shutdown_excluded']=True
    summary['status']='PASS_EXECUTION_NOT_ACCURACY' if resources['complete'] else 'INVALID_RESOURCE_RECORD';write(out/'summary.json',summary)
    if not resources['complete']:raise RuntimeError('Resource sampling incomplete; trial is invalid. See resources.json')
    if not resources['sampling_quality_pass']:raise RuntimeError('Sampling gaps exceeded 300ms; performance result retained but not eligible for ranking')
    print('FULL_CHAIN_EXECUTION_COMPLETE',out,'seconds',round(ready,3),flush=True)

if __name__=='__main__':main()
