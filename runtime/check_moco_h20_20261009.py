"""Standalone real-crop MoCo readiness check, not an end-to-end speed test."""
import argparse, hashlib, json, time
from pathlib import Path

MODEL_SHA = '5baee69ef3c0b6f153eac41543be48dbd2ca6ff8f6327bb44fc0aa61f8fb4eee'
REMOTE_MODEL = '/xusc/ZPH/road_assets_yolo/video_tracking/moco_input/moco-v3-r50-1000ep.pth.tar'
HERE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def export_samples(dest):
    import cv2, numpy as np
    from PIL import Image
    base = HERE.parent / 'hangzhou737B_20261009'
    source = base / 'predictions/737B_new29_timed_v2_20261009T021733Z'
    tensors, provenance = [], []
    # Real boxes, including partial-height queries; no GT or random input.
    for file in sorted(source.glob('*_737B_predictions.json')):
        j = json.loads(file.read_text(encoding='utf8'))
        cap = cv2.VideoCapture(str(base / 'videos' / j['video']))
        try:
            for row in j['frames']:
                ok, frame = cap.read()
                if not ok: raise RuntimeError('Video decode failed')
                eligible = [(i,d) for i,d in enumerate(row['detections'])
                            if d['confidence'] >= .359 and d['class_id'] not in (7,9,10)]
                for i, d in eligible[:2]:
                    for lo, hi in ((0.,1.), (0.,.5)):
                        a,b,c,e = d['bbox_xyxy']; b,e = b+(e-b)*lo,b+(e-b)*hi
                        center=np.array([(a+c)/2,(b+e)/2]); size=np.maximum([c-a,e-b],2)
                        x1,y1=np.maximum(np.floor(center-size/2),[0,0]).astype(int)
                        x2,y2=np.minimum(np.ceil(center+size/2),[frame.shape[1],frame.shape[0]]).astype(int)
                        if x2<=x1 or y2<=y1: continue
                        image=Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))
                        ratio=224/max(image.size)
                        image=image.resize((max(1,round(image.width*ratio)),max(1,round(image.height*ratio))),Image.Resampling.BICUBIC)
                        canvas=Image.new('RGB',(224,224),(124,116,104))
                        canvas.paste(image,((224-image.width)//2,(224-image.height)//2))
                        arr=np.asarray(canvas,dtype=np.float32)/255
                        arr=(arr-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
                        tensors.append(arr.transpose(2,0,1))
                        provenance.append({'video':j['video'],'frame':row['source_frame_index'],'input_index':i,'class_id':d['class_id'],'height_region':[lo,hi]})
                        if len(tensors)==32: break
                    if len(tensors)==32: break
                if len(tensors)==32: break
        finally: cap.release()
        if len(tensors)==32: break
    if len(tensors)!=32: raise RuntimeError('Expected 32 real crops')
    dest=Path(dest)
    if dest.exists() or dest.with_suffix('.json').exists(): raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,inputs=np.stack(tensors).astype(np.float32))
    dest.with_suffix('.json').write_text(json.dumps({'samples_sha256':sha(dest),'samples':provenance,'scope':'readiness samples only, not workload timing'},ensure_ascii=False,indent=2),encoding='utf8')
    print('REAL_SAMPLES_EXPORTED',dest,flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--export-local-samples')
    ap.add_argument('--samples',default=str(HERE/'moco_real_samples_20261009.npz'))
    ap.add_argument('--model',default=REMOTE_MODEL)
    ap.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0'])
    ap.add_argument('--out',default=str(HERE/'moco_h20_readiness.json'))
    ap.add_argument('--self-check',action='store_true')
    args=ap.parse_args()
    if args.self_check:
        compile(Path(__file__).read_text(encoding='utf8'),__file__,'exec')
        print('SELF_CHECK: PASS'); return
    if args.export_local_samples: export_samples(args.export_local_samples); return
    out=Path(args.out)
    if out.exists(): raise FileExistsError('Use a fresh --out; existing results preserved')
    import torch, numpy as np
    from torchvision.models import resnet50
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; refusing CPU fallback')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    modelpath=Path(args.model); samples=Path(args.samples)
    if sha(modelpath)!=MODEL_SHA: raise RuntimeError('MoCo checkpoint identity mismatch')
    meta=json.loads(samples.with_suffix('.json').read_text(encoding='utf8'))
    if sha(samples)!=meta['samples_sha256']: raise RuntimeError('Sample identity mismatch')
    with np.load(samples,allow_pickle=False) as z: array=z['inputs']
    if array.shape!=(32,3,224,224) or array.dtype!=np.float32 or not np.isfinite(array).all(): raise RuntimeError('Invalid input samples')
    print('LOAD_MOCO',args.device,flush=True)
    model=resnet50(weights=None)
    checkpoint=torch.load(modelpath,map_location='cpu',weights_only=True)
    prefix='module.base_encoder.'
    weights={k[len(prefix):]:v for k,v in checkpoint['state_dict'].items() if k.startswith(prefix) and not k.startswith(prefix+'fc.')}
    report=model.load_state_dict(weights,strict=False)
    if set(report.missing_keys)!={'fc.weight','fc.bias'} or report.unexpected_keys: raise RuntimeError(str(report))
    model.fc=torch.nn.Identity();model.eval()
    del checkpoint,weights
    cpu=torch.from_numpy(array)
    with torch.inference_mode(): reference=torch.nn.functional.normalize(model(cpu),dim=-1)
    model.to(device); inputs=cpu.to(device)
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    def forward(x): return torch.nn.functional.normalize(model(x),dim=-1)
    rows=[]
    with torch.inference_mode():
        for _ in range(3): forward(inputs)
        sync()
        first=forward(inputs).cpu();sync()
        individual=torch.cat([forward(inputs[i:i+1]).cpu() for i in range(len(inputs))]);sync()
        for _ in range(10):
            sync(); start=time.perf_counter(); current=forward(inputs);sync()
            rows.append(time.perf_counter()-start)
        final=current.cpu()
    def cosine(a,b): return float(torch.nn.functional.cosine_similarity(a,b,dim=-1).min())
    cpu_cos=cosine(reference,first); batch_cos=cosine(first,individual); repeat_cos=cosine(first,final)
    passed=bool(torch.isfinite(first).all() and first.shape==(32,2048)
                and torch.allclose(first.norm(dim=-1),torch.ones(32),atol=1e-4)
                and min(cpu_cos,batch_cos,repeat_cos)>=.9999)
    result={'status':'PASS' if passed else 'FAIL','device':str(device),'GPU_verified':device.type=='cuda' and passed,
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'model_sha256':MODEL_SHA,'sample_sha256':sha(samples),'fp32':True,'TF32_disabled':device.type=='cuda',
            'CPU_GPU_min_cosine':cpu_cos,'batch1_batch32_min_cosine':batch_cos,'repeat_min_cosine':repeat_cos,
            'max_abs_CPU_GPU_difference':float((reference-first).abs().max()),
            'warmed_batch32_forward_normalize_seconds':rows,'warmup_batches':3,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'not_end_to_end_benchmark':True,'no_training_or_download':True,
            'note':'Readiness uses FP32 and real whole/partial crops. Similar features do not prove identical tracking decisions.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('MOCO_READINESS:',result['status'],'GPU_verified=',result['GPU_verified'],flush=True)
    print('report:',out,flush=True)
    if not passed: raise SystemExit(2)

if __name__=='__main__': main()
