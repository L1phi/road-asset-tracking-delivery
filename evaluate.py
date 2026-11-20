"""Evaluate native AP and fixed-threshold precision, recall and F1."""
import argparse
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=root/'data/road-assets737')
    p.add_argument('--weight', type=Path, default=root/'models/last.pt')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    import yaml
    import ultralytics
    if not torch.cuda.is_available() or torch.__version__!='2.6.0+cu124' or ultralytics.__version__!='8.4.146':
        raise RuntimeError('Use frozen CUDA/PyTorch/Ultralytics environment')
    if a.output.exists():
        raise FileExistsError(a.output)
    data = a.data.resolve()
    rows = json.loads((data/'dataset_manifest.json').read_text(encoding='utf-8'))
    test = [r for r in rows if r['split']=='test']
    assert len(test)==121 and sum(len(r['boxes']) for r in test)==588
    a.output.mkdir(parents=True)
    classes = json.loads((root/'config/classes.json').read_text(encoding='utf-8'))
    cfg={'path':str(data),'train':'images/train_original','val':'images/test','test':'images/test',
         'names':{r['id']:r['label'] for r in classes}}
    cfg_path=a.output.resolve()/'test.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    from ultralytics import YOLO
    result=YOLO(str(a.weight)).val(data=str(cfg_path),split='test',imgsz=960,batch=9,
        conf=.001,iou=.7,max_det=300,nms=True,device=0,save_txt=True,save_conf=True,
        plots=False,project=str(a.output.resolve()),name='native',exist_ok=False)
    from evaluation.frozen_iou50 import evaluate, parse_labels
    gt={Path(r['image']).stem:parse_labels((data/r['label']).read_bytes(),prediction=False) for r in test}
    preds={}
    for r in test:
        name=Path(r['image']).stem
        label=a.output/'native/labels'/(name+'.txt')
        preds[name]=parse_labels(label.read_bytes(),prediction=True) if label.exists() else []
    from run import sha
    report={'native_AP':result.results_dict,'fixed_confidence_iou50':evaluate(gt,preds,.359),
        'weight_sha256':sha(a.weight),'test_images':121,'GT_boxes':588,
        'native_P_R_not_fixed_confidence':True,'matcher':'historical greedy descending IoU same-class one-to-one'}
    (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EVALUATION_SAVED',a.output/'metrics.json')

if __name__=='__main__':
    main()
