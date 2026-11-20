"""Train YOLO26l on the reviewed road asset dataset."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    p.add_argument('--data', type=Path, default=root/'data/road-assets737')
    p.add_argument('--base', type=Path, default=root/'models/yolo26l.pt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    from run import sha
    import torch
    import yaml
    import ultralytics
    if torch.__version__ != '2.6.0+cu124' or ultralytics.__version__ != '8.4.146':
        raise RuntimeError('Use frozen PyTorch/Ultralytics versions')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    if not a.base.is_file() or sha(a.base) != '9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26':
        raise ValueError('Require original general yolo26l.pt with recorded SHA256')
    if a.output.exists():
        raise FileExistsError('Output must be a new directory')
    data = a.data.resolve()
    rows = json.loads((data/'dataset_manifest.json').read_text(encoding='utf-8'))
    degraded = json.loads((data/'videoquality_manifest.json').read_text(encoding='utf-8'))['images']
    assert len(degraded) == 737
    train_rows = [r for r in rows if r['split']=='train_original']
    test_rows = [r for r in rows if r['split']=='test']
    assert len(train_rows)==737 and len(test_rows)==121
    assert not ({r['image_sha256'] for r in train_rows}&{r['image_sha256'] for r in test_rows})
    for row in degraded:
        assert sha(data/row['image'])==row['sha256'] and sha(data/row['label'])==row['label_sha256']
    cfg = yaml.safe_load((root/'training/args_original.yaml').read_text(encoding='utf-8'))
    drop = {'model','data','project','name','save_dir','resume','exist_ok','task','mode','cfg'}
    cfg = {k:v for k,v in cfg.items() if k not in drop}
    assert cfg['epochs']==100 and cfg['val'] is False and cfg['seed']==42
    if a.check_only:
        print('TRAINING_INPUTS_AND_GPU_CHECKED; training not started')
        return
    a.output.mkdir(parents=True)
    names = json.loads((root/'config/classes.json').read_text(encoding='utf-8'))
    dataset = {'path':str(data), 'train':'images/train_videoquality',
               'val':'images/train_videoquality','test':'images/test',
               'names':{r['id']:r['label'] for r in names}}
    data_yaml = a.output.resolve()/'data.yaml'
    data_yaml.write_text(yaml.safe_dump(dataset, allow_unicode=True), encoding='utf-8')
    cfg.update(data=str(data_yaml),project=str(a.output.resolve()),name='train',exist_ok=False,resume=False)
    (a.output/'resolved_train_args.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from types import SimpleNamespace
    torch.set_num_threads(4)
    probe=YOLO(str(a.base)).model.to('cuda').train()
    probe.requires_grad_(True)
    probe.args=SimpleNamespace(**dict(DEFAULT_CFG_DICT,**{k:v for k,v in cfg.items() if k in DEFAULT_CFG_DICT}))
    batch=dict(img=torch.rand(2,3,128,128,device='cuda'),batch_idx=torch.tensor([0,1],device='cuda'),
        cls=torch.zeros(2,1,device='cuda'),bboxes=torch.tensor([[.5,.5,.3,.3],[.5,.5,.3,.3]],device='cuda'))
    loss,_=probe.loss(batch);loss.sum().backward();head=probe.model[-1]
    for branch in (head.cv2,head.cv3,head.one2one_cv2,head.one2one_cv3):
        assert any(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in branch.parameters())
    del probe,batch,loss,head,branch
    torch.cuda.empty_cache()
    YOLO(str(a.base)).train(**cfg)
    import csv
    with (a.output/'train/results.csv').open() as f:
        log=list(csv.DictReader(f))
    assert len(log)==100 and int(log[-1]['epoch'])==100
    assert (a.output/'train/weights/last.pt').is_file()
    print('TRAINING_COMPLETE_USE_EPOCH100_LAST',a.output/'train/weights/last.pt')

if __name__ == '__main__':
    main()
