"""Read-only integrity/GT checks for the delivered training and test corpus."""
import json
import math
from collections import Counter
from pathlib import Path
from run import sha

data=Path('data/road-assets737')
rows=json.loads((data/'dataset_manifest.json').read_text(encoding='utf-8'))
counts=Counter(); images=Counter()
for row in rows:
    assert sha(data/row['image'])==row['image_sha256']
    assert sha(data/row['label'])==row['label_sha256']
    label=(data/row['label']).read_text(encoding='utf-8').splitlines()
    assert len(label)==len(row['boxes'])
    for line in label:
        values=list(map(float,line.split()))
        assert len(values)==5 and all(map(math.isfinite,values))
        c,x,y,w,h=values
        assert int(c)==c and 0<=c<17 and 0<w<=1 and 0<h<=1
        assert -1e-8<=x-w/2<=x+w/2<=1+1e-8
        assert -1e-8<=y-h/2<=y+h/2<=1+1e-8
    counts[row['split']]+=len(label);images[row['split']]+=1
assert images=={'train_original':737,'test':121}
assert counts=={'train_original':3447,'test':588}
hashes={split:{r['image_sha256'] for r in rows if r['split']==split} for split in images}
assert len(hashes['train_original'])==737 and len(hashes['test'])==121
assert not hashes['train_original']&hashes['test']
generated=json.loads((data/'videoquality_manifest.json').read_text(encoding='utf-8'))
assert len(generated['images'])==737
for row in generated['images']:
    assert sha(data/row['image'])==row['sha256']
    assert sha(data/row['label'])==row['label_sha256']
    original=data/'labels/train_original'/Path(row['label']).name
    assert sha(original)==row['label_sha256']
print('DATASET_VERIFIED train_original=737 train_videoquality=737 test=121 GT=3447/588; exact_overlap=0')
