"""Reconstruct training degradation from the recorded recipe; never touch test images."""
import argparse
import json
import shutil
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'data/road-assets737')
    a = p.parse_args()
    import cv2
    import numpy as np
    from PIL import Image, JpegImagePlugin
    from run import sha
    root = Path(__file__).resolve().parent
    recipe = json.loads((root / 'training/experiment_record.json').read_text(encoding='utf-8'))['degradation']
    qt = {int(k): v for k, v in recipe['jpeg_calibration']['qtables'].items()}
    sub = recipe['jpeg_calibration']['subsampling']
    rows = json.loads((a.data / 'dataset_manifest.json').read_text(encoding='utf-8'))
    rows = [r for r in rows if r['split'] == 'train_original']
    assert len(rows) == 737
    output = a.data / 'images/train_videoquality'
    labels = a.data / 'labels/train_videoquality'
    if output.exists() or labels.exists():
        raise FileExistsError('Refusing to overwrite generated training set')
    output.mkdir(parents=True)
    labels.mkdir(parents=True)
    generated = []
    for row in rows:
        source = a.data / row['image']
        assert sha(source) == row['image_sha256']
        with Image.open(source) as im:
            image = im.convert('RGB')
        ratio = min(1, 1920 / image.width, 1080 / image.height)
        image = image.resize((round(image.width*ratio), round(image.height*ratio)), Image.Resampling.LANCZOS)
        rgb = cv2.GaussianBlur(np.asarray(image), (0, 0), sigmaX=.6, sigmaY=.6)
        dest = output / (source.stem + '.jpg')
        Image.fromarray(rgb).save(dest, format='JPEG', qtables=qt, subsampling=sub)
        with Image.open(dest) as im:
            assert im.quantization == qt and JpegImagePlugin.get_sampling(im) == sub
        label = labels / (source.stem + '.txt')
        shutil.copy2(a.data / row['label'], label)
        generated.append({'image':dest.relative_to(a.data).as_posix(),'sha256':sha(dest),
            'label':label.relative_to(a.data).as_posix(),'label_sha256':sha(label)})
    (a.data / 'videoquality_manifest.json').write_text(json.dumps({
        'status':'RECONSTRUCTED_FROM_RECIPE_NOT_HISTORICAL_BYTE_IDENTITY',
        'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,
        'original_test_unchanged':True,'images':generated}, indent=2), encoding='utf-8')
    print('VIDEOQUALITY_TRAIN_IMAGES_READY', len(generated))

if __name__ == '__main__':
    main()
