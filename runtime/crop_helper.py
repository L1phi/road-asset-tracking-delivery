import numpy as np
import torch
import cv2
from PIL import Image

def crop(frame,box,scale):
    b=np.array(box);center=(b[:2]+b[2:])/2;size=np.maximum(b[2:]-b[:2],2)*scale
    x1,y1=np.maximum(np.floor(center-size/2),[0,0]).astype(int)
    x2,y2=np.minimum(np.ceil(center+size/2),[frame.shape[1],frame.shape[0]]).astype(int)
    if x2<=x1 or y2<=y1:raise ValueError('Invalid crop')
    image=Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))
    ratio=224/max(image.size)
    image=image.resize((max(1,round(image.width*ratio)),max(1,round(image.height*ratio))),Image.Resampling.BICUBIC)
    canvas=Image.new('RGB',(224,224),(124,116,104));canvas.paste(image,((224-image.width)//2,(224-image.height)//2))
    a=np.asarray(canvas,dtype=np.float32)/255
    return torch.from_numpy((a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).permute(2,0,1)
