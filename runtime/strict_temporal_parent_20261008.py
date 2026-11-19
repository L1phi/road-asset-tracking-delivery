import numpy as np

def warp_between(trans,a,b):
    M=np.eye(3)
    for f in range(min(a,b)+1,max(a,b)+1):M=trans.get(f,np.eye(3))@M
    return M if b>=a else np.linalg.inv(M)

def geometry(small,large,regions):
    s=np.array(small);b=np.array(large);sz=np.maximum(s[2:]-s[:2],1);bz=np.maximum(b[2:]-b[:2],1)
    inter=np.maximum(np.minimum(s[2:],b[2:])-np.maximum(s[:2],b[:2]),0).prod()/sz.prod()
    if bz[1]<1.4*sz[1] or not .55<=sz[0]/bz[0]<=1.35 or inter<.85:return None
    frac=((s[1]+s[3])/2-b[1])/bz[1]
    # Select the region by geometry ONLY; do not search appearance maxima.
    options=[(abs(frac-(lo+hi)/2)+.15*abs(np.log(sz[1]/(bz[1]*(hi-lo)))),k) for k,(lo,hi) in enumerate(regions) if abs(frac-(lo+hi)/2)<=.18 and .65<=sz[1]/(bz[1]*(hi-lo))<=1.5]
    if not options:return None
    _,k=min(options);return k,float(frac)
