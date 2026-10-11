"""Uncalibrated positive-appraisal readout; no binary warning semantics."""
import hashlib,json
from collections import deque
from pathlib import Path
import numpy as np

class Appraisal:
    def __init__(self,bundle):
        bundle=Path(bundle)
        self.manifest=json.loads((bundle/'manifest.json').read_text())
        path=bundle/'weights.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.manifest['weights_sha256']:
            raise ValueError('Appraisal weight hash mismatch')
        a=np.load(path,allow_pickle=False);self.w=a['weight'];self.b=float(a['bias'])
        if self.w.shape!=(3584,) or not np.isfinite(self.w).all() or not np.isfinite(self.b):
            raise ValueError('Invalid appraisal weights')
        self.reset()
    def reset(self):
        self.window=deque(maxlen=4);self.count=0;self.peak=None
    def score(self,h):
        h=np.asarray(h,dtype=np.float64)
        if h.shape!=(3584,) or not np.isfinite(h).all():raise ValueError('Invalid hidden state')
        raw=float(h@self.w+self.b);self.count+=1;self.window.append(raw)
        mean=float(np.mean(self.window))
        # At four tokens discard short-prefix provisional means.
        if self.count<=4:self.peak=mean
        else:self.peak=max(self.peak,mean)
        return dict(appraisal_raw=raw,appraisal_mean4=mean,appraisal_peak4=self.peak,
                    appraisal_window_ready=self.count>=4,appraisal_experimental=True)
