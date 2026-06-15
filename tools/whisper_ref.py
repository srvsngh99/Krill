import numpy as np, json, hashlib
# deterministic waveform reproducible in Swift: 3s @16k, two sines
sr=16000; N=sr*3
n=np.arange(N)
wav=(0.5*np.sin(2*np.pi*440*n/sr)+0.3*np.sin(2*np.pi*880*n/sr)).astype(np.float32)

from transformers import WhisperFeatureExtractor
fe=WhisperFeatureExtractor(feature_size=80)
print("FE params:",dict(n_fft=fe.n_fft if hasattr(fe,'n_fft') else None,
      hop_length=fe.hop_length, chunk_length=fe.chunk_length,
      sampling_rate=fe.sampling_rate, feature_size=fe.feature_size,
      nb_max_frames=getattr(fe,'nb_max_frames',None)))
mf=np.array(fe.mel_filters)
print("mel_filters shape",mf.shape,"sum",float(mf.sum()),"hash",hashlib.md5(mf.tobytes()).hexdigest()[:8])
print("mel_filters[1,:6]",mf[1,:6].tolist())
print("mel_filters[:6,1]",mf[:6,1].tolist())
out=fe(wav,sampling_rate=sr,return_tensors="np")
feat=out["input_features"][0]  # [80,3000]
print("feat shape",feat.shape)
print("stats mean %.6f std %.6f min %.6f max %.6f"%(feat.mean(),feat.std(),feat.min(),feat.max()))
# golden samples: feat[mel, frame]
pts=[(0,0),(0,1),(0,50),(10,0),(10,100),(40,0),(40,200),(79,0),(79,500),(20,1499),(5,2999)]
for (m,f) in pts:
    print("feat[%d,%d]=%.6f"%(m,f,feat[m,f]))
