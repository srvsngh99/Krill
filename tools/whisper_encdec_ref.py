import numpy as np, glob, os
p=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--mlx-community--whisper-small.en-mlx/snapshots/*/weights.npz'))[0]
z=dict(np.load(p))
def g(k): return z[k].astype(np.float64)

# deterministic synthetic mel [3000,80], reproducible in Swift
T,M=3000,80
mel=np.zeros((T,M))
for t in range(T):
    for m in range(M):
        mel[t,m]=0.1*np.sin(0.05*t+0.3*m)

def gelu(x): 
    from math import sqrt
    import scipy.special as sp
    return x*0.5*(1.0+sp.erf(x/np.sqrt(2.0)))
try:
    import scipy.special
except Exception:
    # erf fallback
    def gelu(x):
        return x*0.5*(1.0+np.vectorize(lambda v: __import__('math').erf(v/np.sqrt(2.0)))(x))

def conv1d(x, w, b, stride, pad):
    # x [L, Cin], w [Cout, K, Cin], b [Cout]
    L,Cin=x.shape; Cout,K,_=w.shape
    xp=np.pad(x,((pad,pad),(0,0)))
    Lout=(xp.shape[0]-K)//stride+1
    out=np.zeros((Lout,Cout))
    for o in range(Cout):
        acc=np.zeros(Lout)
        for k in range(K):
            for t in range(Lout):
                acc[t]+=np.dot(xp[t*stride+k],w[o,k])
        out[:,o]=acc+b[o]
    return out

# conv layers (slow loop; fine once)
def conv1d_fast(x,w,b,stride,pad):
    L,Cin=x.shape;Cout,K,_=w.shape
    xp=np.pad(x,((pad,pad),(0,0)))
    Lout=(xp.shape[0]-K)//stride+1
    # build [Lout, K*Cin]
    cols=np.zeros((Lout,K*Cin))
    for k in range(K):
        idx=np.arange(Lout)*stride+k
        cols[:,k*Cin:(k+1)*Cin]=xp[idx]
    wmat=w.transpose(0,1,2).reshape(Cout,K*Cin)
    return cols@wmat.T+b

x=conv1d_fast(mel,g('encoder.conv1.weight'),g('encoder.conv1.bias'),1,1); x=gelu(x)
x=conv1d_fast(x,g('encoder.conv2.weight'),g('encoder.conv2.bias'),2,1); x=gelu(x)
# sinusoids
def sinusoids(length,ch,maxts=10000.0):
    half=ch//2
    logi=np.log(maxts)/(half-1)
    inv=np.exp(-logi*np.arange(half))
    st=np.arange(length)[:,None]*inv[None,:]
    return np.concatenate([np.sin(st),np.cos(st)],axis=1)
x=x+sinusoids(x.shape[0],x.shape[1])

def ln(x,w,b,eps=1e-5):
    mu=x.mean(-1,keepdims=True);var=x.var(-1,keepdims=True)
    return (x-mu)/np.sqrt(var+eps)*w+b
def lin(x,w,b=None):
    y=x@w.T
    return y+b if b is not None else y
def attn(x,pfx,nhead=12):
    L,D=x.shape;hd=D//nhead;scale=hd**-0.25
    q=lin(x,g(pfx+'query.weight'),g(pfx+'query.bias'))
    k=lin(x,g(pfx+'key.weight'))
    v=lin(x,g(pfx+'value.weight'),g(pfx+'value.bias'))
    q=q.reshape(L,nhead,hd).transpose(1,0,2)*scale
    k=k.reshape(L,nhead,hd).transpose(1,2,0)*scale
    v=v.reshape(L,nhead,hd).transpose(1,0,2)
    s=q@k  # [h,L,L]
    s=s-s.max(-1,keepdims=True)
    w=np.exp(s);w/=w.sum(-1,keepdims=True)
    o=(w@v).transpose(1,0,2).reshape(L,D)
    return lin(o,g(pfx+'out.weight'),g(pfx+'out.bias'))
for i in range(12):
    b=f'encoder.blocks.{i}.'
    x=x+attn(ln(x,g(b+'attn_ln.weight'),g(b+'attn_ln.bias')),b+'attn.')
    h=ln(x,g(b+'mlp_ln.weight'),g(b+'mlp_ln.bias'))
    h=lin(gelu(lin(h,g(b+'mlp1.weight'),g(b+'mlp1.bias'))),g(b+'mlp2.weight'),g(b+'mlp2.bias'))
    x=x+h
x=ln(x,g('encoder.ln_post.weight'),g('encoder.ln_post.bias'))
print("enc shape",x.shape)
print("stats mean %.6f std %.6f min %.6f max %.6f"%(x.mean(),x.std(),x.min(),x.max()))
for (t,c) in [(0,0),(0,1),(0,100),(1,0),(750,384),(1499,767),(100,50),(500,500)]:
    print("enc[%d,%d]=%.6f"%(t,c,x[t,c]))

# ---- decoder parity ----
enc=x.copy()  # [1500,768] encoder output from synthetic mel
tokens=[1,2,3,4,5]
te=g('decoder.token_embedding.weight')  # [vocab,768]
dx=te[tokens]+g('decoder.positional_embedding')[:len(tokens)]
def mha(xq,xkv,pfx,nhead=12,causal=False):
    Lq,D=xq.shape;Lk=xkv.shape[0];hd=D//nhead;scale=hd**-0.25
    q=lin(xq,g(pfx+'query.weight'),g(pfx+'query.bias'))
    k=lin(xkv,g(pfx+'key.weight'))
    v=lin(xkv,g(pfx+'value.weight'),g(pfx+'value.bias'))
    q=q.reshape(Lq,nhead,hd).transpose(1,0,2)*scale
    k=k.reshape(Lk,nhead,hd).transpose(1,2,0)*scale
    v=v.reshape(Lk,nhead,hd).transpose(1,0,2)
    s=q@k
    if causal:
        m=np.triu(np.ones((Lq,Lk)),1)*-1e9
        s=s+m[None]
    s=s-s.max(-1,keepdims=True)
    w=np.exp(s);w/=w.sum(-1,keepdims=True)
    o=(w@v).transpose(1,0,2).reshape(Lq,D)
    return lin(o,g(pfx+'out.weight'),g(pfx+'out.bias'))
for i in range(12):
    b=f'decoder.blocks.{i}.'
    dx=dx+mha(ln(dx,g(b+'attn_ln.weight'),g(b+'attn_ln.bias')),
              ln(dx,g(b+'attn_ln.weight'),g(b+'attn_ln.bias')),b+'attn.',causal=True)
    dx=dx+mha(ln(dx,g(b+'cross_attn_ln.weight'),g(b+'cross_attn_ln.bias')),enc,b+'cross_attn.')
    h=ln(dx,g(b+'mlp_ln.weight'),g(b+'mlp_ln.bias'))
    h=lin(gelu(lin(h,g(b+'mlp1.weight'),g(b+'mlp1.bias'))),g(b+'mlp2.weight'),g(b+'mlp2.bias'))
    dx=dx+h
dx=ln(dx,g('decoder.ln.weight'),g('decoder.ln.bias'))
logits=dx@te.T  # [5,vocab]
print("logits shape",logits.shape)
print("last argmax",int(logits[-1].argmax()),"val %.4f"%logits[-1].max())
for (t,vi) in [(0,0),(0,100),(4,50256),(4,1),(2,2000),(4,13)]:
    print("logit[%d,%d]=%.4f"%(t,vi,logits[t,vi]))
