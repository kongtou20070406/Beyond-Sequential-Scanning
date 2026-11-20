"""Frozen-checkpoint diagnostics. No optimizer, backward pass, or checkpoint writes."""
from pathlib import Path
import argparse, contextlib, hashlib, json, os, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
CODE = Path(__file__).resolve().parents[2]
Q = CODE/'tasks/restoration/haze4k'
os.environ['NSI_HAZE_OUTPUT'] = str(ROOT/'runtime')
sys.path.insert(0, str(Q))
from models import make_model
from clpo2027.real_pair_propagation import RealPairStatePropagation, apply_pair_kernel, OFFSETS
import clpo2027.fp32_inference as fast
fast.use_fp32_inference = lambda *args, **kwargs: False
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
MANIFEST = Q/'DATA_MANIFEST.json'
manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
pixels = None

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def N(x): return x.reshape(x.shape[0], -1, 2, *x.shape[-2:]).square().sum(2).sqrt().amax()
def rms(x): return x.square().mean().sqrt()
def pair(r, i): return torch.stack([r,i], 2).flatten(1,2)
def halves(x):
    p=x.reshape(x.shape[0],-1,2,*x.shape[-2:]); return torch.cat([p[:,:,0],p[:,:,1]],1)
def scaled(x, a): return (x.reshape(x.shape[0],-1,2,*x.shape[-2:])*a.unsqueeze(2)).reshape_as(x)

def retain(h, strength, g=4):
    b,c,y,x=h.shape; z=h.reshape(b,c//(2*g),g,2,y,x)
    lam=strength.reshape(b,c//(2*g),g,y,x).mean(2,keepdim=True).unsqueeze(3)
    eta=(1+lam*z.square().sum(3,keepdim=True).mean(2,keepdim=True)).rsqrt()
    return (z*eta).reshape_as(h),eta,lam

def rjvp(h, v, strength, g=4):
    _,eta,lam=retain(h,strength,g); z=h.reshape(h.shape[0],-1,g,2,*h.shape[-2:]);u=v.reshape_as(z)
    return (eta*u-(lam/g)*eta.pow(3)*z*(z*u).sum((2,3),keepdim=True)).reshape_as(h)

def fields(op, x):
    if isinstance(op,RealPairStatePropagation):
        f=op.Bin(x);kernel,w,r,lam=op.input_fields(f)
        def output(z): return op.Cout(scaled(z,r))
        return f,kernel,scaled(f,w),lam,output,lambda h:kernel
    packed,fr,fi,attention=op._drive_with_attention(x)
    hidden=op._coefficient_hidden(x,packed)
    w,r,lam=op._selective_controls(hidden)
    ar,ai=op._ablation_coefficients(hidden);kernel=torch.stack([ar,ai],3)
    def output(z): return op._interface_readout(halves(scaled(z,r)),attention)
    def dynamic(h):
        hidden=op._coefficient_hidden(x,packed+halves(h))
        ar,ai=op._ablation_coefficients(hidden);return torch.stack([ar,ai],3)
    return pair(fr,fi),kernel,pair(w*fr,w*fi),lam,output,(dynamic if op.ablation=='state_dynamic_transition' else lambda h:kernel)

def mass(k): return k.square().sum(3).sqrt().sum(1)
def valid_row_gain(k):
    # Ignore zero-boundary taps: this is the exact L norm for the pair-max norm.
    a=k.square().sum(3).sqrt().clone(); h,w=k.shape[-2:]
    for j,(dy,dx) in enumerate(OFFSETS):
        if dy==-1:a[:,j,:,0,:]=0
        if dy==1:a[:,j,:,-1,:]=0
        if dx==-1:a[:,j,:,:,0]=0
        if dx==1:a[:,j,:,:,-1]=0
    return a.sum(1)

def local_bound(k, eta, g):
    # Paper's group Jacobian bound beta(s)=sqrt(1-s)*(1+(sqrt(g)-1)*s/2).
    # For state_dynamic this bounds ONLY the frozen-kernel term, not full DT.
    beta=eta*(1+(g**.5-1)*(1-eta.square())/2)
    beta=beta.squeeze(3).expand(-1,-1,g,-1,-1).flatten(1,2)
    padded=F.pad(beta,(1,1,1,1));h,w=k.shape[-2:];bound=torch.zeros_like(beta)
    for j,(dy,dx) in enumerate(OFFSETS):
        bound += k[:,j].square().sum(2).sqrt()*padded[:,:,1+dy:1+dy+h,1+dx:1+dx+w]
    return bound.amax()

def rollout(op,x,mode='native',record=False,extended=15,jacobian=False):
    f,base,d,lam,out,kfun=fields(op,x);g=op.selective_group_size
    def modify(k):
        if mode.startswith('scale_'):return k*float(mode.split('_')[1])
        if mode=='normalize':return k*(.95/(1+mass(k))).unsqueeze(1).unsqueeze(3)
        return k
    h=torch.zeros_like(f);z=torch.zeros_like(f);weights=op.depth_weights(f)
    last_delta=None;last_retained=None;last_kernel=None;steps=[];h4=None
    for j in range(extended if record else op.K):
        kernel=modify(kfun(h));r,eta,_=retain(h,lam,g)
        new=apply_pair_kernel(d+r,kernel);delta=new-h
        if j<op.K:z=z+weights[j]*delta
        if record:
            q=valid_row_gain(kernel)
            row=dict(step=j+1,h=float(N(new)),increment=float(N(delta)),state_rms=float(rms(new)),increment_rms=float(rms(delta)),
                     q=float(q.max()),q_mean=float(q.mean()),fraction_rows_over_one=float((q>1).float().mean()),
                     eta_mean=float(eta.mean()),eta_min=float(eta.min()),local_jacobian_upper=float(local_bound(kernel,eta,g)))
            if last_delta is not None:
                propagated=apply_pair_kernel(r-last_retained,kernel)
                drift=apply_pair_kernel(d+last_retained,kernel-last_kernel)
                row.update(increment_ratio=float(N(delta)/N(last_delta).clamp_min(1e-30)),kernel_drift=float(N(drift)),
                           retained_difference_term=float(N(propagated)),decomposition_error=float(N(delta-propagated-drift)),
                           drift_to_increment=float(N(drift)/N(delta).clamp_min(1e-30)))
            steps.append(row)
        if j==3:h4=new.clone()
        if j==op.K-1:result=out(z);h5=new.clone()
        last_delta=delta;last_retained=r;last_kernel=kernel;h=new
    info={}
    if record:
        info=dict(d_norm=float(N(d)),z_norm=float(N(z)),output_rms=float(rms(result)),weights=weights.cpu().tolist(),
                  strength_mean=float(lam.mean()),strength_min=float(lam.min()),steps=steps,
                  h15_minus_h5_over_h5=float(N(h-h5)/N(h5).clamp_min(1e-30)))
        if hasattr(op,'log_gain'):info['learned_gain']=float(op.log_gain.exp())
    if record and jacobian:
        # Two-scale central differences, on a fixed deterministic random direction.
        gen=torch.Generator(device=h4.device).manual_seed(731)
        v=torch.randn(h4.shape,device=h4.device,generator=gen);v=v/N(v)
        kernel=modify(kfun(h4));frozen=apply_pair_kernel(rjvp(h4,v,lam,g),kernel)
        probes=[]
        for rel in [1e-3,3e-4]:
            eps=rel*max(float(N(h4)),1.)
            hp,hm=h4+eps*v,h4-eps*v
            plus=apply_pair_kernel(d+retain(hp,lam,g)[0],modify(kfun(hp)))
            minus=apply_pair_kernel(d+retain(hm,lam,g)[0],modify(kfun(hm)))
            total=(plus-minus)/(2*eps)
            probes.append(dict(relative_epsilon=rel,total_direction_gain=float(N(total)),frozen_kernel_gain=float(N(frozen)),
                               extra_kernel_feedback_gain=float(N(total-frozen))))
        info['directional_jacobian']=probes
    return result,info

def operators(model):
    return [(n,o) for n,o in model.named_modules() if isinstance(o,RealPairStatePropagation) or hasattr(o,'_drive_with_attention') and hasattr(o,'_ablation_coefficients')]

@contextlib.contextmanager
def patch(model,mode='native',records=None,matched=False,jacobian=False):
    originals=[]
    for name,op in operators(model):
        originals.append((op,op.forward))
        def forward(self,x,name=name):
            y,info=rollout(self,x,mode,record=records is not None,jacobian=jacobian)
            if matched:
                ref,_=rollout(self,x,'native');ratio=rms(ref)/rms(y).clamp_min(1e-20);y=y*ratio
            if records is not None:records.append(dict(module=name,shape=list(x.shape),**info))
            return y
        op.forward=types.MethodType(forward,op)
    assert len(originals)==22
    try:yield
    finally:
        for op,original in originals:op.forward=original

def load(arm):
    model=make_model('gate_only' if arm=='full' else arm)
    if arm=='full':
        for _,op in operators(model):op.arm='full'
    cp=CODE/f'weights/haze4k/{arm}/seed42/last.pth'
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['params'],strict=True)
    return model.cuda().eval().requires_grad_(False),dict(path=str(cp),sha256=sha(cp))

def data(i):
    global pixels
    if pixels is None:
        cache = Path(os.environ.get('NSI_HAZE_CACHE', str(CODE/'data/haze4k/cache')))
        pixels = np.memmap(cache/'test.bin', mode='r', dtype=np.uint8)
    row=manifest['splits']['test'][i];arr=[]
    for key in ['input','target']:
        a=row['arrays'][key];arr.append(np.array(pixels[a['offset']:a['offset']+a['bytes']].reshape(a['shape'])))
    x=torch.from_numpy(arr[0].transpose(2,0,1).copy()).float().cuda().unsqueeze(0)/255
    h,w=x.shape[-2:];x=F.pad(x,(0,(-w)%4,0,(-h)%4),mode='reflect')
    return x,arr[1],row['id']

def psnr(y,gt):
    h,w=gt.shape[:2];p=y[0,:,:h,:w].clamp(0,1).permute(1,2,0).cpu().numpy().astype('float64')
    return float(-10*np.log10(np.mean((p-gt.astype('float64')/255)**2)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','profile','intervene'],required=True);args=parser.parse_args()
    ROOT.mkdir(exist_ok=True);started=time.time();arms=['full','no_gain','free_gain','state_dynamic']
    protocol=dict(arms=arms,profile_indices=list(range(5)),intervention_indices=list(range(0,1000,50)),
                  scales=[.5,.8,1.,1.05,1.1,1.2,1.4],native_size=True,tf32=False,precision='float32',training=False,
                  extended_steps=15,extension='Local module on its native K=5 network input; actual network output uses only original five steps.',
                  jacobian='Random direction at h4, two central-difference scales; not an induced norm certificate.',manifest_sha256=sha(MANIFEST))
    save(ROOT/'protocol.json',protocol)
    with torch.inference_mode():
        if args.phase in ['smoke','profile']:
            summaries=[]
            for arm in arms:
                model,identity=load(arm);x,gt,name=data(0)
                y=model(x)
                with patch(model):yp=model(x)
                parity=float((y-yp).abs().max());assert parity<3e-5,(arm,parity)
                info=dict(arm=arm,checkpoint=identity,parity_max_abs=parity,psnr=psnr(yp,gt),parameters=sum(p.numel() for p in model.parameters()))
                print('PARITY',json.dumps(info),flush=True);summaries.append(info)
                if args.phase=='profile':
                    for i in protocol['profile_indices']:
                        dest=ROOT/f'profile_{arm}_{i:03}.json'
                        if dest.exists():continue
                        x,gt,name=data(i);records=[]
                        with patch(model,records=records,jacobian=True):yp=model(x)
                        save(dest,dict(arm=arm,index=i,id=name,checkpoint=identity,psnr=psnr(yp,gt),records=records))
                        print('PROFILE',arm,i,'seconds',round(time.time()-started,1),flush=True)
                del model,x,y,yp;torch.cuda.empty_cache()
            save(ROOT/f'{args.phase}_summary.json',summaries)
        else:
            results=[]
            for arm in ['full','no_gain']:
                model,identity=load(arm)
                modes=[(f'scale_{s}',False) for s in protocol['scales']] if arm=='full' else [('native',False),('normalize',False),('normalize',True)]
                if arm=='full':modes += [('scale_0.8',True),('scale_1.2',True)]
                for i in protocol['intervention_indices']:
                    dest=ROOT/f'intervene_{arm}_{i:03}.json'
                    if dest.exists():results.append(json.loads(dest.read_text()));continue
                    x,gt,name=data(i);scores=[]
                    for mode,matched in modes:
                        with patch(model,mode,matched=matched):y=model(x)
                        scores.append(dict(mode=mode,output_rms_matched=matched,psnr=psnr(y,gt),finite=bool(torch.isfinite(y).all())))
                    result=dict(arm=arm,index=i,id=name,checkpoint=identity,scores=scores);save(dest,result);results.append(result)
                    print('INTERVENE',arm,i,'seconds',round(time.time()-started,1),flush=True)
                del model,x,y;torch.cuda.empty_cache()
            save(ROOT/'intervention_results.json',results)
    print('COMPLETE',args.phase,round(time.time()-started,1),flush=True)

if __name__=='__main__':main()
