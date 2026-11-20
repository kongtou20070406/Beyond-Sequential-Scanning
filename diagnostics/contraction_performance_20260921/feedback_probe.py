"""Analytic directional derivatives of frozen state dynamics; no parameter updates."""
import contextlib,json,types
import torch
import analyze_weights as a

def probe(op,x):
    f,k,d,lam,out,kfun=a.fields(op,x);g=op.selective_group_size
    h=torch.zeros_like(f);z=torch.zeros_like(f);w=op.depth_weights(f)
    generator=torch.Generator(device=f.device).manual_seed(913)
    rand=torch.randn(f.shape,device=f.device,generator=generator);rand/=a.N(rand)
    v=None;v_rad=None;drive_sens=torch.zeros_like(f);injection=a.apply_pair_kernel(d/a.N(d).clamp_min(1e-20),k)
    rows=[];incs=[]
    for j in range(op.K):
        r,eta,_=a.retain(h,lam,g)
        drive_sens=a.apply_pair_kernel(a.rjvp(h,drive_sens,lam,g),k)+injection
        if v is not None:
            v=a.apply_pair_kernel(a.rjvp(h,v,lam,g),k)
            v_rad=a.apply_pair_kernel(a.rjvp(h,v_rad,lam,g),k)
        new=a.apply_pair_kernel(d+r,k);delta=new-h;z+=w[j]*delta
        if j==0:v=rand.clone();v_rad=new/a.N(new).clamp_min(1e-20)
        row=dict(step=j+1,eta_mean=float(eta.mean()),eta_cubed_mean=float(eta.pow(3).mean()),
                 fraction_eta_below_point1=float((eta<.1).float().mean()),
                 retained_vs_drive_rms=float(a.rms(r)/a.rms(d).clamp_min(1e-20)),
                 retained_vs_drive_norm=float(a.N(r)/a.N(d).clamp_min(1e-20)),
                 random_h1_to_hk_direction_gain=float(a.N(v)),radial_h1_to_hk_direction_gain=float(a.N(v_rad)),
                 constant_drive_direction_gain=float(a.N(drive_sens)),weighted_increment_rms=float(a.rms(w[j]*delta)))
        rows.append(row);h=new
    return out(z),dict(steps=rows)

@contextlib.contextmanager
def patch(model,records):
    originals=[]
    for name,op in a.operators(model):
        originals.append((op,op.forward))
        def forward(self,x,name=name):
            y,info=probe(self,x);records.append(dict(module=name,**info));return y
        op.forward=types.MethodType(forward,op)
    try:yield
    finally:
        for op,fun in originals:op.forward=fun

with torch.inference_mode():
    for arm in ['full','no_gain','free_gain']:
        model,identity=a.load(arm)
        for i in range(5):
            dest=a.ROOT/f'feedback_{arm}_{i:03}.json'
            if dest.exists():continue
            x,gt,image=a.data(i);records=[]
            with patch(model,records):y=model(x)
            base=json.loads((a.ROOT/f'profile_{arm}_{i:03}.json').read_text())
            score=a.psnr(y,gt);assert abs(score-base['psnr'])<1e-4
            a.save(dest,dict(arm=arm,index=i,id=image,psnr=score,checkpoint=identity,records=records))
            print(arm,i,flush=True)
        del model;torch.cuda.empty_cache()
