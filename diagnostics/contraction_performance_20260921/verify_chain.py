"""Check the largest observed no-gain direction by FP64 finite differences."""
import json,torch
import analyze_weights as a
with torch.inference_mode():
    model,identity=a.load('no_gain');name='encoder_level1.0.attn.operator';op=dict(a.operators(model))[name];captured=[]
    hook=op.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    x,gt,image=a.data(0);model(x);hook.remove()
    f,k,d,lam,_,_=a.fields(op,captured[0]);g=op.selective_group_size
    gen=torch.Generator(device=f.device).manual_seed(913);v=torch.randn(f.shape,device=f.device,generator=gen);v/=a.N(v)
    k=k.double();d=d.double();lam=lam.double();v=v.double();h1=a.apply_pair_kernel(d,k);h=h1.clone();v_final=v.clone()
    for _ in range(4):
        v_final=a.apply_pair_kernel(a.rjvp(h,v_final,lam,g),k)
        h=a.apply_pair_kernel(d+a.retain(h,lam,g)[0],k)
    def continuation(h):
        for _ in range(4):h=a.apply_pair_kernel(d+a.retain(h,lam,g)[0],k)
        return h
    rows=[]
    for rel in [1e-4,1e-5,1e-6]:
        eps=rel*float(a.N(h1));fd=(continuation(h1+eps*v)-continuation(h1-eps*v))/(2*eps)
        rows.append(dict(relative_epsilon=rel,fd_gain=float(a.N(fd)),relative_vector_error=float(a.N(fd-v_final)/a.N(v_final))))
    result=dict(arm='no_gain',index=0,module=name,analytic_gain=float(a.N(v_final)),checks=rows,
                scope='Selected after the all-module diagnostic solely to verify its largest measured direction; not an independent generalization sample.')
    assert rows[-1]['relative_vector_error']<1e-5
    a.save(a.ROOT/'chain_validation.json',result);print(json.dumps(result))
