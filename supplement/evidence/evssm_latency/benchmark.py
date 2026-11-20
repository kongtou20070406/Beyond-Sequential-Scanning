"""Paired EVSSM CUDA latency worker. Run only after the training queue exits."""
import hashlib, json, os, statistics, subprocess, sys, time
from pathlib import Path
from functools import partial
import torch
P=Path(__file__).resolve().parent
sys.path[:0]=[str(P/'runtime'), str(P/'adapter')]
import basicsr
from evssm_nsi import build_evssm
from clpo2027.fp32_inference import prepare_fp32_inference
from clpo2027.data import read_tensor

def model(arm):
    m=build_evssm(arm,'test').eval().float()
    m.load_state_dict(torch.load(P/'weights'/f'{arm}.pth',map_location='cpu',weights_only=False)['params'],strict=True)
    assert m.evssm_mixer_count == 48
    return m.cuda().requires_grad_(False)

def profile(m,arm):
    from fvcore.nn import FlopCountAnalysis
    from evssm_nsi import _scan_backend
    import warnings
    sample=torch.zeros(1,3,256,256,device='cuda')
    if arm=='nsi': prepare_fp32_inference(m,enabled=False)
    records=[]; hooks=[]
    if arm=='nsi':
        for op in m.modules():
            if hasattr(op,'state_channels') and hasattr(op,'K') and (hasattr(op,'radial_group_count') or hasattr(op,'selective_group_count')):
                def hook(o,inputs,out):
                    b,_,h,w=inputs[0].shape
                    records.append(int(b*h*w*o.state_channels*o.K*36))
                hooks.append(op.register_forward_hook(hook))
        m(sample)
        for h in hooks: h.remove()
        assert len(records)==48, len(records)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c=FlopCountAnalysis(m,sample).set_op_handle('prim::PythonOp.SelectiveScanCuda',partial(_scan_backend().selective_scan_flop_jit,backend='prefixsum',verbose=False))
        c.set_op_handle('aten::silu',None,'aten::neg',None,'aten::exp',None,'aten::flip',None)
        c.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
        c.total(); ops=dict(c.by_operator()); unsupported=dict(c.unsupported_ops())
    excluded={'layer_norm','batch_norm','group_norm','instance_norm','adaptive_avg_pool2d','upsample_bilinear2d','upsample_nearest2d'}
    traced=sum(int(v) for k,v in ops.items() if k not in excluded)
    if arm=='nsi': prepare_fp32_inference(m,enabled=True)
    return {'input':[1,3,256,256],'fvcore_macs':traced,'propagation_macs':sum(records),'macs':traced+sum(records),'by_operator':ops,'unsupported_ops':unsupported}

def main(index):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    models={a:model(a) for a in ('native','nsi')}
    meta={a:{'parameters':sum(p.numel() for p in m.parameters()),'checkpoint_sha256':hashlib.file_digest((P/'weights'/f'{a}.pth').open('rb'),'sha256').hexdigest()} for a,m in models.items()}
    im=read_tensor(P/'frame.png')[None].cuda(); assert tuple(im.shape)==(1,3,720,1280),im.shape
    with torch.no_grad():
        m=models['nsi']; prepare_fp32_inference(m,enabled=False); ref=m(im); torch.cuda.synchronize()
        ready=prepare_fp32_inference(m,enabled=True); assert ready['fused_blocks']==48,ready
        opt=m(im); torch.cuda.synchronize(); error=(ref-opt).abs().max().item(); del ref,opt
        assert error<1e-4, error
        for arm,m in models.items():
            for _ in range(3): m(im[...,:512,:512].contiguous())
            m(im)
        torch.cuda.synchronize(); rows=[]
        for k in range(12):
            order=['nsi','native'] if (k+index)%2==0 else ['native','nsi']
            for arm in order:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter(); start.record(); y=models[arm](im); end.record(); torch.cuda.synchronize()
                rows.append({'trial':k,'arm':arm,'cuda_ms':start.elapsed_time(end),'wall_ms':(time.perf_counter()-wall)*1000,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}); del y
        if index==0:
            for arm in ('native','nsi'): meta[arm]['profile_256']=profile(models[arm],arm)
    result={'process':index,'arms':meta,'max_abs_diff':error,'rows':rows,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'triton':__import__('triton').__version__,'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
    (P/f'process_{index}.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main(int(sys.argv[1]))
