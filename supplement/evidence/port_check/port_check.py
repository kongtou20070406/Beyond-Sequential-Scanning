"""Compare separately instantiated official and VIVNet port modules on CUDA."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'package'
INPUTS = ROOT / 'port_inputs'
RUNTIME = PACKAGE / 'tasks/restoration/runtime'
SPATIAL = INPUTS / 'supplement/third_party/SpatialMamba/official'
os.environ['SUPPLEMENT_SPATIAL_OFFICIAL'] = str(SPATIAL)
sys.path[:0] = [str(RUNTIME), str(SPATIAL / 'kernels/dwconv2d')]
import clpo2027
clpo2027.__path__.insert(0, str(INPUTS / 'supplement/runtime_overlays/native_blocks/clpo2027'))
clpo2027.__path__.insert(0, str(INPUTS / 'supplement/runtime_overlays/spatial/clpo2027'))
from clpo2027 import mair_vmm_adapter as mair

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Independent MaIR import retains the upstream scanning functions.
official_mair = mair._load_official_mair()
names = ('mamba_ssm','mamba_ssm.ops','mamba_ssm.ops.selective_scan_interface')
old = {k: sys.modules.get(k) for k in names}
mods = [types.ModuleType(k) for k in names]
mods[0].ops = mods[1]
mods[1].selective_scan_interface = mods[2]
mods[2].selective_scan_fn = mair._selective_scan_fn
mods[2].selective_scan_ref = mair._selective_scan_fn
sys.modules.update(dict(zip(names, mods)))
sys.path.insert(0, str(RUNTIME / 'third_party/2025-CVPR-MaIR/basicsr/archs'))
official_unet = load('mairunet_upstream', INPUTS/'mairunet_arch_upstream.py')
for k,v in old.items():
    if v is None: sys.modules.pop(k, None)
    else: sys.modules[k] = v

from clpo2027 import spatial_mamba_adapter as spa
native = load('basicsr.models.archs.VIVNativeBlocks_arch', INPUTS/'supplement/runtime_overlays/native_blocks/basicsr/models/archs/VIVNativeBlocks_arch.py')
matched = load('basicsr.models.archs.MatchedSpatialCheck_arch', INPUTS/'supplement/runtime_overlays/spatial/basicsr/models/archs/VIVNet_MatchedMixers_arch.py')

# The official extension exports only the forward operation. Use the exact
# convolution adjoint for the requested input-gradient comparisons.
def spatial_depthwise_backward(ctx, grad):
    x, inverse = ctx.saved_tensors
    weight = inverse.flip((2, 3))
    padding = (ctx.padding_h, ctx.padding_w)
    dx = torch.nn.grad.conv2d_input(x.shape, weight, grad.contiguous(), padding=padding, groups=x.shape[1])
    dw = torch.nn.grad.conv2d_weight(x, weight.shape, grad.contiguous(), padding=padding, groups=x.shape[1])
    db = grad.sum((0,2,3)) if ctx.is_bias else None
    return dx,dw,db,None,None,None

spa.module.DepthwiseFunction.backward = staticmethod(spatial_depthwise_backward)
native.spatial_official.DepthwiseFunction.backward = staticmethod(spatial_depthwise_backward)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(42)

class Reference(nn.Module):
    def __init__(self, kind, c, d, block):
        super().__init__()
        self.kind, self.block_index = kind, block
        if kind == 'mair_core': self.core = official_mair.VMM(d_model=c, d_state=d, expand=2.0)
        elif kind == 'mair_full': self.core = official_unet.VSSBlock(hidden_dim=c, d_state=d, ssm_ratio=1.2, mlp_ratio=2, norm_layer=nn.LayerNorm, shift_size=4 if block else 0)
        elif kind == 'spatial_core': self.core = spa.module.StructureAwareSSM(d_model=c, d_state=1, expand=2, d_conv=3, dropout=0.)
        else: self.core = spa.module.SpatialMambaBlock(hidden_dim=c, d_state=1, mlp_ratio=4.0, norm_layer=nn.LayerNorm, drop_path=0.0)
    def forward(self,x):
        b,c,h,w = x.shape
        v=x.permute(0,2,3,1).contiguous()
        if self.kind.startswith('mair'):
            k=4 if self.kind=='mair_core' else 8
            base=official_mair.mair_ids_generate((1,1,h,w),scan_len=k)
            shifted=official_mair.mair_shift_ids_generate((1,1,h,w),scan_len=k,shift_len=k//2)
            ids=tuple(i.to(x.device) for i in (*base,*shifted))
            if self.kind=='mair_core': out=self.core(v, ids[2:] if self.block_index else ids[:2])
            else: out=self.core(v.reshape(b,h*w,c),ids,(h,w)).reshape(b,h,w,c)
        else: out=self.core(v)
        return out.permute(0,3,1,2).contiguous()

def compare(kind,c,side,d,block):
    if kind=='mair_core': port=mair.MaIRVMMMixer(c,d_state=d,block_index=block); inner=port.vmm
    elif kind=='mair_full': port=native.MaIRNativeBlock(c,d_state=4*(c//32),block_index=block); inner=port.block
    elif kind=='spatial_core': port=spa.SpatialMambaMixer(c); inner=port.ssm
    else: port=native.SpatialNativeBlock(c); inner=port.block
    ref=Reference(kind,c,d if kind!='mair_full' else 4*(c//32),block)
    ref.core.load_state_dict(inner.state_dict(),strict=True)
    port=port.cuda().eval(); ref=ref.cuda().eval()
    x=torch.randn(1,c,side,side,device='cuda',requires_grad=True)
    y=x.detach().clone().requires_grad_()
    op=port(x); oracle=ref(y)
    probe=torch.randn_like(op)
    gp=torch.autograd.grad(op,x,probe)[0]
    gr=torch.autograd.grad(oracle,y,probe)[0]
    def err(a,b):
        diff=(a-b).abs()
        return {'max_abs':diff.max().item(),'max_rel_eps1e-6':(diff/(b.abs()+1e-6)).max().item()}
    result={'shape':[1,c,side,side], 'block_index':block, 'forward':err(op,oracle), 'input_gradient':err(gp,gr)}
    del port,ref,x,y,op,oracle,gp,gr,probe
    torch.cuda.empty_cache()
    return result

def main():
    stages=[(32,256,4,2),(64,128,8,3),(128,64,16,4),(128,64,16,4),(64,128,8,3),(32,256,4,2),(32,256,4,4)]
    result={'tf32':False,'spatial_backward':'official depthwise extension exports forward only; exact PyTorch convolution adjoint used for input gradients', 'official_paper_reproduction':'not done: no official model weights and test set established locally','comparisons':{},'checkpoints':{}}
    for kind in ('mair_core','mair_full','spatial_core','spatial_full'):
        rows=[]
        for c,side,d,n in stages:
            for block in (0,1) if n>1 else (0,):
                rows.append(compare(kind,c,side,d,block))
        result['comparisons'][kind]=rows
        print(kind, json.dumps(rows), flush=True)
        if kind=='mair_core': model=matched.VIVNetMatchedMixer(mixer_type='mair',K=5); path='weights/ohaze/mair/seed42/last.pth'
        elif kind=='spatial_core': model=matched.VIVNetMatchedMixer(mixer_type='spatial_mamba',K=5); path='weights/supplement/ohaze-spatial-mamba-seed42/last.pth'
        elif kind=='mair_full': model=native.VIVNativeBlocks('mair'); path='weights/supplement/ohaze-mair-full-block-seed42/last.pth'
        else: model=native.VIVNativeBlocks('spatial'); path='weights/supplement/ohaze-spatial-full-block-seed42/last.pth'
        ckpt=torch.load(INPUTS/path,map_location='cpu',weights_only=False)
        state=ckpt.get('params',ckpt)
        model.load_state_dict(state,strict=True)
        counts={}
        target = mair._MAIR.VMM if kind.startswith('mair') else (spa.module.StructureAwareSSM if kind=='spatial_core' else native.spatial_official.SpatialMambaBlock)
        for i,m in enumerate(model.modules()):
            if isinstance(m,target):
                counts[str(i)]=0
                def hook(mod,args,out,key=str(i)): counts[key]+=1
                m.register_forward_hook(hook)
        model.cuda().eval()
        with torch.no_grad(): model(torch.randn(1,3,256,256,device='cuda'))
        result['checkpoints'][kind]={'strict':True,'hook_modules':len(counts),'hook_calls':sum(counts.values()), 'all_once':all(x==1 for x in counts.values()),'hook_type':str(target)}
        if kind=='mair_full':
            result['checkpoints'][kind]['official_vssblock_instances'] = sum(isinstance(m, official_unet.VSSBlock) for m in model.modules())
            result['checkpoints'][kind]['note'] = '22 reconstructed wrappers call official VMM; official VSSBlock forward itself is not executed.'
        del model,ckpt,state
        torch.cuda.empty_cache()
        (ROOT/'A_PORT_CHECK.json').write_text(json.dumps(result,indent=2))
    (ROOT/'A_PORT_CHECK.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
