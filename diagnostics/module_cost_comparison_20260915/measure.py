"""CPU MAC audit of the exact mixer slots; does not benchmark CUDA latency."""
import os,sys,json,hashlib,importlib.util,time,warnings,contextlib,io
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
legacy=load('audit_base',ROOT/'diagnostics/novelty_strengthening_20260910/reconcile_mac_counts.py')
import torch
from fvcore.nn import FlopCountAnalysis
profile=legacy.release_profile
from fvcore.nn.jit_handles import einsum_flop_jit
def exact_einsum(inputs,outputs):
 equation=inputs[0].toIValue().replace(' ','')
 shapes=[v.type().sizes() for v in inputs[1].node().inputs()]
 if equation=='bkdl,kcd->bkcl':
  b,k,d,l=shapes[0];return b*k*d*l*shapes[1][1]
 if equation=='bkrl,kdr->bkdl':
  b,k,r,l=shapes[0];return b*k*r*l*shapes[1][1]
 return einsum_flop_jit(inputs,outputs)

jobs=[('ohaze','ours','O-HAZE_VIVNet_RadialProfiled_K5_P256.yml'),('ohaze','ss2d','O-HAZE_VIVNet_Matched_SS2D_P256.yml'),('gopro','ours','ours.yml'),('gopro','ss2d','ss2d.yml')]
records=[]
for task,arm,filename in jobs:
 cfg=ROOT/'configs/restoration'/task/filename
 torch.manual_seed(42)
 with contextlib.redirect_stdout(io.StringIO()):model=profile.restoration_model(cfg,5).cpu().eval()
 slots={name:m for name,m in model.named_modules() if name.endswith('.attn')}
 assert len(slots)==(22 if task=='ohaze' else 26)
 scan=legacy.count_only_ss2d(model) if arm=='ss2d' else {}
 shapes={};hooks=[]
 for name,m in slots.items():
  def capture(mod,inputs,output,key=name):shapes[key]=[int(v) for v in inputs[0].shape]
  hooks.append(m.register_forward_hook(capture))
 x=torch.zeros(1,3,256,256)
 with torch.inference_mode(),warnings.catch_warnings():
  warnings.simplefilter('ignore')
  a=FlopCountAnalysis(model,x).unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
  a.set_op_handle('aten::einsum',exact_einsum)
  total=a.total();by=a.by_module_and_operator()
 for h in hooks:h.remove()
 rows=[]
 for name,m in slots.items():
  b,c,h,w=shapes[name]
  ops={k:int(round(v)) for k,v in by[name].items() if k in legacy.KEEP_OPERATORS}
  if arm=='ours':
   assert m.operator.__class__.__name__=='RealPairStatePropagation'
   assert m.operator.K==5
   extra=b*h*w*18*c*5
   expected=b*h*w*(2*c*c+75*c+18*5*c)
   spec={'K':5,'real_channels':c,'channel_pairs':c//2}
  else:
   extra=scan[name]['official_scan_macs']
   d=m.d_inner;g=m.k_group;n=m.d_state;r=m.dt_rank
   expected=b*h*w*(3*c*d+9*d+g*d*(2*r+2*n)+g*d*(9*n+1))
   spec={'d_inner':d,'directions':g,'d_state':n,'dt_rank':r}
  macs=sum(ops.values())+extra
  assert macs==expected,(task,arm,name,macs,expected,ops)
  rows.append({'name':name,'shape':shapes[name],'parameters':sum(p.numel() for p in m.parameters()),'projection_generation_conv_macs':sum(ops.values()),'core_propagation_macs':extra,'module_macs':macs,'by_operator':ops,'settings':spec})
 whole=sum(int(round(v)) for k,v in by[''].items() if k in legacy.KEEP_OPERATORS)+sum(r['core_propagation_macs'] for r in rows)
 mixer=sum(r['module_macs'] for r in rows)
 records.append({'task':task,'arm':arm,'config':str(cfg),'config_sha256':legacy.sha(cfg),'whole_model_macs':whole,'mixer_sum_macs':mixer,'carrier_macs':whole-mixer,'mixer_parameters':sum(r['parameters'] for r in rows),'whole_parameters':sum(p.numel() for p in model.parameters()),'slots':rows})
 (OUT/'partial.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',task,arm,'whole_G',whole/1e9,'mixers_G',mixer/1e9,flush=True)
 del a,model,x
import gc;gc.collect()
# Independent executed-operator check of ours for all actual channel widths.
probe=load('module_probe',ROOT/'diagnostics/token_mixer_method_20260912/profile_module_macs.py')
module=probe.load_module()
widths=sorted({r['shape'][1] for rec in records for r in rec['slots']})
checks=[probe.measure(module,c,8,8,5) for c in widths]
comparisons=[]
for task in ['ohaze','gopro']:
 o=next(r for r in records if r['task']==task and r['arm']=='ours');s=next(r for r in records if r['task']==task and r['arm']=='ss2d')
 assert o['carrier_macs']==s['carrier_macs']
 assert [r['shape'] for r in o['slots']]==[r['shape'] for r in s['slots']]
 comparisons.append({'task':task,'mixer_reduction_percent':100*(1-o['mixer_sum_macs']/s['mixer_sum_macs']),'whole_reduction_percent':100*(1-o['whole_model_macs']/s['whole_model_macs']),'core_reduction_percent':100*(1-sum(r['core_propagation_macs'] for r in o['slots'])/sum(r['core_propagation_macs'] for r in s['slots']))})
assert not torch.cuda.is_initialized()
sourcefiles=[Path(__file__),Path(probe.SOURCE),ROOT/'tasks/restoration/runtime/clpo2027/vmamba_ss2d_adapter.py',ROOT/'tasks/restoration/runtime/third_party/VMamba/classification/models/vmamba.py',ROOT/'tasks/restoration/runtime/third_party/VMamba/classification/models/csms6s.py',ROOT/'tools/profile_model.py']
report={'status':'PASS','torch':torch.__version__,'cuda_initialized':False,'input':[1,3,256,256],'records':records,'comparisons':comparisons,'independent_ours_operator_checks':checks,'source_sha256':{str(p):legacy.sha(p) for p in sourcefiles},'rounding_note':'SS2D projection einsums are counted exactly from traced tensor dimensions, avoiding numpy.einsum_path three-significant-digit rounding in the legacy fvcore handler.', 'convention':'Paper MAC convention: conv/linear/matmul/einsum plus explicit propagation. Excludes normalization, activations, pooling, resizing, gates, readout elementwise operations, and layout operations. SS2D core uses official prefixsum estimate; CPU count-only core preserves projection shapes, not numerical output. No GPU timing or inference quality measured.','module_scope':'Entire AttBlock.attn replacement, including its input/output projection and coefficient generation; excludes carrier FFN and other backbone operations.','formulas_per_spatial_position':{'ours_module':'2*C^2+75*C+18*K*C (K=5)','ss2d_module':'3*C*D+9*D+4*D*(2*r+2*N)+4*D*(9*N+1); D=int(.75*C), r=ceil(C/16), N=16','ours_core':'18*K*C','ss2d_core':'4*D*(9*N+1)'},'haze4k':'Same dehazing model structure as O-HAZE; same counts at the same input shape.'}
(OUT/'RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(comparisons,ensure_ascii=False),flush=True)
