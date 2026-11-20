from __future__ import annotations
import hashlib,json,sys,types,os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RELEASE=ROOT
RUNTIME=RELEASE/'tasks/restoration/runtime'
OUT=ROOT/'diagnostics/gain_state_trajectory_20260916'
OHAZE_ROOT = Path(__file__).resolve().parents[2]
GOPRO_ROOT=Path(os.environ.get("NSI_DATA_ROOT", str(ROOT/"data")))/"gopro"
if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))

PAIRS={
 'O-HAZE':{
  'input':OHAZE_ROOT/'test/hazy/42.jpg',
  'crop':'long-side-512',
  'full':{
   'config':RELEASE/'configs/restoration/ohaze/O-HAZE_VIVNet_RadialProfiled_K5_P256.yml',
   'checkpoint':RELEASE/'weights/ohaze/ours/seed42/last.pth',
   'overrides':{},
  },
  'no_gain':{
   'config':RELEASE/'configs/restoration/ohaze/O-HAZE_VIVNet_RadialProfiled_K5_P256.yml',
   'checkpoint':RELEASE/'weights/ohaze/ablations/no_row_mass_projection/seed42/last.pth',
   'overrides':{'type':'VIVNetRadialProfiledAblation','ablation':'no_projection'},
  },
 },
 'GoPro':{
  'input':GOPRO_ROOT/'test/blur/GOPR0854_11_00__000050.png',
  'crop':'center-256',
  'full':{
   'config':RELEASE/'configs/restoration/gopro/ours.yml',
   'checkpoint':RELEASE/'weights/gopro/ours/seed42/last.pth',
   'overrides':{},
  },
  'no_gain':{
   'config':RELEASE/'configs/supplement/gopro-no-normalization-seed42.yml',
   'checkpoint':RELEASE/'weights/supplement/gopro-no-normalization-seed42/last.pth',
   'overrides':{},
  },
 },
}

def sha256(path:Path)->str:
 return hashlib.sha256(path.read_bytes()).hexdigest()

def image_tensor(path:Path)->torch.Tensor:
 with Image.open(path) as image:value=np.asarray(image.convert('RGB'),dtype=np.float32)/255.
 return torch.from_numpy(value).permute(2,0,1).unsqueeze(0)

def prepare(task:str,path:Path)->torch.Tensor:
 x=image_tensor(path)
 if task=='O-HAZE':
  h,w=x.shape[-2:];scale=512/max(h,w);size=(round(h*scale),round(w*scale))
  x=F.interpolate(x,size=size,mode='bicubic',align_corners=False,antialias=True)
  h,w=x.shape[-2:];x=F.pad(x,(0,(-w)%32,0,(-h)%32),mode='reflect')
 else:
  h,w=x.shape[-2:];top=max(0,(h-256)//2);left=max(0,(w-256)//2);x=x[...,top:top+256,left:left+256]
 return x

def build(spec:dict,device:torch.device)->torch.nn.Module:
 from basicsr.models.archs import define_network
 cfg=yaml.safe_load(spec['config'].read_text(encoding='utf-8'))
 network=dict(cfg['network_g']);network['K']=5;network.update(spec['overrides'])
 model=define_network(network)
 saved=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
 state=saved.get('params_ema',saved.get('params',saved))
 model.load_state_dict(state,strict=True)
 return model.to(device=device,dtype=torch.float32).eval()

def operators(model):
 required=('_drive_with_attention','_selective_controls','depth_weights','_interface_readout')
 found=[m for m in model.modules() if (all(hasattr(m,x) for x in required) and (hasattr(m,'_state_depth_step') or hasattr(m,'_step'))) or all(hasattr(m,x) for x in ('input_fields','scale_pairs','Bin','Cout','depth_weights'))]
 if not found:raise RuntimeError('no compatible state propagation operators')
 if any(int(m.K)!=5 for m in found):raise RuntimeError('all operators must use five propagation steps')
 return found

class Recorder:
 def __init__(self):
  self.state_sq=defaultdict(float);self.increment_sq=defaultdict(float);self.count=defaultdict(int)
  self.state_samples=defaultdict(list);self.increment_samples=defaultdict(list)
  self.state_max=defaultdict(float);self.increment_max=defaultdict(float);self.calls=0
 def add(self,step,sr,si,dr,di):
  state_mag=torch.sqrt(sr.detach().float().square()+si.detach().float().square())
  inc_mag=torch.sqrt(dr.detach().float().square()+di.detach().float().square())
  self.state_sq[step]+=float((sr.detach().float().square()+si.detach().float().square()).sum())
  self.increment_sq[step]+=float((dr.detach().float().square()+di.detach().float().square()).sum())
  self.count[step]+=sr.numel()
  self.state_max[step]=max(self.state_max[step],float(state_mag.max()))
  self.increment_max[step]=max(self.increment_max[step],float(inc_mag.max()))
  for value,target in ((state_mag,self.state_samples),(inc_mag,self.increment_samples)):
   flat=value.flatten();stride=max(1,flat.numel()//4096);target[step].append(flat[::stride].cpu().numpy())
 def finish(self):self.calls+=1
 def summary(self):
  rows=[]
  for step in range(1,6):
   s=np.concatenate(self.state_samples[step]);d=np.concatenate(self.increment_samples[step])
   rows.append({
    'step':step,
    'state_rms':(self.state_sq[step]/self.count[step])**.5,
    'state_p95':float(np.quantile(s,.95)),
    'state_p99':float(np.quantile(s,.99)),
    'state_max':self.state_max[step],
    'increment_rms':(self.increment_sq[step]/self.count[step])**.5,
    'increment_p95':float(np.quantile(d,.95)),
    'increment_p99':float(np.quantile(d,.99)),
    'increment_max':self.increment_max[step],
   })
  return rows

def coefficients(op,x,packed):
 hidden=op._coefficient_hidden(x,packed)
 if hasattr(op,'_ablation_coefficients'):return op._ablation_coefficients(hidden)
 if hasattr(op,'dh9_compiler'):return op._q9_coefficients_from_raw(op.dh9_compiler(packed))
 if hasattr(op,'_lightweight_coefficients'):return op._lightweight_coefficients(hidden)
 raise RuntimeError(f'unsupported coefficient path: {type(op).__name__}')

def diagnostic_forward(op,x,recorder):
 if hasattr(op,'input_fields'):
  from clpo2027.real_pair_propagation import _state_step
  feature=op.Bin(x)
  kernel,write,read,strength=op.input_fields(feature)
  drive=op.scale_pairs(feature,write)
  state=torch.zeros_like(feature);readout=torch.zeros_like(feature)
  for step,weight in enumerate(op.depth_weights(feature),1):
   next_state=_state_step(state,drive,kernel,strength,op.group_size)
   delta=next_state-state
   b,c,h,w=state.shape
   pairs=next_state.reshape(b,c//2,2,h,w)
   increments=delta.reshape(b,c//2,2,h,w)
   recorder.add(step,pairs[:,:,0],pairs[:,:,1],increments[:,:,0],increments[:,:,1])
   readout=readout+weight*delta
   state=next_state
  recorder.finish()
  return op.Cout(op.scale_pairs(readout,read))
 packed,fr,fi,attention=op._drive_with_attention(x)
 ar,ai=coefficients(op,x,packed)
 hidden=op._coefficient_hidden(x,packed)
 write,read,strength=op._selective_controls(hidden)
 drive_r,drive_i=write*fr,write*fi
 state_r,state_i=torch.zeros_like(fr),torch.zeros_like(fi)
 readout_r,readout_i=torch.zeros_like(fr),torch.zeros_like(fi)
 for step,weight in enumerate(op.depth_weights(fr).unbind(),1):
  if hasattr(op,'_ablation_coefficients'):next_r,next_i=op._step(state_r,state_i,drive_r,drive_i,strength,ar,ai)
  elif hasattr(op,'_state_depth_step'):next_r,next_i=op._state_depth_step(state_r,state_i,drive_r,drive_i,strength,ar,ai)
  else:next_r,next_i=op._step(state_r,state_i,drive_r,drive_i,strength,ar,ai)
  delta_r,delta_i=next_r-state_r,next_i-state_i
  recorder.add(step,next_r,next_i,delta_r,delta_i)
  readout_r=readout_r+weight*delta_r;readout_i=readout_i+weight*delta_i
  state_r,state_i=next_r,next_i
 recorder.finish()
 return op._interface_readout(torch.cat((read*readout_r,read*readout_i),dim=1),attention)

class Instrumented:
 def __init__(self,model,recorder):self.model=model;self.recorder=recorder;self.original=[]
 def __enter__(self):
  for op in operators(self.model):
   self.original.append((op,op.forward))
   def forward(instance,x,rec=self.recorder):return diagnostic_forward(instance,x,rec)
   op.forward=types.MethodType(forward,op)
  return self
 def __exit__(self,*_):
  for op,forward in self.original:op.forward=forward

def tensor_output(value):return value[-1] if isinstance(value,list) else value

def run_one(task,label,spec,device,x):
 model=build(spec,device)
 for op in operators(model):op._fp32_inference_enabled=False
 with torch.inference_mode():
  normal=tensor_output(model(x))
  recorder=Recorder()
  with Instrumented(model,recorder):observed=tensor_output(model(x))
  error=float((normal-observed).abs().max())
 if error>1e-5:raise AssertionError((task,label,'instrumentation changed output',error))
 result={
  'task':task,'condition':label,'input':str(PAIRS[task]['input']),'input_shape':list(x.shape),
  'config':str(spec['config']),'checkpoint':str(spec['checkpoint']),'checkpoint_sha256':sha256(spec['checkpoint']),
  'operator_calls':recorder.calls,'forward_max_abs_error':error,'trajectory':recorder.summary(),
 }
 del model,normal,observed
 torch.cuda.empty_cache()
 return result

def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device('cuda:0')
 samples=[]
 for task,pair in PAIRS.items():
  x=prepare(task,pair['input']).to(device)
  for label in ('full','no_gain'):samples.append(run_one(task,label,pair[label],device,x))
  del x;torch.cuda.empty_cache()
 comparisons=[]
 for task in PAIRS:
  full=next(x for x in samples if x['task']==task and x['condition']=='full')
  no=next(x for x in samples if x['task']==task and x['condition']=='no_gain')
  rows=[]
  for a,b in zip(full['trajectory'],no['trajectory']):
   rows.append({'step':a['step'],**{f'{key}_ratio_no_gain_over_full':b[key]/a[key] for key in ('state_rms','state_p95','state_p99','increment_rms','increment_p95','increment_p99')}})
  comparisons.append({'task':task,'ratios':rows})
 payload={
  'schema':'nsi.full_vs_no_gain_state_trajectory.v1',
  'evidence_type':'STATIC_CHECKPOINT_DIAGNOSTIC',
  'protocol':'Same fixed input and spatial preprocessing within each task; no parameter updates; aggregate over every NSI block; exact RMS and maxima plus uniformly subsampled P95/P99.',
  'samples':samples,'comparisons':comparisons,
  'limits':['Fixed-checkpoint internal behavior is descriptive and does not replace independently trained quality comparisons.','One preselected input per task; no dataset-level uncertainty claim.'],
 }
 (OUT/'RESULTS.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(payload,ensure_ascii=False),flush=True)
if __name__=='__main__':main()