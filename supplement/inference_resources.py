"""Observe existing inference calls; never repeats images or runs a warm-up forward."""
import collections,json,math,os,statistics,time
from pathlib import Path
import torch

def read_macs(arm):
 p=Path(__file__).resolve().parent/'INFERENCE_MACS.json'
 data=json.loads(p.read_text('utf-8'))
 return data['records'][{'ours':'full'}.get(arm,arm)]

class ResourceMeter:
 def __init__(self,model):
  self.model=model;self.seen=0;self.shapes=[]
  self.hook=model.register_forward_pre_hook(self._capture_shape)
 def _capture_shape(self,module,args):self.shapes.append(list(args[0].shape))
 def begin_image(self):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  self.started=time.perf_counter();self.shapes=[]
 def begin_forward(self):
  torch.cuda.synchronize();self.forward_started=time.perf_counter()
 def end_forward(self):
  torch.cuda.synchronize();self.forward_seconds=time.perf_counter()-self.forward_started
  self.forward_peak=torch.cuda.max_memory_allocated()/2**20
 def finish_image(self):
  torch.cuda.synchronize();self.seen+=1
  return {'forward_wall_seconds':self.forward_seconds,'image_load_transfer_forward_metrics_seconds':time.perf_counter()-self.started,
   'forward_peak_allocated_mib':self.forward_peak,'evaluation_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
   'evaluation_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'model_call_shapes':self.shapes,
   'model_call_count':len(self.shapes),'first_image_of_process':self.seen==1,
   'gpu_name':torch.cuda.get_device_name(),'physical_gpu_visibility':os.environ.get('CUDA_VISIBLE_DEVICES','default'),
   'torch':torch.__version__,'cuda':torch.version.cuda,'precision':'FP32_TF32_DISABLED'}
 def close(self):self.hook.remove()

def summarize_resources(rows,profile):
 records=[r['inference_resources'] for r in rows if 'inference_resources' in r]
 if not records:return {'recorded_images':0,'macs_reference':profile,'note':'No new instrumented inference; preserve available historical records instead of rerunning.'}
 timings=[r['forward_wall_seconds'] for r in records]
 warm=[r['forward_wall_seconds'] for r in records if not r['first_image_of_process']]
 for r in records:
  assert all(math.isfinite(r[k]) and r[k]>=0 for k in ['forward_wall_seconds','forward_peak_allocated_mib','evaluation_peak_allocated_mib','evaluation_peak_reserved_mib'])
 shapes=collections.Counter(tuple(shape) for r in records for shape in r['model_call_shapes'])
 return {'recorded_images':len(records),'total_result_images':len(rows),'macs_reference':profile,
  'forward_wall_seconds_sum':sum(timings),'forward_wall_seconds_mean':statistics.mean(timings),'forward_wall_seconds_median':statistics.median(timings),
  'forward_wall_seconds_p90':sorted(timings)[max(0,math.ceil(.9*len(timings))-1)],
  'excluding_first_process_image_mean_seconds':statistics.mean(warm) if warm else None,
  'first_process_image_forward_seconds':[r['forward_wall_seconds'] for r in records if r['first_image_of_process']],
  'forward_peak_allocated_mib':max(r['forward_peak_allocated_mib'] for r in records),
  'evaluation_peak_allocated_mib':max(r['evaluation_peak_allocated_mib'] for r in records),
  'evaluation_peak_reserved_mib':max(r['evaluation_peak_reserved_mib'] for r in records),
  'model_call_shapes_and_counts':[{'shape':list(k),'calls':v} for k,v in sorted(shapes.items())],
  'gpu_names':sorted({r['gpu_name'] for r in records}),
  'physical_gpu_visibilities':sorted({r['physical_gpu_visibility'] for r in records}),
  'timing_scope':'CUDA-synchronized wall time around existing forward including tile assembly/clamp for tiled evaluation; excludes image decoding, H2D and quality metrics. Dataset evaluation wall time remains separate. No extra warmup/repeated images. Shared host contention and first-call overhead retained.',
  'memory_scope':'PyTorch process allocated/reserved peaks in MiB; forward includes resident model and input/target tensors. Evaluation peak also includes metric workspace. Not whole-device NVML usage; reserved may include allocator cache from earlier calls.',
  'macs_scope':'Reference batch1 RGB256x256 model MACs using recorded convention; per-image/tile call shapes also recorded. Do not treat reference MACs as a whole tiled evaluation count or measured hardware work.'}
