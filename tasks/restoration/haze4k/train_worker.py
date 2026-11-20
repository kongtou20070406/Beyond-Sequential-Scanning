from common import *
import argparse,time,datetime,random,math,traceback
import numpy as np

def atomic_torch(torch,path,obj):
 path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def run():
 torch=setup()
 from models import make_model,loss_fn
 from data import loader,HazePairs
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--canary',type=int,default=0);p.add_argument('--resume',action='store_true');a=p.parse_args()
 plan=json.loads((Q/'PLAN.json').read_text('utf-8'));digest=hashlib.sha256((Q/'PLAN.json').read_bytes()).hexdigest()
 for name,h in plan['source_sha256'].items():assert hashlib.sha256((Q/name).read_bytes()).hexdigest()==h,name
 assert hashlib.sha256((Q/'DATA_MANIFEST.json').read_bytes()).hexdigest()==plan['data_manifest_sha256']
 for name,record in plan['cache_files'].items():
  stat=(CACHE/name).stat();assert stat.st_size==record['bytes'] and stat.st_mtime_ns==record['mtime_ns'],name
 for name,h in json.loads((Q/'RUNTIME_MANIFEST.json').read_text()).items():assert hashlib.sha256((RUNTIME/name).read_bytes()).hexdigest()==h,name
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42);np.random.seed(42)
 m=make_model(a.arm).cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.9),weight_decay=.001,fused=(a.arm=='full'))
 total=plan['updates'];schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total,eta_min=1e-7)
 dest=OUT/('pilot' if a.canary else 'runs')/a.arm;dest.mkdir(parents=True,exist_ok=True)
 current=0;prior=0.;deadline=None
 if not a.canary:
  if (dest/'COMPLETE.json').exists():raise FileExistsError('Completed arm must be reused')
  if (dest/'RESUME.json').exists():
   if not a.resume:raise FileExistsError('Use explicit resume for existing arm')
   pointer=json.loads((dest/'RESUME.json').read_text());assert hashlib.sha256(Path(pointer['path']).read_bytes()).hexdigest()==pointer['sha256'];saved=torch.load(pointer['path'],map_location='cpu',weights_only=False)
   assert saved['plan_sha256'] in [digest,*plan.get('compatible_resume_plan_sha256',[])]
   m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler']);current=saved['update'];prior=saved['wall_seconds']
   torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);del saved
  else:
   atomic_torch(torch,dest/'initial.pth',{'params':{k:v.detach().cpu() for k,v in m.state_dict().items()},'seed':42,'plan_sha256':digest})
 save(dest/'START.json',{'status':'PILOT' if a.canary else 'TRAINING','arm':a.arm,'pid':os.getpid(),'start_update':current,'resume_plan_compatibility':plan.get('compatible_resume_plan_sha256',[]),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':digest})
 end=current+a.canary if a.canary else total;iterator=iter(loader(current,end));start=time.perf_counter();recent=[]
 def checkpoint_model():
  path=dest/f'resume_{(current//2000)%2}.pth'
  atomic_torch(torch,path,{'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'update':current,'wall_seconds':prior+time.perf_counter()-start,'plan_sha256':digest,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
  save(dest/'RESUME.json',{'path':str(path),'update':current,'plan_sha256':digest,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 while current<end:
  if (OUT/'STOP').exists():
   if not a.canary:checkpoint_model()
   raise TimeoutError('Frozen budget deadline or STOP reached')
  before=time.perf_counter();batch=next(iterator);x=batch['input'].cuda(non_blocking=True);y=batch['target'].cuda(non_blocking=True)
  opt.zero_grad(set_to_none=True);prediction=m(x);loss=loss_fn(prediction,y)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),.01,error_if_nonfinite=True);opt.step();schedule.step();torch.cuda.synchronize()
  current+=1;recent.append(time.perf_counter()-before);recent=recent[-100:]
  if current<=3 or current%100==0 or current==end:
   row={'arm':a.arm,'update':current,'total':total,'loss':float(loss.detach()),'gradient_norm':float(grad),'lr':opt.param_groups[0]['lr'],'seconds_per_step':sum(recent)/len(recent),'wall_seconds':prior+time.perf_counter()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'images':list(batch['id']),'positions':batch['position'].tolist(),'peak_mib':torch.cuda.max_memory_allocated()/2**20}
   if a.arm=='free_gain':row['gains']=[float(z.log_gain.detach().exp()) for z in m.clpos()]
   save(dest/'PROGRESS.json',row)
   with (dest/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
  if not a.canary and current%2000==0:checkpoint_model()
 if a.canary:
  # Use training images for inference timing. Do not inspect official test quality.
  m.eval();ds=HazePairs();timings=[]
  if a.arm=='cspn':m.set_backend('eager')
  with torch.no_grad():
   for i in [0,100,1000]:
    (raw,_),name=ds.pair(i);h,w=raw.shape[:2];x=torch.from_numpy(np.ascontiguousarray(raw.transpose(2,0,1))).float().unsqueeze(0).cuda()/255
    x=torch.nn.functional.pad(x,(0,(-w)%4,0,(-h)%4),mode='reflect');before=time.perf_counter();pred=m(x);torch.cuda.synchronize();timings.append(time.perf_counter()-before);assert torch.isfinite(pred).all()
  save(dest/'PILOT_COMPLETE.json',{'status':'PASS','steps':a.canary,'median_train_seconds':float(np.median(recent[20:])),'inference_seconds':timings,'plan_sha256':digest,'peak_mib':torch.cuda.max_memory_allocated()/2**20,'parameters':sum(p.numel() for p in m.parameters()),'mean_train_seconds':float(np.mean(recent[20:]))})
 else:
  path=dest/'last.pth';atomic_torch(torch,path,{'params':{k:v.detach().cpu() for k,v in m.state_dict().items()},'update':current,'plan_sha256':digest})
  save(dest/'COMPLETE.json',{'status':'TRAINED','arm':a.arm,'update':current,'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'wall_seconds':prior+time.perf_counter()-start,'plan_sha256':digest})
if __name__=='__main__':
 try:run()
 except Exception:
  print(traceback.format_exc(),flush=True);raise

