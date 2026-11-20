"""Train or evaluate one frozen Haze4K configuration."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
Q=Path(__file__).resolve().parent
ROOT=Q.parents[2]
PROTOCOL=dict(id='direct_rgb_ssim_v1',color_space='RGB',image_pre_pooling=False,window_size=11,window_sigma=1.5,data_range=255,k1=.01,k2=.03,backend='NumPy/OpenCV valid Gaussian window',numeric_precision='float64 metric values')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');t.replace(p)

def prepare_cache(data_root,cache,splits):
 import numpy as np
 from PIL import Image
 manifest=read(Q/'DATA_MANIFEST.json');cache.mkdir(parents=True,exist_ok=True)
 for split in splits:
  path=cache/f'{split}.bin';receipt=cache/f'{split}.json'
  if path.exists() and receipt.exists():
   old=read(receipt)
   if old['manifest_sha256']==digest(Q/'DATA_MANIFEST.json') and old['cache_sha256']==digest(path):continue
   raise ValueError('Cache identity changed: '+str(path))
  temp=path.with_suffix('.tmp')
  with temp.open('wb') as out:
   for row in manifest['splits'][split]:
    for role in ['input','target']:
     src=row['source'][role];relative=Path(*Path(src['archive_path']).parts[1:]);image_path=data_root/relative
     if digest(image_path)!=src['sha256']:raise ValueError('Dataset file differs from the frozen split: '+str(relative))
     with Image.open(image_path) as im:arr=np.asarray(im.convert('RGB'),dtype=np.uint8)
     spec=row['arrays'][role]
     assert list(arr.shape)==spec['shape'] and out.tell()==spec['offset'] and arr.nbytes==spec['bytes']
     out.write(arr.tobytes())
  temp.replace(path)
  save(receipt,dict(manifest_sha256=digest(Q/'DATA_MANIFEST.json'),cache_sha256=digest(path),images=len(manifest['splits'][split])))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--id',required=True);p.add_argument('--action',choices=['train','eval'],required=True)
 p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();arm=a.id.removeprefix('haze4k-').removesuffix('-seed42')
 config=read(ROOT/'configs/haze4k.json')
 if a.dry_run:
  print(json.dumps(dict(id=a.id,action=a.action,arm=arm,config=config,metric_protocol=PROTOCOL),indent=2));return
 a.output=a.output.resolve();cache=a.output/'cache';os.environ['NSI_HAZE_OUTPUT']=str(a.output);os.environ['NSI_HAZE_CACHE']=str(cache)
 prepare_cache(a.data_root.resolve(),cache,['train'] if a.action=='train' else ['test'])
 if a.action=='train':
  args=[sys.executable,'-X','utf8','-B',str(Q/'train_worker.py'),'--arm',arm]
  if (a.output/'runs'/arm/'RESUME.json').exists():args.append('--resume')
  subprocess.run(args,check=True,cwd=Q)
  import shutil
  dest=a.output/'models/net_g_100000.pth';dest.parent.mkdir(exist_ok=True)
  shutil.copy2(a.output/'runs'/arm/'last.pth',dest);return
 if a.checkpoint is None:raise ValueError('Evaluation requires --checkpoint')
 from common import setup
 torch=setup()
 import numpy as np
 from models import make_model
 from data import HazePairs
 from basicsr.metrics.psnr_ssim import _ssim
 model=make_model(arm).cuda().float().eval().requires_grad_(False)
 checkpoint=a.checkpoint.resolve();cp_sha=digest(checkpoint)
 model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['params'],strict=True)
 if arm=='cspn':model.set_backend('eager')
 ds=HazePairs('test');rows=[];start=time.perf_counter()
 assert len(ds.rows)==1000
 with torch.inference_mode():
  for i in range(1000):
   (raw,gt),name=ds.pair(i);h,w=raw.shape[:2]
   x=torch.from_numpy(np.ascontiguousarray(raw.transpose(2,0,1))).float().unsqueeze(0).cuda()/255
   x=torch.nn.functional.pad(x,(0,(-w)%4,0,(-h)%4),mode='reflect')
   pred=model(x)[:,:,:h,:w].clamp(0,1)
   if not torch.isfinite(pred).all():raise FloatingPointError(name)
   pred=pred[0].permute(1,2,0).cpu().numpy().astype(np.float64);target=np.asarray(gt,dtype=np.float64)/255
   mse=float(np.mean((pred-target)**2))
   row=dict(image=name,psnr=float(-10*np.log10(max(mse,1e-30))),ssim=float(_ssim(pred*255,target*255,255.)))
   rows.append(row)
   if (i+1)%50==0:save(a.output/'EVAL_PROGRESS.json',dict(completed=i+1,total=1000))
 assert digest(checkpoint)==cp_sha
 save(a.output/'results/FINAL.json',dict(status='COMPLETE',id=a.id,count=1000,checkpoint_sha256=cp_sha,metric_protocol=PROTOCOL,
    protocol='1000 native whole images; RGB; reflect pad to multiple of four; crop before metrics; clip prediction to [0,1]; no pre-pooling',
    psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),images=rows,seconds=time.perf_counter()-start))

if __name__=='__main__':main()
