from pathlib import Path
import os,sys,json,hashlib
Q=Path(__file__).resolve().parent
OUT=Path(os.environ.get('NSI_HAZE_OUTPUT',str(Q.parents[2]/'runs/haze4k')))
CACHE=Path(os.environ.get('NSI_HAZE_CACHE',str(OUT/'cache')))
RUNTIME=Q/'runtime'
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(OUT/'compiler_cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR',str(OUT/'compiler_cache/triton'))
sys.path.insert(0,str(RUNTIME))
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def setup():
 import torch
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch._dynamo.config.recompile_limit=64
 return torch
