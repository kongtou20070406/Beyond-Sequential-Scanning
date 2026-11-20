"""Train the context-generator CSPN with the recorded paired sample order."""
from pathlib import Path
import contextlib
import copy
import hashlib
import json
import random
import time

import numpy as np
import torch

from supplement.runtime import ROOT, activate, digest


@contextlib.contextmanager
def sample_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def prepare_cache(data, cache):
    from basicsr.utils import imfrombytes
    cache.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in (data / 'train/hazy').glob('*.jpg'))
    if [int(Path(n).stem) for n in names] != list(range(1, 41)):
        raise ValueError('Use the 40 O-HAZE training images with IDs 1 to 40.')
    rows = []
    for name in names:
        record = {'image':name}
        for role, folder in [('input','hazy'),('target','gt')]:
            path = data/'train'/folder/name
            output = cache/(Path(name).stem+'_'+role+'.npy')
            source_hash = digest(path)
            receipt = output.with_suffix('.sha256')
            if not output.exists() or not receipt.exists() or receipt.read_text() != source_hash:
                array = imfrombytes(path.read_bytes(), float32=False)
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                    raise ValueError('The training cache requires uint8 BGR images.')
                np.save(output,array)
                receipt.write_text(source_hash,encoding='utf-8')
            record[role] = str(output)
            record[role+'_source_sha256'] = source_hash
        rows.append(record)
    return rows


class KeyedDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.arrays = {}

    def __len__(self):
        return 40

    def __getitem__(self, key):
        from basicsr.data.transforms import paired_random_crop, augment
        from basicsr.utils import img2tensor
        from torchvision.transforms import functional as TF
        index, seed = key
        row = self.rows[index]
        for role in ['input','target']:
            if (index,role) not in self.arrays:
                self.arrays[index,role] = np.load(row[role],mmap_mode='r')
        source, target = self.arrays[index,'input'], self.arrays[index,'target']
        assert source.shape == target.shape
        with sample_rng(seed):
            gt, lq = paired_random_crop(target,source,256,1,row['image'])
            gt, lq = gt.astype(np.float32)/255., lq.astype(np.float32)/255.
            gt, lq = augment([gt,lq],True,True)
            gt, lq = img2tensor([gt,lq],bgr2rgb=True,float32=True)
            if random.randint(0,2) == 1:
                lq, gt = TF.adjust_gamma(lq,1), TF.adjust_gamma(gt,1)
            if random.randint(0,2) == 1:
                factor = 1+(0.2-0.4*np.random.rand())
                lq, gt = TF.adjust_saturation(lq,factor), TF.adjust_saturation(gt,factor)
        return {'lq':lq,'gt':gt,'image':row['image'],'sample_index':index,'augmentation_seed':seed}


class KeyedBatchSampler:
    def __init__(self,start=0,stop=55800):
        self.start,self.stop,self.orders=start,stop,{}

    def keys(self,update):
        keys=[]
        for sample in range(2*update,2*update+2):
            epoch,position=divmod(sample,8000)
            if epoch not in self.orders:
                rng=torch.Generator().manual_seed(epoch)
                self.orders[epoch]=(torch.randperm(8000,generator=rng)%40).tolist()
            seed=int.from_bytes(hashlib.sha256(f'OHAZE:42:{epoch}:{position}'.encode()).digest()[:4],'little')
            keys.append((self.orders[epoch][position],seed))
        return keys

    def __iter__(self):
        for update in range(self.start,self.stop):
            yield self.keys(update)

    def __len__(self):
        return self.stop-self.start


def worker_init(worker):
    torch.set_num_threads(1)


def train(args,row):
    import yaml
    from supplement.run import save
    activate(row,args.output)
    from basicsr.models.image_restoration_model import ImageRestorationModel
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Use CUDA for the recorded FP32 training recipe.')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch._dynamo.config.recompile_limit=32
    if (args.output/'TRAIN_COMPLETE.json').exists():
        raise FileExistsError('Use a new directory for a new training run.')
    resume_path=args.output/'resume.pt'
    if (args.output/'INIT.json').exists() and not (args.resume and resume_path.exists()):
        raise FileExistsError('Use --resume or a new output directory.')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config=yaml.safe_load((ROOT/row['config']).read_text())
    assert config['manual_seed']==row['seed']==42 and config['train']['total_iter']==row['updates']==55800
    config.update(is_train=True,dist=False,rank=0,world_size=1)
    model=ImageRestorationModel(copy.deepcopy(config))
    initial=ROOT/row['initialization_checkpoint']
    expected=json.loads((ROOT/'weights/checksums.json').read_text())['files'][row['initialization_checkpoint']]
    if digest(initial)!=expected['sha256']:
        raise ValueError('The initial weight identity changed.')
    model.net_g.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['params'],strict=True)
    model.net_g.set_backend('compiled')
    samples=prepare_cache(args.data_root,args.output/'training_cache')
    identity={'id':row['id'],'seed':42,'updates':55800,'precision':'FP32_TF32_DISABLED',
              'initial_checkpoint_sha256':digest(initial),'config_sha256':digest(ROOT/row['config']),
              'sample_sources':[{k:v for k,v in s.items() if k not in ('input','target')} for s in samples],
              'torch':torch.__version__,'cuda':torch.version.cuda}
    current=0
    prior_seconds=0.
    if args.resume:
        state=torch.load(resume_path,map_location='cpu',weights_only=False)
        assert state['identity']==identity
        model.net_g.load_state_dict(state['params'],strict=True)
        for optimizer,saved in zip(model.optimizers,state['optimizers'],strict=True):
            optimizer.load_state_dict(saved)
        for scheduler,saved in zip(model.schedulers,state['schedulers'],strict=True):
            scheduler.load_state_dict(saved)
        current=state['iter']
        prior_seconds=state['wall_seconds']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        save(args.output/'INIT.json',identity)
    def finite_loss(module,inputs,value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('The loss is not finite.')
    model.cri_pix.register_forward_hook(finite_loss)
    model.cri_fft.register_forward_hook(finite_loss)
    def finite_grad(optimizer,a,kw):
        gradients=[p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if not bool(torch.isfinite(torch.stack(torch._foreach_norm(gradients))).all()):
            raise FloatingPointError('A gradient is not finite.')
    model.optimizer_g.register_step_pre_hook(finite_grad)
    loader=torch.utils.data.DataLoader(KeyedDataset(samples),batch_sampler=KeyedBatchSampler(current,55800),
        num_workers=8,pin_memory=True,worker_init_fn=worker_init,persistent_workers=True,prefetch_factor=2,
        generator=torch.Generator().manual_seed(42))
    started=time.perf_counter()
    def checkpoint():
        data={'identity':identity,'iter':current,'wall_seconds':prior_seconds+time.perf_counter()-started,
              'params':{k:v.detach().cpu() for k,v in model.net_g.state_dict().items()},
              'optimizers':[o.state_dict() for o in model.optimizers],
              'schedulers':[s.state_dict() for s in model.schedulers],
              'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
              'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        temp=resume_path.with_suffix('.tmp')
        torch.save(data,temp)
        temp.replace(resume_path)
    for batch in loader:
        if (args.output/'STOP').exists():
            checkpoint()
            return {'status':'PAUSED','iter':current}
        model.update_learning_rate(current+1,warmup_iter=-1)
        model.feed_data(batch,is_val=False)
        model.optimize_parameters(current+1,None)
        current+=1
        if current%200==0 or current==55800:
            save(args.output/'PROGRESS.json',{'iter':current,'total':55800,'losses':{k:float(v) for k,v in model.get_current_log().items()}})
        if current%5000==0 or current==55800:
            checkpoint()
    models=args.output/'models'
    models.mkdir(exist_ok=True)
    endpoint=models/'net_g_55800.pth'
    torch.save({'params':{k:v.detach().cpu() for k,v in model.net_g.state_dict().items()}},endpoint)
    result={'status':'COMPLETE','iter':current,'checkpoint_sha256':digest(endpoint),**identity}
    save(args.output/'TRAIN_COMPLETE.json',result)
    return result
