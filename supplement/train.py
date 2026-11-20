"""Train a new fixed endpoint with the recorded loss and optimizer settings."""
from pathlib import Path
import copy
import hashlib
import json
import logging
import math
import os
import random
import time

from supplement.runtime import ROOT, activate, digest


def state_digest(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def train(args, row):
    import numpy as np
    import torch
    import yaml
    from supplement.run import save
    activate(row, args.output)
    from basicsr.models import create_model
    from basicsr.train import create_train_val_dataloader
    from basicsr.utils import get_root_logger, set_random_seed
    from basicsr.utils.options import parse
    from supplement.execution_policy import apply_execution_policy
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('The recorded training protocol requires CUDA.')
    torch.cuda.set_device(args.device)
    if (args.output / 'TRAIN_COMPLETE.json').exists():
        raise FileExistsError('This training run is complete. Use a new output directory.')
    resume_path = args.output / 'resume.pt'
    if (args.output / 'INIT.json').exists() and not (args.resume and resume_path.exists()):
        raise FileExistsError('This directory contains a training attempt. Use --resume or a new output.')
    config = yaml.safe_load((ROOT / row['config']).read_text('utf-8'))
    dataset = row['dataset']
    data = config['datasets']['train']
    data['dataroot_gt'] = str((args.data_root / 'train' / ('gt' if dataset == 'ohaze' else 'sharp')).resolve())
    data['dataroot_lq'] = str((args.data_root / 'train' / ('hazy' if dataset == 'ohaze' else 'blur')).resolve())
    if dataset == 'gopro':
        data['meta_info_file'] = str(ROOT / 'supplement/reference/train_order.txt')
    config['path']['pretrain_network_g'] = None
    config['path']['resume_state'] = None
    resolved = args.output / 'config.yml'
    resolved.write_text(yaml.safe_dump(config, sort_keys=False), 'utf-8')
    os.environ['NSI_RUN_ROOT'] = str(args.output.resolve())
    config = parse(str(resolved), is_train=True)
    config.update(dist=False, rank=0, world_size=1)
    for key in ['experiments_root', 'models', 'training_states', 'log', 'visualization']:
        Path(config['path'][key]).mkdir(parents=True, exist_ok=True)
    assert config['manual_seed'] == row['seed'] and config['train']['total_iter'] == row['updates']
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    set_random_seed(row['seed'])
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO,
                             log_file=str(args.output / 'training.log'))
    loader, sampler, _, _, total = create_train_val_dataloader(config, logger)
    assert len(loader.dataset) == (40 if dataset == 'ohaze' else 2103)
    model = create_model(copy.deepcopy(config))
    initialization_hash = None
    if dataset == 'gopro':
        initial = ROOT / 'supplement/reference/random_initialization_seed42.pt'
        saved = torch.load(initial, map_location='cpu', weights_only=True)
        model.net_g.load_state_dict(saved['params'], strict=True)
        torch.set_rng_state(saved['torch_rng_after_initialization'])
        initialization_hash = digest(initial)
    blocks = apply_execution_policy(model.net_g, row['arm'])
    identity = {'id': row['id'], 'seed': row['seed'], 'updates': total,
                'initial_state_sha256': state_digest(model.net_g.state_dict()),
                'shared_initialization_file_sha256': initialization_hash,
                'published_config_sha256': digest(ROOT / row['config']),
                'runtime_manifest_sha256': digest(ROOT / 'supplement/runtime_manifests' / (row['runtime'] + '.json')),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(), 'activation_checkpoint_blocks': blocks,
                'precision': 'BF16 microbatch 8, effective batch 32' if dataset == 'gopro' else 'FP32 batch 2',
                'resume_limit': 'Worker augmentation RNG restarts when the process resumes.'}
    current = epoch = next_batch = 0
    prior_seconds = 0.0
    if args.resume:
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        for key in ['id', 'published_config_sha256', 'runtime_manifest_sha256']:
            assert saved['identity'][key] == identity[key]
        model.net_g.load_state_dict(saved['params'], strict=True)
        for obj, state in zip(model.optimizers, saved['optimizers'], strict=True):
            obj.load_state_dict(state)
        for obj, state in zip(model.schedulers, saved['schedulers'], strict=True):
            obj.load_state_dict(state)
        current, epoch, next_batch = saved['iter'], saved['epoch'], saved['next_batch']
        prior_seconds = saved['wall_seconds']
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'])
        torch.cuda.set_rng_state_all(saved['rng_cuda'])
    else:
        save(args.output / 'INIT.json', identity)
    step = model.optimizer_g.step
    def guarded_step(*a, **kw):
        gradients = [p.grad for p in model.net_g.parameters() if p.grad is not None]
        if not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
            raise FloatingPointError('The optimizer update was refused because a gradient is not finite.')
        return step(*a, **kw)
    model.optimizer_g.step = guarded_step
    start = time.perf_counter()
    def checkpoint(batch):
        value = {'identity': identity, 'iter': current, 'epoch': epoch, 'next_batch': batch,
                 'wall_seconds': prior_seconds + time.perf_counter() - start,
                 'params': {k: v.detach().cpu() for k, v in model.net_g.state_dict().items()},
                 'optimizers': [o.state_dict() for o in model.optimizers],
                 'schedulers': [s.state_dict() for s in model.schedulers],
                 'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                 'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}
        temp = resume_path.with_suffix('.tmp')
        torch.save(value, temp)
        temp.replace(resume_path)
    torch.cuda.reset_peak_memory_stats()
    while current < total:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < next_batch:
                continue
            if current >= total:
                break
            if (args.output / 'STOP').exists():
                checkpoint(batch_index)
                return {'status': 'PAUSED', 'iter': current}
            model.update_learning_rate(current + 1, warmup_iter=config['train'].get('warmup_iter', -1))
            model.feed_data(batch, is_val=False)
            model.optimize_parameters(current + 1, None)
            current += 1
            if current % 200 == 0 or current == total:
                losses = {k: float(v) for k, v in model.get_current_log().items()}
                assert all(math.isfinite(v) for v in losses.values())
                save(args.output / 'PROGRESS.json', {'iter': current, 'total': total, 'losses': losses})
            if current % 1000 == 0 or current == total:
                checkpoint(batch_index + 1)
        next_batch = 0
        epoch += 1
    endpoint = args.output / 'models' / f'net_g_{total}.pth'
    endpoint.parent.mkdir(exist_ok=True)
    torch.save({'params': model.net_g.state_dict()}, endpoint)
    result = {**identity, 'status': 'COMPLETE', 'iter': current,
              'checkpoint': endpoint.relative_to(args.output).as_posix(), 'checkpoint_sha256': digest(endpoint),
              'wall_seconds': prior_seconds + time.perf_counter() - start,
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
    save(args.output / 'TRAIN_COMPLETE.json', result)
    return result
