"""Verify or measure the paper's default FP32 inference implementation."""
from pathlib import Path
import argparse
import collections
import copy
import hashlib
import json
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'benchmark', 'evaluate'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Run a smaller, labelled numerical smoke check.')
    parser.add_argument('--start-with', choices=['ours', 'ss2d'], default='ours')
    parser.add_argument('--inference-mode', choices=['native', 'tiled'], default='native')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and (args.action != 'verify' or args.limit < 1):
        parser.error('--limit requires verify and a positive integer')
    return args


def paired_images(root, split):
    from clpo2027.data import ImagePair
    if (root / split / 'blur').is_dir():
        rows = []
        for path in sorted((root / split / 'blur').glob('*.png')):
            sequence, frame = path.name.split('__', 1)
            rows.append(ImagePair(sequence + '/' + frame, path, root / split / 'sharp' / path.name))
    else:
        from clpo2027.gopro_data import load_gopro_pairs
        rows = load_gopro_pairs(root, split)
    if len(rows) != {'train': 2103, 'test': 1111}[split] or not all(r.gt_path.is_file() for r in rows):
        raise ValueError('Use the complete paired GoPro split.')
    return rows


def main():
    args = arguments()
    methods = ['ours', 'ss2d'] if args.action == 'benchmark' else ['ours']
    expected = json.loads((ROOT / 'weights/checksums.json').read_text())['files']
    identity = {}
    for method in methods:
        path = ROOT / f'weights/gopro/{method}/seed42/last.pth'
        relative = path.relative_to(ROOT).as_posix()
        if digest(path) != expected[relative]['sha256']:
            raise ValueError('Checkpoint checksum mismatch: ' + relative)
        identity[method] = {'checkpoint': relative, 'checkpoint_sha256': digest(path),
                            'config': f'configs/restoration/gopro/{method}.yml'}
    assert identity['ours']['checkpoint_sha256'] == '780f48f7d7fef8a4c7b2fa51cd5b495de544905d3c5896af40e4e5d7a864686e'
    description = {'action': args.action, 'inference_mode': args.inference_mode, 'selected': identity, 'training': False,
                   'backend': 'joint_heads_direct_rollout_fp32', 'precision': 'FP32_TF32_DISABLED',
                   'scope': 'smoke' if args.limit else '55-image panel' if args.action == 'verify'
                   else 'one fixed image' if args.action == 'benchmark' else '1111-image official test'}
    if args.dry_run:
        print(json.dumps(description, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'tasks/restoration/runtime'))
    import torch
    import yaml
    from basicsr.models.archs import define_network
    from clpo2027.data import read_tensor
    from clpo2027.fp32_inference import prepare_fp32_inference
    from clpo2027.gopro_eval import halo_tiled_inference, rgb_psnr, rgb_ssim
    from tasks.restoration.evaluate_gopro import TensorOutput
    if not torch.cuda.is_available():
        raise RuntimeError('Use CUDA for the FP32 inference implementation.')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    models = {}
    for method, record in identity.items():
        config = yaml.safe_load((ROOT / record['config']).read_text())
        model = define_network(dict(config['network_g']))
        saved = torch.load(ROOT / record['checkpoint'], map_location='cpu', weights_only=True)
        model.load_state_dict(saved.get('params_ema', saved.get('params', saved)), strict=True)
        models[method] = TensorOutput(model).cuda().float().eval().requires_grad_(False)
    reference = copy.deepcopy(models['ours']) if args.action == 'verify' else None
    if reference is not None:
        prepare_fp32_inference(reference, enabled=False)
    description['inference'] = prepare_fp32_inference(models['ours'])
    description.update(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                       cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
    save(args.output / 'IDENTITY.json', description)
    def full(model, image):
        if args.inference_mode == 'native':
            return model(image)
        return halo_tiled_inference(model, image, core_tile=384, halo=64, overlap=64)
    test = paired_images(args.data_root, 'test')
    with torch.inference_mode():
        if args.action == 'benchmark':
            image = read_tensor(test[0].hazy_path)[None].cuda()
            crop = image[..., :512, :512].contiguous()
            for model in models.values():
                for _ in range(3):
                    model(crop)
                full(model, image)
            rows = []
            initial = ['ours', 'ss2d'] if args.start_with == 'ours' else ['ss2d', 'ours']
            for scope, x, fn in [('crop', crop, lambda m,x: m(x)), (args.inference_mode, image, full)]:
                for repeat in range(12):
                    order = initial if repeat % 2 == 0 else initial[::-1]
                    for position, method in enumerate(order):
                        torch.cuda.synchronize()
                        allocated = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start = time.perf_counter()
                        begin.record()
                        output = fn(models[method], x)
                        end.record()
                        torch.cuda.synchronize()
                        wall_ms = 1000 * (time.perf_counter()-start)
                        rows.append({'scope':scope, 'repeat':repeat, 'position':position, 'method':method,
                                     'cuda_ms':begin.elapsed_time(end), 'wall_ms':wall_ms,
                                     'peak_delta_mib':(torch.cuda.max_memory_allocated()-allocated)/2**20})
                        del output
            summary = {scope: {method: {key:statistics.median(r[key] for r in rows if r['scope']==scope and r['method']==method)
                       for key in ['cuda_ms','wall_ms','peak_delta_mib']} for method in methods} for scope in ['crop',args.inference_mode]}
            save(args.output / 'FINAL.json', {'status':'COMPLETE', 'image_id':test[0].image_id,
                                           'summary':summary, 'rows':rows, **description})
        else:
            selected = test
            if args.action == 'verify':
                selected = []
                for split in ['train','test']:
                    groups = collections.defaultdict(list)
                    for row in paired_images(args.data_root, split):
                        groups[row.image_id.split('/')[0]].append(row)
                    for rows in groups.values():
                        selected.extend([rows[len(rows)//2]] if split == 'train' else [rows[0],rows[len(rows)//2],rows[-1]])
                assert len(selected) == 55
                selected = selected[:args.limit] if args.limit else selected
            rows = []
            with (args.output / 'per_image.jsonl').open('w', encoding='utf-8') as stream:
                for pair in selected:
                    image, target = [read_tensor(p)[None].cuda() for p in [pair.hazy_path,pair.gt_path]]
                    value = full(models['ours'], image)
                    result = {'image_id':pair.image_id, 'psnr_db':rgb_psnr(value.clamp(0,1),target),
                              'ssim':rgb_ssim(value.clamp(0,1),target)}
                    if reference is not None:
                        original = full(reference, image)
                        result['max_abs_error'] = (value-original).abs().max().item()
                        result['elements_pass'] = bool(torch.allclose(value, original, atol=1e-5, rtol=1e-4))
                        result['psnr_abs_difference'] = abs(result['psnr_db']-rgb_psnr(original.clamp(0,1),target))
                        result['ssim_abs_difference'] = abs(result['ssim']-rgb_ssim(original.clamp(0,1),target))
                        if not result['elements_pass'] or result['psnr_abs_difference'] > 1e-4 or result['ssim_abs_difference'] > 1e-5:
                            raise ValueError('Numerical comparison failed: '+pair.image_id)
                        del original
                    rows.append(result)
                    stream.write(json.dumps(result)+'\n')
                    stream.flush()
                    del image,target,value
            save(args.output / 'FINAL.json', {'status':'COMPLETE', 'count':len(rows),
                 'psnr_db':statistics.mean(r['psnr_db'] for r in rows), 'ssim':statistics.mean(r['ssim'] for r in rows),
                 'per_image_sha256':digest(args.output/'per_image.jsonl'), **description})


if __name__ == '__main__':
    main()
