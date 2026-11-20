"""Reproduce one supplementary endpoint or protocol comparison."""
from pathlib import Path
import argparse
import collections
import importlib.util
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supplement.runtime import digest, experiment, network, read
from tasks.restoration.metric_contract import direct_ssim_protocol


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--action', choices=['eval', 'train', 'check', 'profile'], default='eval')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def pairs(root):
    from clpo2027.data import ImagePair
    flat = root / 'test/blur'
    if flat.is_dir():
        output = []
        for path in sorted(flat.glob('*.png')):
            sequence, frame = path.name.split('__', 1)
            output.append(ImagePair(image_id=sequence + '/' + frame, hazy_path=path,
                                    gt_path=root / 'test/sharp' / path.name))
    else:
        from clpo2027.gopro_data import load_gopro_pairs
        output = load_gopro_pairs(root, split='test')
    if len(output) != 1111 or not all(p.gt_path.is_file() for p in output):
        raise ValueError('GoPro evaluation requires all 1111 paired test images.')
    return output


def macs(row):
    if row['mixer'] == 'cspn_context':
        return {'status': 'not_reported', 'parameters': 2230624}
    if row['dataset'] == 'ohaze':
        return read(ROOT / 'supplement/evidence/spatial_mamba/MACS.json')
    data = read(ROOT / 'supplement/evidence/INFERENCE_MACS_REFERENCE.json')
    arm = row.get('arm', row['mixer'])
    arm = 'full' if arm in ['ours', 'full_repeat', 'no_projection', 'joint'] else arm
    return data['records'][arm]


def check(row, model, checkpoint):
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = state.get('params_ema', state.get('params', state))
    model.load_state_dict(state, strict=True)
    result = {'id': row['id'], 'strict_load': {'status': 'passed', 'state_keys': len(state)},
              'parameters': sum(p.numel() for p in model.parameters()),
              'checkpoint_sha256': digest(checkpoint), 'forward': {'status': 'not_run'}}
    if row['mixer'] in ['spatial_mamba', 'ss2d', 'mair',
                        'mair_native_blocks', 'spatial_native_blocks']:
        result['forward'] = {'status': 'skipped', 'reason': 'The official scan requires CUDA.'}
    else:
        model = model.float().eval()
        with torch.inference_mode():
            image = torch.zeros(1, 3, 16, 16)
            value = model(image)
            value = value[-1] if isinstance(value, list) else value
        if value.shape != image.shape or not torch.isfinite(value).all():
            raise ValueError('CPU shape or finite-value check failed.')
        result['forward'] = {'status': 'passed', 'shape': list(value.shape), 'device': 'cpu'}
    return result


def load_prefix(output, identity, ids):
    identity_path = output / 'IDENTITY.json'
    path = output / 'per_image.jsonl'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('The output belongs to a different checkpoint, runtime, or protocol.')
    if path.exists() and path.stat().st_size and not identity_path.exists():
        raise ValueError('The saved prefix has no experiment identity.')
    rows = [json.loads(s) for s in path.read_text('utf-8').splitlines() if s.strip()] if path.exists() else []
    if len(rows) > len(ids) or [r['image_id'] for r in rows] != ids[:len(rows)]:
        raise ValueError('The saved result prefix does not match the test split.')
    if any(r.get('checkpoint_sha256') != identity['checkpoint_sha256'] for r in rows):
        raise ValueError('The saved prefix contains a different checkpoint hash.')
    if any(not math.isfinite(r[k]) for r in rows for k in ['psnr_db', 'ssim']):
        raise ValueError('The saved prefix contains a nonfinite metric.')
    return rows


def evaluate(args, row, model, checkpoint):
    import torch
    from supplement.inference_resources import ResourceMeter, summarize_resources
    from clpo2027.data import read_tensor
    from clpo2027.gopro_eval import halo_tiled_inference, rgb_psnr, rgb_ssim
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Use CUDA for the recorded evaluation protocol.')
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('params_ema', state.get('params', state)), strict=True)
    model = model.to(args.device).float().eval().requires_grad_(False)
    from clpo2027.fp32_inference import prepare_fp32_inference
    inference = prepare_fp32_inference(model)
    output = args.output / 'results'
    output.mkdir(parents=True, exist_ok=True)
    metric_protocol = direct_ssim_protocol(
        backend='torch.conv2d.valid' if row['dataset'] == 'gopro' else 'pytorch_msssim.ssim',
        data_range=1,
        numeric_precision='FP32',
    )
    identity = {'id': row['id'], 'checkpoint_sha256': digest(checkpoint),
                'config_sha256': digest(ROOT / row['config']), 'protocol': row['protocol'],
                'metric_protocol': metric_protocol,
                'runtime_manifest_sha256': digest(ROOT / 'supplement/runtime_manifests' / (row['runtime'] + '.json')),
                'precision': 'FP32_TF32_DISABLED', 'inference': inference}
    if (output / 'FINAL.json').exists():
        raise FileExistsError('The evaluation is complete. Use another output for a new evaluation.')
    path = output / 'per_image.jsonl'
    if row['dataset'] == 'gopro':
        data = pairs(args.data_root)
        ids = [p.image_id for p in data]
    else:
        data = [f'{i}.jpg' for i in range(41, 46)]
        ids = list(data)
        from tasks.restoration.evaluate_ohaze import image_tensor, resize_long_side, pad32, metric
    rows = load_prefix(output, identity, ids)
    save(output / 'IDENTITY.json', identity)
    prefix = len(rows)
    meter = ResourceMeter(model)
    started = time.perf_counter()
    with torch.inference_mode(), path.open('a', encoding='utf-8') as stream:
        for index in range(prefix, len(data)):
            meter.begin_image()
            if row['dataset'] == 'gopro':
                pair = data[index]
                image = read_tensor(pair.hazy_path)[None].to(args.device)
                target = read_tensor(pair.gt_path)[None].to(args.device)
                meter.begin_forward()
                if row['protocol'] == 'tiled':
                    prediction = halo_tiled_inference(model, image, core_tile=384, halo=64, overlap=64)
                    prediction = prediction.clamp(0, 1)
                    meter.end_forward()
                else:
                    prediction = model(image)
                    prediction = prediction[-1] if isinstance(prediction, list) else prediction
                    meter.end_forward()
                    assert prediction.shape == target.shape and torch.isfinite(prediction).all()
                    prediction = prediction.clamp(0, 1)
                psnr, ssim = rgb_psnr(prediction, target), rgb_ssim(prediction, target)
            else:
                name = data[index]
                image = resize_long_side(image_tensor(args.data_root / 'test/hazy' / name)).to(args.device)
                target = resize_long_side(image_tensor(args.data_root / 'test/gt' / name)).to(args.device)
                padded, height, width = pad32(image)
                meter.begin_forward()
                prediction = model(padded)
                prediction = prediction[-1] if isinstance(prediction, list) else prediction
                prediction = prediction[..., :height, :width]
                meter.end_forward()
                psnr, ssim = metric(prediction, target)
            if not math.isfinite(psnr) or not math.isfinite(ssim):
                raise ValueError('A metric is not finite.')
            record = {'image_id': ids[index], 'psnr_db': float(psnr), 'ssim': float(ssim),
                      'checkpoint_sha256': identity['checkpoint_sha256'],
                      'inference_resources': meter.finish_image()}
            rows.append(record)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            save(output / 'PROGRESS.json', {'count': len(rows), 'total': len(ids)})
            del image, target, prediction
    meter.close()
    final = {**identity, 'pair_count': len(rows), 'status': 'COMPLETE',
             'psnr_db': sum(r['psnr_db'] for r in rows) / len(rows),
             'ssim': sum(r['ssim'] for r in rows) / len(rows),
             'per_image_sha256': digest(path), 'resumed_prefix': prefix,
             'this_attempt_wall_seconds': time.perf_counter() - started,
             'inference_resources': summarize_resources(rows, macs(row))}
    if row['dataset'] == 'gopro':
        videos = collections.defaultdict(list)
        for record in rows:
            videos[record['image_id'].split('/')[0]].append(record)
        save(output / 'per_video.json', {k: {'count': len(v), **{
            m: sum(r[m] for r in v) / len(v) for m in ['psnr_db', 'ssim']}}
            for k, v in videos.items()})
    save(output / 'FINAL.json', final)
    return final


def main():
    args = arguments()
    row = experiment(args.id)
    if args.action == 'train' and not row.get('trainable', True):
        raise ValueError('This row is a fixed-weight evaluation. It has no new training run.')
    if args.dry_run:
        print(json.dumps({'id': args.id, 'action': args.action, 'config': row['config'],
                          'checkpoint': str(args.checkpoint or row['checkpoint']),
                          'runtime': row.get('runtime'), 'protocol': row['protocol']}))
        return
    if args.action in ['train', 'eval'] and args.data_root is None:
        raise ValueError('Set --data-root.')
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4 if args.action == 'eval' and row['dataset'] == 'gopro' else 1)
    torch.set_num_interop_threads(1)
    if args.action == 'train':
        if row['mixer'] == 'cspn_context':
            from supplement.cspn_training import train
        else:
            from supplement.train import train
        result = train(args, row)
    else:
        model, _ = network(row, args.output)
        checkpoint = args.checkpoint or ROOT / row['checkpoint']
        if args.action == 'check':
            result = check(row, model, checkpoint)
            save(args.output / 'CHECK.json', result)
        elif args.action == 'profile':
            from supplement.count_ops import profile
            result = profile(row, model, args.device)
            save(args.output / 'MACS.json', result)
        else:
            result = evaluate(args, row, model, checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
