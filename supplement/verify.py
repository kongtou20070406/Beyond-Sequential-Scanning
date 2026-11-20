"""Verify recorded supplementary results without model inference."""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import json
import math
import re
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tasks.restoration.metric_contract import (  # noqa: E402
    recorded_result,
    require_direct_ssim,
    result_metric,
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metric(value, name):
    if name == 'ssim':
        return result_metric(value, name)
    aliases = {'psnr': ['psnr', 'psnr_db', 'native_psnr_db', 'average_psnr']}
    return next(float(value[k]) for k in aliases[name] if k in value)


def verify_result(root, row):
    path = root / row['recorded_result']
    final = read(path)
    if 'ssim' in row.get('expected', {}) or row['dataset'] in {'ohaze', 'gopro', 'haze4k'}:
        require_direct_ssim(row, final, path)
    if 'per_image_sha256' in final:
        per_image = path.parent / 'per_image.jsonl'
        if digest(per_image) != final['per_image_sha256']:
            raise ValueError(f'Result hash mismatch: {row["id"]}')
        images = [json.loads(s) for s in per_image.read_text().splitlines() if s.strip()]
    else:
        images = final['images']
    expected_count = {'gopro': 1111, 'haze4k': 1000}.get(row['dataset'], 5)
    if len(images) != expected_count:
        raise ValueError(f'Image count mismatch: {row["id"]}')
    ids = [r.get('image_id', r.get('image', r.get('name'))) for r in images]
    if len(set(ids)) != expected_count:
        raise ValueError(f'Duplicate image ID: {row["id"]}')
    for name in ['psnr', 'ssim']:
        values = [metric(r, name) for r in images]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'Nonfinite metric: {row["id"]}')
        mean = statistics.mean(values)
        if abs(mean - metric(final, name)) > 1e-9:
            raise ValueError(f'Mean mismatch: {row["id"]}/{name}')
        # Original table values can be rounded.
        reported = result_metric(final, name)
        if name in row.get('expected', {}) and abs(reported - row['expected'][name]) > (0.0001 if name == 'psnr' else 0.00001):
            raise ValueError(f'Expected metric mismatch: {row["id"]}/{name}')
    checkpoint = root / row['checkpoint']
    if final.get('checkpoint_sha256') and digest(checkpoint) != final['checkpoint_sha256']:
        raise ValueError(f'Checkpoint link mismatch: {row["id"]}')
    videos = {}
    if row['dataset'] == 'gopro':
        groups = collections.defaultdict(list)
        for image, identity in zip(images, ids):
            groups[identity.split('/')[0]].append(image)
        if len(groups) != 11:
            raise ValueError('GoPro must contain 11 test sequences.')
        videos = {k: {'count': len(v), 'psnr_db': statistics.mean(metric(r, 'psnr') for r in v),
                      'ssim': statistics.mean(metric(r, 'ssim') for r in v)} for k, v in groups.items()}
        stored_path = path.parent / 'per_video.json'
        if stored_path.exists():
            stored = read(stored_path)
            for name, value in videos.items():
                if stored[name]['count'] != value['count']:
                    raise ValueError('Video count mismatch.')
                for key in ['psnr_db', 'ssim']:
                    if abs(stored[name][key] - value[key]) > 1e-9:
                        raise ValueError('Video mean mismatch.')
    return {'id': row['id'], 'count': expected_count, 'psnr': result_metric(final, 'psnr'),
            'ssim': result_metric(final, 'ssim'), 'videos': len(videos), 'ids': ids}


def scan(root):
    failures = []
    private_path = re.compile(r'(?<![A-Za-z])(?:[A-Za-z]:[/\\]|/ro' r'ot/|/ho' r'me/|/auto' r'dl-tmp/)')
    extensions = {'.py', '.json', '.jsonl', '.md', '.yaml', '.yml', '.txt', '.csv', '.patch', '.toml', '.sh'}
    for path in root.rglob('*'):
        if path.is_symlink():
            failures.append({'file': path.relative_to(root).as_posix(), 'issue': 'symbolic link'})
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix in ['.pyc', '.pyd', '.so']:
            failures.append({'file': relative, 'issue': 'machine-specific generated file'})
        if path.suffix in ['.pt', '.pth'] and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if not name.endswith('data.pkl'):
                        continue
                    text = archive.read(name).decode('latin1')
                    if private_path.search(text):
                        failures.append({'file': relative, 'issue': 'private path or identity in checkpoint metadata'})
        if path.suffix not in extensions:
            continue
        text = path.read_text(encoding='utf-8')
        if path.name == 'verify.py':
            # The scanner contains the detection literals, not experiment paths.
            continue
        if private_path.search(text):
            failures.append({'file': relative, 'issue': 'private path or identity in text'})
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = read(ROOT / 'experiments.json')['experiments']
    checked = []
    for row in rows:
        source = recorded_result(row)
        if source:
            checked.append(verify_result(ROOT, {**row, 'recorded_result': source}))
    gopro = [row['ids'] for row in checked if row['count'] == 1111]
    if gopro and not all(ids == gopro[0] for ids in gopro):
        raise ValueError('GoPro conditions do not have the same ordered test IDs.')
    for row in checked:
        row.pop('ids')
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text('utf-8-sig'), filename=path.relative_to(ROOT).as_posix())
    failures = scan(ROOT)
    provenance = ROOT / 'supplement/SOURCE_PROVENANCE.json'
    if provenance.exists():
        for row in read(provenance):
            if digest(ROOT / row['published']) != row['published_sha256']:
                failures.append({'file': row['published'], 'issue': 'published source-provenance checksum mismatch'})
    manifest = ROOT / 'supplement/PACKAGE_MANIFEST.json'
    if manifest.exists():
        recorded_files = read(manifest)['files']
        actual_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p != manifest}
        if actual_files != set(recorded_files):
            failures.append({'issue': 'package file inventory mismatch',
                             'extra': sorted(actual_files - set(recorded_files)),
                             'missing': sorted(set(recorded_files) - actual_files)})
        for name, value in recorded_files.items():
            if digest(ROOT / name) != value:
                failures.append({'file': name, 'issue': 'package checksum mismatch'})
    result = {'all_pass': not failures, 'recorded_results': checked, 'failures': failures,
              'coverage': 'File hashes, exact result means, IDs, per-video records, Python syntax, and anonymous file scan. No inference.'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', 'utf-8')
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
