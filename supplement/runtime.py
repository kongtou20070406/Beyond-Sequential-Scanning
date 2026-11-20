"""Use a recorded runtime without changing the main paper runtime."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def experiment(ident):
    return next(row for row in read(ROOT / 'experiments.json')['experiments'] if row['id'] == ident)


def activate(row, output):
    """Copy the main runtime and apply only the recorded changed files."""
    base = ROOT / 'tasks/restoration/runtime'
    tag = row.get('runtime', 'base')
    if tag == 'base':
        runtime = base
    else:
        runtime = Path(output) / '_runtime'
        shutil.copytree(base, runtime, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        overlay = ROOT / 'supplement/runtime_overlays' / tag
        shutil.copytree(overlay, runtime, dirs_exist_ok=True)
        expected = read(ROOT / 'supplement/runtime_manifests' / f'{tag}.json')
        for name, value in expected.items():
            if digest(runtime / name) != value:
                raise ValueError(f'Runtime checksum mismatch: {tag}/{name}')
    if tag in ['spatial', 'native_blocks']:
        official = ROOT / 'supplement/third_party/SpatialMamba/official'
        os.environ['SUPPLEMENT_SPATIAL_OFFICIAL'] = str(official)
        # A user can select a local build. No machine-specific build is bundled.
        build = os.environ.get('SUPPLEMENT_SPATIAL_EXTENSIONS')
        if build:
            sys.path.insert(0, str(Path(build).resolve()))
    sys.path.insert(0, str(runtime))
    return runtime


def network(row, output):
    import yaml
    activate(row, output)
    from basicsr.models.archs import define_network
    config = yaml.safe_load((ROOT / row['config']).read_text(encoding='utf-8'))
    return define_network(dict(config['network_g'])), config
