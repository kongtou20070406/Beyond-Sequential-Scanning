"""Reject damaged supplementary evidence and preserve protocol separation."""
import json
from pathlib import Path
import tempfile
import unittest

from supplement.verify import digest, verify_result
from supplement.run import load_prefix
from tasks.restoration.metric_contract import DIRECT_RGB_SSIM


class EvidenceTests(unittest.TestCase):
    def test_resume_rejects_unidentified_or_mixed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {'checkpoint_sha256': 'endpoint-A', 'protocol': 'native'}
            record = {'image_id': 'frame', 'checkpoint_sha256': 'endpoint-B', 'psnr_db': 26., 'ssim': .94}
            (root / 'per_image.jsonl').write_text(json.dumps(record) + '\n')
            with self.assertRaisesRegex(ValueError, 'no experiment identity'):
                load_prefix(root, identity, ['frame'])
            (root / 'IDENTITY.json').write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError, 'different checkpoint hash'):
                load_prefix(root, identity, ['frame'])

    def fixture(self, root):
        folder = root / 'results'
        folder.mkdir()
        (root / 'weight.pth').write_bytes(b'fixed endpoint')
        images = [{'image_id': f'{i}.jpg', 'psnr_db': 26.0, 'ssim': .94} for i in range(41, 46)]
        path = folder / 'per_image.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in images))
        final = {'psnr_db': 26.0, 'ssim': .94,
                 'metric_protocol': DIRECT_RGB_SSIM,
                 'per_image_sha256': digest(path),
                 'checkpoint_sha256': digest(root / 'weight.pth')}
        (folder / 'FINAL.json').write_text(json.dumps(final))
        return {'id': 'fixture', 'dataset': 'ohaze', 'checkpoint': 'weight.pth',
                'recorded_result': 'results/FINAL.json'}, images, final

    def test_corrupt_image_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, _, _ = self.fixture(root)
            self.assertEqual(verify_result(root, row)['count'], 5)
            path = root / 'results/per_image.jsonl'
            path.write_text(path.read_text().replace('26.0', '25.0', 1))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_result(root, row)

    def test_self_consistent_hash_cannot_hide_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, images, final = self.fixture(root)
            images[-1]['image_id'] = images[0]['image_id']
            path = root / 'results/per_image.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in images))
            final['per_image_sha256'] = digest(path)
            (root / 'results/FINAL.json').write_text(json.dumps(final))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                verify_result(root, row)

    def test_replaced_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, _, _ = self.fixture(root)
            (root / 'weight.pth').write_bytes(b'another endpoint')
            with self.assertRaisesRegex(ValueError, 'Checkpoint link'):
                verify_result(root, row)

    def test_native_registry_reuses_main_weights(self):
        root = Path(__file__).resolve().parents[1]
        rows = json.loads((root / 'experiments.json').read_text())['experiments']
        registry = {r['id']: r for r in rows}
        native = [r for r in rows if 'gopro-native' in r['suites']]
        self.assertGreaterEqual(len(native), 2)
        for row in native:
            main = registry[f'gopro-{row["mixer"]}-seed{row["seed"]}']
            self.assertEqual(row['checkpoint'], main['checkpoint'])
            self.assertFalse(row['trainable'])
            self.assertEqual(row['protocol'], 'native')
            self.assertNotIn('gopro', row['suites'])


if __name__ == '__main__':
    unittest.main()
