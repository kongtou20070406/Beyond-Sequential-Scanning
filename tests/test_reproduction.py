"""Check data identity and result collection without a GPU."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paper
import prepare_data
from tasks.restoration.metric_contract import DIRECT_RGB_SSIM


class DataTests(unittest.TestCase):
    def test_same_size_corrupt_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, target = root / "source", root / "target"
            source.write_bytes(b"abcd")
            target.write_bytes(b"dcba")
            with self.assertRaises(FileExistsError):
                prepare_data.place(source, target, copy=True)
            self.assertEqual(target.read_bytes(), b"dcba")

    def test_ohaze_split_preserves_pairs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, target = root / "source", root / "target"
            for folder, marker in (("GT", "GT"), ("hazy", "hazy")):
                (source / folder).mkdir(parents=True)
                for i in range(1, 46):
                    (source / folder / f"{i:02d}_outdoor_{marker}.jpg").write_bytes(bytes([i]))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_data.prepare_ohaze(source, target, copy=True)
            self.assertEqual(len(list((target / "train/gt").iterdir())), 40)
            self.assertEqual({p.stem for p in (target / "test/hazy").iterdir()},
                             {"41", "42", "43", "44", "45"})
            self.assertEqual((target / "test/gt/41.jpg").read_bytes(), bytes([41]))


class CollectionTests(unittest.TestCase):
    def test_gopro_uses_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output = root / "output"
            rows = []
            for i in range(3):
                ident = f"gopro-ours-{i}"
                rows.append({"id": ident, "dataset": "gopro", "label": "Ours",
                             "seed": 42 + i, "suites": ["gopro"], "expected": {}})
                result = output / ident / "results"
                result.mkdir(parents=True)
                (result / "FINAL.json").write_text(json.dumps({
                    "psnr_db": 29 + i,
                    "ssim": .9 + .01 * i,
                    "metric_protocol": DIRECT_RGB_SSIM,
                }))
            rows.append({"id": "haze4k-ours", "dataset": "haze4k", "label": "Ours",
                         "seed": 42, "suites": ["haze4k"], "expected": {}})
            result = output / "haze4k-ours" / "results"
            result.mkdir(parents=True)
            (result / "FINAL.json").write_text(json.dumps({
                "psnr": 31,
                "ssim": .98,
                "metric_protocol": DIRECT_RGB_SSIM,
            }))
            (root / "experiments.json").write_text(json.dumps({"experiments": rows}))
            with patch.object(paper, "ROOT", root), patch("sys.argv", ["paper.py", "collect", "--output", str(output)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    paper.main()
            value = json.loads((output / "SUMMARY.json").read_text())
            stats = value["gopro_sample_statistics"]["Ours"]["psnr"]
            self.assertEqual(stats, {"mean": 30, "sample_std": 1})

    def test_missing_result_fails(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "experiments.json").write_text(json.dumps({"experiments": [
                {"id": "missing", "dataset": "ohaze", "label": "Ours", "suites": ["ohaze-main"]}]}))
            with patch.object(paper, "ROOT", root), patch("sys.argv", ["paper.py", "collect"]):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    paper.main()
            self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
