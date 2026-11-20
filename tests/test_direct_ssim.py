"""Check the direct SSIM contract and recorded-result collection on CPU."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import paper
from tasks.restoration.metric_contract import (
    historical_pooled_only,
    recorded_result,
    require_direct_ssim,
    result_metric,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


class DirectSsimTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
    def test_torch_backends_agree_without_pre_pooling(self):
        from tasks.restoration.evaluate_ohaze import metric
        from tasks.restoration.runtime.clpo2027.gopro_eval import rgb_ssim

        torch.manual_seed(7)
        prediction = torch.rand(1, 3, 37, 41)
        target = (prediction * 0.91 + 0.03).clamp(0, 1)
        pytorch_msssim_value = metric(prediction, target)[1]
        direct_conv_value = rgb_ssim(prediction, target)
        self.assertLess(abs(pytorch_msssim_value - direct_conv_value), 2e-6)

    def test_contract_rejects_pooled_result_and_ssim_alias(self):
        row = {"dataset": "ohaze"}
        pooled = {"protocol": "RGB PSNR; pooled SSIM", "ssim": 0.94}
        with self.assertRaisesRegex(ValueError, "historical evidence"):
            require_direct_ssim(row, pooled, "old.json")
        with self.assertRaisesRegex(ValueError, "canonical direct SSIM"):
            result_metric({"average_ssim": 0.94}, "ssim")

    def test_bundled_reassessment_covers_all_current_ohaze_rows(self):
        root = Path(paper.__file__).resolve().parent
        registry = json.loads((root / "experiments.json").read_text())["experiments"]
        current = [
            row for row in registry
            if row["dataset"] == "ohaze" and not historical_pooled_only(row)
        ]
        direct = [row for row in current if "ssim" in row.get("expected", {})]
        self.assertEqual(len(direct), 23)
        for row in direct:
            source = recorded_result(row)
            self.assertIsNotNone(source)
            value = json.loads((root / source).read_text())
            require_direct_ssim(row, value, source)
            self.assertAlmostEqual(value["ssim"], row["expected"]["ssim"], places=12)

        groups = list((
            root / "supplement/evidence/ssim_reassessment_20260912/groups"
        ).glob("*/FINAL.json"))
        self.assertEqual(len(groups), 38)
        self.assertEqual(sum(len(json.loads(path.read_text())["images"])
                             for path in groups), 190)

    def test_recorded_ohaze_collection_uses_direct_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            argv = [
                "paper.py", "collect", "--suite", "ohaze-main",
                "--recorded", "--output", str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()), patch("sys.argv", argv):
                paper.main()
            summary = json.loads((output / "SUMMARY.json").read_text())
            self.assertEqual(summary["missing"], [])
            self.assertTrue(summary["results"])
            self.assertTrue(all(
                row["metric_protocol"] == "direct_rgb_ssim_v1"
                for row in summary["results"]
            ))
            self.assertTrue(all(abs(row["psnr_difference"]) < 1e-12
                                for row in summary["results"]))

    def test_historical_rows_are_excluded_without_fallback(self):
        expected = {
            "ohaze-step_conditioned_rules-seed42",
            "ohaze-equal_parameter_real-seed42",
            "ohaze-alternative_generator-seed42",
            "ohaze-conjugate_symmetric-seed42",
            "ohaze-p0t_diagonal_delay-seed42",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            argv = [
                "paper.py", "collect", "--suite", "ohaze-ablations",
                "--recorded", "--output", str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()), patch("sys.argv", argv):
                paper.main()
            summary = json.loads((output / "SUMMARY.json").read_text())
            self.assertEqual(summary["missing"], [])
            self.assertEqual({row["id"] for row in summary["excluded"]}, expected)


if __name__ == "__main__":
    unittest.main()
