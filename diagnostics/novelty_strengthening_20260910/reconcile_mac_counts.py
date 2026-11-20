"""Reconcile static restoration counts on CPU without training or image data.

SS2D uses a count-only core: its projection einsums keep their actual shapes,
and the official prefix-sum scan estimate is added separately. This does not
verify scan outputs, accuracy, runtime, or gradients. Other models use their
ordinary CPU forward. No release or active experiment file is changed.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import types
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "diagnostics/novelty_strengthening_20260910"
CODE = ROOT
RUNTIME = CODE / "tasks/restoration/runtime"
sys.path[:0] = [str(CODE / "tools"), str(CODE), str(RUNTIME)]

import torch
from fvcore.nn import FlopCountAnalysis
import profile_model as release_profile

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

KEEP_OPERATORS = {"conv", "linear", "matmul", "bmm", "mm", "addmm", "einsum"}
ALLOWED_REMOVALS = {"layer_norm", "batch_norm", "group_norm", "instance_norm",
                    "adaptive_avg_pool2d", "upsample_bilinear2d", "upsample_nearest2d"}
SOURCE_PATHS = [
    CODE / "tools/profile_model.py",
    RUNTIME / "basicsr/models/archs/VIVNet_arch.py",
    RUNTIME / "basicsr/models/archs/VIVNet_MatchedMixers_arch.py",
    RUNTIME / "basicsr/models/archs/VIVNet_ProfiledCLPO_arch.py",
    RUNTIME / "basicsr/models/archs/VIVNet_SelectiveCLPO_arch.py",
    RUNTIME / "clpo2027/model.py",
    RUNTIME / "clpo2027/mixers.py",
    RUNTIME / "clpo2027/vmamba_ss2d_adapter.py",
    RUNTIME / "third_party/VMamba/classification/models/vmamba.py",
    RUNTIME / "third_party/VMamba/classification/models/csms6s.py",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_only_ss2d(model):
    records = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "SS2D":
            continue
        assert module.channel_first and module.k_group == 4
        assert not module.disable_z and module.with_dconv
        assert module.forward_core.keywords["selective_scan_backend"] == "oflex"
        assert isinstance(module.out_norm, torch.nn.LayerNorm)

        def core(current, x, *, key=name):
            batch, width, height, image_width = [int(v) for v in x.shape]
            length = height * image_width
            groups, state, rank = current.k_group, current.d_state, current.dt_rank
            assert width == current.d_inner
            # Cross-scan/merge only rearrange and add tensors; these operations
            # are excluded by the historical MAC convention. Values below are
            # used only to trace the two original projection einsums.
            xs = torch.stack([x.flatten(2)] * groups, dim=1)
            projected = torch.einsum("b k d l, k c d -> b k c l", xs, current.x_proj_weight)
            dts, _, _ = torch.split(projected, [rank, state, state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, current.dt_projs_weight)
            scan = 9 * batch * length * (groups * width) * state + batch * (groups * width) * length
            records[key] = dict(input=[batch, width, height, image_width],
                                d_state=state, dt_rank=rank, directions=groups,
                                official_scan_macs=scan,
                                default_layer_norm_estimate=5 * batch * width * length)
            y = dts.sum(dim=1).reshape(batch, width, height, image_width)
            return current.out_norm(y).to(x.dtype)

        module.forward_core = types.MethodType(core, module)
    assert records == {}
    return records


def profile(task, mixer, file_name, expected_g, expected_params_m):
    config = CODE / "configs/restoration" / task / file_name
    torch.manual_seed(42)
    with contextlib.redirect_stdout(io.StringIO()):
        model = release_profile.restoration_model(config, 5).cpu().eval()
    assert not torch.cuda.is_initialized()
    count_records = count_only_ss2d(model) if mixer == "ss2d" else {}
    sample = torch.zeros(1, 3, 256, 256)
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis = FlopCountAnalysis(model, sample)
        analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings("none")
        traced = int(round(analysis.total()))
        by_operator = {k: int(round(v)) for k, v in analysis.by_operator().items()}
        unsupported = dict(analysis.unsupported_ops())
        propagation, propagation_records = (release_profile.propagation_macs(model, sample)
                                            if mixer in {"ours", "cspn_style"} else (0, []))
    if mixer == "ss2d":
        assert len(count_records) in (22, 26)
        assert by_operator.get("layer_norm", 0) == sum(r["default_layer_norm_estimate"] for r in count_records.values())
    added_scan = sum(row["official_scan_macs"] for row in count_records.values())
    removed = {k: v for k, v in by_operator.items() if k not in KEEP_OPERATORS}
    assert set(removed) <= ALLOWED_REMOVALS, removed
    raw_total = traced + propagation + added_scan
    unified = raw_total - sum(removed.values())
    params = sum(p.numel() for p in model.parameters())
    # These checks bind the CPU accounting to the already published model
    # configurations and counts; they are not accuracy comparisons.
    assert abs(raw_total / 1e9 - expected_g) < 0.00051, (task, mixer, raw_total, expected_g)
    assert abs(params / 1e6 - expected_params_m) < 0.00051, (task, mixer, params, expected_params_m)
    if task == "ohaze" and mixer == "ss2d":
        assert raw_total == 36827962448, raw_total
    assert not torch.cuda.is_initialized()
    result = dict(task=task, mixer=mixer, config=str(config), config_sha256=sha(config),
                  input=[1, 3, 256, 256], parameters=params,
                  historical_rounded_macs_g=expected_g,
                  raw_fvcore_plus_custom_macs=raw_total,
                  by_operator=by_operator, removed_operators=removed,
                  explicit_propagation_macs=propagation,
                  propagation_records=propagation_records,
                  explicit_scan_macs=added_scan, scan_records=count_records,
                  unified_macs=unified, unified_macs_g=unified / 1e9,
                  unsupported_ops=unsupported,
                  count_only_scan=(mixer == "ss2d"))
    del model, sample, analysis
    gc.collect()
    return result


def main():
    output = WORK / "MAC_COUNT_CPU_RECONCILIATION.json"
    assert not output.exists(), "Preserve the existing audit; inspect it before rerunning."
    sources = {str(p): sha(p) for p in SOURCE_PATHS}
    jobs = [
        ("ohaze", "ss2d", "O-HAZE_VIVNet_Matched_SS2D_P256.yml", 36.828, 2.470),
        ("gopro", "ss2d", "ss2d.yml", 82.285, 6.301),
        ("ohaze", "native", "O-HAZE_VIVNet_Baseline.yml", 31.804, 2.860),
        ("ohaze", "dwconv", "O-HAZE_VIVNet_Matched_DWConv_P256.yml", 28.384, 2.849),
        ("ohaze", "mdta", "O-HAZE_VIVNet_Matched_MDTA_P256.yml", 26.576, 2.476),
        ("ohaze", "swin", "O-HAZE_VIVNet_Matched_Swin_P256.yml", 34.546, 2.942),
        ("ohaze", "cspn_style", "O-HAZE_VIVNet_Matched_CSPN_STYLE_P256.yml", 24.332, 2.240),
        ("ohaze", "ours", "O-HAZE_VIVNet_RadialProfiled_K5_P256.yml", 25.641, 2.248),
        ("gopro", "native", "original.yml", 79.134, 7.422),
        ("gopro", "cspn_style", "cspn_style.yml", 59.133, 5.792),
        ("gopro", "ours", "ours.yml", 61.474, 5.807),
    ]
    result = dict(status="RUNNING", utc_started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  device="cpu", cpu_threads=1, optimizer_updates=0, test_images_read=0,
                  purpose="Static MAC accounting only; count-only SS2D does not validate numerical outputs.",
                  convention="Conv/linear/matrix/einsum MACs, official VMamba scan estimate, and explicit 9-tap propagation; exclude normalization, pooling, interpolation, activation, gating, padding, and rearrangement.",
                  source_sha256=sources, records=[])
    for job in jobs:
        started = time.perf_counter()
        record = profile(*job)
        record["audit_wall_seconds"] = time.perf_counter() - started
        result["records"].append(record)
        output.with_suffix(".partial.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(job[0], job[1], record["parameters"], record["raw_fvcore_plus_custom_macs"], record["unified_macs"], flush=True)
    assert sources == {str(p): sha(p) for p in SOURCE_PATHS}
    assert not torch.cuda.is_initialized()
    result.update(status="PASS", cuda_initialized=False,
                  utc_finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  script_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS", output, flush=True)


if __name__ == "__main__":
    main()
