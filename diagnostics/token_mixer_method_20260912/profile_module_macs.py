"""Count the released module's executed forward operations and plot the sweeps."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tasks/restoration/runtime/clpo2027/real_pair_propagation.py"
DEFAULT_OUTPUT = ROOT / "runs/module_macs"


def load_module():
    spec = importlib.util.spec_from_file_location("released_real_pair", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExecutedMacCounter(TorchDispatchMode):
    """Observe conv work and the scalar products inside the nine-neighbor kernel.

    One convolution multiply-accumulate is one MAC. A two-channel neighbor
    transform has four scalar products, counted as four MACs under the paper's
    propagation convention. This is an operation count, not a hardware timer.
    """

    def __init__(self):
        super().__init__()
        self.in_pair_kernel = False
        self.conv_macs = 0
        self.pair_macs = 0
        self.conv_calls = 0
        self.pair_calls = 0
        self.pair_product_calls = 0
        self.conv_records = []
        self.pair_records = []
        self.other_ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.convolution.default, torch.ops.aten.conv2d.default):
            weight = args[1]
            # weight.shape[1] already accounts for convolution groups.
            macs = result.numel() * weight[0].numel()
            self.conv_macs += macs
            self.conv_calls += 1
            self.conv_records.append(dict(weight=list(weight.shape),
                                          output=list(result.shape), macs=macs))
        elif self.in_pair_kernel and func == torch.ops.aten.mul.Tensor:
            self.pair_macs += result.numel()
            self.pair_product_calls += 1
        else:
            self.other_ops[str(func)] += 1
        return result


def measure(module, channels, height, width, rounds):
    torch.manual_seed(42)
    model = module.RealPairStatePropagation(channels, K=rounds, group_size=4).eval()
    # The operation count does not depend on parameter values.
    sample = torch.randn(1, channels, height, width)
    original = module.apply_pair_kernel
    counter = ExecutedMacCounter()

    def observed_pair_kernel(value, kernel):
        counter.pair_calls += 1
        before = counter.pair_macs
        counter.in_pair_kernel = True
        try:
            result = original(value, kernel)
        finally:
            counter.in_pair_kernel = False
        counter.pair_records.append(dict(input=list(value.shape),
                                         kernel=list(kernel.shape),
                                         macs=counter.pair_macs-before))
        return result

    module.apply_pair_kernel = observed_pair_kernel
    started = time.perf_counter()
    try:
        with torch.inference_mode(), counter:
            result = model(sample)
        assert result.shape == sample.shape and torch.isfinite(result).all().item()
    finally:
        module.apply_pair_kernel = original
    n = height * width
    # Independently derived counts check the observed forward operation counts.
    expected_conv = n * (2 * channels**2 + 75 * channels)
    expected_pair = n * 18 * channels * rounds
    assert counter.conv_macs == expected_conv, (counter.conv_macs, expected_conv,
                                               counter.conv_records, counter.other_ops)
    assert counter.pair_macs == expected_pair
    assert counter.conv_calls == 7
    assert counter.pair_calls == rounds
    assert counter.pair_product_calls == 36 * rounds
    return dict(N=n, C=channels, K=rounds, H=height, W=width,
                parameters=sum(p.numel() for p in model.parameters()),
                convolution_macs=counter.conv_macs,
                propagation_macs=counter.pair_macs,
                total_macs=counter.conv_macs + counter.pair_macs,
                conv_calls=counter.conv_calls, propagation_calls=counter.pair_calls,
                propagation_product_calls=counter.pair_product_calls,
                elapsed_counting_seconds=time.perf_counter()-started,
                conv_records=counter.conv_records, propagation_records=counter.pair_records)


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 10.8, "axes.labelsize": 10.2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    colors = {"full": "#1866A3", "propagation": "#CF7332"}
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.65))
    settings = [
        ("N", r"(a) Spatial positions $N$", r"$N=H\times W$ (thousands)",
         r"$C=32,\ K=5$", "Linear in spatial positions"),
        ("C", r"(b) Channels $C$", r"Real channel count $C$",
         r"$H=W=128,\ K=5$", "Channel projections\nadd quadratic cost"),
        ("K", r"(c) Rounds $K$", r"Round count $K$",
         r"$H=W=128,\ C=32$", "Each round adds\nthe same propagation cost"),
    ]
    for ax, (variable, title, xlabel, fixed, takeaway) in zip(axes, settings):
        points = [row for row in rows if row["sweep"] == variable]
        x = np.array([row[variable] for row in points], dtype=float)
        if variable == "N":
            x /= 1000
        full = np.array([row["total_macs"] for row in points]) / 1e6
        propagation = np.array([row["propagation_macs"] for row in points]) / 1e6
        ax.fill_between(x, propagation, full, color=colors["full"], alpha=.055)
        ax.plot(x, full, "o-", color=colors["full"], lw=2, ms=4.5,
                label="Full propagation module")
        ax.plot(x, propagation, "s--", color=colors["propagation"], lw=1.8, ms=4,
                label="Repeated neighborhood propagation")
        ax.set_title(title, loc="left", pad=26, fontweight="semibold")
        ax.text(0, 1.03, fixed, transform=ax.transAxes, fontsize=9.3, color="#536170")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MACs (millions)")
        ax.grid(axis="y", color="#DEE4EA", linewidth=.65)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0 if variable != "K" else .5)
        ax.text(.5, -.35, takeaway, transform=ax.transAxes, ha="center",
                va="top", fontsize=9.3, color="#344453")
        if variable == "N":
            ax.set_xticks([0, 16, 32, 48, 64])
            ax.annotate(r"$256\times256$", (x[-1], full[-1]), xytext=(-6, -19),
                        textcoords="offset points", ha="right", fontsize=9,
                        color=colors["full"])
        elif variable == "C":
            ax.set_xticks([8, 32, 64, 96, 128])
        else:
            ax.set_xticks([1, 3, 5, 7, 9, 12])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01),
               ncol=2, frameon=False, fontsize=9.6, handlelength=2.6,
               columnspacing=2)
    fig.subplots_adjust(left=.075, right=.985, top=.75, bottom=.30, wspace=.45)
    for extension in ["svg", "pdf", "png"]:
        fig.savefig(output / f"module_macs_scaling.{extension}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = args.output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = json.loads((output / "module_macs_data.json").read_text())["rows"]
        plot(rows, output)
        return
    torch.set_num_threads(4)
    module = load_module()
    records = []
    cache = {}
    settings = [("N", 32, s, s, 5) for s in [32, 64, 96, 128, 192, 256]]
    settings += [("C", c, 128, 128, 5) for c in [8, 16, 24, 32, 48, 64, 96, 128]]
    settings += [("K", 32, 128, 128, k) for k in [1, 2, 3, 5, 7, 9, 12]]
    for sweep, c, h, w, k in settings:
        key = (c, h, w, k)
        if key not in cache:
            cache[key] = measure(module, c, h, w, k)
        records.append(dict(sweep=sweep, **cache[key]))
        (output / "module_macs_progress.json").write_text(json.dumps(
            dict(completed=len(records), planned=len(settings), last=[sweep,c,h,w,k])))
    # Model parameters and all one-time conv calls remain unchanged across K.
    rounds_rows = [r for r in records if r["sweep"] == "K"]
    assert len({r["parameters"] for r in rounds_rows}) == 1
    assert len({r["convolution_macs"] for r in rounds_rows}) == 1
    report = dict(status="passed", source=str(SOURCE),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  environment=dict(python=sys.version, torch=torch.__version__, device="CPU",
                                   dtype="float32", threads=torch.get_num_threads()),
                  batch_size=1, group_size_pairs=4, plotted_points=len(records),
                  distinct_forward_executions=len(cache),
                  method="Actual eager forward execution; aten convolution and scalar products inside apply_pair_kernel are observed with TorchDispatchMode.",
                  scope="One RealPairStatePropagation including its input/output projections and kernel/control generator; excludes outer carrier FFN and backbone.",
                  convention="Conv MACs plus four MACs per two-channel neighbor transform. Bias, normalization, activations, pointwise gating, readout elementwise ops, and tensor layout ops are excluded, as in the release counter.",
                  validation_formula_g4="N * (2*C*C + 75*C + 18*K*C)",
                  linearity=dict(N="linear for both curves", C="propagation linear; whole module includes 2*N*C^2 channel projections", K="propagation linear; full module affine with fixed generation/projection cost"),
                  rows=records)
    (output / "module_macs_data.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    fields = ["sweep", "N", "C", "K", "H", "W", "parameters", "convolution_macs", "propagation_macs", "total_macs"]
    with (output / "module_macs_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    plot(records, output)
    print(json.dumps({key:report[key] for key in ["status", "plotted_points", "distinct_forward_executions", "linearity"]}))


if __name__ == "__main__":
    main()
