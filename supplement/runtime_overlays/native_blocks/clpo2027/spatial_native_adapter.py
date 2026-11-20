"""Load the unmodified official Spatial-Mamba block implementation.

Model construction and strict checkpoint loading work without the optional CUDA
depthwise-convolution build. Evaluation still requires that official build.
"""

import importlib
import os
from pathlib import Path
import sys
import types


OFFICIAL = Path(os.environ["SUPPLEMENT_SPATIAL_OFFICIAL"])
sys.path.insert(0, str(OFFICIAL / "kernels" / "dwconv2d"))
# The official Python module imports its CUDA scan at module load time. A stub
# is enough for CPU construction and strict state loading; its functions fail
# clearly if somebody tries to run inference without the official build.
if "selective_scan_cuda_oflex_rh" not in sys.modules:
    scan_stub = types.ModuleType("selective_scan_cuda_oflex_rh")

    def _missing_scan(*args, **kwargs):
        raise RuntimeError(
            "Official Spatial-Mamba selective-scan CUDA build is required for inference."
        )

    scan_stub.fwd = _missing_scan
    scan_stub.bwd = _missing_scan
    sys.modules["selective_scan_cuda_oflex_rh"] = scan_stub
package = "spatial_mamba_native_upstream"
if package not in sys.modules:
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(OFFICIAL / "classification" / "models")]
    sys.modules[package] = namespace
module = importlib.import_module(package + ".spatialmamba")


def require_cuda_extension():
    if module.DepthwiseFunction is None:
        raise ImportError(
            "Official Spatial-Mamba dwconv2d is required for evaluation. "
            "Set SUPPLEMENT_SPATIAL_EXTENSIONS to its local build directory."
        )
