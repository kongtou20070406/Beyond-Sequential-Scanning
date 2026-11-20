"""NCHW slot adapter around unmodified official Spatial-Mamba Python modules."""
import importlib
from pathlib import Path
import sys, types, os
from torch import nn

OFFICIAL = Path(os.environ['SUPPLEMENT_SPATIAL_OFFICIAL'])
sys.path.insert(0, str(OFFICIAL / 'kernels' / 'dwconv2d'))
package = 'spatial_mamba_upstream'
if package not in sys.modules:
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(OFFICIAL / 'classification' / 'models')]
    sys.modules[package] = namespace
module = importlib.import_module(package + '.spatialmamba')
if module.DepthwiseFunction is None:
    raise ImportError('Official dwconv2d build is required; no replacement implementation is allowed.')

class SpatialMambaMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ssm = module.StructureAwareSSM(d_model=channels, d_state=1, expand=2, d_conv=3, dropout=0.)
    def forward(self, x):
        return self.ssm(x.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()

def build_spatial_mamba(channels):
    return SpatialMambaMixer(channels)
