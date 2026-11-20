"""Build the restoration architectures used by the paper supplement."""

import importlib


ARCHITECTURES = {
    "VIVNet": "VIVNet_arch",
    "VIVNetMatchedMixer": "VIVNet_MatchedMixers_arch",
    "VIVNetRadialStateDepthProfiledCLPO": "VIVNet_ProfiledCLPO_arch",
    "VIVNetRadialProfiledAblation": "VIVNet_RadialProfiledAblation_arch",
    "VIVNetPhysicalQ9RadialProfiledCLPO": "VIVNet_PhysicalQ9RadialProfiledCLPO_arch",
    "VIVNativeBlocks": "VIVNativeBlocks_arch",
}


def define_network(opt):
    options = dict(opt)
    name = options.pop("type")
    if name not in ARCHITECTURES:
        raise ValueError(f"The paper has no architecture named {name}.")
    module = importlib.import_module(f"basicsr.models.archs.{ARCHITECTURES[name]}")
    return getattr(module, name)(**options)
