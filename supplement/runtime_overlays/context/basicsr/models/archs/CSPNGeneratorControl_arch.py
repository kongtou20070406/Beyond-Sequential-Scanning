"""Keep the CSPN recurrence and change only its affinity generator."""
from __future__ import annotations

import copy
import torch
from torch import nn
import torch.nn.functional as F

from basicsr.models.archs.VIVNet_MatchedMixers_arch import VIVNetMatchedMixer
from basicsr.models.archs.VIVNet_ProfiledCLPO_arch import VIVNetRadialStateDepthProfiledCLPO
from clpo2027.mixers import CSPNOriginalStyleMixer


def anchored_rollout(initial, center, off, steps=5):
    """Use the published neighbor order, initial anchor and zero boundary."""
    height, width = initial.shape[-2:]
    state = initial
    for _ in range(steps):
        padded = F.pad(state, (1, 1, 1, 1))
        result = center * initial
        for index, (dy, dx) in enumerate(CSPNOriginalStyleMixer._OFFSETS):
            neighbor = padded[:, :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            result = result + off[:, :, index] * neighbor
        state = result
    return state


_compiled_rollout = None


def compiled_rollout(initial, center, off, steps=5):
    global _compiled_rollout
    if _compiled_rollout is None:
        _compiled_rollout = torch.compile(anchored_rollout, fullgraph=True, dynamic=False)
    return _compiled_rollout(initial, center, off, steps)


class ContextAffinity(nn.Module):
    """Copy the paper's hidden compiler and emit eight real affinities."""

    def __init__(self, reference_operator, head_seed):
        super().__init__()
        d = reference_operator.state_channels
        self.channels = 2 * d
        self.groups = reference_operator.selective_group_count
        self.coeff_in = copy.deepcopy(reference_operator.coeff_in)
        self.coeff_dw = copy.deepcopy(reference_operator.coeff_dw)
        self.coeff_dw2 = copy.deepcopy(reference_operator.coeff_dw2)
        self.coeff_gamma = nn.Parameter(reference_operator.coeff_gamma.detach().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.coeff_out = nn.Conv2d(d, 8 * self.channels, 1, groups=self.groups, bias=True)

    def hidden(self, state):
        first, second = state.chunk(2, dim=1)
        source = torch.stack((first, second), dim=2).flatten(1, 2)
        projected = self.coeff_in(source)
        primary = F.gelu(self.coeff_dw(projected))
        secondary = F.gelu(self.coeff_dw2(projected))
        return primary + self.coeff_gamma.view(1, -1, 1, 1) * secondary

    def forward(self, state):
        batch, channels, height, width = state.shape
        raw = self.coeff_out(self.hidden(state))
        # Each hidden group receives four paired state coordinates. Restore
        # the original C-channel order before the CSPN normalization.
        raw = raw.view(batch, self.groups, 4, 2, 8, height, width)
        return raw.permute(0, 3, 1, 2, 4, 5, 6).reshape(batch, 8 * channels, height, width)


class ControlledCSPN(CSPNOriginalStyleMixer):
    backend = 'eager'

    def forward(self, x):
        initial = self.state_in(x)
        center, off = self._compile_affinity(initial)
        if initial.is_cuda:
            fn = compiled_rollout if self.backend == 'compiled' else anchored_rollout
            # Match the FP32 recurrence used by the existing Linux CUDA bridge.
            state = fn(initial.float(), center.float(), off.float(), self.steps).to(initial.dtype)
        else:
            state = anchored_rollout(initial, center, off, self.steps)
        return self.state_out(state)


class CSPNGeneratorControl(VIVNetMatchedMixer):
    """Use one VIVNet carrier and one of two affinity generators."""

    def __init__(self, *, generator='original', compiler_seed=42, **network):
        if generator not in ('original', 'context'):
            raise ValueError(generator)
        network = dict(network)
        network['mixer_type'] = 'cspn_style'
        super().__init__(**network)
        mixers = [(name, module) for name, module in self.named_modules()
                  if isinstance(module, CSPNOriginalStyleMixer)]
        references = None
        if generator == 'context':
            reference_config = {key: value for key, value in network.items() if key != 'mixer_type'}
            reference_config.update(rho_max=.95, group_size=4, linear_tilt_init=.20, curved_tilt_init=.01)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(compiler_seed)
                reference = VIVNetRadialStateDepthProfiledCLPO(**reference_config)
            references = list(reference.clpos())
            if len(references) != len(mixers):
                raise AssertionError('The CSPN and paper mixer positions differ')
        for index, (_, module) in enumerate(mixers):
            module.__class__ = ControlledCSPN
            module.backend = 'eager'
            if references is not None:
                if module.channels != 2 * references[index].state_channels:
                    raise AssertionError('The matched hidden width differs')
                module.affinity_head = ContextAffinity(references[index], 20260911 + 101 * index)
        self.generator = generator

    def set_backend(self, backend):
        if backend not in ('eager', 'compiled'):
            raise ValueError(backend)
        for module in self.modules():
            if isinstance(module, ControlledCSPN):
                module.backend = backend
