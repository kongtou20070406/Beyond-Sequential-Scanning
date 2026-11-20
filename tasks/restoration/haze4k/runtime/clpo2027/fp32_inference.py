"""Install the paper's FP32 inference path after loading fixed model weights."""

import torch

BACKEND = 'joint_heads_direct_rollout_fp32'


def _forward(self, x):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Use the FP32 inference model under torch.inference_mode().')
    _refresh_depth_weights(self)
    from .fp32_heads import fused_heads
    from .fp32_rollout import direct_rollout
    if hasattr(self, 'pair_layout_version'):
        feature = self.Bin(x)
        hidden = self.coefficient_hidden(feature)
        ar, ai, write, read, strength = fused_heads(self, hidden)
        value = direct_rollout(write * feature[:, 0::2], write * feature[:, 1::2],
                               strength, ar, ai, self._inference_depth_weights,
                               read, self.selective_group_size, pair_output=True)
        return self.Cout(value)
    packed, fr, fi, attention = self._drive_with_attention(x)
    hidden = self._coefficient_hidden(x, packed)
    ar, ai, write, read, strength = fused_heads(self, hidden)
    value = direct_rollout(write * fr, write * fi, strength, ar, ai,
                           self._inference_depth_weights, read, self.selective_group_size)
    return self._interface_readout(value, attention)


def _refresh_depth_weights(op):
    signature = tuple((id(p), p._version, p.device, p.dtype)
                      for p in (op.linear_tilt_raw, op.curved_tilt_raw))
    if signature != getattr(op, '_inference_depth_signature', None):
        op._inference_depth_weights = op.depth_weights(op.linear_tilt_raw).detach()
        op._inference_depth_signature = signature


def use_fp32_inference(op, x):
    """Select fixed-weight CUDA FP32 inference. Keep gradients and autocast intact."""
    if (op.training or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
            or getattr(op, '_fp32_inference_enabled', True) is False
            or not x.is_cuda or x.dtype != torch.float32):
        return False
    from basicsr.models.archs.VIVNet_ProfiledCLPO_arch import RadialStateDepthProfiledOperator
    from .real_pair_propagation import RealPairStatePropagation
    return (type(op) in (RadialStateDepthProfiledOperator, RealPairStatePropagation)
            and op.K == 5 and op.tap_count == 9 and op.selective_group_size == 4
            and all(p.is_cuda and p.device == x.device and p.dtype == torch.float32
                    for p in op.parameters()))


def fp32_inference(op, x):
    """Run the fused operator on the full input extent."""
    return _forward(op, x)


def prepare_fp32_inference(model, *, enabled=True):
    """Set inference selection. The default model forward selects this path too.

    Use enabled=False for reference measurements. No parameter or state key changes.
    The cache refreshes after parameter changes. Training uses the normal forward.
    """
    from basicsr.models.archs.VIVNet_ProfiledCLPO_arch import (
        RadialStateDepthProfiledOperator, VIVNetRadialStateDepthProfiledCLPOMixer)
    from .real_pair_propagation import RealPairStatePropagation
    entries = [m.operator for m in model.modules()
               if type(m) is VIVNetRadialStateDepthProfiledCLPOMixer
               and type(m.operator) in (RadialStateDepthProfiledOperator, RealPairStatePropagation)
               and m.operator.K == 5 and m.operator.tap_count == 9
               and m.operator.selective_group_size == 4]
    ready = 0
    for op in entries:
        op._fp32_inference_enabled = enabled
        if enabled and not op.training and all(p.is_cuda and p.dtype == torch.float32 for p in op.parameters()):
            with torch.inference_mode():
                _refresh_depth_weights(op)
            ready += 1
    return {'backend': BACKEND if ready else 'model_defined', 'fused_blocks': ready}
