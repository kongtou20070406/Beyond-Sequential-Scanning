"""Install the paper's FP32 inference path after loading fixed model weights."""
import types

import torch

BACKEND = 'joint_heads_direct_rollout_fp32'


def _forward(self, x):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Use the FP32 inference model under torch.inference_mode().')
    signature = tuple((id(p), p._version, p.device, p.dtype)
                      for p in (self.linear_tilt_raw, self.curved_tilt_raw))
    if signature != self._inference_depth_signature:
        raise RuntimeError('The depth parameters changed. Prepare the inference model again.')
    from .fp32_heads import fused_heads
    from .fp32_rollout import direct_rollout
    packed, fr, fi, attention = self._drive_with_attention(x)
    hidden = self._coefficient_hidden(x, packed)
    ar, ai, write, read, strength = fused_heads(self, hidden)
    value = direct_rollout(write * fr, write * fi, strength, ar, ai,
                           self._inference_depth_weights, read, self.selective_group_size)
    return self._interface_readout(value, attention)


def prepare_fp32_inference(model):
    """Use the fused path for full five-step radial models. Keep other models intact.

    Call this function after load_state_dict(), to(device), float(), and eval().
    Training and fixed-weight intervention tools use the differentiable model
    without calling this function. This function does not change state_dict().
    """
    from basicsr.models.archs.VIVNet_ProfiledCLPO_arch import (
        RadialStateDepthProfiledOperator, VIVNetRadialStateDepthProfiledCLPOMixer)
    if model.training:
        raise ValueError('Call eval() before preparing FP32 inference.')
    entries = [m.operator for m in model.modules()
               if type(m) is VIVNetRadialStateDepthProfiledCLPOMixer
               and type(m.operator) is RadialStateDepthProfiledOperator
               and m.operator.K == 5]
    if not entries:
        return {'backend': 'model_defined', 'fused_blocks': 0}
    for op in entries:
        if op.K != 5 or op.tap_count != 9 or op.selective_group_size != 4:
            raise ValueError('The fused path requires five steps, nine taps, and group width four.')
        if any(not p.is_cuda or p.dtype != torch.float32 for p in op.parameters()):
            raise ValueError('The fused path requires CUDA FP32 model parameters.')
    with torch.inference_mode():
        for op in entries:
            op._inference_depth_weights = op.depth_weights(op.linear_tilt_raw).detach()
            op._inference_depth_signature = tuple((id(p), p._version, p.device, p.dtype)
                                                  for p in (op.linear_tilt_raw, op.curved_tilt_raw))
            op.forward = types.MethodType(_forward, op)
    return {'backend': BACKEND, 'fused_blocks': len(entries)}
