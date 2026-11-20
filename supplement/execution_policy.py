"""Recorded execution-only activation recomputation for the state-dependent arm."""
import torch
from torch.utils.checkpoint import checkpoint
from basicsr.models.archs.VIVNet_arch import AttBlock

def apply_execution_policy(model,arm):
 names=[]
 if arm!='state_dynamic':return names
 for name,block in model.named_modules():
  if not isinstance(block,AttBlock):continue
  original=block.forward
  def forward(x,_original=original,_block=block):
   if _block.training and torch.is_grad_enabled():
    return checkpoint(_original,x,use_reentrant=False,preserve_rng_state=True)
   return _original(x)
  block.forward=forward;names.append(name)
 assert len(names)==26,len(names)
 return names
