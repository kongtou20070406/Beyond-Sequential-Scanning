import types,torch
import clpo2027.real_pair_propagation as rp
from fused_state import state_step
from compiled_heads import normalizer

def install_fast_training(model):
 rp._compiled_state_step=lambda:state_step
 count=0
 for op in model.modules():
  if not isinstance(op,rp.RealPairStatePropagation):continue
  original=op.input_fields
  def fields(self,feature,_original=original):
   if not (self.training and torch.is_grad_enabled() and feature.is_cuda):return _original(feature)
   hidden=self.coefficient_hidden(feature)
   return normalizer()(self.coeff_out(hidden),self.selective_head(hidden),self.group_count,self.group_size,self.rho_max,self.gate_range)
  op.input_fields=types.MethodType(fields,op);op.use_checkpoint=False;count+=1
 assert count==22,count
 return model
