import torch,types
from functools import lru_cache
from clpo2027.real_pair_propagation import RealPairStatePropagation

def normalize_fields(raw,control,groups,group_size,rho,gate_range):
 b,_,h,w=raw.shape;d=groups*group_size
 raw=raw.reshape(b,groups,2,9,group_size,h,w).permute(0,3,1,4,2,5,6).reshape(b,9,d,2,h,w)
 mass=raw.square().sum(3).sqrt().sum(1,keepdim=True).unsqueeze(3)
 kernel=raw*(rho/(1+mass))
 control=control.reshape(b,groups,3,group_size,h,w).permute(0,2,1,3,4,5).reshape(b,3,d,h,w)
 return kernel,1+gate_range*control[:,0].tanh(),1+gate_range*control[:,1].tanh(),torch.nn.functional.softplus(control[:,2])
@lru_cache(None)
def normalizer():return torch.compile(normalize_fields,fullgraph=True,dynamic=False)
def install_heads(m):
 for op in m.modules():
  if not isinstance(op,RealPairStatePropagation):continue
  def fields(self,feature):
   hidden=self.coefficient_hidden(feature)
   return normalizer()(self.coeff_out(hidden),self.selective_head(hidden),self.group_count,self.group_size,self.rho_max,self.gate_range)
  op.input_fields=types.MethodType(fields,op)
 return m
