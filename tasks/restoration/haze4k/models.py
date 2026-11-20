from common import *
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from clpo2027.real_pair_propagation import RealPairStatePropagation,_state_step,_compiled_state_step
from basicsr.models.archs.VIVNet_ProfiledCLPO_arch import VIVNetRadialStateDepthProfiledCLPO
ARMS=('full','simple','gate_only','readout_only','convex_readout','free_gain','cspn','ss2d','native','dwconv','swin','cspn_style','no_gain','state_dynamic')
class ControlledPair(RealPairStatePropagation):
 def input_fields(self,feature):
  kernel,write,read,strength=super().input_fields(feature)
  if self.arm in ('simple','readout_only'):strength=torch.zeros_like(strength)
  if self.arm=='free_gain':kernel=kernel*self.log_gain.exp()
  return kernel,write,read,strength
 def depth_weights(self,reference):
  if self.arm in ('simple','gate_only'):return torch.ones(self.K,device=reference.device,dtype=reference.dtype)
  return super().depth_weights(reference)
 def forward(self,x):
  if self.arm!='convex_readout':return super().forward(x)
  feature=self.Bin(x);kernel,write,read,strength=self.input_fields(feature)
  drive=self.scale_pairs(feature,write);state=torch.zeros_like(feature);readout=torch.zeros_like(feature)
  step=_compiled_state_step() if x.is_cuda and torch.is_grad_enabled() else _state_step
  for weight in self.depth_weights(feature)/self.K:
   if self.use_checkpoint and torch.is_grad_enabled():state=checkpoint(step,state,drive,kernel,strength,self.group_size,use_reentrant=False,preserve_rng_state=False)
   else:state=step(state,drive,kernel,strength,self.group_size)
   readout=readout+weight*state
  return self.Cout(self.scale_pairs(readout,read))
def make_model(arm):
 if arm not in ARMS:raise ValueError(arm)
 torch.manual_seed(42)
 if arm=='state_dynamic':
  from basicsr.models.archs.VIVNet_RadialProfiledAblation_arch import VIVNetRadialProfiledAblation
  model=VIVNetRadialProfiledAblation(K=5,rho_max=.95,group_size=4,linear_tilt_init=.20,curved_tilt_init=.01,ablation='state_dynamic_transition')
  if sum(p.numel() for p in model.parameters())!=2248140:raise AssertionError('Unexpected state_dynamic parameter count')
  return model
 if arm=='no_gain':
  from basicsr.models.archs.VIVNet_RadialProfiledAblation_arch import VIVNetRadialProfiledAblation
  return VIVNetRadialProfiledAblation(K=5,rho_max=.95,group_size=4,linear_tilt_init=.20,curved_tilt_init=.01,ablation='no_projection')
 if arm in ('native','dwconv','swin','cspn_style'):
  from basicsr.models.archs.VIVNet_arch import VIVNet
  from basicsr.models.archs.VIVNet_MatchedMixers_arch import VIVNetMatchedMixer
  kwargs=dict(dim=32,num_blocks=[2,3,4],num_refinement_blocks=4,ffn_expansion_factor=3,bias=False)
  return VIVNet(**kwargs) if arm=='native' else VIVNetMatchedMixer(**kwargs,mixer_type=arm,K=5)
 if arm=='cspn':
  from context_cspn import CSPNGeneratorControl
  model=CSPNGeneratorControl(generator='context',K=5);model.set_backend('compiled')
 elif arm=='ss2d':
  from basicsr.models.archs.VIVNet_MatchedMixers_arch import VIVNetMatchedMixer
  model=VIVNetMatchedMixer(mixer_type='cspn_style' if arm=='cspn' else 'ss2d',K=5)
 else:
  model=VIVNetRadialStateDepthProfiledCLPO(K=5)
  for m in model.modules():
   if isinstance(m,RealPairStatePropagation):
    m.__class__=ControlledPair;m.arm=arm
    if arm=='free_gain':m.log_gain=nn.Parameter(torch.zeros(()))
 if arm=='full':
  from fast_training import install_fast_training
  model=install_fast_training(model)
 return model
def loss_fn(pred,target):
 delta=torch.fft.fft2(pred)-torch.fft.fft2(target)
 return (pred-target).abs().mean()+.1*torch.view_as_real(delta).abs().mean()

