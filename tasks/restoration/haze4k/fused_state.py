import torch
import triton
import triton.language as tl

@triton.jit
def gate_forward(S,D,T,Z,N:tl.constexpr,G:tl.constexpr,BLOCK:tl.constexpr):
 pos=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 bg=tl.program_id(1);lanes=tl.arange(0,8)
 off=(bg*8+lanes[:,None])*N+pos[None,:]
 st=tl.load(S+off,pos[None,:]<N,0)
 r=tl.arange(0,4)
 strength=tl.sum(tl.load(T+(bg*4+r[:,None])*N+pos[None,:],pos[None,:]<N,0),0)*0.25
 energy=tl.sum(st*st,0)*0.25
 scale=tl.rsqrt(1.+strength*energy)
 dr=tl.load(D+off,pos[None,:]<N,0)
 tl.store(Z+off,dr+st*scale[None,:],pos[None,:]<N)

@triton.jit
def stencil_forward(Z,K,Y,N:tl.constexpr,P:tl.constexpr,H:tl.constexpr,W:tl.constexpr,TOTAL:tl.constexpr,BLOCK:tl.constexpr):
 j=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=j<TOTAL
 pix=j%N;p=(j//N)%P;b=j//(N*P);row=pix//W;col=pix%W
 rr=tl.full((BLOCK,),0,tl.float32);ii=tl.full((BLOCK,),0,tl.float32)
 for t in tl.static_range(9):
  dy=(0,-1,-1,-1,0,0,1,1,1)[t];dx=(0,-1,0,1,-1,1,-1,0,1)[t]
  near=pix+dy*W+dx;mask=valid&(row+dy>=0)&(row+dy<H)&(col+dx>=0)&(col+dx<W)
  zr=tl.load(Z+(b*P*2+p*2)*N+near,mask,0);zi=tl.load(Z+(b*P*2+p*2+1)*N+near,mask,0)
  ka=tl.load(K+((b*9+t)*P*2+p*2)*N+pix,valid,0);kb=tl.load(K+((b*9+t)*P*2+p*2+1)*N+pix,valid,0)
  rr=rr+ka*zr-kb*zi;ii=ii+kb*zr+ka*zi
 tl.store(Y+(b*P*2+p*2)*N+pix,rr,valid);tl.store(Y+(b*P*2+p*2+1)*N+pix,ii,valid)

@triton.jit
def stencil_backward(Z,K,DY,DZ,DK,N:tl.constexpr,P:tl.constexpr,H:tl.constexpr,W:tl.constexpr,TOTAL:tl.constexpr,BLOCK:tl.constexpr):
 j=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=j<TOTAL
 pix=j%N;p=(j//N)%P;b=j//(N*P);row=pix//W;col=pix%W;base=(b*P*2+p*2)*N
 gr=tl.load(DY+base+pix,valid,0);gi=tl.load(DY+base+N+pix,valid,0)
 dr=tl.full((BLOCK,),0,tl.float32);di=tl.full((BLOCK,),0,tl.float32)
 for t in tl.static_range(9):
  dy=(0,-1,-1,-1,0,0,1,1,1)[t];dx=(0,-1,0,1,-1,1,-1,0,1)[t]
  near=pix+dy*W+dx;mask=valid&(row+dy>=0)&(row+dy<H)&(col+dx>=0)&(col+dx<W)
  zr=tl.load(Z+base+near,mask,0);zi=tl.load(Z+base+N+near,mask,0)
  ko=((b*9+t)*P*2+p*2)*N
  tl.store(DK+ko+pix,gr*zr+gi*zi,valid);tl.store(DK+ko+N+pix,-gr*zi+gi*zr,valid)
  prev=pix-dy*W-dx;pmask=valid&(row-dy>=0)&(row-dy<H)&(col-dx>=0)&(col-dx<W)
  yr=tl.load(DY+base+prev,pmask,0);yi=tl.load(DY+base+N+prev,pmask,0)
  ka=tl.load(K+ko+prev,pmask,0);kb=tl.load(K+ko+N+prev,pmask,0)
  dr=dr+ka*yr+kb*yi;di=di-kb*yr+ka*yi
 tl.store(DZ+base+pix,dr,valid);tl.store(DZ+base+N+pix,di,valid)

@triton.jit
def gate_backward(S,T,DZ,DS,DT,N:tl.constexpr,BLOCK:tl.constexpr):
 pos=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);bg=tl.program_id(1);lanes=tl.arange(0,8);r=tl.arange(0,4)
 off=(bg*8+lanes[:,None])*N+pos[None,:]
 st=tl.load(S+off,pos[None,:]<N,0);dz=tl.load(DZ+off,pos[None,:]<N,0)
 toff=(bg*4+r[:,None])*N+pos[None,:]
 strength=tl.sum(tl.load(T+toff,pos[None,:]<N,0),0)*0.25
 energy=tl.sum(st*st,0)*0.25;g=tl.rsqrt(1.+strength*energy);dot=tl.sum(st*dz,0);g3=g*g*g
 ds=g[None,:]*dz-st*(strength*g3*dot*0.25)[None,:]
 dt=-0.125*energy*g3*dot
 tl.store(DS+off,ds,pos[None,:]<N);tl.store(DT+toff,tl.broadcast_to(dt[None,:],(4,BLOCK)),pos[None,:]<N)

class FusedState(torch.autograd.Function):
 @staticmethod
 def forward(ctx,state,drive,kernel,strength,group_size):
  assert group_size==4 and state.is_cuda
  assert all(x.dtype==torch.float32 for x in (state,drive,kernel,strength))
  state,drive,kernel,strength=[x.contiguous() for x in (state,drive,kernel,strength)]
  b,c,h,w=state.shape;n=h*w;p=c//2
  z=torch.empty_like(state);y=torch.empty_like(state)
  gate_forward[(triton.cdiv(n,128),b*c//8)](state,drive,strength,z,n,c//8,128,enable_fp_fusion=False)
  stencil_forward[(triton.cdiv(b*p*n,128),)](z,kernel,y,n,p,h,w,b*p*n,128,enable_fp_fusion=False)
  ctx.save_for_backward(state,kernel,strength,z);ctx.shape=(b,c,h,w)
  return y
 @staticmethod
 def backward(ctx,dy):
  state,kernel,strength,z=ctx.saved_tensors;b,c,h,w=ctx.shape;n=h*w;p=c//2
  dy=dy.contiguous();dz=torch.empty_like(state);dk=torch.empty_like(kernel);ds=torch.empty_like(state);dt=torch.empty_like(strength)
  stencil_backward[(triton.cdiv(b*p*n,128),)](z,kernel,dy,dz,dk,n,p,h,w,b*p*n,128,enable_fp_fusion=False)
  gate_backward[(triton.cdiv(n,128),b*c//8)](state,strength,dz,ds,dt,n,128,enable_fp_fusion=False)
  return ds,dz,dk,dt,None

def state_step(state,drive,kernel,strength,group_size):
 return FusedState.apply(state,drive,kernel,strength,group_size)
