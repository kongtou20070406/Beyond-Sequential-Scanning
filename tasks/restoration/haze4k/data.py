from common import *
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
class HazePairs(Dataset):
 def __init__(self,split='train',start=0,steps=50000,crop=256):
  manifest=json.loads((Q/'DATA_MANIFEST.json').read_text('utf-8'))
  self.rows=manifest['splits'][split];self.path=str(CACHE/f'{split}.bin');self.split=split
  self.start=start*2;self.count=(steps-start)*2;self.crop=crop;self.map=None;self.orders={}
 def __len__(self):return self.count
 def pair(self,index):
  if self.map is None:self.map=np.fromfile(self.path,dtype=np.uint8)
  row=self.rows[index];images=[]
  for role in ('input','target'):
   a=row['arrays'][role];images.append(self.map[a['offset']:a['offset']+a['bytes']].reshape(a['shape']))
  return images,row['id']
 def __getitem__(self,index):
  position=self.start+index;epoch,slot=divmod(position,len(self.rows))
  if epoch not in self.orders:self.orders={epoch:np.random.default_rng(42+epoch).permutation(len(self.rows))}
  which=int(self.orders[epoch][slot]);(x,y),name=self.pair(which)
  rng=np.random.default_rng(np.random.SeedSequence([42,epoch,slot]))
  h,w=x.shape[:2];p=self.crop
  if h<p or w<p:
   padding=((0,max(0,p-h)),(0,max(0,p-w)),(0,0));x=np.pad(x,padding,mode='reflect');y=np.pad(y,padding,mode='reflect')
  top=int(rng.integers(0,x.shape[0]-p+1));left=int(rng.integers(0,x.shape[1]-p+1));x=x[top:top+p,left:left+p];y=y[top:top+p,left:left+p]
  if rng.random()<.5:x=x[:,::-1];y=y[:,::-1]
  if rng.random()<.5:x=x[::-1];y=y[::-1]
  k=int(rng.integers(4));x=np.rot90(x,k);y=np.rot90(y,k)
  return {'input':torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()/255,'target':torch.from_numpy(np.ascontiguousarray(y.transpose(2,0,1))).float()/255,'id':name,'position':position}
def loader(start,steps,workers=0):
 return DataLoader(HazePairs(start=start,steps=steps),batch_size=2,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
