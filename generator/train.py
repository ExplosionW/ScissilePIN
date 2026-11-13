"""Reproduce the 18-target adaptation and standard-DPO training stages."""
import argparse,json,os,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import AA,STOP,configure_dpo
from training import TrainingMixture,model_state,nll,sha,save,write_json
from targets import ROOT,TARGETS


def load_training(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    m=TrainingMixture();m.load_state_dict(ck['model'],strict=True)
    return m.to(device).eval(),ck

def reset(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def logps(model,x,y,t):
    flags=torch.ones(len(x),device=x.device,dtype=torch.bool)
    labels=torch.cat([x,torch.full((len(x),1),STOP,device=x.device,dtype=torch.long)],1)
    return F.log_softmax(model(x,y,t,flags),-1).gather(-1,labels[:,:,None]).squeeze(-1).sum(1)

def pair_forward(model,c,r,y,t):
    values=logps(model,torch.cat([c,r]),torch.cat([y,y]),torch.cat([t,t]))
    return values.chunk(2)

def dpo_loss(a,b,ref):return F.softplus(-.1*((a-b)-(ref[:,0]-ref[:,1]))).mean()

@torch.no_grad()
def reference_probs(model,c,r,y,t):
    out=[]
    for lo in range(0,len(c),128):
        a,b=pair_forward(model,c[lo:lo+128],r[lo:lo+128],y[lo:lo+128],t[lo:lo+128])
        out.append(torch.stack([a,b],1))
    return torch.cat(out)

def load_pairs(seed,device):
    rows=pd.read_csv(ROOT/f'data/preferences_SEED{seed}.csv')
    request=json.loads((ROOT/'data/requests_18.json').read_text())['observed']
    lookup={(t,int(ix)):c for t,tmp in enumerate(request) for ix,c in zip(tmp['fit_indices'],tmp['conditions'])}
    encode=lambda col:torch.tensor([[AA.index(a) for a in s] for s in rows[col]],device=device)
    y=torch.tensor(np.asarray([lookup[int(r.target),int(r.fit_index)] for r in rows.itertuples()],np.float32),device=device)
    t=torch.tensor(rows.target.to_numpy(),device=device)
    groups=[np.flatnonzero((rows.target==j)&(rows.split=='train')) for j in range(18)]
    if not all(len(g)>0 for g in groups):raise ValueError('Every target needs training pairs.')
    return rows,encode('chosen'),encode('rejected'),y,t,groups

def task_indices(groups,seed,step,device):
    rng=np.random.default_rng(2026110810+seed*1000+step)
    counts=np.bincount((np.arange(128)+(step-1)*128)%18,minlength=18)
    return [torch.tensor(rng.choice(g,size=int(counts[t]),replace=True),device=device) for t,g in enumerate(groups)]

def mean_task_gradient(model,loss_fn,indices,rank=0,world=1):
    params=[p for p in model.parameters() if p.requires_grad]
    g=torch.zeros((18,sum(p.numel() for p in params)),device=params[0].device)
    losses=torch.zeros(18,device=params[0].device)
    for t in range(rank,18,world):
        loss=loss_fn(indices[t]);losses[t]=loss.detach()
        derivatives=torch.autograd.grad(loss,params,allow_unused=True)
        offset=0
        for p,d in zip(params,derivatives):
            if d is not None:g[t,offset:offset+p.numel()]=d.detach().flatten()
            offset+=p.numel()
    if world>1:dist.all_reduce(g);dist.all_reduce(losses)
    if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite gradient')
    return g.mean(0),losses

def assign(model,g):
    offset=0
    for p in model.parameters():
        if p.requires_grad:
            p.grad=g[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
    assert offset==len(g)

class CE(nn.Module):
    def __init__(self,model):super().__init__();self.net=model
    def forward(self,x,y,t,flags,positive,fraction):
        _,a,b=self.net.components(x,y,t,flags)
        return .5*(nll(a,x)+nll(b,x))*(1+(flags&positive).float())/(1+.5*fraction)

def ce_optimizer(model):
    groups=[[],[],[]]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        i=0 if '.student.' in name and not any(k in name for k in ['.channel_mod.','.rel_']) else 1 if name.startswith('experts.') and not any(k in name for k in ['.channel_mod.','.rel_']) else 2
        groups[i].append(p)
    return torch.optim.Adam([dict(params=g,lr=lr) for g,lr in zip(groups,[5e-6,5e-5,5e-4])],eps=1e-7)

def arrays(device,split):
    z=np.load(ROOT/f'data/{split}.npz');labels=z['labels']
    return dict(x=torch.tensor(z['tokens'],device=device,dtype=torch.long),y=torch.tensor(z['conditions'],device=device),target=torch.tensor(labels.argmax(1),device=device),pos=torch.tensor(np.stack([(labels[:,t]>1)&(labels[:,t]>np.delete(labels,t,1).max(1)) for t in range(18)],1),device=device))

@torch.no_grad()
def evaluate(model,data,rank=0,world=1):
    model.eval();rows=[];dev=next(model.parameters()).device
    for t in range(rank,18,world):
        sums=np.zeros(4);selective=0
        for lo in range(0,len(data['x']),256):
            x=data['x'][lo:lo+256];y=data['y'][lo:lo+256]
            ids=torch.full((len(x),),t,device=dev,dtype=torch.long);flags=torch.ones(len(x),device=dev,dtype=torch.bool)
            a=nll(model(x,y,ids,flags),x);b=nll(model(x,y,ids,~flags),x);mask=data['pos'][lo:lo+len(x),t]
            sums+=np.array([float(a.sum()),float(b.sum()),float(a[mask].sum()),len(x)]);selective+=int(mask.sum())
        rows.append(dict(target=t,target_name=TARGETS[t],cond_nll=sums[0]/sums[3],uncond_nll=sums[1]/sums[3],joint_nll=(sums[0]+sums[1])/(2*sums[3]),selective_nll=sums[2]/selective if selective else None,selective_count=selective))
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,rows);rows=sum(gathered,[])
    rows.sort(key=lambda x:x['target'])
    return dict(by_target=rows,joint_nll=float(np.mean([r['joint_nll'] for r in rows])))

def rng_state():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,np=np.random.get_state(),python=random.getstate())
def restore_rng(s):
    torch.set_rng_state(s['cpu']);np.random.set_state(s['np']);random.setstate(s['python'])
    if s['cuda'] is not None:torch.cuda.set_rng_state(s['cuda'])
def barrier(world):
    if world>1:dist.barrier()
def rank_rng(world):
    values=[None]*world
    if world>1:dist.all_gather_object(values,rng_state())
    else:values[0]=rng_state()
    return values

def run(args,rank,world,device):
    initial=args.initialization or ROOT/f'initialization/{"seed"+str(args.seed) if args.stage=="adapt" else "adapted_SEED"+str(args.seed)}.pt'
    reset(2026112700+1000*args.seed if args.stage=='adapt' else 2026110800+1000*args.seed)
    model,original=load_training(initial,device)
    if original['seed']!=args.seed:raise ValueError('Initialization seed mismatch')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest=dict(seed=args.seed,stage=args.stage,world_size=world,initialization_sha256=sha(initial),code_sha256=sha(Path(__file__)),model_sha256=sha(ROOT/'model.py'),torch=str(torch.__version__),device=device.type,data={name:sha(ROOT/'data'/name) for name in ['fit.npz','development.npz','requests_18.json',f'preferences_SEED{args.seed}.csv']})
    if rank==0:
        if (out/'config.json').exists():
            if not args.resume or json.loads((out/'config.json').read_text())!=manifest:raise ValueError('Use a new output directory or matching --resume')
        else:
            if any(out.iterdir()):raise FileExistsError('Output directory is not empty')
            write_json(out/'config.json',manifest)
    barrier(world)
    if (out/'done.json').exists():
        d=json.loads((out/'done.json').read_text());assert sha(out/d['file'])==d['sha256'];return
    development=arrays(device,'development');history=[];start=1
    def payload(step):return dict(model=model_state(model),model_name='PRISM generator',arm='standard',architecture='original',mixture_mode='uniform',seed=args.seed,stage=args.stage,epoch=step,world_size=world)
    if args.stage=='adapt':
        # The extra source load preserves the training construction sequence.
        source,_=load_training(initial,device);del source
        opt=ce_optimizer(model);reset(2026112700+1000*args.seed)
        val=evaluate(model,development,rank,world);best=val['joint_nll'];best_epoch=0
        if rank==0 and not args.resume:save(out/'best.pt',payload(0))
        wrap=CE(model)
        if world>1:wrap=DDP(wrap,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=True)
        fit=arrays(device,'fit');n=len(fit['x']);fraction=float(fit['pos'][torch.arange(n,device=device),fit['target']].float().mean())
        end=60
    else:
        _,c,r,y,t,groups=load_pairs(args.seed,device)
        refs=reference_probs(model,c,r,y,t);configure_dpo(model)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-6,eps=1e-7)
        reset(2026110810+1000*args.seed);end=1000
    if args.resume and (out/'resume.pt').exists():
        ck=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rank_rng'][rank]);start=ck['epoch']+1;history=ck['history']
        if args.stage=='adapt':best=ck['best'];best_epoch=ck['best_epoch']
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    for step in range(start,end+1):
        if args.stage=='adapt':
            model.train();rng=np.random.default_rng(2026112700+1000*args.seed+step);order=rng.permutation(n);flags=rng.random(n)<.5;losses=[]
            for lo in range(0,n,128):
                ix=torch.tensor(order[lo:lo+128],device=device);f=torch.tensor(flags[lo:lo+128],device=device)
                assert len(ix)%world==0
                size=len(ix)//world;local=slice(rank*size,(rank+1)*size);j=ix[local];tt=fit['target'][j]
                opt.zero_grad(set_to_none=True);loss=wrap(fit['x'][j],fit['y'][j],tt,f[local],fit['pos'][j,tt],fraction).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();opt.step();losses.append(float(loss.detach()))
            val=evaluate(model,development,rank,world)
            if val['joint_nll']<best:
                best=val['joint_nll'];best_epoch=step
                if rank==0:save(out/'best.pt',payload(step))
            record=dict(epoch=step,loss=float(np.mean(losses)),validation=val,best_epoch=best_epoch)
        else:
            model.eval();ix=task_indices(groups,args.seed,step,device)
            fn=lambda j:dpo_loss(*pair_forward(model,c[j],r[j],y[j],t[j]),refs[j])
            g,losses=mean_task_gradient(model,fn,ix,rank,world);opt.zero_grad(set_to_none=True);assign(model,g)
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
            opt.step();record=dict(step=step,mean_task_loss=float(losses.mean()),gradient_norm=float(norm))
        if args.stage=='adapt' or step%25==0:
            for name,p in model.named_parameters():
                if name in frozen and not torch.equal(p,frozen[name]):raise RuntimeError('Frozen parameter changed')
            history.append(record);rngs=rank_rng(world)
            if rank==0:
                ck=payload(step);ck.update(optimizer=opt.state_dict(),rank_rng=rngs,history=history)
                if args.stage=='adapt':ck.update(best=best,best_epoch=best_epoch)
                save(out/'resume.pt',ck);write_json(out/'history.json',history);print(json.dumps(record),flush=True)
            barrier(world)
    if rank==0:
        name='last.pt' if args.stage=='adapt' else f'generator_SEED{args.seed}.pt'
        save(out/name,payload(end));write_json(out/'done.json',dict(file=name,sha256=sha(out/name),seed=args.seed,stage=args.stage,updates=end))
    barrier(world)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['adapt','dpo'],default='dpo');p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path,required=True);p.add_argument('--initialization',type=Path);p.add_argument('--resume',action='store_true')
    args=p.parse_args();world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world not in (1,2):p.error('Use one or two processes')
    device=torch.device(f'cuda:{local}' if args.device=='cuda' else args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:
        if device.type!='cuda':p.error('Distributed training requires CUDA')
        dist.init_process_group('nccl')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:run(args,rank,world,device)
    finally:
        if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
