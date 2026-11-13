"""Generate target-specific peptides with PRISM generator."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from model import AA,START,load
from sampling import sample
from targets import TARGETS,REQUESTS,templates,target_index,known_for_request
from training import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--target',choices=TARGETS,required=True)
    p.add_argument('--request',choices=REQUESTS,default='g1_original50')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026111201)
    p.add_argument('--mode',choices=['selective','unconditional'],default='selective')
    p.add_argument('--temperature',type=float)
    p.add_argument('--per-template',type=int,default=400)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    if args.per_template<1:p.error('--per-template must be positive')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,ck=load(args.checkpoint,args.device)
    _,profiles=templates(args.target,args.request);t=target_index(args.target)
    y=np.repeat(profiles,args.per_template,axis=0);conditional=args.mode=='selective'
    if not conditional:y=np.zeros_like(y)
    temperature=args.temperature if args.temperature is not None else (1.2 if conditional else 1.)
    main_request=args.request in ('observed','counterfactual')
    kernel='subtract_log' if main_request else 'canonical'
    sampling_seed=args.seed+t*10 if main_request else args.seed
    known=known_for_request(args.request);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as stream:
        wr=csv.writer(stream)
        wr.writerow(['sample_id','sequence','template','target','target_index','request','training_seed','replicate','sampling_seed','stopped_normally','raw_length','valid','filter_reason'])
        for lo,tokens,lengths,stopped in sample(model,y,t,conditional,temperature,sampling_seed,args.batch,kernel):
            for j,(row,length,stop) in enumerate(zip(tokens,lengths,stopped)):
                tok=row[:length];seq=''.join(AA[i] if i<20 else '^' if i==START else '$' for i in tok)
                valid=bool(stop and length==10 and np.all(tok<20))
                reason=('no_stop' if not stop else 'special_token' if np.any(tok>=20) else 'length' if length!=10 else 'known_sequence' if seq in known else '')
                wr.writerow([lo+j,seq,(lo+j)//args.per_template,args.target,t,args.request,ck['seed'],args.seed,sampling_seed,bool(stop),int(length),valid,reason])
    meta=dict(model='PRISM generator',checkpoint_sha256=sha(args.checkpoint),training_seed=ck['seed'],target=args.target,target_index=t,target_order=TARGETS,request=args.request,rep=args.seed,sampling_seed=sampling_seed,kernel=kernel,temperature=temperature,batch=args.batch,attempts=len(y),raw_sha256=sha(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Saved {len(y)} {args.target} attempts to {args.output}')
if __name__=='__main__':main()
