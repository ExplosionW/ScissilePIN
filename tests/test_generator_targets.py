"""End-to-end target identities and equal-task DPO on the released package."""
import json,hashlib,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'generator'))
from targets import TARGETS,REQUESTS,templates,target_index,pool_identity
from scoring import rank_profiles
from sampling import sample
from model import load,configure_dpo
from train import task_indices,mean_task_gradient,load_pairs,load_training,pair_forward,reference_probs,dpo_loss

class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_all_request_targets_and_scoring_columns(self):
        fit=np.load(R/'generator/data/fit.npz');fitseq=set(fit['sequences'])
        seq=np.array(['AAAAAAAAAA','CCCCCCCCCC','DDDDDDDDDD'])
        for name in TARGETS:
            t=target_index(name)
            for req in REQUESTS:
                item,y=templates(name,req)
                self.assertEqual(y.shape,(50,18))
                if 'sequences' in item:self.assertTrue(set(item['sequences'])<=fitseq)
            p=np.zeros((3,18),np.float32);p[0,t]=2;p[1,t]=3;p[1,(t+1)%18]=4;p[2,t]=.9
            v=rank_profiles(seq,p,name,TARGETS)
            self.assertEqual(v['top'][0],0);self.assertEqual(v['joint'].tolist(),[True,False,False])
            with self.assertRaises(ValueError):rank_profiles(seq,p,name,TARGETS[::-1])
            self.assertEqual(pool_identity(pd.DataFrame({'target':[name],'target_index':[t],'request':['observed']}))['target'],name)
        with self.assertRaises(ValueError):pool_identity(pd.DataFrame({'target':['MMP1'],'target_index':[4],'request':['observed']}))
    def test_all_checkpoints_and_samplers(self):
        release=json.loads((R/'generator/configs/release.json').read_text())
        for seed in range(3):
            path=R/f'generator/checkpoints/generator_SEED{seed}.pt'
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),release['checkpoints'][path.name]['sha256'])
            m,ck=load(path)
            self.assertEqual((ck['seed'],ck['arm'],ck['architecture']),(seed,'standard','original'))
            for name in TARGETS:
                _,y=templates(name,'g1_original50')
                for kernel in ('canonical','subtract_log'):
                    batches=list(sample(m,y[:1],target_index(name),True,1.2,42,1,kernel))
                    self.assertEqual(batches[0][1].shape,(1,15))
    def test_equal_task_weighting_not_example_weighting(self):
        m=torch.nn.Linear(1,1,bias=False);groups=[torch.arange(1+t%3) for t in range(18)]
        def loss(i):return ((m(torch.ones(len(i),1))-len(i))**2).mean()
        actual,_=mean_task_gradient(m,loss,groups)
        expected=torch.autograd.grad(sum(loss(i) for i in groups)/18,m.weight)[0].flatten()
        torch.testing.assert_close(actual,expected)
    def test_training_pairs_and_update_all_seeds(self):
        for seed in range(3):
            rows,c,r,y,t,groups=load_pairs(seed,'cpu')
            self.assertEqual(set(rows.target),set(range(18)))
            self.assertFalse(set(rows.loc[rows.split=='train','chosen']) & set(rows.loc[rows.split!='train','chosen']))
            m,_=load_training(R/f'generator/initialization/adapted_SEED{seed}.pt','cpu');configure_dpo(m)
            # One real pair for each task; variable tasks must flow into both log probabilities.
            ix=torch.tensor([g[0] for g in groups]);c,r,y,t=c[ix],r[ix],y[ix],t[ix]
            refs=reference_probs(m,c,r,y,t);batch=[torch.tensor([i]) for i in range(18)]
            before={n:p.detach().clone() for n,p in m.named_parameters()}
            gradient,_=mean_task_gradient(m,lambda i:dpo_loss(*pair_forward(m,c[i],r[i],y[i],t[i]),refs[i]),batch)
            from train import assign
            opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-6,eps=1e-7)
            assign(m,gradient);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            self.assertTrue(any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if p.requires_grad))
            for n,p in m.named_parameters():
                if not p.requires_grad:self.assertTrue(torch.equal(p,before[n]))
            ids=task_indices(groups,seed,1,'cpu');self.assertEqual(sum(map(len,ids)),128)
            self.assertEqual([len(x) for x in ids],[8,8]+[7]*16)

if __name__=='__main__':unittest.main()
