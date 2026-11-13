"""Score generated sequences with the released three-seed ScissilePIN predictor ensemble."""
from pathlib import Path
import sys,argparse,subprocess,json,re
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'predictor'));from predict import load,predict

sys.path.append(str(R/'generator'))
from scoring import rank_profiles
from targets import pool_identity,known_for_request

def main():
    p=argparse.ArgumentParser();p.add_argument('--pools',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--precision',choices=['bf16','fp32'],default='bf16');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];files=sorted(a.pools.glob('*_generator*_seed*_sample*_T*_*.csv'))
    if not files:raise ValueError('No ScissilePIN pools found; run generator or pass a pool directory')
    for file in files:
        raw=pd.read_csv(file,keep_default_na=False);identity=pool_identity(raw);target=identity['target'];known=known_for_request(identity['request']);attempts=len(raw);d=raw;d=d[d.stopped_normally.astype(str).str.lower().eq('true')&d.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10}')&~d.sequence.isin(known)].drop_duplicates('sequence');seq=d.sequence.tolist()
        if not seq:raise ValueError(f'No novel canonical peptides in {file.name}')
        work=a.output/file.stem;work.mkdir(exist_ok=True);inp=work/'peptides.csv';pd.DataFrame({'sequence':seq}).to_csv(inp,index=False);feat=work/'features.npz'
        if not feat.exists():subprocess.run([sys.executable,str(R/'predictor/extract_features.py'),'--input',str(inp),'--output',str(feat),'--device',a.device,'--precision',a.precision],check=True)
        z=np.load(feat);assert z['sequences'].tolist()==seq
        all_pred=[]
        for seed in range(3):
            net,targets=load(R/f'predictor/checkpoints/seed{seed}.pt',a.device);all_pred.append(predict(net,seq,z['esm33'],a.device));del net
        mean=np.stack(all_pred).mean(0);result=rank_profiles(seq,mean,target,targets)
        t=result['target_index'];activity=result['activity'];margin=result['margin'];selectivity=result['selectivity'];joint=result['joint'];top=result['top'];top24=top[:24]
        off=np.delete(mean,t,axis=1)
        np.savez_compressed(work/'predictions.npz',sequences=seq,targets=targets,requested_target=target,requested_target_index=t,predictions=np.stack(all_pred),mean=mean)
        match=re.search(r'_seed(\d+)_sample(\d+)_T([\d.]+)_(\w+)$',file.stem)
        if not match:raise ValueError(f'Unexpected pool name: {file.stem}')
        selected=pd.DataFrame({'rank':np.arange(1,len(top)+1),'sequence':np.asarray(seq)[top],'target':target,'target_index':t,'target_activity':activity[top],'max_competitor':off.max(1)[top],'bottleneck':np.minimum(activity-1,margin)[top],'predicted_joint':joint[top]})
        selected.to_csv(work/'selected_top100.csv',index=False)
        rows.append(dict(pool=file.stem,target=target,target_index=t,request=identity['request'],seed=int(match[1]),replicate=int(match[2]),temperature=float(match[3]),mode=match[4],scorer='ScissilePIN ensemble (seeds 0,1,2)',attempts=attempts,unique_novel=len(seq),target_predicted_activity_mean=float(activity.mean()),mean_contrast_selectivity_mean=float(selectivity.mean()),predicted_target_leads_fraction=float((margin>0).mean()),predicted_joint_unique=int(joint.sum()),predicted_joint_per_attempt=float(joint.sum()/attempts),top24_n=len(top24),top100_n=len(top),top24_joint_fraction=float(joint[top24].mean()),top100_joint_fraction=float(joint[top].mean()),top100_selectivity_mean=float(selectivity[top].mean())))
        print(json.dumps(rows[-1]),flush=True)
    table=pd.DataFrame(rows);table.to_csv(a.output/'predicted_function.csv',index=False)
    metrics=[c for c in table.select_dtypes(include='number').columns if c not in ['seed','replicate','temperature','target_index']]
    byseed=table.groupby(['target','request','mode','temperature','seed'])[metrics].mean();byseed.to_csv(a.output/'predicted_function_by_seed.csv')
    byseed.groupby(['target','request','mode','temperature'])[metrics].agg(['mean','std']).to_csv(a.output/'predicted_function_summary.csv')
if __name__=='__main__':main()
