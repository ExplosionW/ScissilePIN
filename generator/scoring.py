"""Target-aware ranking of saved predictor profiles."""
import numpy as np
from targets import TARGETS,target_index
from selection_rules import select

def rank_profiles(sequences,profiles,target,order):
    if list(order)!=TARGETS:raise ValueError('Predictor enzyme order differs from generator target_order.json')
    t=target_index(target);p=np.asarray(profiles)
    if p.shape!=(len(sequences),18) or not np.isfinite(p).all():raise ValueError('Expected finite N x 18 predictions')
    activity=p[:,t];off=np.delete(p,t,axis=1);margin=activity-off.max(1)
    return dict(target_index=t,activity=activity,max17=off.max(1),selectivity=activity-off.mean(1),margin=margin,joint=(activity>1)&(margin>0),top=select(sequences,p,100,t))
