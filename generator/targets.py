"""Shared target, request and scoring identities for PRISM generator."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
TARGETS=json.loads((ROOT/'data/target_order.json').read_text())
REQUESTS=['g1_original50','fit_paper','observed','counterfactual']
def target_index(name):
    if name not in TARGETS:raise ValueError(f'Unknown target {name}; choose one of {TARGETS}')
    return TARGETS.index(name)
def templates(name,request):
    t=target_index(name)
    if request in ('observed','counterfactual'):
        item=json.loads((ROOT/'data/requests_18.json').read_text())[request][t]
        assert item['name']==name
    elif request in ('g1_original50','fit_paper'):
        item=json.loads((ROOT/'data/templates_18.json').read_text())[name][request]
    else:raise ValueError('Unknown request')
    y=np.asarray(item['conditions'],np.float32)
    if y.shape!=(50,18) or not np.isfinite(y).all():raise ValueError('Expected 50 finite profiles in the released target order.')
    return item,y
def known_for_request(request):
    if request not in REQUESTS:raise ValueError('Unknown request')
    name='excluded_main.txt' if request in ('observed','counterfactual') else 'excluded_supplement.txt'
    return set((ROOT/'data'/name).read_text().splitlines())
def pool_identity(frame):
    result={}
    for key in ('target','target_index','request'):
        if key not in frame or frame[key].nunique()!=1:raise ValueError(f'Pool must contain exactly one {key}')
        result[key]=frame[key].iloc[0]
    if int(result['target_index'])!=target_index(result['target']) or result['request'] not in REQUESTS:raise ValueError('Pool target, column index and request disagree.')
    return result
