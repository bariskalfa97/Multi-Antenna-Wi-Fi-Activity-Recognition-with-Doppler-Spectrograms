"""Check split/window parity against the pinned authors' function and Adam numerics."""
import ast
import json
import pickle
import tempfile
from pathlib import Path
import numpy as np
import torch
from train import DopplerWindows, partition_bounds, ReferenceAdam, ReferenceEpochSampler, ROOT
from validate_fusion import fuse_logits

reference = ROOT/'Results/reproduction_setup/20260930T105538Z/reference/dataset_utility.py'
tree = ast.parse(reference.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='create_windows_antennas')
namespace = {'np': np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(reference), 'exec'), namespace)
checks = []
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)/'S1a'; folder.mkdir()
    rng = np.random.default_rng(7)
    array = rng.random((2400,100))
    for code in ['E','L','W','R','J']:
        for ant in range(4):
            with (folder/f'S1a_{code}_stream_{ant}.txt').open('wb') as f:
                pickle.dump(array, f)
    centered = array-array.mean(axis=0,keepdims=True)
    for partition in ['train','val','test','full']:
        ds=DopplerWindows(Path(tmp), ['S1a'], 30, partition=partition, reference_windows=True)
        a,b=partition_bounds(len(array), partition)
        ref,_=namespace['create_windows_antennas']([np.stack([centered[a:b].T]*4)], [0],340,30)
        np.testing.assert_array_equal(ds.starts[0], np.array(list(range(a,b-340,30))))
        assert len(ref)==len(ds.starts[0])
        for i,window in enumerate(ref):
            np.testing.assert_array_equal(ds[i][0][0].numpy(),window[0].T.astype(np.float32))
        sampler=ReferenceEpochSampler(ds,32)
        indices=list(sampler)
        assert len(indices)%32==0 and len(indices)>=len(ds)
        assert len(set(indices))==len(ds)
        assert indices[:4]==[0,len(ref),2*len(ref),3*len(ref)]
    assert partition_bounds(2400,'val')[0]-partition_bounds(2400,'train')[1]==31
    assert partition_bounds(2400,'test')[0]-partition_bounds(2400,'val')[1]==31
checks.append('Exact feature/window parity with authors function for train, val, test, full; 31-row gaps; repeat batching')
# Independent NumPy calculation includes tiny gradients to exercise epsilon placement.
p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64));optim=ReferenceAdam([p])
x=np.array([1.,2.]);m=np.zeros(2);v=np.zeros(2)
for t,g in enumerate([np.array([1e-8,-.3]),np.array([2e-8,.2])],1):
    p.grad=torch.tensor(g);optim.step()
    m=.9*m+.1*g;v=.999*v+.001*g*g
    x-=1e-4*np.sqrt(1-.999**t)/(1-.9**t)*m/(np.sqrt(v)+1e-7)
    np.testing.assert_allclose(p.detach().numpy(),x,rtol=0,atol=1e-14)
checks.append('Adam matches independent reference update including epsilon placement')
# Exhaust all 5**4 vote patterns; reference branch condition equals >=3 agreement.
import itertools
for votes in itertools.product(range(5),repeat=4):
    logits=torch.nn.functional.one_hot(torch.tensor(votes),5).float()[None]
    pred,majority=fuse_logits(logits)
    labels,counts=np.unique(votes,return_counts=True)
    if len(labels)==1: expected=labels[0]
    elif len(labels)>2 or sorted(counts)[-1]==sorted(counts)[-2]: expected=logits.sum(1).argmax().item()
    else: expected=labels[counts.argmax()]
    assert pred.item()==expected
checks.append('Fusion agrees with authors branch logic for all 625 vote patterns')
print(json.dumps({'passed':checks},indent=2))
