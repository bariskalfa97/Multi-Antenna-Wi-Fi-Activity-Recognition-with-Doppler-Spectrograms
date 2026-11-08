"""Inspect saved validation fusion errors; no training or test evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import ArrayReader,CLASS_NAMES


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',type=Path,required=True)
 args=p.parse_args();run=args.run
 meta=json.loads((run/'comparison.json').read_text())
 windows=json.loads((run/'windows.json').read_text())
 data=np.load(run/'predictions.npz',allow_pickle=False)
 assert len(windows)==len(data['truth'])
 assert meta['sessions']==['S1c'], 'This audit is specifically for S1c validation'
 config=json.loads((Path(meta['checkpoint']).parent/'config.json').read_text())
 root=Path(config['data_root']);out=run/'jumping_analysis';out.mkdir(exist_ok=True)
 ids=np.flatnonzero(data['truth']==4)
 details=[];summary={};selected=[]
 for rec in sorted({windows[i]['recording'] for i in ids}):
  ix=np.array([i for i in ids if windows[i]['recording']==rec]);ok=data['fused'][ix]==4;vote=data['majority'][ix]
  summary[rec]=dict(windows=len(ix),correct=int(ok.sum()),predictions=np.bincount(data['fused'][ix],minlength=5).tolist(),
    majority_correct=int((vote&ok).sum()),majority_wrong=int((vote&~ok).sum()),
    fallback_correct=int((~vote&ok).sum()),fallback_wrong=int((~vote&~ok).sum()),
    antenna_correct=(data['individual'][ix]==4).sum(0).tolist())
  # Median chronological example from each group: explicit, reproducible selection.
  for correct in [False,True]:
   candidates=ix[ok==correct]
   if len(candidates):selected.append(int(candidates[len(candidates)//2]))
 for i in ids:
  logits=data['logits'][i];shift=logits-logits.max(1,keepdims=True);prob=np.exp(shift);prob/=prob.sum(1,keepdims=True)
  details.append(dict(index=int(i),**windows[i],approx_start_seconds=windows[i]['start']*.006,
    correct=bool(data['fused'][i]==4),fused=CLASS_NAMES[int(data['fused'][i])],
    rule='majority' if data['majority'][i] else 'logit_sum',
    antenna_predictions=[CLASS_NAMES[int(v)] for v in data['individual'][i]],
    logits=logits.tolist(),probabilities=prob.tolist(),summed_logits=logits.sum(0).tolist()))
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 (out/'jumping_windows.json').write_text(json.dumps(details,indent=2))
 cache={}
 panels=[]
 for i in selected:
  rec=windows[i]['recording'];start=windows[i]['start'];panel=[]
  for antenna in range(4):
   path=root/'S1c'/f'{rec}_stream_{antenna}.txt'
   if path not in cache:
    with path.open('rb') as f:a=ArrayReader(f).load()
    cache[path]=(a-a.mean(axis=0,keepdims=True)).astype(np.float32)
   panel.append(cache[path][start:start+340])
  panels.append(panel)
 # Shared scale across all examples and antennas, centered linear power.
 bound=max(float(np.abs(a).max()) for panel in panels for a in panel)
 fig,axes=plt.subplots(4,len(selected),figsize=(17,10),layout='constrained')
 for col,(i,panel) in enumerate(zip(selected,panels)):
  rec=windows[i]['recording'];start=windows[i]['start']*.006
  for ant,a in enumerate(panel):
   ax=axes[ant,col];im=ax.imshow(a.T,origin='lower',aspect='auto',extent=[start,start+2.04,-.5,99.5],cmap='RdBu_r',vmin=-bound,vmax=bound)
   title=f"Antenna {ant}: {CLASS_NAMES[int(data['individual'][i,ant])]}"
   if ant==0:title=f"{rec} | fused: {CLASS_NAMES[int(data['fused'][i])]}\n"+title
   ax.set_title(title,fontsize=10)
   if col==0:ax.set_ylabel('Doppler bin')
   if ant==3:ax.set_xlabel('Approximate time (s)')
 fig.suptitle('Jumping: failed and correct windows, as seen by the CNN',fontsize=16)
 fig.colorbar(im,ax=axes,label='Linear power minus per-bin recording mean',shrink=.75)
 fig.savefig(out/'jumping_spectrograms.png',dpi=140);plt.close(fig)
 fig,axes=plt.subplots(1,len(selected),figsize=(17,5),layout='constrained')
 for ax,i in zip(axes,selected):
  scores=np.vstack([data['logits'][i],data['logits'][i].sum(0)])
  im=ax.imshow(scores,aspect='auto',cmap='coolwarm')
  for row in range(5):
   for col in range(5):ax.text(col,row,f'{scores[row,col]:.1f}',ha='center',va='center',fontsize=8)
  ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASS_NAMES,yticklabels=['Ant 0','Ant 1','Ant 2','Ant 3','Sum'])
  ax.tick_params(axis='x',rotation=65)
  ax.set_title(f"{windows[i]['recording']} at {windows[i]['start']*.006:.2f}s\nFused: {CLASS_NAMES[int(data['fused'][i])]}\n{'Majority' if data['majority'][i] else 'Logit sum'}",fontsize=10)
 fig.suptitle('Raw class scores: argmax per antenna; sum is used only without a majority',fontsize=14)
 fig.savefig(out/'jumping_scores.png',dpi=140);plt.close(fig)
 print(json.dumps(summary,indent=2));print('Selected examples:',selected);print('Saved:',out)
 for row in details:
  if row['index'] in selected:print(json.dumps(row))

if __name__=='__main__':main()
