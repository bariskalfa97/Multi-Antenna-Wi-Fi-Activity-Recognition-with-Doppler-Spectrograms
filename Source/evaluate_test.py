# Evaluate a frozen SHARP checkpoint on reserved S1 and external test sessions.
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from architecture import ActivityCNN
from train import DopplerWindows, CLASS_NAMES, ROOT, partition_bounds
from validate_fusion import group_traces, fuse_logits, metrics


def evaluate(model, dataset, device, batch_size):
    logits_all, truth, keys = [], [], []
    for recording, streams in sorted(group_traces(dataset).items()):
        ids = [streams[a] for a in range(4)]
        starts = dataset.starts[ids[0]]
        for offset in range(0, len(starts), batch_size):
            subset = starts[offset:offset+batch_size]
            x = np.stack([np.stack([dataset.arrays[i][s:s+340] for i in ids]) for s in subset])
            with torch.inference_mode():
                logits = model(torch.from_numpy(x).reshape(-1,1,340,100).to(device))
            logits_all.append(logits.reshape(len(subset),4,5).cpu())
            truth.extend([dataset.labels[ids[0]]]*len(subset))
            keys.extend(dict(recording=recording,start=int(s)) for s in subset)
    logits = torch.cat(logits_all)
    y = np.asarray(truth)
    fused, majority = fuse_logits(logits)
    individual = logits.argmax(-1).numpy()
    result = dict(single_antenna=metrics(np.repeat(y,4), individual.reshape(-1)),
                  decision_fusion=metrics(y, fused.numpy()),
                  per_antenna={str(a):metrics(y,individual[:,a]) for a in range(4)},
                  majority_windows=int(majority.sum()), fallback_windows=int((~majority).sum()))
    return result, dict(logits=logits.numpy(), truth=y, individual=individual,
                        fused=fused.numpy(), majority=majority.numpy()), keys


def main():
    parser=argparse.ArgumentParser(description='Evaluate a frozen checkpoint on reserved and external sessions')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true',help='Resume an interrupted output directory with the identical checkpoint')
    parser.add_argument('--batch-size',type=int,default=32,help='Aligned windows per inference batch')
    args=parser.parse_args()
    if args.batch_size<1: parser.error('Batch size must be positive')
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=checkpoint['config']
    if config.get('protocol')!='sharp' or config.get('max_windows_per_trace',0):
        raise ValueError('Requires full-data SHARP checkpoint')
    if config['classes']!=CLASS_NAMES: raise ValueError('Class order mismatch')
    if config['val_sessions']!=config['train_sessions'] or config.get('val_partition')!='val':
        raise ValueError('Unexpected training split')
    if set(config['test_sessions']) & set(config['train_sessions']):
        raise ValueError('External sessions overlap training')
    groups={'S1':(config['train_sessions'],'test')}
    for session in config['test_sessions']:
        name=session[:-1]
        groups.setdefault(name,([], 'full'))[0].append(session)
    root=Path(config['data_root'])
    for sessions,_ in groups.values():
        for session in sessions:
            if not (root/session).is_dir(): raise FileNotFoundError(root/session)
    output=args.output or ROOT/'Results'/'test'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume and args.output is None: parser.error('--resume requires --output')
    output.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model=ActivityCNN(5).to(device);model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    model.requires_grad_(False)
    result=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest,
                checkpoint_epoch=checkpoint['epoch'],classes=CLASS_NAMES,device=str(device),
                config=config,window_rows=340,stride=config['eval_stride'],
                note='Frozen checkpoint. Whole-recording centering matches reference. Overlapping windows are correlated.',
                sets={})
    if args.resume:
        saved=json.loads((output/'comparison.json').read_text())
        if saved['checkpoint_sha256']!=digest or saved['config']!=config:
            raise ValueError('Resume checkpoint/config mismatch')
        result=saved
    for name,(sessions,partition) in groups.items():
        if name in result['sets']:
            for suffix in ['predictions.npz','windows.json','manifest.json']:
                if not (output/f'{name}_{suffix}').is_file(): raise ValueError('Incomplete saved set')
            print('Reusing completed set:',name,flush=True)
            continue
        dataset=DopplerWindows(root,sessions,stride=config['eval_stride'],
            preprocessing=config['preprocessing'],partition=partition,reference_windows=True)
        if name=='S1':
            for row in dataset.manifest:
                _,val_end=partition_bounds(row['shape'][0],'val')
                assert row['first_window_start']>=val_end+31
        res,arrays,keys=evaluate(model,dataset,device,args.batch_size)
        res.update(sessions=sessions,partition=partition,recordings=len(group_traces(dataset)))
        res['per_session']={}
        for session in sessions:
            mask=np.array([k['recording'].startswith(session+'_') for k in keys])
            res['per_session'][session]=dict(
                single_antenna=metrics(np.repeat(arrays['truth'][mask],4),arrays['individual'][mask].reshape(-1)),
                decision_fusion=metrics(arrays['truth'][mask],arrays['fused'][mask]))
        result['sets'][name]=res
        np.savez_compressed(output/f'{name}_predictions.npz',**arrays)
        (output/f'{name}_windows.json').write_text(json.dumps(keys))
        (output/f'{name}_manifest.json').write_text(json.dumps(dataset.manifest,indent=2))
        (output/'comparison.json').write_text(json.dumps(result,indent=2))
        print(f"{name}: {res['decision_fusion']['samples']} windows; single {res['single_antenna']['accuracy']:.4%}; fused {res['decision_fusion']['accuracy']:.4%}; fused F1 {res['decision_fusion']['macro_f1']:.4f}",flush=True)
        del dataset
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Checkpoint changed during evaluation')
    result['checkpoint_unchanged']=True
    (output/'comparison.json').write_text(json.dumps(result,indent=2))
    lines=['# Frozen-checkpoint test results','',f'Checkpoint: {args.checkpoint.resolve()}',
           f'SHA256: {digest}',f'Epoch: {checkpoint["epoch"]}','',
           '| Set | Fused windows | Single accuracy | Fused accuracy | Fused macro F1 |',
           '|---|---:|---:|---:|---:|']
    for name,res in result['sets'].items():
        a,b=res['single_antenna'],res['decision_fusion']
        lines.append(f"| {name} | {b['samples']} | {a['accuracy']:.2%} | {b['accuracy']:.2%} | {b['macro_f1']:.4f} |")
    lines.extend(['','S1 uses the reserved chronological test portions. S2-S7 use complete external sessions.',
                  'All preprocessing and fusion settings are frozen. No retraining or checkpoint selection uses these results.',
                  'Window stride 30 creates correlated overlapping samples. Whole-recording centering follows the reference.'])
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
    for ax,(name,res) in zip(axes.flat,result['sets'].items()):
        cm=np.asarray(res['decision_fusion']['confusion_matrix'])
        pct=cm/np.maximum(cm.sum(1,keepdims=True),1)*100
        ax.imshow(pct,vmin=0,vmax=100,cmap='Blues')
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{pct[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if pct[i,j]>50 else 'black')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,
               title=f"{name}: {res['decision_fusion']['accuracy']:.1%}",xlabel='Predicted',ylabel='True')
        ax.tick_params(axis='x',rotation=45)
    for ax in list(axes.flat)[len(result['sets']):]:ax.axis('off')
    fig.suptitle('Frozen final checkpoint: fused test confusion matrices (% of true class)')
    fig.savefig(output/'test_confusion.png',dpi=140);plt.close(fig)
    print('Saved:',output,flush=True)

if __name__=='__main__': main()
