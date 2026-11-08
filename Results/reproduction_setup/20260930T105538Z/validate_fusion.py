"""Compare single-antenna and four-antenna decision fusion on validation only.

No training or test-session loading. Uses the checkpoint's preprocessing and
validation sessions. Windows are paired by recording and exact start index.
Three or four agreeing argmax labels win; otherwise sum raw logits and argmax.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from architecture import ActivityCNN
from train import DopplerWindows, CLASS_NAMES, ROOT


def fuse_logits(logits):
    """logits: [windows, 4 antennas, 5 classes]. Return labels and vote mask."""
    if logits.ndim != 3 or logits.shape[1:] != (4, 5):
        raise ValueError('Expected [windows, 4, 5] logits')
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite logits')
    votes = logits.argmax(dim=-1)
    counts = torch.nn.functional.one_hot(votes, num_classes=5).sum(dim=1)
    agreement, winner = counts.max(dim=-1)
    majority = agreement >= 3
    combined = logits.sum(dim=1).argmax(dim=-1)
    return torch.where(majority, winner, combined), majority


def metrics(truth, prediction):
    cm = np.bincount(truth*5 + prediction, minlength=25).reshape(5,5)
    tp = np.diag(cm)
    support = cm.sum(1)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2*tp / np.maximum(cm.sum(0)+support, 1)
    return dict(samples=int(cm.sum()), accuracy=float(tp.sum()/cm.sum()),
                macro_f1=float(f1.mean()), confusion_matrix=cm.tolist(),
                per_class={name:dict(precision=float(precision[i]), recall=float(recall[i]),
                                    f1=float(f1[i]), support=int(support[i]))
                           for i,name in enumerate(CLASS_NAMES)})


def group_traces(dataset):
    groups = defaultdict(dict)
    for i, row in enumerate(dataset.manifest):
        key = row['recording']
        antenna = row['antenna']
        if antenna in groups[key]:
            raise ValueError(f'Duplicate stream {key}/{antenna}')
        groups[key][antenna] = i
    for key, streams in groups.items():
        if set(streams) != {0,1,2,3}:
            raise ValueError(f'{key}: expected antenna IDs 0,1,2,3')
        ids = [streams[a] for a in range(4)]
        if len({dataset.arrays[i].shape for i in ids}) != 1:
            raise ValueError(f'{key}: antenna trace shapes do not match')
        if len({dataset.labels[i] for i in ids}) != 1:
            raise ValueError(f'{key}: inconsistent labels')
        if any(not np.array_equal(dataset.starts[ids[0]], dataset.starts[i]) for i in ids):
            raise ValueError(f'{key}: window starts do not match')
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch-size', type=int, default=32, help='Number of aligned windows, each with 4 streams')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config['classes'] != CLASS_NAMES:
        raise ValueError('Checkpoint class order does not match loader')
    if config.get('max_windows_per_trace', 0):
        raise ValueError('Use a full-data checkpoint, not a smoke-test run')
    preprocessing = config.get('preprocessing')
    if preprocessing == 'stored linear normalized power, float32':
        preprocessing = 'raw'
    if preprocessing not in {'raw','trace_mean'}:
        raise ValueError('Unknown checkpoint preprocessing')
    sessions = config['val_sessions']
    if set(sessions) & (set(config['train_sessions']) | set(config['test_sessions'])):
        raise ValueError('Validation sessions overlap training/test')
    data_root = args.data_root or Path(config['data_root'])
    dataset = DopplerWindows(data_root, sessions, stride=340, preprocessing=preprocessing)
    groups = group_traces(dataset)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    model = ActivityCNN(5).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_logits, all_labels, window_keys = [], [], []
    with torch.inference_mode():
        for recording, streams in sorted(groups.items()):
            ids = [streams[a] for a in range(4)]
            starts = dataset.starts[ids[0]]
            for offset in range(0,len(starts),args.batch_size):
                subset = starts[offset:offset+args.batch_size]
                batch = np.stack([np.stack([dataset.arrays[i][s:s+340] for i in ids]) for s in subset])
                x = torch.from_numpy(batch).reshape(-1,1,340,100).to(device)
                logits = model(x).reshape(len(subset),4,5).cpu()
                all_logits.append(logits)
                all_labels.extend([dataset.labels[ids[0]]]*len(subset))
                window_keys.extend([dict(recording=recording,start=int(s)) for s in subset])
    logits = torch.cat(all_logits)
    truth = np.array(all_labels)
    individual = logits.argmax(-1).numpy()
    fused, majority = fuse_logits(logits)
    result = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_epoch=checkpoint['epoch'],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        sessions=sessions, preprocessing=preprocessing, classes=CLASS_NAMES,
        window_rows=340, stride=340, recordings=len(groups), aligned_windows=len(truth),
        rule='at least 3 of 4 argmax labels agree; otherwise argmax of summed raw logits',
        majority_windows=int(majority.sum()), fallback_windows=int((~majority).sum()),
        single_antenna=metrics(np.repeat(truth,4),individual.reshape(-1)),
        per_antenna={str(a):metrics(truth,individual[:,a]) for a in range(4)},
        decision_fusion=metrics(truth,fused.numpy()))
    expected = checkpoint.get('val_metrics')
    if expected and expected.get('confusion_matrix') != result['single_antenna']['confusion_matrix']:
        raise RuntimeError('Single-antenna confusion matrix does not reproduce the saved validation checkpoint')
    result['saved_validation_reproduced'] = bool(expected)
    output = args.output or ROOT/'Results'/'fusion'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(result,indent=2))
    (output/'windows.json').write_text(json.dumps(window_keys,indent=2))
    np.savez_compressed(output/'predictions.npz',logits=logits.numpy(),truth=truth,
                        individual=individual,fused=fused.numpy(),majority=majority.numpy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,key,title in zip(axes,['single_antenna','decision_fusion'],['Individual antennas','Decision fusion']):
        cm=np.asarray(result[key]['confusion_matrix'])
        percentages=100*cm/np.maximum(cm.sum(1,keepdims=True),1)
        im=ax.imshow(percentages,vmin=0,vmax=100,cmap='Blues')
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{percentages[i,j]:.1f}%',ha='center',va='center',color='white' if percentages[i,j]>50 else 'black')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,
               xlabel='Predicted class',ylabel='True class',title=f"{title}: accuracy {result[key]['accuracy']:.1%}")
        ax.tick_params(axis='x',rotation=35)
    fig.colorbar(im,ax=axes,label='Percentage of true-class samples')
    fig.savefig(output/'confusion_comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps(result,indent=2))
    print('Saved:',output)


if __name__ == '__main__':
    main()
