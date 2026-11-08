"""Train ActivityCNN on extracted numerical Doppler traces.

Run from any directory: python3 /home/ubuntu/Deep\ Learning/Source/train.py
Defaults: train S1a,S1b; validate S1c; test S7a. This is an explicit
session-held-out protocol, not an exact reproduction of SHARP's 60/20/20 split.
Each antenna is an independent sample; reported metrics are BEFORE decision
fusion. Default input is linear power minus its per-bin full-trace mean.
This matches SHARP offline preprocessing; it uses future samples within each
recording and is not a causal streaming transform. Test evaluation is opt-in.
Only E/L/W/R/J activities are used; repetition suffixes such as J2 are retained
as separate recordings. Other activity codes are reported and excluded.
"""
import argparse
import bisect
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import random
import re
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from architecture import ActivityCNN

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ['empty', 'sitting', 'walking', 'running', 'jumping']
CODES = ['E', 'L', 'W', 'R', 'J']
PATTERN = re.compile(r'(?P<session>S\d+[a-z])_(?P<activity>[A-Z]+)(?P<repeat>\d*)_stream_(?P<antenna>\d+)\.txt')


class ArrayReader(pickle.Unpickler):
    """Limit legacy dataset deserialization to its NumPy array types."""
    def find_class(self, module, name):
        core = np._core if hasattr(np, '_core') else np.core
        allowed = {
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): core.multiarray._reconstruct,
        }
        if (module, name) not in allowed:
            raise ValueError(f'Unexpected serialized type: {module}.{name}')
        return allowed[module, name]


class DopplerWindows(Dataset):
    """Keep each trace once in RAM and slice windows on demand."""
    def __init__(self, root, sessions, stride, limit=0, preprocessing="trace_mean"):
        self.arrays, self.labels, self.starts, self.ends = [], [], [], []
        self.manifest, self.ignored = [], Counter()
        total = 0
        for session in sessions:
            folder = root / session
            if not folder.is_dir():
                raise FileNotFoundError(f'Missing session: {folder}')
            before = len(self.arrays)
            for path in sorted(folder.glob('*.txt')):
                match = PATTERN.fullmatch(path.name)
                if match is None or match['session'] != session:
                    raise ValueError(f'Unrecognized filename: {path}')
                code = match['activity']
                if code not in CODES:
                    self.ignored[code] += 1
                    continue
                with path.open('rb') as f:
                    array = ArrayReader(f).load()
                if (not isinstance(array, np.ndarray) or array.ndim != 2
                        or array.shape[1] != 100 or array.shape[0] < 340
                        or array.dtype.kind != 'f'):
                    raise ValueError(f'Invalid trace shape/type: {path}')
                if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1.00001:
                    raise ValueError(f'Expected finite normalized powers in [0,1]: {path}')
                if preprocessing == 'trace_mean':
                    # Reference: subtract temporal mean separately for each bin
                    # and antenna, BEFORE extracting windows. Keep signed values.
                    array = array - np.mean(array, axis=0, keepdims=True)
                elif preprocessing != 'raw':
                    raise ValueError(f'Unknown preprocessing: {preprocessing}')
                array = np.asarray(array, dtype=np.float32, order='C')
                starts = np.arange(0, len(array) - 340 + 1, stride)
                if limit and len(starts) > limit:
                    starts = starts[np.linspace(0, len(starts)-1, limit, dtype=int)]
                label = CODES.index(code)
                self.arrays.append(array)
                self.labels.append(label)
                self.starts.append(starts)
                total += len(starts)
                self.ends.append(total)
                self.manifest.append(dict(path=str(path.resolve()), session=session,
                    recording=f"{session}_{code}{match['repeat']}",
                    antenna=int(match['antenna']), label=label,
                    shape=list(array.shape), windows=len(starts)))
            if len(self.arrays) == before:
                raise ValueError(f'No baseline traces in {folder}')
        missing = set(range(len(CODES))) - set(self.labels)
        if missing:
            raise ValueError(f'Split missing classes: {[CLASS_NAMES[i] for i in missing]}')

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        trace = bisect.bisect_right(self.ends, index)
        previous = self.ends[trace-1] if trace else 0
        start = self.starts[trace][index - previous]
        return torch.from_numpy(self.arrays[trace][start:start+340]).unsqueeze(0), self.labels[trace]


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    confusion = torch.zeros(5, 5, dtype=torch.long)
    total_loss, count = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            predictions = logits.detach().argmax(1)
            confusion += torch.bincount((targets*5 + predictions).cpu(), minlength=25).reshape(5,5)
            total_loss += loss.item()*len(targets)
            count += len(targets)
    cm = confusion.numpy()
    tp = np.diag(cm)
    f1 = 2*tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return dict(loss=total_loss/count, accuracy=float(tp.sum()/count),
                macro_f1=float(f1.mean()), samples=count, confusion_matrix=cm.tolist(),
                per_class={name: dict(f1=float(f1[i]), support=int(cm[i].sum()))
                           for i,name in enumerate(CLASS_NAMES)})


def initialize_weights(model):
    """Glorot initialization with zero biases; preserve the diagram's layers."""
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def check_learning_signal(model, dataset, device):
    """Check real training examples from every class before optimization."""
    indices = []
    seen = set()
    for i, label in enumerate(dataset.labels):
        if label not in seen:
            previous = dataset.ends[i-1] if i else 0
            indices.append(previous + len(dataset.starts[i]) // 2)
            seen.add(label)
    x = torch.stack([dataset[i][0] for i in indices]).to(device)
    y = torch.tensor([dataset[i][1] for i in indices], device=device)
    model.eval()
    features = []
    hook = model.fusion.register_forward_hook(lambda module, inputs, output: features.append(output))
    try:
        logits = model(x) 
        nn.functional.cross_entropy(logits, y).backward()
        fraction = float((features[0] > 0).float().mean())
        grads = {name: float(p.grad.norm()) for name, p in model.named_parameters()
                 if p.grad is not None}
        if fraction == 0 or any(not np.isfinite(v) for v in grads.values()):
            raise RuntimeError('Initial features are all zero or gradients are non-finite.')
        if any(grads.get(name, 0) == 0 for name in
               ['conv_branch.0.weight', 'deep_branch.5.weight', 'fusion.0.weight', 'dense.weight']):
            raise RuntimeError('No initial gradient in a feature branch; inspect initialization.')
        return dict(fusion_positive_fraction=fraction, gradient_norms=grads)
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, help='Extracted doppler_traces directory containing S1a etc.')
    p.add_argument('--train-sessions', nargs='+', default=['S1a','S1b'])
    p.add_argument('--val-sessions', nargs='+', default=['S1c'])
    p.add_argument('--test-sessions', nargs='+', default=['S7a'])
    p.add_argument('--preprocessing', choices=['trace_mean', 'raw'], default='trace_mean')
    p.add_argument('--evaluate-test', action='store_true', help='Evaluate held-out test only after tuning is complete')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--stride', type=int, default=170, help='Training window stride; evaluation uses 340')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--output', type=Path, help='New output directory (must not already exist)')
    p.add_argument('--max-windows-per-trace', type=int, default=0,
                   help='Limit windows for smoke tests only; 0 uses all windows')
    args = p.parse_args()
    if min(args.epochs,args.batch_size,args.stride,args.patience) < 1 or args.lr <= 0 or args.max_windows_per_trace < 0:
        p.error('Epochs, batch size, stride, patience and lr must be positive; window limit nonnegative')
    splits = [args.train_sessions, args.val_sessions, args.test_sessions]
    if any(len(s) != len(set(s)) for s in splits) or any(set(splits[i]) & set(splits[j]) for i in range(3) for j in range(i)):
        p.error('Sessions must be unique and disjoint across training, validation, and test')
    if args.data_root is None:
        candidates = sorted(p for p in (ROOT/'Data').rglob('doppler_traces') if p.is_dir())
        if len(candidates) != 1:
            p.error('Specify --data-root: expected exactly one extracted doppler_traces folder')
        args.data_root = candidates[0]
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    print('Device:', device, flush=True)
    datasets = {}
    for split, sessions in [('train',args.train_sessions), ('val',args.val_sessions), ('test',args.test_sessions)]:
        if split == 'test' and not args.evaluate_test:
            continue
        ds = DopplerWindows(args.data_root, sessions, args.stride if split == 'train' else 340,
                            args.max_windows_per_trace, args.preprocessing)
        datasets[split] = ds
        print(f'{split}: {len(ds.manifest)} traces, {len(ds)} windows; excluded {dict(ds.ignored)}', flush=True)
    path_sets = [{r['path'] for r in ds.manifest} for ds in datasets.values()]
    if any(path_sets[i] & path_sets[j] for i in range(len(path_sets)) for j in range(i)):
        raise ValueError('Same resolved trace occurs in multiple splits')
    output = args.output or ROOT/'Results'/'training'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(classes=CLASS_NAMES, class_codes=CODES, input_shape=[1,340,100],
                  preprocessing=args.preprocessing, evaluation='single antenna',
                  actual_device=str(device), torch_version=str(torch.__version__),
                  initialization='xavier_uniform_weights_zero_bias')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'splits.json').write_text(json.dumps({k:d.manifest for k,d in datasets.items()},indent=2))
    loaders = {k:DataLoader(d,batch_size=args.batch_size,shuffle=(k=='train'),num_workers=0,
                generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type=='cuda')
               for k,d in datasets.items()}
    model = ActivityCNN(num_classes=5).to(device)
    initialize_weights(model)
    signal = check_learning_signal(model, datasets['train'], device)
    (output/'initial_learning_signal.json').write_text(json.dumps(signal, indent=2))
    print('Initial learning signal: PASS', flush=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    print('Parameters:',sum(p.numel() for p in model.parameters()), 'Output:', output,flush=True)
    best_loss, stale, history = float('inf'), 0, []
    best_f1 = -1.0
    for epoch in range(1,args.epochs+1):
        start = time.monotonic()
        train = run_epoch(model,loaders['train'],device,optimizer)
        val = run_epoch(model,loaders['val'],device)
        history.append(dict(epoch=epoch,train=train,val=val,seconds=time.monotonic()-start))
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print(f"Epoch {epoch}: train loss {train['loss']:.4f}, acc {train['accuracy']:.3f}; "
              f"val loss {val['loss']:.4f}, acc {val['accuracy']:.3f}, F1 {val['macro_f1']:.3f}",flush=True)
        if val['macro_f1'] > best_f1:
            best_f1 = val['macro_f1']
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch,
                            config=config, val_metrics=val), output/'best_f1.pt')
        if val['loss'] < best_loss:
            best_loss, stale = val['loss'], 0
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                            epoch=epoch,config=config,val_metrics=val),output/'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping',flush=True)
                break
    if not args.evaluate_test:
        print('Validation run complete; held-out test was not loaded or evaluated.', flush=True)
        return
    checkpoint = torch.load(output/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test = run_epoch(model,loaders['test'],device)
    test['checkpoint_epoch'] = checkpoint['epoch']
    test['smoke_test'] = bool(args.max_windows_per_trace)
    (output/'test_metrics.json').write_text(json.dumps(test,indent=2))
    print(f"Test accuracy {test['accuracy']:.3f}; macro F1 {test['macro_f1']:.3f}",flush=True)
    if args.max_windows_per_trace:
        print('SMOKE TEST ONLY: these metrics are not a benchmark result.',flush=True)


if __name__ == '__main__':
    main()
