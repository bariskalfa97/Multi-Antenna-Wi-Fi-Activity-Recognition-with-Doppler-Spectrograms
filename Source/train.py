# Train ActivityCNN on extracted numerical Doppler traces.
#
# Run from any directory: python3 '/home/ubuntu/Deep Learning/Source/train.py'
# Default protocol: SHARP chronological 60/20/20 partitions within S1a/S1b/S1c,
# with 31-row gaps and stride 30. --protocol session preserves the earlier
# session-held-out experiment. Held-out tests are opt-in.
# Each antenna is an independent sample; reported metrics are BEFORE decision
# fusion. Default input is linear power minus its per-bin full-trace mean.
# This matches SHARP offline preprocessing; it uses future samples within each
# recording and is not a causal streaming transform. Test evaluation is opt-in.
# Only E/L/W/R/J activities are used; repetition suffixes such as J2 are retained
# as separate recordings. Other activity codes are reported and excluded.
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
PATTERN = re.compile(r'(?P<session>S\d+[a-z])_(?P<activity>[A-Z]+)(?P<repeat>(?:_?\d+)?)_stream_(?P<antenna>\d+)\.txt')


class ArrayReader(pickle.Unpickler):
    # Limit legacy dataset deserialization to its NumPy array types.
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



def partition_bounds(length, partition):
    # Reference chronological split; the STFT gap is ceil(31/1) rows.
    train_len, val_len = int(length * 0.6), int(length * 0.2)
    val_start = train_len + 31
    test_start = val_start + val_len + 31
    bounds = {'full': (0, length), 'train': (0, train_len),
              'val': (val_start, val_start + val_len), 'test': (test_start, length)}
    if partition not in bounds:
        raise ValueError(f'Unknown partition: {partition}')
    return bounds[partition]

class DopplerWindows(Dataset):
    # Keep each trace once in RAM and slice windows on demand.
    def __init__(self, root, sessions, stride, limit=0, preprocessing="trace_mean",
                 partition="full", reference_windows=False):
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
                begin, end = partition_bounds(len(array), partition)
                # Reference create_windows_antennas uses range(0, length-340, stride).
                stop = end - 340 + (0 if reference_windows else 1)
                starts = np.arange(begin, stop, stride)
                if not len(starts):
                    raise ValueError(f'No complete windows in {partition}: {path}')
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
                    shape=list(array.shape), windows=len(starts),
                    partition=partition, partition_start=begin, partition_end=end,
                    first_window_start=int(starts[0]), last_window_start=int(starts[-1])))
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


class ReferenceEpochSampler(torch.utils.data.Sampler):
    # Match repeat-before-batch: full batches, with training continuing across epochs.
    #
    #     Keras restarts validation evaluation each epoch. Antennas are ordered within
    #     each window as in expand_antennas; seeded PyTorch shuffling is framework-specific.
    #     
    def __init__(self, dataset, batch_size, shuffle=False, seed=42):
        self.order = []
        groups = {}
        for i, row in enumerate(dataset.manifest):
            groups.setdefault(row['recording'], []).append(i)
        for ids in groups.values():
            ids.sort(key=lambda i: dataset.manifest[i]['antenna'])
            if [dataset.manifest[i]['antenna'] for i in ids] != [0,1,2,3]:
                raise ValueError('SHARP requires four aligned antenna streams')
            if any(not np.array_equal(dataset.starts[i], dataset.starts[ids[0]]) for i in ids):
                raise ValueError('SHARP antenna windows are not aligned')
            for j in range(len(dataset.starts[ids[0]])):
                self.order.extend((dataset.ends[i-1] if i else 0) + j for i in ids)
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.size = ((len(self.order) + batch_size - 1)//batch_size)*batch_size
        self.cycle, self.cursor = [], 0

    def __len__(self):
        return self.size

    def __iter__(self):
        if not self.shuffle:
            for i in range(self.size):
                yield self.order[i % len(self.order)]
            return
        for _ in range(self.size):
            if self.cursor == len(self.cycle):
                self.cycle = torch.randperm(len(self.order), generator=self.generator).tolist()
                self.cursor = 0
            index = self.order[self.cycle[self.cursor]]
            self.cursor += 1
            yield index


def augment_spectrograms(inputs, max_shift, gain_jitter, noise_std):
    # Apply training-only, physically modest spectrogram perturbations.
    #
    #     A zero-filled time shift models uncertainty in activity alignment. Zero is
    #     the baseline after per-bin mean subtraction. Gain and additive noise model
    #     small changes in received power. The caller applies this only in training.
    #     
    if inputs.ndim != 4 or inputs.shape[1:] != (1, 340, 100):
        raise ValueError(f'Expected [batch, 1, 340, 100], got {tuple(inputs.shape)}')
    if max_shift < 0 or gain_jitter < 0 or noise_std < 0:
        raise ValueError('Augmentation magnitudes must be non-negative')
    augmented = inputs.clone()
    if max_shift:
        shifts = torch.randint(-max_shift, max_shift + 1, (len(inputs),), device=inputs.device)
        shifted = torch.zeros_like(augmented)
        for index, shift in enumerate(shifts.tolist()):
            if shift >= 0:
                shifted[index, :, shift:] = augmented[index, :, :340-shift]
            else:
                shifted[index, :, :shift] = augmented[index, :, -shift:]
        augmented = shifted
    if gain_jitter:
        gains = torch.empty((len(inputs), 1, 1, 1), device=inputs.device)
        gains.uniform_(1.0-gain_jitter, 1.0+gain_jitter)
        augmented = augmented * gains
    if noise_std:
        augmented = augmented + torch.randn_like(augmented) * noise_std
    return augmented


def run_epoch(model, loader, device, optimizer=None, augmentation=None):
    model.train(optimizer is not None)
    confusion = torch.zeros(5, 5, dtype=torch.long)
    total_loss, count = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if augmentation is not None:
                    inputs = augment_spectrograms(inputs, **augmentation)
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


class ReferenceAdam(torch.optim.Optimizer):
    # Keras Adam update: lr*sqrt(1-beta2**t)/(1-beta1**t), epsilon=1e-7.
    def __init__(self, params, lr=1e-4):
        super().__init__(params, dict(lr=lr, betas=(0.9, 0.999), eps=1e-7))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state['step'] += 1
                state['m'].add_(p.grad - state['m'], alpha=1-b1)
                state['v'].add_(p.grad.square() - state['v'], alpha=1-b2)
                alpha = group['lr'] * (1-b2**state['step'])**0.5 / (1-b1**state['step'])
                p.addcdiv_(state['m'], state['v'].sqrt().add_(group['eps']), value=-alpha)
        return loss


def initialize_weights(model):
    # Glorot initialization with zero biases; preserve the diagram's layers.
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def check_learning_signal(model, dataset, device):
    # Check real training examples from every class before optimization.
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
    p = argparse.ArgumentParser(description='Train the SHARP-style Wi-Fi activity classifier', formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, help='Extracted doppler_traces directory containing S1a etc.')
    p.add_argument('--protocol', choices=['sharp','session'], default='sharp')
    p.add_argument('--train-sessions', nargs='+')
    p.add_argument('--val-sessions', nargs='+')
    p.add_argument('--test-sessions', nargs='+')
    p.add_argument('--preprocessing', choices=['trace_mean', 'raw'], default='trace_mean')
    p.add_argument('--evaluate-test', action='store_true', help='Evaluate held-out test only after tuning is complete')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--stride', type=int, default=30, help='SHARP uses this stride in every split; session evaluation uses 340')
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--augment', action='store_true',
                   help='Apply training-only time shift, gain jitter, and Gaussian noise')
    p.add_argument('--augment-max-shift', type=int, default=15,
                   help='Maximum zero-filled time shift in Doppler rows')
    p.add_argument('--augment-gain-jitter', type=float, default=0.1,
                   help='Maximum multiplicative gain deviation around one')
    p.add_argument('--augment-noise-std', type=float, default=0.01,
                   help='Additive Gaussian noise standard deviation in centered power units')
    p.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    p.add_argument('--output', type=Path, help='New output directory (must not already exist)')
    p.add_argument('--max-windows-per-trace', type=int, default=0,
                   help='Limit windows for smoke tests only; 0 uses all windows')
    args = p.parse_args()
    if args.train_sessions is None:
        args.train_sessions = ['S1a','S1b','S1c'] if args.protocol == 'sharp' else ['S1a','S1b']
    if args.val_sessions is None:
        args.val_sessions = list(args.train_sessions) if args.protocol == 'sharp' else ['S1c']
    if args.test_sessions is None:
        args.test_sessions = (['S2a','S2b','S3a','S4a','S4b','S5a','S6a','S6b','S7a']
                              if args.protocol == 'sharp' else ['S7a'])
    if args.protocol == 'sharp' and (args.val_sessions != args.train_sessions or args.preprocessing != 'trace_mean'):
        p.error('SHARP requires identical train/val session lists and trace_mean preprocessing')
    if (min(args.epochs,args.batch_size,args.stride,args.patience) < 1 or args.lr <= 0
            or args.max_windows_per_trace < 0 or args.augment_max_shift < 0
            or args.augment_gain_jitter < 0 or args.augment_noise_std < 0):
        p.error('Epochs, batch size, stride, patience, and learning rate must be positive; other magnitudes non-negative')
    splits = [args.train_sessions, args.val_sessions, args.test_sessions]
    if any(len(s) != len(set(s)) for s in splits):
        p.error('Session lists must not contain duplicates')
    if set(args.test_sessions) & (set(args.train_sessions) | set(args.val_sessions)):
        p.error('External test sessions must not overlap train/val sessions')
    if args.protocol == 'session' and set(args.train_sessions) & set(args.val_sessions):
        p.error('Session protocol requires disjoint train/val sessions')
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
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    print('Device:', device, flush=True)
    datasets = {}
    specs = [('train', args.train_sessions, 'train' if args.protocol == 'sharp' else 'full'),
             ('val', args.val_sessions, 'val' if args.protocol == 'sharp' else 'full')]
    if args.evaluate_test:
        if args.protocol == 'sharp':
            specs.append(('test_s1', args.train_sessions, 'test'))
        specs.extend((f'test_{session}', [session], 'full') for session in args.test_sessions)
    for split, sessions, partition in specs:
        ds = DopplerWindows(args.data_root, sessions,
                            args.stride,
                            args.max_windows_per_trace, args.preprocessing, partition=partition,
                            reference_windows=args.protocol == 'sharp')
        datasets[split] = ds
        print(f'{split}: {len(ds.manifest)} traces, {len(ds)} windows; excluded {dict(ds.ignored)}', flush=True)
    # Shared recordings are intentional only for the disjoint chronological partitions.
    manifests = [(name, row) for name, ds in datasets.items() for row in ds.manifest]
    for i, (split, row) in enumerate(manifests):
        for other_split, other in manifests[:i]:
            if split != other_split and row['path'] == other['path']:
                if max(row['partition_start'], other['partition_start']) < min(row['partition_end'], other['partition_end']):
                    raise ValueError('Overlapping recording intervals across splits')
    output = args.output or ROOT/'Results'/'training'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(classes=CLASS_NAMES, class_codes=CODES, input_shape=[1,340,100],
                  preprocessing=args.preprocessing, evaluation='single antenna',
                  actual_device=str(device), torch_version=str(torch.__version__),
                  initialization='xavier_uniform_weights_zero_bias',
                  val_partition='val' if args.protocol == 'sharp' else 'full',
                  eval_stride=args.stride,
                  reference_windows=args.protocol == 'sharp',
                  selected_checkpoint='final.pt' if args.protocol == 'sharp' else 'best.pt',
                  adam_epsilon=1e-7 if args.protocol == 'sharp' else 1e-8,
                  early_stopping_min_delta=0.0,
                  monitoring_batches='repeat_before_batch' if args.protocol == 'sharp' else 'finite',
                  reference_commit='5bc1788896e800218cd167734f6c11a05b88e983')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'splits.json').write_text(json.dumps({k:d.manifest for k,d in datasets.items()},indent=2))
    loaders = {}
    for name, dataset in datasets.items():
        sampler = (ReferenceEpochSampler(dataset, args.batch_size, shuffle=name=='train', seed=args.seed)
                   if args.protocol == 'sharp' and name in ('train','val') else None)
        loaders[name] = DataLoader(dataset, batch_size=args.batch_size,
                    shuffle=(name=='train' and sampler is None), sampler=sampler, num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type=='cuda')
    model = ActivityCNN(num_classes=5).to(device)
    initialize_weights(model)
    signal = check_learning_signal(model, datasets['train'], device)
    (output/'initial_learning_signal.json').write_text(json.dumps(signal, indent=2))
    print('Initial learning signal: PASS', flush=True)
    optimizer = (ReferenceAdam(model.parameters(), lr=args.lr) if args.protocol == 'sharp'
                 else torch.optim.Adam(model.parameters(), lr=args.lr))
    augmentation = (dict(max_shift=args.augment_max_shift,
                         gain_jitter=args.augment_gain_jitter,
                         noise_std=args.augment_noise_std) if args.augment else None)
    if augmentation:
        print('Training augmentation:', augmentation, flush=True)
    print('Parameters:',sum(p.numel() for p in model.parameters()), 'Output:', output,flush=True)
    best_loss, stale, history = float('inf'), 0, []
    best_f1 = -1.0
    for epoch in range(1,args.epochs+1):
        start = time.monotonic()
        train = run_epoch(model,loaders['train'],device,optimizer,augmentation)
        val = run_epoch(model,loaders['val'],device)
        history.append(dict(epoch=epoch,train=train,val=val,seconds=time.monotonic()-start))
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print(f"Epoch {epoch}: train loss {train['loss']:.4f}, acc {train['accuracy']:.3f}; "
              f"val loss {val['loss']:.4f}, acc {val['accuracy']:.3f}, F1 {val['macro_f1']:.3f}",flush=True)
        if val['macro_f1'] > best_f1:
            best_f1 = val['macro_f1']
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch,
                            config=config, val_metrics=val, val_metrics_repeated=args.protocol=='sharp'), output/'best_f1.pt')
        if val['loss'] < best_loss:
            best_loss, stale = val['loss'], 0
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                            epoch=epoch,config=config,val_metrics=val,
                            val_metrics_repeated=args.protocol=='sharp'),output/'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping',flush=True)
                break
    # Monitoring uses the reference repeated batches. Final reporting counts each window once.
    final_val = run_epoch(model, DataLoader(datasets['val'], batch_size=args.batch_size), device)
    torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, config=config,
                    val_metrics=final_val, val_metrics_repeated=False), output/'final.pt')
    (output/'final_validation.json').write_text(json.dumps(final_val, indent=2))
    print(f"Final checkpoint: {output/'final.pt'}; validation accuracy {final_val['accuracy']:.3f}", flush=True)
    if not args.evaluate_test:
        print('Validation run complete; held-out test partitions were not evaluated.', flush=True)
        return
    checkpoint = torch.load(output/config['selected_checkpoint'], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    tests = {}
    for name, loader in loaders.items():
        if name.startswith('test_'):
            tests[name] = run_epoch(model, loader, device)
            tests[name].update(checkpoint_epoch=checkpoint['epoch'], smoke_test=bool(args.max_windows_per_trace))
            print(f"{name}: accuracy {tests[name]['accuracy']:.3f}; macro F1 {tests[name]['macro_f1']:.3f}", flush=True)
    (output/'test_metrics.json').write_text(json.dumps(tests, indent=2))



if __name__ == '__main__':
    main()
