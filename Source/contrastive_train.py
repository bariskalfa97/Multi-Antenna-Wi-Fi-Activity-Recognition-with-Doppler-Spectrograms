# Domain-aware supervised-contrastive pretraining for the unchanged ActivityCNN.
#
# The projection head exists only during pretraining. Each contrastive batch
# contains paired events from two S1 sessions: the four aligned antennas of an
# event are positives, as are same-activity events from the other session.
# S1c is kept entirely separate for validation.
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from architecture import ActivityCNN
from train import (CLASS_NAMES, ROOT, DopplerWindows, ReferenceAdam,
                   ReferenceEpochSampler, augment_spectrograms,
                   check_learning_signal, initialize_weights, run_epoch)


class ProjectionHead(nn.Module):
    # Temporary head, discarded before deployment and final evaluation.
    def __init__(self, input_dim=3 * 170 * 50, hidden_dim=128, output_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, features):
        return nn.functional.normalize(self.layers(features), dim=1)


class ContrastiveWindows(Dataset):
    # Expose session and physical-event identifiers alongside each window.
    def __init__(self, base):
        self.base = base
        self.session_ids, self.event_ids = {}, {}
        session_number = {session: number for number, session in
                          enumerate(sorted({row['session'] for row in base.manifest}))}
        event_number = {}
        for trace, row in enumerate(base.manifest):
            offset = base.ends[trace - 1] if trace else 0
            for position, start in enumerate(base.starts[trace]):
                index = offset + position
                key = (row['recording'], int(start))
                event_number.setdefault(key, len(event_number))
                self.session_ids[index] = session_number[row['session']]
                self.event_ids[index] = event_number[key]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, self.session_ids[index], self.event_ids[index]


class CrossSessionAlignedSampler(Sampler):
    # Build batches with two matched four-antenna events per activity pair.
    def __init__(self, dataset, batch_size, seed):
        if batch_size < 8 or batch_size % 8:
            raise ValueError('Contrastive batch size must be a multiple of eight')
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        events = {}
        base = dataset.base
        for trace, row in enumerate(base.manifest):
            offset = base.ends[trace - 1] if trace else 0
            for position, start in enumerate(base.starts[trace]):
                key = (row['label'], row['session'], row['recording'], int(start))
                events.setdefault(key, []).append((row['antenna'], offset + position))
        self.by_label_session = {}
        for (label, session, _recording, _start), members in events.items():
            members.sort()
            if [antenna for antenna, _ in members] != [0, 1, 2, 3]:
                raise ValueError('Every contrastive event needs four aligned antennas')
            self.by_label_session.setdefault((label, session), []).append([index for _, index in members])
        self.sessions_by_label = {}
        for label, session in self.by_label_session:
            self.sessions_by_label.setdefault(label, []).append(session)
        if any(len(sessions) < 2 for sessions in self.sessions_by_label.values()):
            raise ValueError('Each activity needs examples from two training sessions')
        self.batches = (len(dataset) + batch_size - 1) // batch_size

    def __len__(self):
        return self.batches * self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed)
        labels = sorted(self.sessions_by_label)
        for _ in range(self.batches):
            batch = []
            for _ in range(self.batch_size // 8):
                label = rng.choice(labels)
                first, second = rng.sample(self.sessions_by_label[label], 2)
                batch.extend(rng.choice(self.by_label_session[label, first]))
                batch.extend(rng.choice(self.by_label_session[label, second]))
            yield from batch


def domain_augment(inputs, max_shift, gain_jitter, noise_std, time_mask, frequency_mask):
    # Mild shifts and masks; zero is the centred-power baseline.
    augmented = augment_spectrograms(inputs, max_shift, gain_jitter, noise_std)
    for sample in range(len(augmented)):
        width = int(torch.randint(0, time_mask + 1, (), device=inputs.device))
        if width:
            start = int(torch.randint(0, 340 - width + 1, (), device=inputs.device))
            augmented[sample, :, start:start + width, :] = 0
        width = int(torch.randint(0, frequency_mask + 1, (), device=inputs.device))
        if width:
            start = int(torch.randint(0, 100 - width + 1, (), device=inputs.device))
            augmented[sample, :, :, start:start + width] = 0
    return augmented


def supervised_contrastive_loss(features, labels, sessions, events, temperature):
    # Positive pairs match an event or match activity across sessions.
    similarity = features @ features.T / temperature
    nonself = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
    same_label = labels[:, None].eq(labels[None, :])
    positives = same_label & (sessions[:, None].ne(sessions[None, :]) |
                              events[:, None].eq(events[None, :])) & nonself
    positive_count = positives.sum(1)
    if (positive_count == 0).any():
        raise RuntimeError('Every anchor needs a cross-session or aligned-antenna positive')
    masked = similarity.masked_fill(~nonself, float('-inf'))
    log_probability = similarity - torch.logsumexp(masked, dim=1, keepdim=True)
    return -((log_probability * positives).sum(1) / positive_count).mean()


def pretrain_epoch(encoder, projector, loader, optimizer, device, args):
    encoder.train(); projector.train()
    total, count = 0.0, 0
    for inputs, labels, sessions, events in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        sessions, events = sessions.to(device), events.to(device)
        first = domain_augment(inputs, args.time_shift, args.gain_jitter, args.noise_std,
                               args.time_mask, args.frequency_mask)
        second = domain_augment(inputs, args.time_shift, args.gain_jitter, args.noise_std,
                                args.time_mask, args.frequency_mask)
        encoded = torch.cat((encoder.forward_features(first), encoder.forward_features(second)))
        loss = supervised_contrastive_loss(
            projector(encoded), torch.cat((labels, labels)), torch.cat((sessions, sessions)),
            torch.cat((events, events)), args.temperature)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite contrastive loss')
        loss.backward(); optimizer.step()
        total += loss.item() * len(inputs); count += len(inputs)
    return total / count


def choose_device(request):
    if request == 'auto':
        return torch.device('cuda' if torch.cuda.is_available()
                            else 'mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(request)


def main():
    parser = argparse.ArgumentParser(description='Run supervised-contrastive pretraining and classifier fine-tuning')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--pretrain-epochs', type=int, default=10)
    parser.add_argument('--finetune-epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--time-shift', type=int, default=15)
    parser.add_argument('--gain-jitter', type=float, default=0.05)
    parser.add_argument('--noise-std', type=float, default=0.005)
    parser.add_argument('--time-mask', type=int, default=20)
    parser.add_argument('--frequency-mask', type=int, default=6)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--max-windows-per-trace', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (min(args.pretrain_epochs, args.finetune_epochs, args.batch_size, args.patience) < 1
            or args.batch_size % 8 or args.lr <= 0 or args.time_shift < 0
            or min(args.gain_jitter, args.noise_std, args.time_mask, args.frequency_mask) < 0
            or args.temperature <= 0 or args.max_windows_per_trace < 0):
        parser.error('Invalid parameters; batch size must be a multiple of eight')
    if args.data_root is None:
        candidates = sorted(p for p in (ROOT / 'Data').rglob('doppler_traces') if p.is_dir())
        if len(candidates) != 1:
            parser.error('Specify --data-root')
        args.data_root = candidates[0]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    train_sessions, val_sessions = ['S1a', 'S1b'], ['S1c']
    train = DopplerWindows(args.data_root, train_sessions, 30, args.max_windows_per_trace,
                           'trace_mean', 'full', True)
    validation = DopplerWindows(args.data_root, val_sessions, 30, args.max_windows_per_trace,
                                'trace_mean', 'full', True)
    output = args.output or ROOT / 'Results' / 'contrastive_domain' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    config = dict(method='domain-aware supervised contrastive pretraining followed by classifier fine-tuning',
        data_root=str(args.data_root), classes=CLASS_NAMES, input_shape=[1, 340, 100],
        preprocessing='trace_mean', train_sessions=train_sessions, val_sessions=val_sessions,
        test_sessions=['S2a','S2b','S3a','S4a','S4b','S5a','S6a','S6b','S7a'],
        protocol='session_held_out', val_partition='full', eval_stride=30, reference_windows=True,
        selected_checkpoint='best_f1.pt',
        arguments={key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()}, actual_device=str(device),
        projection_head='25500 -> 128 -> 64; discarded before fine-tuning',
        positives='same physical event across antennas OR same activity across S1a/S1b')
    (output / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    (output / 'splits.json').write_text(json.dumps({'train': train.manifest, 'val': validation.manifest}, indent=2))
    contrastive_train = ContrastiveWindows(train)
    pretrain_loader = DataLoader(contrastive_train, batch_size=args.batch_size,
        sampler=CrossSessionAlignedSampler(contrastive_train, args.batch_size, args.seed), num_workers=0)
    finetune_loader = DataLoader(train, batch_size=args.batch_size,
        sampler=ReferenceEpochSampler(train, args.batch_size, shuffle=True, seed=args.seed), num_workers=0)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    encoder = ActivityCNN(5).to(device); initialize_weights(encoder)
    signal = check_learning_signal(encoder, train, device)
    (output / 'initial_learning_signal.json').write_text(json.dumps(signal, indent=2))
    projector = ProjectionHead().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=args.lr)
    history = []
    print('Device:', device, 'classifier parameters:', sum(p.numel() for p in encoder.parameters()), flush=True)
    print('Temporary projection parameters:', sum(p.numel() for p in projector.parameters()), flush=True)
    print('Validation holds out S1c completely; S2--S7 are not loaded.', flush=True)
    for epoch in range(1, args.pretrain_epochs + 1):
        loss = pretrain_epoch(encoder, projector, pretrain_loader, optimizer, device, args)
        history.append(dict(stage='pretrain', epoch=epoch, contrastive_loss=loss))
        print(f'Pretrain epoch {epoch}: contrastive loss {loss:.4f}', flush=True)
    torch.save(dict(encoder_state_dict=encoder.state_dict(), projector_state_dict=projector.state_dict(), config=config), output / 'pretrain.pt')
    del projector
    optimizer = ReferenceAdam(encoder.parameters(), lr=args.lr)
    best_loss, stale, best_f1 = float('inf'), 0, -1.0
    for epoch in range(1, args.finetune_epochs + 1):
        training = run_epoch(encoder, finetune_loader, device, optimizer)
        val = run_epoch(encoder, val_loader, device)
        history.append(dict(stage='finetune', epoch=epoch, train=training, val=val))
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(f"Fine-tune epoch {epoch}: train loss {training['loss']:.4f}, acc {training['accuracy']:.3f}; "
              f"val loss {val['loss']:.4f}, acc {val['accuracy']:.3f}, F1 {val['macro_f1']:.3f}", flush=True)
        if val['macro_f1'] > best_f1:
            best_f1 = val['macro_f1']
            torch.save(dict(model_state_dict=encoder.state_dict(), epoch=epoch, config=config,
                            val_metrics=val, val_metrics_repeated=False), output / 'best_f1.pt')
        if val['loss'] < best_loss:
            best_loss, stale = val['loss'], 0
            torch.save(dict(model_state_dict=encoder.state_dict(), epoch=epoch, config=config,
                            val_metrics=val, val_metrics_repeated=False), output / 'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping', flush=True); break
    final_val = run_epoch(encoder, val_loader, device)
    torch.save(dict(model_state_dict=encoder.state_dict(), epoch=epoch, config=config,
                    val_metrics=final_val, val_metrics_repeated=False), output / 'final.pt')
    (output / 'final_validation.json').write_text(json.dumps(final_val, indent=2))
    print(f'Final checkpoint: {output / "final.pt"}; validation accuracy {final_val["accuracy"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
