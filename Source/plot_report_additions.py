"""Create additional figures for the Wi-Fi HAR report from frozen experiment records."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Report' / 'figures'
FROZEN = ROOT / 'Results' / 'test' / '20260930T113028986949Z' / 'comparison.json'
BASE = ROOT / 'Results' / 'fusion' / '20260930T143644520675Z' / 'comparison.json'
CONTRAST = ROOT / 'Results' / 'fusion' / '20260930T142837168964Z' / 'comparison.json'
TIME_SHIFT = ROOT / 'Results' / 'fusion' / 'session_holdout_time_shift' / 'comparison.json'
PLOTS = ROOT / 'Data' / 'Doppler_plots'
CLASSES = ['empty', 'sitting', 'walking', 'running', 'jumping']


def load(path):
    return json.loads(path.read_text())


def frozen_session_fusion():
    record = load(FROZEN)['sets']
    names = list(record)
    single = [100 * record[name]['single_antenna']['accuracy'] for name in names]
    fused = [100 * record[name]['decision_fusion']['accuracy'] for name in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.bar(x - .19, single, .38, label='Individual antennas', color='#93c5fd')
    ax.bar(x + .19, fused, .38, label='Decision fusion', color='#2563eb')
    for xx, value in zip(x - .19, single):
        ax.text(xx, value - 2.6, f'{value:.1f}', ha='center', va='top', rotation=90, fontsize=7)
    for xx, value in zip(x + .19, fused):
        ax.text(xx, value - 2.6, f'{value:.1f}', ha='center', va='top', rotation=90, fontsize=7,
                color='white')
    ax.set(xticks=x, xticklabels=names, ylim=(70, 101), ylabel='Accuracy (%)')
    ax.grid(axis='y', alpha=.25)
    fig.legend(ncol=2, loc='upper center', bbox_to_anchor=(.5, .99), frameon=False)
    fig.subplots_adjust(top=.80, bottom=.16, left=.09, right=.98)
    fig.savefig(OUT / 'frozen_session_fusion.png', dpi=220)
    plt.close(fig)


def heldout_class_recall():
    base = load(BASE)['decision_fusion']['per_class']
    shifted = load(TIME_SHIFT)['decision_fusion']['per_class']
    contrast = load(CONTRAST)['decision_fusion']['per_class']
    series = [
        ('Supervised baseline', [100 * base[c]['recall'] for c in CLASSES], '#34a853', -.27),
        ('Time-shift augmentation', [100 * shifted[c]['recall'] for c in CLASSES], '#4f86c6', 0),
        ('Domain-aware contrastive', [100 * contrast[c]['recall'] for c in CLASSES], '#f59e0b', .27),
    ]
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(7.9, 3.8))
    for label, values, color, offset in series:
        ax.bar(x + offset, values, .27, label=label, color=color)
        for xx, value in zip(x + offset, values):
            if value >= 18:
                ax.text(xx, value - 6, f'{value:.0f}', ha='center', va='top', fontsize=6.5, color='white')
            else:
                ax.text(xx, value + 3, f'{value:.0f}', ha='center', va='bottom', fontsize=6.5)
    ax.set(xticks=x, xticklabels=CLASSES, ylim=(0, 105), ylabel='Fused recall (%)')
    ax.grid(axis='y', alpha=.25)
    fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, .99), frameon=False, fontsize=7.5)
    fig.subplots_adjust(top=.80, bottom=.16, left=.10, right=.98)
    fig.savefig(OUT / 'heldout_class_recall.png', dpi=220)
    plt.close(fig)


def doppler_examples():
    # Representative recorded Doppler spectrograms, one for each class.
    files = [
        ('empty', 'csi_doppler_activity_S1a_E.png'),
        ('sitting', 'csi_doppler_activity_S1a_L.png'),
        ('walking', 'csi_doppler_activity_S1a_W.png'),
        ('running', 'csi_doppler_activity_S1a_R.png'),
        ('jumping', 'csi_doppler_activity_S1a_J1.png'),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.3), constrained_layout=True)
    for ax, (label, filename) in zip(axes, files):
        ax.imshow(plt.imread(PLOTS / filename))
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    fig.savefig(OUT / 'doppler_examples.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_session_fusion()
    heldout_class_recall()
    doppler_examples()
