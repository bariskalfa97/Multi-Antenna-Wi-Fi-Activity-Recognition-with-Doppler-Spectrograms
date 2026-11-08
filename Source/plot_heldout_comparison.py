"""Create report figures for the matched S1a/S1b-to-S1c experiment."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'Results/fusion/20260930T143644520675Z/comparison.json'
CONTRAST = ROOT / 'Results/fusion/20260930T142837168964Z/comparison.json'
BASE_HISTORY = ROOT / 'Results/session_holdout_baseline/history.json'
CONTRAST_HISTORY = ROOT / 'Results/contrastive_domain/20260930T140121748559Z/history.json'
OUT = ROOT / 'Report/figures'
NAMES = ['empty', 'sitting', 'walking', 'running', 'jumping']


def load(path):
    return json.loads(path.read_text())


def matrix_figure():
    results = [('Supervised baseline', load(BASE)), ('Domain-aware contrastive', load(CONTRAST))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (name, result) in zip(axes, results):
        cm = np.asarray(result['decision_fusion']['confusion_matrix'])
        values = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        im = ax.imshow(values, vmin=0, vmax=100, cmap='Blues')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{values[i,j]:.0f}', ha='center', va='center',
                        color='white' if values[i,j] > 55 else 'black', fontsize=8)
        ax.set(title=f'{name}\nFused accuracy {result["decision_fusion"]["accuracy"]:.1%}',
               xticks=range(5), yticks=range(5), xticklabels=NAMES, yticklabels=NAMES,
               xlabel='Predicted class', ylabel='True class')
        ax.tick_params(axis='x', rotation=35)
    fig.colorbar(im, ax=axes, label='Percentage of true class')
    fig.savefig(OUT / 'heldout_fusion_comparison.png', dpi=200)


def curve_figure():
    histories = [('Supervised baseline', load(BASE_HISTORY)),
                 ('Domain-aware contrastive', load(CONTRAST_HISTORY))]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for name, history in histories:
        fine = [row for row in history if row.get('stage', 'finetune') == 'finetune']
        epochs = [row['epoch'] for row in fine]
        axes[0].plot(epochs, [row['val']['accuracy'] for row in fine], marker='o', label=name)
        axes[1].plot(epochs, [row['val']['macro_f1'] for row in fine], marker='o', label=name)
    axes[0].set(xlabel='Fine-tuning epoch', ylabel='Held-out S1c accuracy', ylim=(0.6, 0.76))
    axes[1].set(xlabel='Fine-tuning epoch', ylabel='Held-out S1c macro F1', ylim=(0.6, 0.76))
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(OUT / 'heldout_learning_curves.png', dpi=200)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    matrix_figure(); curve_figure()
