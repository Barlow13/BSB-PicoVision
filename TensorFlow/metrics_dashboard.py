"""
Brady Barlow
Oklahoma State University

BSB-PicoVision — Unified Metrics Dashboard
Compares 6 model configurations (3 width multipliers × 2 input resolutions)
and optionally runs sample images through each TFLite model to visualize
predictions.

Outputs everything into a Metrics/ folder:
  Combined views  → Metrics/combined_*.png
  Individual views → Metrics/individual_*.png
  Inference demo   → Metrics/inference_demo.png  (if --samples provided)

Usage:
    # Generate all metric plots (no inference):
    python metrics_dashboard.py

    # Generate metrics + run sample images through all TFLite models:
    python metrics_dashboard.py --samples person.jpg dog.jpg cat.jpg none.jpg

    # Custom run directories:
    python metrics_dashboard.py --runs export_a1.0_640 export_a1.0_96

Notes:
    - Run from the TensorFlow/ project directory
    - Requires: matplotlib, numpy, pandas
    - TFLite inference additionally requires: tensorflow, opencv-python (cv2)
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter


# ═══════════════════════════════════════════════════════════════════════
#  IEEE LIGHT THEME — Oklahoma State University accent palette
# ═══════════════════════════════════════════════════════════════════════
#
# Designed for IEEE-style journal figures:
#   • Serif typography, pure-white panels, black titles
#   • OSU orange (Pantone 165 C) reserved as a brand accent and to mark
#     the proposed BSB-PicoVision result
#   • Companion ramp: warm tan, saddle brown, cool slate, deep navy, black
#     — six perceptually distinct, print-safe tones in the same family
# ───────────────────────────────────────────────────────────────────────

ORANGE       = '#FF7300'   # OSU Pantone 165 C — primary brand accent
SADDLE       = '#8B4513'   # warm earthy brown
NAVY         = '#2E3F5C'   # deep cool slate
SLATE        = '#5C7180'   # mid slate gray
TAN          = '#C19A6B'   # warm tan
BLACK        = '#1A1A1A'   # near-black
GRAY_DK      = '#3A3A3A'   # body / tick text
GRAY_MD      = '#6B6B6B'   # secondary text & captions
GRAY_LT      = '#BFBFBF'   # axis spines / weak rules

# ─── LIGHT MODE BACKGROUND ──────────────────────────────────────
BG           = '#FFFFFF'   # pure white page
PANEL        = '#FFFFFF'   # pure white axes — no warm cream tint
GRID         = '#E2E2E2'   # neutral light gray grid
TEXT         = BLACK
TEXT_DIM     = GRAY_MD
TITLE_COLOR  = BLACK       # IEEE: titles are black, not coloured
BAR_EDGE     = '#FFFFFF'

# 6-run accent ramp.  Order: full-resolution variants first (warm OSU
# family), then progressively cooler / darker hues for the reduced
# configurations, with near-black reserved for the deployment target.
RUN_COLORS       = [ORANGE,    TAN,       SADDLE,    SLATE,     NAVY,      BLACK]
RUN_COLORS_LIGHT = ['#FFC299', '#E2CCAE', '#C49066', '#A2AEB8', '#8AA0BC', '#9A9A9A']

CLASSES = ['person', 'dog', 'cat', 'none']

# Default run layout (can be overridden via --runs)
# Ordered: full → only resolution reduced → only width reduced (×2) → final
DEFAULT_NAMES = [
    'Reference (640×640 · α1.0)',          # full resolution, full width — baseline
    'Low-Res (96×96 · α1.0)',              # only resolution reduced
    'Mid-Width (640×640 · α0.5)',          # half width, full res
    'Mid-Width Low-Res (96×96 · α0.5)',    # half width, low res
    'Slim-Width (640×640 · α0.35)',        # slim width, full res
    'BSB-PicoVision (96×96 · α0.35)',      # final deployment target — both reduced
]
DEFAULT_DIRS  = [
    './export_a1.0_640',
    './export_a1.0_96',
    './export_a0.5_b6_640',
    './export_a0.5_b6_96',
    './export_a0.35_b6_640',
    './export_a0.35_b6_96',
]

# ─── DEMO IMAGES ────────────────────────────────────────────────────
# Set these paths to sample images for the inference demo.
# The demo runs automatically when any path is non-empty.
# Pass --samples on the command line to override these.
DEMO_IMAGES = {
    'PERSON_IMAGE':    '../datasets/coco/images/train2017/000000369840.jpg',
    'DOG_IMAGE':       '../datasets/coco/images/train2017/000000181763.jpg',
    'CAT_IMAGE':       '../datasets/coco/images/train2017/000000448627.jpg',
    'NONE_IMAGE':      '../datasets/coco/images/train2017/000000296367.jpg',
    'MULTI_IMAGE':     '../datasets/coco/images/train2017/000000380252.jpg',   # person + dog + cat
}


def apply_theme():
    plt.rcParams.update({
        'figure.facecolor':   BG,
        'axes.facecolor':     PANEL,
        'axes.edgecolor':     GRAY_LT,
        'axes.linewidth':     0.8,
        'axes.labelcolor':    TEXT,
        'axes.labelsize':     9.5,
        'axes.titlecolor':    TITLE_COLOR,
        'text.color':         TEXT,
        'xtick.color':        GRAY_DK,
        'ytick.color':        GRAY_DK,
        'xtick.labelsize':    8.5,
        'ytick.labelsize':    8.5,
        'xtick.major.size':   3,
        'ytick.major.size':   3,
        'xtick.major.width':  0.7,
        'ytick.major.width':  0.7,
        'grid.color':         GRID,
        'grid.alpha':         1.0,
        'grid.linewidth':     0.5,
        'grid.linestyle':     '-',
        'font.family':        'serif',
        'font.serif':         ['DejaVu Serif', 'Times New Roman', 'Liberation Serif',
                               'Nimbus Roman', 'serif'],
        'mathtext.fontset':   'dejavuserif',
        'font.size':          9,
        'axes.titlesize':     10.5,
        'axes.titleweight':   'bold',
        'axes.titlepad':      8,
        'figure.titlesize':   13,
        'figure.titleweight': 'bold',
        'legend.fontsize':    8,
        'legend.frameon':     True,
        'legend.framealpha':  0.95,
        'legend.edgecolor':   GRAY_LT,
        'legend.facecolor':   '#FFFFFF',
        'legend.borderpad':   0.5,
        'legend.handlelength': 1.8,
        'savefig.facecolor':  BG,
        'savefig.edgecolor':  BG,
    })


def lkw():
    """Legend keyword style — IEEE: subtle border, white fill, dark labels."""
    return dict(framealpha=0.95, facecolor='#FFFFFF', edgecolor=GRAY_LT, labelcolor=TEXT)


def _lighten(hex_color, amount=0.4):
    """Mix a color toward white by `amount` (0 = original, 1 = white).
    Used for paired fills (e.g. macro vs micro F1) without resorting to
    print-unfriendly hatching.
    """
    r, g, b = to_rgb(hex_color)
    a = float(np.clip(amount, 0.0, 1.0))
    return (r + (1 - r) * a, g + (1 - g) * a, b + (1 - b) * a)


def make_cmap(hex_color):
    """Single-hue sequential colormap from white → hex_color."""
    r, g, b = to_rgb(hex_color)
    return LinearSegmentedColormap.from_list('osu', [
        (0.00, '#FFFFFF'),
        (0.25, (1-0.30*(1-r), 1-0.30*(1-g), 1-0.30*(1-b))),
        (0.55, (1-0.65*(1-r), 1-0.65*(1-g), 1-0.65*(1-b))),
        (0.85, hex_color),
        (1.00, hex_color),
    ])


# Single brand colormap used for all confusion-matrix panels so cross-run
# magnitudes are comparable on the page.
BRAND_CMAP = None  # populated lazily after apply_theme()


def _human_format(x, _pos=None):
    """Format large numbers as 67K, 1.9M, etc."""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


def _annotate_bars_vertical(ax, bars_list, vals_list, fontsize=6.5,
                            fmt='.0%'):
    """
    Annotate grouped bars with a single rotated value label per bar,
    placed just above the bar.  Vertical orientation keeps tightly
    packed groups (e.g. six per cluster) readable without the cross-bar
    overlap that staggered horizontal labels produce.
    """
    for bars, vals in zip(bars_list, vals_list):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.015,
                    f'{v:{fmt}}',
                    ha='center', va='bottom', fontsize=fontsize,
                    color=GRAY_DK, rotation=90)


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_run(name, d):
    """Load all data for a single run directory."""
    meta = json.load(open(os.path.join(d, 'metadata.json')))
    stats = json.load(open(os.path.join(d, 'confusion_stats.json')))
    hist = pd.read_csv(os.path.join(d, 'training_history.csv'))

    per_class = {}
    for cls in CLASSES:
        s = stats[cls]
        tp, fp, fn = s['TP'], s['FP'], s['FN']
        p = tp/(tp+fp) if (tp+fp) else 0
        r = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        tn = s['TN']
        spec = tn/(tn+fp) if (tn+fp) else 0
        per_class[cls] = {'precision': p, 'recall': r, 'f1': f1,
                          'specificity': spec, 'TP': tp, 'FP': fp,
                          'FN': fn, 'TN': tn}

    tflite_path = os.path.join(d, 'BSB-PicoVision.tflite')
    tflite_kb = os.path.getsize(tflite_path) / 1024 if os.path.exists(tflite_path) else 0

    trainable = 0
    summary_path = os.path.join(d, 'model_summary.txt')
    if os.path.exists(summary_path):
        for line in open(summary_path):
            if 'Trainable params:' in line and 'Non' not in line:
                trainable = int(line.split(':')[1].strip().split('(')[0].strip().replace(',',''))

    return {
        'name': name, 'dir': d,
        'macro_f1': meta['macro_f1'], 'micro_f1': meta['micro_f1'],
        'per_class': per_class, 'history': hist,
        'tflite_kb': tflite_kb, 'trainable': trainable,
        'input_shape': meta['input_shape'],
        'thresholds': meta.get('thresholds', [0.5]*len(CLASSES)),
        'quantization': meta.get('quantization', 'float16'),
        'epochs': len(hist),
        'best_val_loss': hist['val_loss'].min(),
        'best_val_auc': hist['val_auc'].max(),
        'best_val_acc': hist['val_bin_acc'].max(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  INDIVIDUAL PLOT FUNCTIONS  (each returns a figure)
# ═══════════════════════════════════════════════════════════════════════

def _plot_training_metric(runs, train_col, val_col, title):
    """Generic single training curve plot."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_title(title)
    ax.set_xlabel('Epoch'); ax.set_ylabel(title)
    ax.grid(True); ax.set_axisbelow(True)
    for i, rd in enumerate(runs):
        ep = np.arange(1, rd['epochs']+1)
        ax.plot(ep, rd['history'][train_col], color=RUN_COLORS_LIGHT[i],
                linewidth=0.9, alpha=0.55, linestyle='--')
        ax.plot(ep, rd['history'][val_col], color=RUN_COLORS[i],
                linewidth=1.6, label=rd['name'])
    ax.legend(ncol=2, **lkw())
    _add_train_val_legend(fig)
    _add_source_note(fig, SOURCE_TRAIN)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_loss(runs):
    return _plot_training_metric(runs, 'loss', 'val_loss', 'Loss')

def plot_accuracy(runs):
    return _plot_training_metric(runs, 'bin_acc', 'val_bin_acc', 'Binary Accuracy')

def plot_auc(runs):
    return _plot_training_metric(runs, 'auc', 'val_auc', 'AUC')

def plot_precision_curve(runs):
    return _plot_training_metric(runs, 'precision', 'val_precision', 'Precision')

def plot_recall_curve(runs):
    return _plot_training_metric(runs, 'recall', 'val_recall', 'Recall')


def plot_lr_schedule(runs):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, which='both'); ax.set_axisbelow(True)
    for i, rd in enumerate(runs):
        ep = np.arange(1, rd['epochs']+1)
        ax.plot(ep, rd['history']['learning_rate'], color=RUN_COLORS[i],
                linewidth=1.6, label=rd['name'])
    ax.legend(ncol=2, **lkw())
    _add_source_note(fig, SOURCE_TRAIN)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_per_class_bars(runs, metric_key, metric_name):
    """Per-class grouped bar for a given metric."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.set_title(f'Per-Class {metric_name}')
    x = np.arange(len(CLASSES))
    n = len(runs)
    w = 0.8 / n
    all_bars = []; all_vals = []
    for i, rd in enumerate(runs):
        vals = [rd['per_class'][c][metric_key] for c in CLASSES]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, color=RUN_COLORS[i],
                      label=rd['name'], edgecolor=BAR_EDGE, linewidth=0.6, zorder=3)
        all_bars.append(bars); all_vals.append(vals)
    _annotate_bars_vertical(ax, all_bars, all_vals, fontsize=6, fmt='.0%')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, 1.22)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=min(n, 3), **lkw())
    _add_source_note(fig, SOURCE_KERAS_VAL)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_overall_f1(runs):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_title('Overall F1 Scores')
    x = np.arange(len(runs))
    w = 0.36
    macro = [rd['macro_f1'] for rd in runs]
    micro = [rd['micro_f1'] for rd in runs]
    # Macro: solid filled bar.  Micro: same hue but lightened — distinguishable
    # without the print-unfriendly hatching pattern.
    micro_face = [_lighten(c, 0.45) for c in RUN_COLORS[:len(runs)]]
    b1 = ax.bar(x - w/2, macro, w, color=RUN_COLORS[:len(runs)],
                edgecolor=GRAY_DK, linewidth=0.5, zorder=3)
    b2 = ax.bar(x + w/2, micro, w, color=micro_face,
                edgecolor=GRAY_DK, linewidth=0.5, zorder=3)
    for bar, v in zip(b1, macro):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                f'{v:.1%}', ha='center', va='bottom', fontsize=7, color=GRAY_DK)
    for bar, v in zip(b2, micro):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                f'{v:.1%}', ha='center', va='bottom', fontsize=7, color=GRAY_DK)
    ax.set_xticks(x)
    ax.set_xticklabels([rd['name'] for rd in runs], rotation=20, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.10)
    ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.legend([Patch(facecolor=GRAY_DK, edgecolor=GRAY_DK),
               Patch(facecolor=_lighten(GRAY_DK, 0.45), edgecolor=GRAY_DK)],
              ['Macro F1', 'Micro F1'], loc='upper right', **lkw())
    _add_source_note(fig, SOURCE_KERAS_VAL)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def _label_hbar(ax, bars, labels, max_val, fontsize=8.5):
    """Place labels inside bars when wide enough, outside when narrow."""
    for bar, label in zip(bars, labels):
        w = bar.get_width()
        cy = bar.get_y() + bar.get_height() / 2
        if w > max_val * 0.40:
            ax.text(w - max_val * 0.015, cy, label,
                    ha='right', va='center', fontsize=fontsize,
                    color='#FFFFFF')
        else:
            ax.text(w + max_val * 0.015, cy, label,
                    ha='left', va='center', fontsize=fontsize,
                    color=GRAY_DK)


def plot_model_size(runs):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_title('TFLite Model Size')
    sizes = [rd['tflite_kb'] for rd in runs]
    bars = ax.barh(range(len(runs)), sizes, color=RUN_COLORS[:len(runs)],
                   edgecolor=GRAY_DK, linewidth=0.5, zorder=3, height=0.6)
    labels = [f'{v/1024:.1f} MB' if v >= 1024 else f'{v:.0f} KB' for v in sizes]
    _label_hbar(ax, bars, labels, max(sizes), fontsize=8.5)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([rd['name'] for rd in runs])
    ax.set_xlabel('Size (KB)')
    ax.grid(axis='x'); ax.set_axisbelow(True)
    ax.invert_yaxis()
    _add_source_note(fig, SOURCE_TFLITE_FILE)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_trainable_params(runs):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_title('Trainable Parameters')
    params = [rd['trainable'] for rd in runs]
    bars = ax.barh(range(len(runs)), params, color=RUN_COLORS[:len(runs)],
                   edgecolor=GRAY_DK, linewidth=0.5, zorder=3, height=0.6)
    labels = [_human_format(v) for v in params]
    _label_hbar(ax, bars, labels, max(params), fontsize=8.5)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([rd['name'] for rd in runs])
    ax.set_xlabel('Parameters')
    ax.xaxis.set_major_formatter(FuncFormatter(_human_format))
    ax.grid(axis='x'); ax.set_axisbelow(True)
    ax.invert_yaxis()
    _add_source_note(fig, SOURCE_KERAS_MODEL)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_confusion_heatmap(runs, row_type='counts'):
    """Single row of confusion heatmaps for all runs.
    Uses a single OSU-orange colormap so magnitudes are comparable across
    panels and the figure stays cohesive in print.
    """
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(2.6*n, 3.4))
    if n == 1:
        axes = [axes]

    is_rates = (row_type == 'rates')
    col_labels = ['Precision', 'Recall', 'F1', 'Specificity'] if is_rates else ['TP', 'FP', 'FN', 'TN']
    subtitle = 'Derived Rates' if is_rates else 'Raw Counts'

    fig.suptitle(f'Confusion Statistics — {subtitle}',
                 fontweight='bold', color=TITLE_COLOR, y=0.99)

    cmap = make_cmap(ORANGE)

    for col, rd in enumerate(runs):
        ax = axes[col]

        if is_rates:
            matrix = np.array([[rd['per_class'][c][k]
                                for k in ['precision','recall','f1','specificity']]
                               for c in CLASSES])
            norm = matrix
        else:
            matrix = np.array([[rd['per_class'][c][k] for k in ['TP','FP','FN','TN']]
                               for c in CLASSES], dtype=float)
            norm = matrix / (matrix.max(axis=0, keepdims=True) + 1e-9)

        ax.imshow(norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        for i in range(len(CLASSES)):
            for j in range(4):
                val = matrix[i, j]
                tc = '#FFFFFF' if norm[i, j] > 0.55 else TEXT
                txt = f'{val:.1%}' if is_rates else f'{int(val):,}'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=7.5, color=tc)

        ax.set_xticks(range(4))
        ax.set_xticklabels(col_labels, fontsize=7.5)
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels([c.capitalize() for c in CLASSES], fontsize=8)
        ax.set_title(rd['name'], fontsize=8.5, color=TEXT, pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRAY_LT); spine.set_linewidth(0.5)

    _add_source_note(fig, SOURCE_KERAS_VAL)
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    return fig


def plot_radar(runs):
    fig = plt.figure(figsize=(5.6, 5.6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_title('Per-Class F1', pad=18)
    angles = np.linspace(0, 2*np.pi, len(CLASSES), endpoint=False).tolist()
    angles += angles[:1]
    for i, rd in enumerate(runs):
        vals = [rd['per_class'][c]['f1'] for c in CLASSES] + \
               [rd['per_class'][CLASSES[0]]['f1']]
        ax.plot(angles, vals, color=RUN_COLORS[i], linewidth=1.4, label=rd['name'])
        ax.fill(angles, vals, color=RUN_COLORS[i], alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=9, color=TEXT)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7, color=GRAY_MD)
    ax.set_facecolor(PANEL)
    ax.spines['polar'].set_color(GRAY_LT)
    ax.grid(color=GRID, linewidth=0.5)
    # Legend below the figure with extra padding so it never overlaps the wheel.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22),
              ncol=min(len(runs), 2), **lkw())
    _add_source_note(fig, SOURCE_KERAS_VAL)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


def _scatter_label_placements(runs):
    """Decide a leader-line offset for each scatter label that avoids
    collisions between points clustered close together.  Points are
    grouped by proximity in axes coordinates; within a cluster the labels
    fan out vertically in alphabetical name order.
    """
    pts = [(i, rd['tflite_kb'], rd['macro_f1']) for i, rd in enumerate(runs)]
    # Cluster by normalised distance in data coords.
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    xr = max(xs) - min(xs) or 1.0
    yr = max(ys) - min(ys) or 1.0

    visited = [False] * len(pts)
    clusters = []
    for i in range(len(pts)):
        if visited[i]:
            continue
        cluster = [i]; visited[i] = True
        for j in range(i+1, len(pts)):
            if visited[j]:
                continue
            dx = (pts[i][1] - pts[j][1]) / xr
            dy = (pts[i][2] - pts[j][2]) / yr
            if (dx*dx + dy*dy) ** 0.5 < 0.10:
                cluster.append(j); visited[j] = True
        clusters.append(cluster)

    placements = {}
    for cluster in clusters:
        # Sort cluster top→bottom by macro_f1 then stack labels with even spacing.
        cluster.sort(key=lambda i: -pts[i][2])
        k = len(cluster)
        for rank, idx in enumerate(cluster):
            dy = 12 + 14 * (rank - (k - 1) / 2)
            placements[idx] = ((14, dy), 'left', 'center')
    return placements


def plot_efficiency_scatter(runs):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_title('Accuracy vs. Model Size')
    placements = _scatter_label_placements(runs)
    for i, rd in enumerate(runs):
        ax.scatter(rd['tflite_kb'], rd['macro_f1'], s=80,
                   color=RUN_COLORS[i], zorder=5,
                   edgecolor=GRAY_DK, linewidth=0.7)
        offset, ha, va = placements[i]
        # leader line from point to label
        ax.annotate(rd['name'], (rd['tflite_kb'], rd['macro_f1']),
                    textcoords='offset points', xytext=offset,
                    fontsize=8, color=TEXT,
                    ha=ha, va=va,
                    arrowprops=dict(arrowstyle='-', color=GRAY_LT,
                                    lw=0.5, shrinkA=4, shrinkB=2))
    ax.set_xlabel('TFLite Size (KB)')
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0.55, 1.05)
    sizes = [rd['tflite_kb'] for rd in runs]
    ax.set_xlim(-max(sizes) * 0.05, max(sizes) * 1.35)
    ax.grid(True); ax.set_axisbelow(True)
    _add_source_note(fig, SOURCE_MIXED)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_val_loss_convergence(runs):
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.set_title('Validation Loss Convergence')
    for i, rd in enumerate(runs):
        ep = np.arange(1, rd['epochs']+1)
        ax.plot(ep, rd['history']['val_loss'], color=RUN_COLORS[i], linewidth=1.4,
                label=f"{rd['name']} (min={rd['best_val_loss']:.4f})", zorder=3)
        best_idx = rd['history']['val_loss'].idxmin()
        ax.scatter(best_idx+1, rd['best_val_loss'], color=RUN_COLORS[i],
                   s=28, zorder=5, edgecolor=GRAY_DK, linewidth=0.6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Loss')
    ax.grid(True); ax.set_axisbelow(True)
    ax.legend(ncol=2, **lkw())
    _add_source_note(fig, SOURCE_TRAIN)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def _add_train_val_legend(fig):
    """Caption clarifying solid vs dashed lines."""
    fig.text(0.01, 0.005,
             'Solid: validation.  Dashed: training.',
             ha='left', va='bottom', fontsize=7.5, color=GRAY_MD)


# Data-source labels — surfaced under each plot so readers can tell
# Keras-derived metrics from TFLite-derived ones at a glance.
SOURCE_TRAIN = 'Keras training history (per-epoch)'
SOURCE_KERAS_VAL = 'Keras float model — validation set'
SOURCE_TFLITE_FILE = 'TFLite int8 file size'
SOURCE_KERAS_MODEL = 'Keras model graph'
SOURCE_TFLITE_INFER = 'TFLite int8 inference (quantized)'
SOURCE_MIXED = 'Macro F1: Keras val  ·  Size: TFLite int8'
SOURCE_DASHBOARD = 'Sources: Keras training/val · TFLite int8 file'


def _add_source_note(fig, source):
    """Bottom-right caption: 'Source: …'."""
    fig.text(0.99, 0.005, f'Source: {source}',
             ha='right', va='bottom', fontsize=7.5, color=GRAY_MD)


# ═══════════════════════════════════════════════════════════════════════
#  COMBINED PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def combined_training_curves(runs):
    """All 5 training metrics + LR schedule in 2×3 grid with a shared legend."""
    metrics = [
        ('loss','val_loss','Loss'), ('bin_acc','val_bin_acc','Binary Accuracy'),
        ('auc','val_auc','AUC'), ('precision','val_precision','Precision'),
        ('recall','val_recall','Recall'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    axes = axes.flatten()
    handles = None
    for idx, (tc, vc, title) in enumerate(metrics):
        ax = axes[idx]
        ax.set_title(title)
        ax.set_xlabel('Epoch'); ax.grid(True); ax.set_axisbelow(True)
        for i, rd in enumerate(runs):
            ep = np.arange(1, rd['epochs']+1)
            ax.plot(ep, rd['history'][tc], color=RUN_COLORS_LIGHT[i],
                    linewidth=0.8, alpha=0.55, linestyle='--')
            ax.plot(ep, rd['history'][vc], color=RUN_COLORS[i],
                    linewidth=1.4, label=rd['name'])
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    # LR panel
    ax_lr = axes[5]
    ax_lr.set_title('Learning Rate Schedule')
    ax_lr.set_xlabel('Epoch'); ax_lr.set_yscale('log')
    ax_lr.grid(True, which='both'); ax_lr.set_axisbelow(True)
    for i, rd in enumerate(runs):
        ep = np.arange(1, rd['epochs']+1)
        ax_lr.plot(ep, rd['history']['learning_rate'], color=RUN_COLORS[i],
                   linewidth=1.4, label=rd['name'])
    # Single shared legend at the bottom — far less visual clutter than per-axes legends.
    fig.legend(handles, labels, loc='lower center', ncol=min(len(runs), 6),
               bbox_to_anchor=(0.5, 0.0), **lkw())
    fig.suptitle('BSB-PicoVision — Training Curves Comparison', y=0.995)
    _add_train_val_legend(fig)
    _add_source_note(fig, SOURCE_TRAIN)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    return fig


def _combined_grouped_bar(ax, runs, metric_key, metric_name, show_legend=False):
    """Grouped bar helper for combined view (legend handled at figure level)."""
    x = np.arange(len(CLASSES))
    n = len(runs)
    w = 0.8 / n
    all_bars = []; all_vals = []
    for i, rd in enumerate(runs):
        vals = [rd['per_class'][c][metric_key] for c in CLASSES]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, color=RUN_COLORS[i], label=rd['name'],
                      edgecolor=BAR_EDGE, linewidth=0.5, zorder=3)
        all_bars.append(bars); all_vals.append(vals)
    _annotate_bars_vertical(ax, all_bars, all_vals, fontsize=5.5, fmt='.0%')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, 1.22)
    ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.set_title(f'Per-Class {metric_name}')
    if show_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                  ncol=min(n, 3), **lkw())


def combined_model_metrics(runs):
    """Per-class P/R/F1 + overall F1 + model size + trainable params."""
    fig = plt.figure(figsize=(11.5, 8.0))
    fig.suptitle('BSB-PicoVision — Model Configuration Comparison', y=0.985)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.32,
                          left=0.06, right=0.97, top=0.92, bottom=0.13)

    _combined_grouped_bar(fig.add_subplot(gs[0,0]), runs, 'precision', 'Precision')
    _combined_grouped_bar(fig.add_subplot(gs[0,1]), runs, 'recall', 'Recall')
    _combined_grouped_bar(fig.add_subplot(gs[0,2]), runs, 'f1', 'F1 Score')

    # Overall F1
    ax_ov = fig.add_subplot(gs[1,0])
    ax_ov.set_title('Overall F1 Scores')
    x = np.arange(len(runs)); w = 0.36
    macro = [rd['macro_f1'] for rd in runs]; micro = [rd['micro_f1'] for rd in runs]
    micro_face = [_lighten(c, 0.45) for c in RUN_COLORS[:len(runs)]]
    b1 = ax_ov.bar(x-w/2, macro, w, color=RUN_COLORS[:len(runs)],
                   edgecolor=GRAY_DK, linewidth=0.5, zorder=3)
    b2 = ax_ov.bar(x+w/2, micro, w, color=micro_face,
                   edgecolor=GRAY_DK, linewidth=0.5, zorder=3)
    for bar, v in zip(b1, macro):
        ax_ov.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                   f'{v:.0%}', ha='center', va='bottom', fontsize=6, color=GRAY_DK)
    for bar, v in zip(b2, micro):
        ax_ov.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                   f'{v:.0%}', ha='center', va='bottom', fontsize=6, color=GRAY_DK)
    ax_ov.set_xticks(x); ax_ov.set_xticklabels([rd['name'] for rd in runs],
                                               fontsize=7, rotation=22, ha='right')
    ax_ov.set_ylabel('F1 Score')
    ax_ov.set_ylim(0, 1.10); ax_ov.grid(axis='y'); ax_ov.set_axisbelow(True)
    ax_ov.legend([Patch(facecolor=GRAY_DK, edgecolor=GRAY_DK),
                  Patch(facecolor=_lighten(GRAY_DK, 0.45), edgecolor=GRAY_DK)],
                 ['Macro F1', 'Micro F1'], loc='upper right', **lkw())

    # Model size
    ax_sz = fig.add_subplot(gs[1,1])
    ax_sz.set_title('TFLite Model Size')
    sizes = [rd['tflite_kb'] for rd in runs]
    bars = ax_sz.barh(range(len(runs)), sizes, color=RUN_COLORS[:len(runs)],
                      edgecolor=GRAY_DK, linewidth=0.5, zorder=3, height=0.6)
    sz_labels = [f'{v/1024:.1f} MB' if v >= 1024 else f'{v:.0f} KB' for v in sizes]
    _label_hbar(ax_sz, bars, sz_labels, max(sizes), fontsize=7.5)
    ax_sz.set_yticks(range(len(runs)))
    ax_sz.set_yticklabels([rd['name'] for rd in runs], fontsize=7.5)
    ax_sz.set_xlabel('Size (KB)'); ax_sz.grid(axis='x'); ax_sz.set_axisbelow(True)
    ax_sz.invert_yaxis()

    # Trainable params
    ax_p = fig.add_subplot(gs[1,2])
    ax_p.set_title('Trainable Parameters')
    params = [rd['trainable'] for rd in runs]
    bars = ax_p.barh(range(len(runs)), params, color=RUN_COLORS[:len(runs)],
                     edgecolor=GRAY_DK, linewidth=0.5, zorder=3, height=0.6)
    p_labels = [_human_format(v) for v in params]
    _label_hbar(ax_p, bars, p_labels, max(params), fontsize=7.5)
    ax_p.set_yticks(range(len(runs)))
    ax_p.set_yticklabels([rd['name'] for rd in runs], fontsize=7.5)
    ax_p.set_xlabel('Parameters')
    ax_p.xaxis.set_major_formatter(FuncFormatter(_human_format))
    ax_p.grid(axis='x'); ax_p.set_axisbelow(True); ax_p.invert_yaxis()

    # Shared run legend below the per-class panels
    handles = [Patch(facecolor=RUN_COLORS[i], edgecolor=BAR_EDGE,
                     label=rd['name']) for i, rd in enumerate(runs)]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(runs), 6),
               bbox_to_anchor=(0.5, 0.0), **lkw())
    _add_source_note(fig, SOURCE_DASHBOARD)
    return fig


def combined_confusion(runs):
    """2-row confusion heatmaps: raw counts + derived rates.
    Both rows share a single OSU-orange colormap for visual consistency.
    """
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(2.6*n, 6.0))
    fig.suptitle('BSB-PicoVision — Confusion Statistics Comparison', y=0.99)
    if n == 1:
        axes = axes.reshape(2, 1)

    cmap = make_cmap(ORANGE)

    for col, rd in enumerate(runs):
        # Row 0 — counts
        ax = axes[0, col]
        mat = np.array([[rd['per_class'][c][k] for k in ['TP','FP','FN','TN']]
                        for c in CLASSES], dtype=float)
        norm = mat / (mat.max(axis=0, keepdims=True) + 1e-9)
        ax.imshow(norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(4):
                tc = '#FFFFFF' if norm[i,j] > 0.55 else TEXT
                ax.text(j, i, f'{int(mat[i,j]):,}', ha='center', va='center',
                        fontsize=7, color=tc)
        ax.set_xticks(range(4)); ax.set_xticklabels(['TP','FP','FN','TN'], fontsize=7.5)
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels([c.capitalize() for c in CLASSES], fontsize=8)
        ax.set_title(f'{rd["name"]}\nRaw Counts', fontsize=8.5, pad=6, color=TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRAY_LT); sp.set_linewidth(0.5)

        # Row 1 — rates
        ax2 = axes[1, col]
        rates = np.array([[rd['per_class'][c][k]
                           for k in ['precision','recall','f1','specificity']]
                          for c in CLASSES])
        ax2.imshow(rates, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(4):
                tc = '#FFFFFF' if rates[i,j] > 0.55 else TEXT
                ax2.text(j, i, f'{rates[i,j]:.1%}', ha='center', va='center',
                         fontsize=7, color=tc)
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['Precision','Recall','F1','Specificity'], fontsize=7)
        ax2.set_yticks(range(len(CLASSES)))
        ax2.set_yticklabels([c.capitalize() for c in CLASSES], fontsize=8)
        ax2.set_title('Derived Rates', fontsize=8.5, pad=6, color=TEXT)
        for sp in ax2.spines.values():
            sp.set_edgecolor(GRAY_LT); sp.set_linewidth(0.5)

    _add_source_note(fig, SOURCE_KERAS_VAL)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    return fig


def combined_efficiency_dashboard(runs):
    """Single-page summary: scatter, radar, degradation, convergence, table."""
    fig = plt.figure(figsize=(13.5, 8.6))

    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.42,
                  left=0.05, right=0.97, top=0.88, bottom=0.07)
    fig.suptitle('BSB-PicoVision — Efficiency & Quality Dashboard', y=0.97)

    # ── Scatter ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Accuracy vs. Model Size')
    placements = _scatter_label_placements(runs)
    for i, rd in enumerate(runs):
        ax1.scatter(rd['tflite_kb'], rd['macro_f1'], s=60, color=RUN_COLORS[i],
                    zorder=5, edgecolor=GRAY_DK, linewidth=0.6)
        offset, ha, va = placements[i]
        ax1.annotate(rd['name'], (rd['tflite_kb'], rd['macro_f1']),
                     textcoords='offset points', xytext=offset,
                     fontsize=7, color=TEXT, ha=ha, va=va,
                     arrowprops=dict(arrowstyle='-', color=GRAY_LT,
                                     lw=0.4, shrinkA=3, shrinkB=2))
    ax1.set_xlabel('TFLite Size (KB)'); ax1.set_ylabel('Macro F1')
    ax1.set_ylim(0.55, 1.05); ax1.grid(True); ax1.set_axisbelow(True)
    sizes = [rd['tflite_kb'] for rd in runs]
    ax1.set_xlim(-max(sizes) * 0.05, max(sizes) * 1.40)

    # ── Radar ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0,1], polar=True)
    # Generous pad keeps the top spoke label clear of the figure title.
    ax2.set_title('Per-Class F1', pad=22)
    base_angles = np.linspace(0, 2*np.pi, len(CLASSES), endpoint=False).tolist()
    angles = base_angles + [base_angles[0]]
    for i, rd in enumerate(runs):
        vals = [rd['per_class'][c]['f1'] for c in CLASSES] + [rd['per_class'][CLASSES[0]]['f1']]
        ax2.plot(angles, vals, color=RUN_COLORS[i], linewidth=1.2, label=rd['name'])
        ax2.fill(angles, vals, color=RUN_COLORS[i], alpha=0.06)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=8, color=TEXT)
    ax2.set_ylim(0, 1.0); ax2.set_yticks([.25,.5,.75,1.0])
    ax2.set_yticklabels(['25%','50%','75%','100%'], fontsize=6.5, color=GRAY_MD)
    ax2.set_facecolor(PANEL)
    ax2.spines['polar'].set_color(GRAY_LT)
    ax2.grid(color=GRID, linewidth=0.5)

    # ── Degradation ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('F1 Relative to Reference')
    baseline = runs[0]['macro_f1']
    ax3.barh(range(len(runs)), [rd['macro_f1'] for rd in runs],
             color=RUN_COLORS[:len(runs)], edgecolor=GRAY_DK, linewidth=0.5,
             height=0.55, zorder=3)
    for i, rd in enumerate(runs):
        pct = rd['macro_f1'] / baseline * 100
        ax3.text(rd['macro_f1']+0.006, i, f"{rd['macro_f1']:.1%}  ({pct:.0f}%)",
                 va='center', fontsize=7, color=GRAY_DK)
    ax3.set_yticks(range(len(runs)))
    ax3.set_yticklabels([rd['name'] for rd in runs], fontsize=7.5)
    ax3.set_xlim(0.5, 1.12); ax3.set_xlabel('Macro F1')
    ax3.grid(axis='x'); ax3.set_axisbelow(True); ax3.invert_yaxis()

    # ── Val loss convergence ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1,0:2])
    ax4.set_title('Validation Loss Convergence')
    for i, rd in enumerate(runs):
        ep = np.arange(1, rd['epochs']+1)
        ax4.plot(ep, rd['history']['val_loss'], color=RUN_COLORS[i], linewidth=1.3,
                 label=f"{rd['name']} (min={rd['best_val_loss']:.4f})", zorder=3)
        best_idx = rd['history']['val_loss'].idxmin()
        ax4.scatter(best_idx+1, rd['best_val_loss'], color=RUN_COLORS[i],
                    s=22, zorder=5, edgecolor=GRAY_DK, linewidth=0.5)
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('Validation Loss')
    ax4.grid(True); ax4.set_axisbelow(True)
    ax4.legend(ncol=2, fontsize=7, **lkw())

    # ── Summary table — IEEE-style: plain monochrome ────────────────
    ax5 = fig.add_subplot(gs[1,2])
    ax5.set_title('Summary')
    ax5.axis('off')
    short_names = []
    for rd in runs:
        n = rd['name']
        idx = n.find('(')
        if idx > 0:
            label = n[:idx].strip(); qual = n[idx:].strip('() ')
        else:
            parts = n.split()
            label = parts[0]
            qual  = next((p for p in parts[1:] if any(c.isdigit() for c in p)), '')
        if len(label) > 12:
            pieces = label.replace(' ', '-').split('-')
            longest = max(pieces, key=len)
            label = longest if len(longest) >= 5 else label[:12]
        short_names.append(f'{label}\n{qual[:14]}' if qual else label)
    headers = ['Metric'] + short_names
    rows = [
        ['Macro F1']      + [f"{rd['macro_f1']:.1%}" for rd in runs],
        ['Micro F1']      + [f"{rd['micro_f1']:.1%}" for rd in runs],
        ['Best Val Loss'] + [f"{rd['best_val_loss']:.4f}" for rd in runs],
        ['Best Val AUC']  + [f"{rd['best_val_auc']:.4f}" for rd in runs],
        ['Best Val Acc']  + [f"{rd['best_val_acc']:.1%}" for rd in runs],
        ['TFLite Size']   + [f"{rd['tflite_kb']:.0f} KB" if rd['tflite_kb']<1024
                              else f"{rd['tflite_kb']/1024:.1f} MB" for rd in runs],
        ['Trainable']     + [_human_format(rd['trainable']) for rd in runs],
        ['Epochs']        + [str(rd['epochs']) for rd in runs],
        ['Input']         + [f"{rd['input_shape'][0]}×{rd['input_shape'][0]}" for rd in runs],
    ]
    table = ax5.table(cellText=rows, colLabels=headers,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(6.5); table.scale(1.0, 1.25)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(GRAY_LT)
        cell.set_linewidth(0.4)
        cell.set_facecolor('#FFFFFF')
        cell.set_text_props(color=TEXT)
        if key[0] == 0:
            # Header row: solid OSU orange band, white text.
            cell.set_facecolor(ORANGE)
            cell.set_text_props(color='#FFFFFF', fontweight='bold')
        elif key[1] == 0:
            # First column: bold metric labels.
            cell.set_text_props(fontweight='bold', color=TEXT)
    _add_source_note(fig, SOURCE_DASHBOARD)
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  TFLITE INFERENCE DEMO
# ═══════════════════════════════════════════════════════════════════════

def run_inference_demo(runs, sample_items, out_dir):
    """
    Run sample images through all TFLite models and produce comparison figures.
    
    sample_items: list of (label, path) tuples
        label  — display name shown above the image (e.g. "Person", "Multi-Class")
        path   — file path to the image
    
    Generates:
        inference_demo.png            — all samples combined
        inference_person.png          — individual per-sample (one each)
        inference_dog.png
        ...
    """
    try:
        import cv2
        import tensorflow as tf
    except ImportError as e:
        print(f"⚠  Skipping inference demo — missing dependency: {e}")
        print("   Install with: pip install tensorflow opencv-python")
        return

    # Validate
    valid_items = []
    for label, p in sample_items:
        if os.path.isfile(p):
            valid_items.append((label, p))
        else:
            print(f"⚠  Sample not found, skipping: {p}")
    if not valid_items:
        print("⚠  No valid sample images found. Skipping inference demo.")
        return

    # Load all TFLite interpreters
    interpreters = []
    for rd in runs:
        tflite_path = os.path.join(rd['dir'], 'BSB-PicoVision.tflite')
        if not os.path.exists(tflite_path):
            print(f"⚠  TFLite not found for {rd['name']}: {tflite_path}")
            interpreters.append(None)
            continue
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        img_size = inp['shape'][1]
        quant = rd['quantization']
        in_scale = inp.get('quantization', (0.0, 0))[0]
        in_zero  = inp.get('quantization', (0.0, 0))[1]
        out_scale = out.get('quantization', (0.0, 0))[0]
        out_zero  = out.get('quantization', (0.0, 0))[1]
        interpreters.append({
            'interp': interp, 'inp': inp, 'out': out,
            'img_size': img_size, 'quant': quant,
            'in_scale': in_scale, 'in_zero': in_zero,
            'out_scale': out_scale, 'out_zero': out_zero,
            'thresholds': np.array(rd['thresholds'], dtype=np.float32),
        })

    def preprocess(img_bgr, info):
        img = cv2.resize(img_bgr, (info['img_size'], info['img_size']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_f = img.astype(np.float32) / 255.0
        if info['quant'] == 'int8' and info['in_scale'] > 0:
            return np.round(img_f / info['in_scale'] + info['in_zero']).astype(np.int8)
        return img_f

    def infer(img_proc, info):
        tensor = np.expand_dims(img_proc, 0)
        info['interp'].set_tensor(info['inp']['index'], tensor)
        info['interp'].invoke()
        raw = info['interp'].get_tensor(info['out']['index'])[0]
        if info['quant'] == 'int8' and info['out_scale'] > 0:
            raw = (raw.astype(np.float32) - info['out_zero']) * info['out_scale']
        return raw

    def _run_all_models(img_bgr):
        """Return list of probability arrays, one per model."""
        all_probs = []
        for info in interpreters:
            if info is None:
                all_probs.append(None)
                continue
            proc = preprocess(img_bgr, info)
            probs = infer(proc, info)
            all_probs.append(np.clip(probs, 0.0, 1.0))
        return all_probs

    def _draw_inference_row(ax_img, ax_bar, img_rgb, label, filename,
                            all_probs, show_legend=False):
        """Draw one image + bar chart row onto the given axes."""
        ax_img.imshow(img_rgb)
        ax_img.set_title(f'{label}\n{filename}', fontsize=9, color=TEXT)
        ax_img.axis('off')

        x = np.arange(len(CLASSES))
        n_runs = len(runs)
        w = 0.8 / n_runs

        all_bars = []; all_vals = []
        for mi, (rd, info, probs) in enumerate(zip(runs, interpreters, all_probs)):
            if probs is None:
                continue
            bars = ax_bar.bar(x + mi*w - 0.4 + w/2, probs, w,
                              color=RUN_COLORS[mi],
                              edgecolor=BAR_EDGE, linewidth=0.5, zorder=3,
                              label=rd['name'] if show_legend else None)
            all_bars.append(bars); all_vals.append(probs)

            # Per-class decision threshold tick marks (kept subtle)
            for ci, th in enumerate(info['thresholds']):
                bx = x[ci] + mi*w - 0.4 + w/2
                ax_bar.plot([bx - w*0.4, bx + w*0.4], [th, th],
                            color=RUN_COLORS[mi], linewidth=0.8,
                            linestyle=':', alpha=0.5, zorder=4)

        _annotate_bars_vertical(ax_bar, all_bars, all_vals, fontsize=6, fmt='.0%')

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([c.capitalize() for c in CLASSES])
        ax_bar.set_ylim(0, 1.18)
        ax_bar.set_ylabel('Probability')
        ax_bar.grid(axis='y'); ax_bar.set_axisbelow(True)
        if show_legend:
            ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                          ncol=min(len(runs), 3), **lkw())

    # ── Pre-compute all inferences ──────────────────────────────────
    results = []  # list of (label, filename, img_rgb, all_probs)
    for label, img_path in valid_items:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"⚠  Could not read image: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        all_probs = _run_all_models(img_bgr)
        filename = os.path.basename(img_path)
        results.append((label, filename, img_rgb, all_probs))

    if not results:
        print("⚠  No images could be read.")
        return

    # ── Combined view (all samples) ─────────────────────────────────
    n_imgs = len(results)
    fig, axes = plt.subplots(n_imgs, 2, figsize=(9.5, 2.6 * n_imgs),
                              gridspec_kw={'width_ratios': [1, 2.6]})
    if n_imgs == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle('BSB-PicoVision — Inference Demo', y=0.99)

    for row, (label, filename, img_rgb, all_probs) in enumerate(results):
        _draw_inference_row(axes[row, 0], axes[row, 1],
                            img_rgb, label, filename, all_probs,
                            show_legend=(row == n_imgs - 1))

    _add_source_note(fig, SOURCE_TFLITE_INFER)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(out_dir, 'inference_demo.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out_path}")

    # ── Individual views (one per sample) ───────────────────────────
    for label, filename, img_rgb, all_probs in results:
        fig_i, axes_i = plt.subplots(1, 2, figsize=(9.5, 3.4),
                                      gridspec_kw={'width_ratios': [1, 2.6]})
        fig_i.suptitle(f'Inference — {label}', y=0.99)
        _draw_inference_row(axes_i[0], axes_i[1],
                            img_rgb, label, filename, all_probs,
                            show_legend=True)
        _add_source_note(fig_i, SOURCE_TFLITE_INFER)
        fig_i.tight_layout(rect=[0, 0.07, 1, 0.94])

        safe_label = label.lower().replace(' ', '_').replace('-', '_')
        out_i = os.path.join(out_dir, f'inference_{safe_label}.png')
        fig_i.savefig(out_i, dpi=300, bbox_inches='tight',
                      facecolor=fig_i.get_facecolor())
        plt.close(fig_i)
        print(f"  ✓ {out_i}")


# ═══════════════════════════════════════════════════════════════════════
#  SAVE HELPER
# ═══════════════════════════════════════════════════════════════════════

def save(fig, path):
    """IEEE-friendly export: 300 dpi raster, tight crop, white background."""
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='BSB-PicoVision Unified Metrics Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python metrics_dashboard.py
  python metrics_dashboard.py --samples person.jpg dog.jpg cat.jpg none.jpg
  python metrics_dashboard.py --runs export_a1.0_640 export_a0.35_b6_96 --names "Reference" "BSB-PicoVision"
  python metrics_dashboard.py --out Results
        """)
    parser.add_argument('--runs', nargs='+', default=None,
                        help='Export directories for each run '
                             '(default: export_a1.0_640, export_a1.0_96, '
                             'export_a0.5_b6_640, export_a0.5_b6_96, '
                             'export_a0.35_b6_640, export_a0.35_b6_96)')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Display names for each run (must match --runs count)')
    parser.add_argument('--out', type=str, default='./Metrics',
                        help='Output directory (default: ./Metrics)')
    parser.add_argument('--samples', nargs='+', default=None,
                        help='Image paths to run through TFLite models for inference demo')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300, IEEE print-ready)')
    args = parser.parse_args()

    run_dirs  = args.runs  or DEFAULT_DIRS
    run_names = args.names or DEFAULT_NAMES[:len(run_dirs)]
    if len(run_names) != len(run_dirs):
        print(f"Error: --names count ({len(run_names)}) must match --runs count ({len(run_dirs)})")
        sys.exit(1)

    for d in run_dirs:
        if not os.path.isdir(d):
            print(f"Error: directory not found: {d}")
            sys.exit(1)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    apply_theme()

    print("Loading run data...")
    runs = []
    for name, d in zip(run_names, run_dirs):
        rd = load_run(name, d)
        runs.append(rd)
        print(f"  {name}: {rd['epochs']} epochs, macro_f1={rd['macro_f1']:.4f}, "
              f"tflite={rd['tflite_kb']:.0f}KB")

    # ── COMBINED VIEWS ──────────────────────────────────────────────
    print("\nGenerating combined views...")
    save(combined_training_curves(runs),       os.path.join(out_dir, 'combined_training_curves.png'))
    save(combined_model_metrics(runs),         os.path.join(out_dir, 'combined_model_metrics.png'))
    save(combined_confusion(runs),             os.path.join(out_dir, 'combined_confusion_matrices.png'))
    save(combined_efficiency_dashboard(runs),  os.path.join(out_dir, 'combined_efficiency_dashboard.png'))

    # ── INDIVIDUAL VIEWS ────────────────────────────────────────────
    print("\nGenerating individual views...")
    save(plot_loss(runs),            os.path.join(out_dir, 'individual_loss.png'))
    save(plot_accuracy(runs),        os.path.join(out_dir, 'individual_accuracy.png'))
    save(plot_auc(runs),             os.path.join(out_dir, 'individual_auc.png'))
    save(plot_precision_curve(runs), os.path.join(out_dir, 'individual_precision_curve.png'))
    save(plot_recall_curve(runs),    os.path.join(out_dir, 'individual_recall_curve.png'))
    save(plot_lr_schedule(runs),     os.path.join(out_dir, 'individual_lr_schedule.png'))

    save(plot_per_class_bars(runs, 'precision', 'Precision'),
         os.path.join(out_dir, 'individual_per_class_precision.png'))
    save(plot_per_class_bars(runs, 'recall', 'Recall'),
         os.path.join(out_dir, 'individual_per_class_recall.png'))
    save(plot_per_class_bars(runs, 'f1', 'F1 Score'),
         os.path.join(out_dir, 'individual_per_class_f1.png'))

    save(plot_overall_f1(runs),      os.path.join(out_dir, 'individual_overall_f1.png'))
    save(plot_model_size(runs),      os.path.join(out_dir, 'individual_model_size.png'))
    save(plot_trainable_params(runs),os.path.join(out_dir, 'individual_trainable_params.png'))

    save(plot_confusion_heatmap(runs, 'counts'),
         os.path.join(out_dir, 'individual_confusion_counts.png'))
    save(plot_confusion_heatmap(runs, 'rates'),
         os.path.join(out_dir, 'individual_confusion_rates.png'))

    save(plot_radar(runs),              os.path.join(out_dir, 'individual_radar.png'))
    save(plot_efficiency_scatter(runs), os.path.join(out_dir, 'individual_efficiency_scatter.png'))
    save(plot_val_loss_convergence(runs),
         os.path.join(out_dir, 'individual_val_loss_convergence.png'))

    # ── INFERENCE DEMO ─────────────────────────────────────────────
    # Build list of (label, path) tuples
    sample_items = []

    if args.samples:
        # From command line — use filename stem as label
        for p in args.samples:
            stem = os.path.splitext(os.path.basename(p))[0]
            sample_items.append((stem.replace('_', ' ').title(), p))
    else:
        # From DEMO_IMAGES config — derive label from key name
        for key, path in DEMO_IMAGES.items():
            if path and os.path.isfile(path):
                # 'PERSON_IMAGE' → 'Person', 'MULTI_IMAGE' → 'Multi-Class'
                label = key.replace('_IMAGE', '').replace('_', '-').title()
                if label.lower() == 'multi':
                    label = 'Multi-Class'
                sample_items.append((label, path))
        if sample_items:
            print(f"\nUsing {len(sample_items)} demo images from DEMO_IMAGES config...")

    if sample_items:
        print("Running inference demo...")
        run_inference_demo(runs, sample_items, out_dir)
    else:
        print("\nNo demo images found. Set paths in DEMO_IMAGES or use --samples.")

    # ── SUMMARY ─────────────────────────────────────────────────────
    all_files = sorted(f for f in os.listdir(out_dir) if f.endswith('.png'))
    print(f"\n{'='*60}")
    print(f"  Done! {len(all_files)} plots saved to {out_dir}/")
    print(f"{'='*60}")
    for f in all_files:
        print(f"  • {f}")


if __name__ == '__main__':
    main()