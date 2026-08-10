"""Plot the Pade roots of a results directory in the complex plane, coloured by cluster.

The clustering pipeline reports only cluster statistics, never per-point labels, so the
membership is recovered here by replaying the same steps clustering.clustering() takes:
the imag_err filter, the MAD outlier trim, StandardScaler, then DBSCAN at each epsilon
listed in output.dat. Each reported cluster is matched back to a DBSCAN label by its
mean and size, so a mismatch raises rather than mislabels the plot.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.home() / 'rvp' / 'rvp-fork' / 'src'))
from clustering import mad_outlier_mask

FONT = 'Courier New'
# mathtext has its own font stack, so the subscript in the title and the exponent in the
# y label would fall back to the default sans unless it is pointed at Courier too.
plt.rcParams.update({
    'font.family': 'monospace',
    'font.monospace': [FONT, 'Courier', 'DejaVu Sans Mono'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': FONT,
    'mathtext.it': f'{FONT}:italic',
    'mathtext.bf': f'{FONT}:bold',
})

IMAG_SCALE = 1e-6
IMAG_UNIT = r'$10^{-6}$'

SURFACE = '#fcfcfb'
BLACK = '#000000'
INK = '#0b0b0b'
# Between the palette's secondary (#52514e) and muted (#898781) ink; 5.6:1 on the surface.
INK_SECONDARY = '#686663'
GRID = '#e1e0d9'
REJECTED = '#d03b3b'
# Categorical slots 1, 3 and 7 of the reference palette. Validated all-pairs against
# the rejected-outlier red on the light surface: worst CVD dE 9.9, worst normal dE 16.3.
CLUSTER_COLORS = ['#2a78d6', '#1baf7a', '#4a3aa7']
CLUSTER_MARKERS = ['o', 's', '^']
# (numsides, style=2, angle) is matplotlib's asterisk marker: eight unfilled spokes.
MEAN_MARKER = (8, 2, 0)

# Fraction of the data span left blank inside the frame; the right side gets more so
# the outermost outliers and the cluster labels do not crowd the border.
PAD_LEFT, PAD_RIGHT, PAD_Y = 0.06, 0.22, 0.06


WORD_NUMBER = re.compile(r'([A-Za-z]+)(\d+)')


def format_title(dir_name: str) -> str:
    """results_A2_root1_fit1 -> 'A_2 root 1 fit 1', with the point group subscripted.

    The leading token is the Mulliken symbol of the irreducible representation, so its
    digit is a subscript; every other trailing digit is just an index needing a space.
    Upright \\mathrm keeps the symbol in the same face as the rest of the title.
    """
    parts = []
    for i, part in enumerate(dir_name.replace('results_', '').split('_')):
        match = WORD_NUMBER.fullmatch(part)
        if match is None:
            parts.append(part)
        elif i == 0:
            parts.append(rf'$\mathrm{{{match.group(1)}}}_{{{match.group(2)}}}$')
        else:
            parts.append(f'{match.group(1)} {match.group(2)}')
    return ' '.join(parts)


def read_reported_clusters(output_dat: Path) -> pd.DataFrame:
    lines = output_dat.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.split()[:2] == ['cluster', 'grade'])
    columns = lines[header].split()
    rows = [line.split() for line in lines[header + 1:] if line.strip()]
    return pd.DataFrame(rows, columns=columns).astype(float)


def recover_membership(results_dir: Path, outlier_sigma: float = 6.0):
    data = pd.read_csv(results_dir / 'clustering_input.csv')
    filtered = data[abs(data['imag_err'] / data['imag']) < 0.25].drop('imag_err', axis='columns').reset_index(drop=True)

    outliers = mad_outlier_mask(filtered, ['real', 'imag'], outlier_sigma)
    rejected = filtered[outliers].reset_index(drop=True)
    core = filtered[~outliers].reset_index(drop=True)

    scaled = pd.DataFrame(StandardScaler().fit_transform(core), index=core.index, columns=core.columns)
    min_samples = min(max(round(len(core) * 0.08), 1), 100)

    reported = read_reported_clusters(results_dir / 'output.dat')
    labels_by_eps: dict[float, np.ndarray] = {}
    members = []

    for row in reported.itertuples():
        if row.epsilon not in labels_by_eps:
            labels_by_eps[row.epsilon] = DBSCAN(eps=row.epsilon,
                                                min_samples=min_samples).fit(scaled[['real', 'imag']]).labels_
        labels = labels_by_eps[row.epsilon]

        match = None
        for label in np.unique(labels[labels >= 0]):
            mask = labels == label
            if (mask.sum() == round(row.size)
                    and np.isclose(core.loc[mask, 'real'].mean(), row.real_mean, rtol=1e-6)
                    and np.isclose(core.loc[mask, 'imag'].mean(), row.imag_mean, rtol=1e-6)):
                match = mask
                break
        if match is None:
            raise ValueError(f'could not match reported cluster {row.cluster:g} at epsilon {row.epsilon} '
                             f'in {results_dir.name}')
        members.append(match)

    return filtered, core, rejected, reported, members


def plot(results_dir: Path, output_path: Path, zoom: bool = False) -> Path:
    filtered, core, rejected, reported, members = recover_membership(results_dir)

    clustered = np.logical_or.reduce(members) if members else np.zeros(len(core), dtype=bool)
    unclustered = core[~clustered]

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    if zoom:
        ax.scatter(unclustered['real'], unclustered['imag'] / IMAG_SCALE, marker='x', s=20, linewidths=0.8,
                   color=BLACK, alpha=0.8, zorder=2, label=f'Unclustered ({len(unclustered)})')

        for i, (row, mask) in enumerate(zip(reported.itertuples(), members)):
            color, marker = CLUSTER_COLORS[i % len(CLUSTER_COLORS)], CLUSTER_MARKERS[i % len(CLUSTER_MARKERS)]
            ax.scatter(core.loc[mask, 'real'], core.loc[mask, 'imag'] / IMAG_SCALE, marker=marker, s=26,
                       facecolor=color, edgecolor=SURFACE, linewidths=0.6, zorder=3,
                       label=f'Cluster {row.cluster:g} ({int(row.size)})')
    else:
        # At full range a cluster spans a fraction of a pixel, so splitting the points by
        # membership buys nothing: one series of x marks, and the means carry the clusters.
        ax.scatter(core['real'], core['imag'] / IMAG_SCALE, marker='x', s=20, linewidths=0.8,
                   color=BLACK, alpha=0.8, zorder=2, label=f'Results ({len(core)})')

    if not rejected.empty:
        ax.scatter(rejected['real'], rejected['imag'] / IMAG_SCALE, marker='D', s=17, facecolor='none',
                   edgecolor=REJECTED, linewidths=0.9, zorder=4, label=f'Rejected outliers ({len(rejected)})')

    for i, row in enumerate(reported.itertuples()):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        # A surface-coloured asterisk underneath widens each spoke into its own outline,
        # so the mean stays legible sitting on top of its own cluster.
        ax.scatter(row.real_mean, row.imag_mean / IMAG_SCALE, marker=MEAN_MARKER, s=70,
                   color=SURFACE, linewidths=2.3, zorder=5)
        ax.scatter(row.real_mean, row.imag_mean / IMAG_SCALE, marker=MEAN_MARKER, s=70,
                   color=color, linewidths=1.15, zorder=6)
        ax.annotate(f'C{row.cluster:g}', (row.real_mean, row.imag_mean / IMAG_SCALE),
                    textcoords='offset points', xytext=(13, 9), fontsize=9, color=BLACK, weight='bold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=SURFACE, edgecolor='none', alpha=0.8))

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle='none', marker=MEAN_MARKER, markersize=7,
                          markeredgecolor=BLACK, markeredgewidth=1.0))
    labels.append('Cluster mean')

    # Above the axes rather than inside them: the outliers reach the corners, and a
    # boxed legend would sit on top of the very points this plot exists to show.
    legend_columns = 3
    legend_rows = -(-len(labels) // legend_columns)
    legend = ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=legend_columns,
                       fontsize=8.5, frameon=False, handletextpad=0.4, columnspacing=1.4)
    for text in legend.get_texts():
        text.set_color(BLACK)

    if zoom:
        extent = core[clustered]
        # Zoom limits are set off the clustered points alone, then widened, so the frame
        # keeps enough of the surrounding field for the clusters to read as clusters.
        span_x = 1.6 * (extent['real'].max() - extent['real'].min())
        span_y = 1.6 * (extent['imag'].max() - extent['imag'].min())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        subtitle = f'Zoom on the clustered region; {len(filtered)} points plotted in full'
    else:
        extent = filtered
        span_x = extent['real'].max() - extent['real'].min()
        span_y = extent['imag'].max() - extent['imag'].min()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        ax.xaxis.set_major_locator(MaxNLocator(6))
        subtitle = (f'All {len(filtered)} points surviving the imag_err filter, '
                    f'including the {len(rejected)} rejected outliers')

    ax.set_xlim(extent['real'].min() - PAD_LEFT * span_x, extent['real'].max() + PAD_RIGHT * span_x)
    ax.set_ylim((extent['imag'].min() - PAD_Y * span_y) / IMAG_SCALE,
                (extent['imag'].max() + PAD_Y * span_y) / IMAG_SCALE)

    ax.set_xlabel('Re[E] (a.u.)', fontsize=10, color=BLACK)
    ax.set_ylabel(f'Im[E] ({IMAG_UNIT} a.u.)', fontsize=10, color=BLACK)
    # Title and subtitle clear the legend, which sits above the axes and grows upward.
    ax.set_title(format_title(results_dir.name), fontsize=12, color=INK,
                 loc='left', pad=40 + 15 * legend_rows)
    ax.text(0, 1.045 + 0.055 * legend_rows, subtitle, transform=ax.transAxes, fontsize=8.5,
            color=INK_SECONDARY)

    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.9)
    ax.tick_params(which='both', top=True, right=True, direction='in', colors=BLACK,
                   labelsize=9, length=4, width=0.9)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot Pade roots coloured by cluster for a results directory.')
    parser.add_argument('results_dirs', nargs='+', help='Directories such as results_A2_root1_fit1')
    parser.add_argument('--zoom', action='store_true', help='Also write a zoom on the clustered region')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exit_code = 0

    for raw_dir in args.results_dirs:
        results_dir = Path(raw_dir)
        try:
            print(f'wrote {plot(results_dir, results_dir / "cluster_plot.png")}')
            if args.zoom:
                print(f'wrote {plot(results_dir, results_dir / "cluster_plot_zoom.png", zoom=True)}')
        except Exception as exc:
            exit_code = 1
            print(f'{results_dir.name}: {exc}')

    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
