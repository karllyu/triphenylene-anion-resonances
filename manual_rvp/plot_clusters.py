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

# Some roots throw rejected outliers orders of magnitude further out than the points that
# survive, far enough that framing on them squeezes every real point into a pixel. Below
# this share of either axis the full-range view is framed on the surviving points instead
# and the outliers are left off, which the subtitle says outright. The A2 roots split
# cleanly on it: the crowded ones sit at 0.3-1.3%, the rest at 11% and up.
OUTLIER_CROWDING = 0.05

# Im[E] axis factors set by hand, overriding the automatic step. Both of these land on
# 10^-3 on their own, which leaves every tick a fraction; a decade finer reads better.
IMAG_SCALE_OVERRIDES = {
    'results_A2_root2_fit2': 1e-4,
    'results_A2_root3_fit2': 1e-4,
}

# Where a cluster label sits relative to its mean, in points. Roots whose clusters are
# a hair apart in Re[E] would otherwise stack their labels on the same pixel, so each
# label falls through to the first offset that clears the ones already placed.
LABEL_OFFSETS = [(13, 9), (13, -17), (-32, 9), (-32, -17)]
LABEL_CLEARANCE = (30, 12)


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


def engineering_scale(max_abs: float) -> tuple[float, int]:
    """Power-of-1000 factor that puts the largest Im[E] tick in [1, 1000).

    Roots differ by orders of magnitude in how far their outliers reach - some stay
    within 1e-4 a.u. of the real axis, others run past 0.5 - so the axis factor is
    read off the data rather than fixed at the 1e-6 that suits the tightest root.
    """
    if not np.isfinite(max_abs) or max_abs == 0:
        return 1.0, 0
    exponent = 3 * int(np.floor(np.log10(max_abs) / 3))
    return 10.0 ** exponent, exponent


def tick_decimals(ticks: list[float], limit: int = 10) -> int:
    """Fewest decimals that still labels every tick at its true value.

    Re[E] spans anything from 1e-5 wide (a zoom) to order 1 (a root whose outliers
    scatter across the plane). Too few decimals and a label rounds away from the
    gridline it names; too many and the axis carries zeros that say nothing - so the
    precision is taken from the tick spacing.
    """
    if len(ticks) < 2:
        return 2
    step = min(abs(later - earlier) for earlier, later in zip(ticks, ticks[1:]))
    for decimals in range(limit + 1):
        if all(abs(float(f'{tick:.{decimals}f}') - tick) <= 1e-3 * step for tick in ticks):
            return decimals
    return limit


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
        # Zoom limits are set off the clustered points alone, then widened, so the frame
        # keeps enough of the surrounding field for the clusters to read as clusters.
        extent = core[clustered]
        span_x = 1.6 * (extent['real'].max() - extent['real'].min())
        span_y = 1.6 * (extent['imag'].max() - extent['imag'].min())
        locator = MaxNLocator(5)
        show_rejected = True
        subtitle = f'Zoom on the clustered region; {len(filtered)} points plotted in full'
    else:
        spans = [(core[axis].max() - core[axis].min(), filtered[axis].max() - filtered[axis].min())
                 for axis in ('real', 'imag')]
        show_rejected = not any(kept / whole < OUTLIER_CROWDING for kept, whole in spans if whole)

        extent = filtered if show_rejected else core
        span_x = extent['real'].max() - extent['real'].min()
        span_y = extent['imag'].max() - extent['imag'].min()
        locator = MaxNLocator(6)
        # Both subtitles stay under the frame width; a longer one widens the saved canvas
        # past the axes, since the figure is cropped tight around whatever it draws.
        subtitle = (f'All {len(filtered)} points surviving the imag_err filter, '
                    f'including the {len(rejected)} rejected outliers') if show_rejected else (
            f'{len(core)} points surviving the imag_err filter; '
            f'{len(rejected)} rejected outliers off scale')

    xlim = (extent['real'].min() - PAD_LEFT * span_x, extent['real'].max() + PAD_RIGHT * span_x)
    ylim = (extent['imag'].min() - PAD_Y * span_y, extent['imag'].max() + PAD_Y * span_y)
    # Fixed before anything is drawn: every y value below is divided by this factor.
    override = IMAG_SCALE_OVERRIDES.get(results_dir.name)
    imag_scale, imag_exponent = ((override, round(np.log10(override))) if override
                                 else engineering_scale(max(abs(ylim[0]), abs(ylim[1]))))

    if zoom:
        ax.scatter(unclustered['real'], unclustered['imag'] / imag_scale, marker='x', s=20, linewidths=0.8,
                   color=BLACK, alpha=0.8, zorder=2, label=f'Unclustered ({len(unclustered)})')

        for i, (row, mask) in enumerate(zip(reported.itertuples(), members)):
            color, marker = CLUSTER_COLORS[i % len(CLUSTER_COLORS)], CLUSTER_MARKERS[i % len(CLUSTER_MARKERS)]
            ax.scatter(core.loc[mask, 'real'], core.loc[mask, 'imag'] / imag_scale, marker=marker, s=26,
                       facecolor=color, edgecolor=SURFACE, linewidths=0.6, zorder=3,
                       label=f'Cluster {row.cluster:g} ({int(row.size)})')
    else:
        # At full range a cluster spans a fraction of a pixel, so splitting the points by
        # membership buys nothing: one series of x marks, and the means carry the clusters.
        ax.scatter(core['real'], core['imag'] / imag_scale, marker='x', s=20, linewidths=0.8,
                   color=BLACK, alpha=0.8, zorder=2, label=f'Results ({len(core)})')

    if not rejected.empty and show_rejected:
        # A zoom on the clustered region can exclude every outlier; a legend entry for a
        # series with nothing on screen would send the reader hunting for marks.
        inside = rejected['real'].between(*xlim) & rejected['imag'].between(*ylim)
        ax.scatter(rejected['real'], rejected['imag'] / imag_scale, marker='D', s=17, facecolor='none',
                   edgecolor=REJECTED, linewidths=0.9, zorder=4,
                   label=f'Rejected outliers ({len(rejected)})' if inside.any() else '_nolegend_')

    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[0] / imag_scale, ylim[1] / imag_scale)

    placed: list[tuple[float, float]] = []
    for i, row in enumerate(reported.itertuples()):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        mean = (row.real_mean, row.imag_mean / imag_scale)
        # A surface-coloured asterisk underneath widens each spoke into its own outline,
        # so the mean stays legible sitting on top of its own cluster.
        ax.scatter(*mean, marker=MEAN_MARKER, s=70, color=SURFACE, linewidths=2.3, zorder=5)
        ax.scatter(*mean, marker=MEAN_MARKER, s=70, color=color, linewidths=1.15, zorder=6)

        anchor = ax.transData.transform(mean)
        offset = next((candidate for candidate in LABEL_OFFSETS
                       if all(abs(anchor[0] + candidate[0] - x) > LABEL_CLEARANCE[0]
                              or abs(anchor[1] + candidate[1] - y) > LABEL_CLEARANCE[1]
                              for x, y in placed)),
                      LABEL_OFFSETS[-1])
        placed.append((anchor[0] + offset[0], anchor[1] + offset[1]))
        ax.annotate(f'C{row.cluster:g}', mean, textcoords='offset points', xytext=offset,
                    fontsize=9, color=BLACK, weight='bold', zorder=7,
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

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(FormatStrFormatter(
        f'%.{tick_decimals([tick for tick in locator.tick_values(*xlim) if xlim[0] <= tick <= xlim[1]])}f'))
    # The y values are already scaled by hand, so suppress the offset text that would
    # otherwise factor them a second time.
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)

    unit = f'$10^{{{imag_exponent}}}$ a.u.' if imag_exponent else 'a.u.'
    ax.set_xlabel('Re[E] (a.u.)', fontsize=10, color=BLACK)
    ax.set_ylabel(f'Im[E] ({unit})', fontsize=10, color=BLACK)
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
