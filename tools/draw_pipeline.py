"""Draw Rohan Sai's Michigan model pipeline from the published split manifest."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


def draw():
    manifest = json.loads((ROOT / 'docs/site-data.json').read_text(encoding='utf-8'))
    full, sample = manifest['full_rows'], manifest['browser_rows']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(16, 14), facecolor='#09121d')
    ax.set(xlim=(0, 16), ylim=(0, 14))
    ax.set_axis_off()
    ink, muted, gold, blue = '#eef4fa', '#b4c5d5', '#f5c663', '#87bbf5'

    def box(x, y, w, h, title, body, accent=blue):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.14',
                                   linewidth=1.3, edgecolor=accent, facecolor='#142334', zorder=2))
        ax.text(x + 0.20, y + h - 0.24, title, color=accent, fontsize=12, weight='bold', va='top', zorder=3)
        ax.text(x + 0.20, y + h - 0.61, body, color=ink, fontsize=10.2, va='top', linespacing=1.55, zorder=3)

    def arrow(points, color=blue, label=None, label_xy=None):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.5, zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=14,
                                     linewidth=1.5, color=color, zorder=1))
        if label:
            ax.text(*label_xy, label, color=muted, fontsize=9, ha='center', va='center',
                    bbox={'facecolor': '#09121d', 'edgecolor': 'none', 'pad': 3}, zorder=4)

    ax.text(0.5, 13.65, 'MICHIGAN OUTAGE MODEL · PIPELINE', color=ink, fontsize=21, weight='bold')
    ax.text(0.5, 13.28, 'Project by Rohan Sai  |  County-level, next-three-hour high-burden classification',
            color=gold, fontsize=11)

    box(0.6, 11.95, 7.0, 0.95, 'OUTAGE HISTORY', 'EAGLE-I county outage observations')
    box(8.4, 11.95, 7.0, 0.95, 'WEATHER + CONTEXT', 'IEM ASOS observations, county geography and calendar')
    arrow([(4.1, 11.95), (4.1, 11.67), (8.0, 11.67), (8.0, 11.35)])
    arrow([(11.9, 11.95), (11.9, 11.67), (8.0, 11.67), (8.0, 11.35)])

    box(0.6, 10.1, 14.8, 1.25, 'AUDITED PREPARED DATASET',
        f"{manifest['original_rows']:,} county/time windows · 83 counties · sources already joined in the supplied CSV\n"
        'Verify checksum, county/time keys, duplicate records, chronological splits and complete targets.')
    arrow([(8, 10.1), (8, 9.8)])
    box(0.6, 8.55, 14.8, 1.25, 'TARGET CHECKS + 67 NUMERIC PREDICTORS',
        'High burden = a next-three-hour hourly mean reaches the county training-period p90 threshold.\n'
        'Exclude future outcomes, IDs and split fields; retain missing inputs for native model handling.')

    for center in [2.8, 8.0, 13.2]:
        arrow([(8, 8.55), (8, 8.28), (center, 8.28), (center, 7.95)])

    box(0.6, 6.65, 4.4, 1.3, 'TRAIN · 2014–2022',
        f"Full research: {full['train']:,} rows\nBrowser sample: {sample['train']:,} rows", gold)
    box(5.8, 6.65, 4.4, 1.3, 'VALIDATE · 2023',
        f"Full research: {full['validation']:,} rows\nBrowser sample: {sample['validation']:,} rows", gold)
    box(11.0, 6.65, 4.4, 1.3, 'RESERVE TEST · 2024',
        f"Full research: {full['test']:,} rows\nBrowser sample: {sample['test']:,} rows")

    arrow([(2.8, 6.65), (2.8, 6.25)], gold)
    box(0.6, 4.55, 4.4, 1.7, 'FIT BOOSTED TREES',
        '180 boosting iterations · 15 leaves\nInverse-frequency sample weights\nMissing values handled natively', gold)
    arrow([(8.0, 6.65), (8.0, 6.25)], gold)
    box(5.8, 4.55, 4.4, 1.7, 'SELECT ALERT THRESHOLD',
        'Use 2023 scores only\nMaximize recall with FPR ≤ 2%\nand ≥ 25 alerts; otherwise abstain', gold)
    arrow([(5.0, 5.4), (5.8, 5.4)], gold, 'scores', (5.4, 5.68))

    arrow([(2.8, 4.55), (2.8, 3.75), (5.8, 3.75)], gold)
    arrow([(8, 4.55), (8, 4.2)], gold)
    box(5.8, 3.05, 4.4, 1.15, 'FREEZE MODEL + THRESHOLD',
        'All choices fixed before test scoring', gold)

    arrow([(13.2, 6.65), (13.2, 4.2)])
    box(11.0, 3.05, 4.4, 1.15, 'EVALUATE ON 2024',
        'Accuracy, F1, ROC/PR curves,\nconfusion matrix + no-alert baseline')
    arrow([(10.2, 3.65), (11.0, 3.65)], gold)

    box(0.6, 1.0, 4.4, 1.2, 'PREDICTION INPUTS',
        'Selected historical rows or a CSV\nOnly the 67 predictors enter the model')
    box(5.8, 1.0, 9.6, 1.2, 'PREDICT → SCORE → HIGH-BURDEN ALERT / NO ALERT',
        'Apply the same fitted model and 2023-selected threshold.\nActual labels are displayed afterward for comparison; predictions do not retrain the model.', gold)
    arrow([(5.0, 1.6), (5.8, 1.6)])
    arrow([(8.0, 3.05), (8.0, 2.2)], gold)
    ax.text(0.6, 0.48, 'Chronology is preserved. The 2024 test does not select the model or threshold.', color=gold, fontsize=11)
    ax.text(0.6, 0.12, 'Historical county-burden research · uncalibrated scores · full-research and browser-sample results are labelled separately.',
            color=muted, fontsize=9.3)

    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.99)
    (ROOT / 'docs/pipeline').mkdir(exist_ok=True)
    (ROOT / '.temp').mkdir(exist_ok=True)
    svg_path = ROOT / 'docs/pipeline/michigan_pipeline.svg'
    fig.savefig(svg_path, facecolor=fig.get_facecolor(), metadata={'Date': None})
    svg_path.write_text(svg_path.read_text(encoding='utf-8').replace(
        "font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif"), encoding='utf-8')
    fig.savefig(ROOT / '.temp/michigan_pipeline.png', dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Created the website SVG and high-resolution PNG pipeline diagram.')


if __name__ == '__main__':
    draw()
