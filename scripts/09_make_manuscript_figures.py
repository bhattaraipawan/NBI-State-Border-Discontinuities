from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / 'generated'
TOP = RES / 'top5_exact_outputs'
DATA = ROOT / 'data' / 'processed'
RAW = ROOT / 'data' / 'raw'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Tinos'],
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def clean_axes(ax):
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save(fig, out, name):
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{name}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def fig1(out):
    df = pd.read_csv(RES / 'National_2025_NBI_High_Priority_Borders.csv').sort_values('screening_score')
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bars = ax.barh(df['border_pair'], df['screening_score'], height=0.68)
    ax.set_xlabel('Screening score (pre-outcome diagnostics)')
    ax.set_xlim(0, max(85, df['screening_score'].max() + 7))
    for bar, val in zip(bars, df['screening_score']):
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=9)
    clean_axes(ax)
    fig.tight_layout()
    save(fig, out, 'Fig1')


def fig2(out):
    with open(RES / 'Top5_Exact_RD_Results.json', encoding='utf-8') as f:
        meta = json.load(f)
    top5 = meta['selection']['screening_top5']
    rd = pd.read_csv(TOP / 'rd_results.csv')
    d = rd[(rd['bandwidth'] == 10) & (rd['outcome'] == 'lowest_rating')].copy()
    d = d.set_index('border_pair').loc[top5].reset_index()
    y = np.arange(len(d))
    est = d['adjusted_estimate_a_minus_b'].to_numpy()
    lo = d['adjusted_ci_low'].to_numpy()
    hi = d['adjusted_ci_high'].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.errorbar(est, y, xerr=np.vstack([est-lo, hi-est]), fmt='o', capsize=3, linewidth=1.1)
    ax.axvline(0, linestyle='--', linewidth=0.9)
    ax.set_yticks(y, d['border_pair'])
    ax.invert_yaxis()
    ax.set_xlabel('Adjusted rating discontinuity (State A - State B), rating points')
    ax.set_xlim(min(-0.95, float(lo.min()) - 0.08), max(0.85, float(hi.max()) + 0.08))
    clean_axes(ax)
    fig.tight_layout()
    save(fig, out, 'Fig2')


def fig3(out):
    bridges = pd.read_csv(DATA / 'Top5_Exact_Border_Bridge_Data_25mi.csv.gz')
    d = bridges[bridges['border_pair'] == 'KS-MO'].copy()
    states = gpd.read_file('zip://' + str(RAW / 'tl_2025_us_state.zip'))
    states = states[states['STUSPS'].isin(['KS', 'MO'])].to_crs(4326)
    geom = {row.STUSPS: row.geometry for _, row in states.iterrows()}
    border = geom['KS'].boundary.intersection(geom['MO'].boundary)

    fig, ax = plt.subplots(figsize=(5.8, 6.2))
    sc = ax.scatter(d['longitude'], d['latitude'], c=d['lowest_rating'], s=7, alpha=0.72,
                    vmin=0, vmax=9, linewidths=0)
    parts = list(border.geoms) if hasattr(border, 'geoms') else [border]
    for part in parts:
        if hasattr(part, 'xy'):
            x, y = part.xy
            ax.plot(x, y, linewidth=1.0)
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_xlim(d['longitude'].min()-0.05, d['longitude'].max()+0.05)
    ax.set_ylim(d['latitude'].min()-0.03, d['latitude'].max()+0.03)
    clean_axes(ax)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label('Minimum component rating')
    cbar.set_ticks(np.arange(0, 10, 1))
    fig.tight_layout()
    save(fig, out, 'Fig3')


def fig4(out):
    b = pd.read_csv(TOP / 'binned_means.csv')
    d = b[b['border_pair'] == 'KS-MO'].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    left = d[d['bin_center'] < 0]
    right = d[d['bin_center'] > 0]
    ax.plot(left['bin_center'], left['mean_rating'], marker='o', markersize=4, linewidth=1.1, label='Kansas')
    ax.plot(right['bin_center'], right['mean_rating'], marker='o', markersize=4, linewidth=1.1, label='Missouri')
    ax.axvline(0, linestyle='--', linewidth=0.9)
    ax.set_xlabel('Signed distance from border (mi; Kansas negative, Missouri positive)')
    ax.set_ylabel('Mean minimum component rating')
    ax.legend(frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save(fig, out, 'Fig4')


def climate_row(clim, spec, outcome='lowest_rating', bw=10):
    x = clim[(clim['border_pair'] == 'KS-MO') &
             (clim['bandwidth_miles'] == bw) &
             (clim['specification'] == spec) &
             (clim['outcome'] == outcome)]
    if len(x) != 1:
        raise ValueError((spec, outcome, bw, len(x)))
    return x.iloc[0]


def fig5(out):
    clim = pd.read_csv(RES / 'KS_MO_Climate_Model_Results.csv')
    models = pd.read_csv(RES / 'KS_MO_Model_Results.csv')
    with open(RES / 'KS_MO_Final_Robustness_Results.json', encoding='utf-8') as f:
        rob = json.load(f)
    with open(RES / 'KS_MO_Confounder_Matching_Results.json', encoding='utf-8') as f:
        match = json.load(f)

    rows = []
    r = climate_row(clim, 'Bridge/traffic adjusted')
    rows.append(('Bridge/traffic adjusted', r.estimate_a_minus_b, r.ci_low, r.ci_high))
    r = climate_row(clim, 'Climate-adjusted (winter temperature + annual precipitation)')
    rows.append(('Winter climate adjusted', r.estimate_a_minus_b, r.ci_low, r.ci_high))
    r = climate_row(clim, 'Common-age (9.0-91.0 y) + climate')
    rows.append(('Common-age + climate', r.estimate_a_minus_b, r.ci_low, r.ci_high))
    r = climate_row(clim, 'State-owned/state-maintained + climate')
    rows.append(('State agency + climate', r.estimate_a_minus_b, r.ci_low, r.ci_high))
    for label, spec in [
        ('Donut: exclude <0.5 mi', 'Donut: exclude <0.5 mile'),
        ('Donut: exclude <1.0 mi', 'Donut: exclude <1.0 mile'),
        ('Exclude shared-state structures', 'Exclude shared-state structures'),
    ]:
        r = next(x for x in rob['donut'] if x['specification'] == spec)
        rows.append((label, r['estimate_a_minus_b'], r['ci_low'], r['ci_high']))
    r = models[(models['specification'] == 'Expanded overlap weighting') &
               (models['outcome'] == 'lowest_rating')].iloc[0]
    rows.append(('Expanded overlap weighting', r.estimate_a_minus_b, r.ci_low, r.ci_high))
    ms = next(x for x in match['pair_results']['KS-MO']['matched_sets']
              if x['label'] == 'Exposure-tight all-owner match')
    r = next(x for x in ms['stats'] if x['outcome'] == 'lowest_rating')
    rows.append(('Exposure-tight matched pairs', r['mean_a_minus_b'], r['ci_low'], r['ci_high']))

    labels = [x[0] for x in rows]
    est = np.array([x[1] for x in rows])
    lo = np.array([x[2] for x in rows])
    hi = np.array([x[3] for x in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.errorbar(est, y, xerr=np.vstack([est-lo, hi-est]), fmt='o', capsize=3, linewidth=1.1)
    ax.axvline(0, linestyle='--', linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel('Kansas - Missouri rating difference (rating points)')
    ax.set_xlim(min(-0.05, float(lo.min())-0.05), max(0.95, float(hi.max())+0.05))
    clean_axes(ax)
    fig.tight_layout()
    save(fig, out, 'Fig5')


def fig6(out):
    with open(RES / 'KS_MO_Final_Robustness_Results.json', encoding='utf-8') as f:
        rob = json.load(f)
    d = pd.DataFrame(rob['placebo']).sort_values('cutoff')
    x = d['cutoff'].to_numpy()
    est = d['pseudo_left_minus_right'].to_numpy()
    lo = d['ci_low'].to_numpy()
    hi = d['ci_high'].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(x, est, yerr=np.vstack([est-lo, hi-est]), fmt='o', capsize=3, linewidth=1.1)
    ax.axhline(0, linestyle='--', linewidth=0.9)
    ax.axvline(0, linestyle=':', linewidth=0.8)
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_xlabel('Fake cutoff location relative to true border (mi)')
    ax.set_ylabel('Pseudo-discontinuity (rating points)')
    clean_axes(ax)
    fig.tight_layout()
    save(fig, out, 'Fig6')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(RES / 'manuscript_figures'))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6]:
        fn(out)
    print(f'Wrote manuscript figures to {out}')


if __name__ == '__main__':
    main()
