"""Render publication figures from the archived additional-study tables."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NETS = ['IEEE 33', 'SimBench MV', 'SimBench assets']
LABELS = ['IEEE 33-bus feeder', 'SimBench aggregate factors', 'SimBench original asset profiles']
NAVY, TEAL, AMBER, GRAY = '#123C68', '#087F86', '#C88908', '#687887'


def save(fig, out, number):
    fig.savefig(out / f'Figure_{number}.pdf', bbox_inches='tight')
    fig.savefig(out / f'Figure_{number}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results/epsr-2026'))
    ap.add_argument('--output', type=Path, default=Path('manuscript/epsr_figures'))
    args = ap.parse_args(argv)
    p, out = args.results, args.output
    out.mkdir(parents=True, exist_ok=True)
    o, b, r = [pd.read_csv(p / name) for name in ['overall.csv', 'block_results.csv', 'matched_coverage.csv']]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'axes.labelsize': 10, 'axes.titlesize': 11, 'legend.fontsize': 9})
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), layout='constrained')
    for ax, net, label in zip(axes, NETS, LABELS):
        t = o[o.network == net].set_index('controller')
        for j, (policy, color) in enumerate([('binary', NAVY), ('ternary', AMBER)]):
            modes = [f'Agreement {policy}', f'Composite {policy}']
            vals = 100 * t.loc[modes, 'violation_rate'].to_numpy()
            bars = ax.bar(np.arange(2) + (j - .5) * .30, vals, width=.28, color=color)
            for bar, mode in zip(bars, modes):
                ax.annotate(str(int(t.loc[mode, 'violations'])), (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
        ax.axhline(100*t.loc['Ungated', 'violation_rate'], color=GRAY, linestyle='--', lw=1.4)
        ax.axhline(100*t.loc['Always half', 'violation_rate'], color=TEAL, linestyle=':', lw=1.6)
        ax.set_xticks([0, 1], ['Agreement score', 'Composite score'])
        ax.set_ylabel('Violations (%)')
        ax.set_title(label, loc='left', fontweight='bold')
        ax.set_ylim(0, max(t.loc[['Agreement binary', 'Composite binary', 'Agreement ternary', 'Composite ternary', 'Ungated'], 'violation_rate']) * 125 + .025)
        ax.grid(axis='y', alpha=.15)
    handles = [plt.Rectangle((0,0),1,1,color=NAVY,label='Binary policy'),
               plt.Rectangle((0,0),1,1,color=AMBER,label='Three-regime policy'),
               Line2D([0],[0],color=GRAY,ls='--',label='Ungated'),
               Line2D([0],[0],color=TEAL,ls=':',label='Always half')]
    fig.legend(handles=handles, loc='outside upper center', ncol=2, frameon=False)
    save(fig, out, 2)

    modes = ['Ungated', 'Always half', 'Agreement ternary', 'Composite ternary', 'Reference forecast', 'Stress screen']
    mode_labels = ['Ungated', 'Always half', 'Agreement, three regimes', 'Composite, three regimes', 'Reference forecast', 'Stress screen']
    scenarios = ['Clean', 'Missing', 'Delayed', 'Noisy', 'Bias', 'Disturbance', 'Reference outage', 'Reference noisy', 'Common bias']
    scenario_labels = ['Clean', 'Missing', 'Delay', 'Noise', 'Bias', 'Disturbance', 'Ref. outage', 'Ref. noise', 'Common bias']
    tables = []
    for net in NETS:
        t = b[b.network == net].groupby(['controller', 'scenario']).violation_rate.mean().unstack()
        tables.append(100*t.loc[modes, scenarios].to_numpy())
    maximum = max(float(x.max()) for x in tables)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 6.8), layout='constrained')
    for ax, vals, label in zip(axes, tables, LABELS):
        im = ax.imshow(vals, cmap='Blues', vmin=0, vmax=maximum, aspect='auto')
        for i in range(len(modes)):
            for j in range(len(scenarios)):
                ax.text(j, i, f'{vals[i,j]:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if vals[i,j] > .60*maximum else NAVY)
        ax.set_yticks(range(len(modes)), mode_labels)
        ax.set_xticks(range(len(scenarios)), scenario_labels)
        ax.tick_params(axis='both', length=0, labelsize=9)
        ax.set_title(label, loc='left', fontweight='bold')
    fig.colorbar(im, ax=axes, location='right', shrink=.85, label='Realised constraint violations (%)', fraction=.025)
    save(fig, out, 3)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.2), layout='constrained')
    for ax, net, label in zip(axes, NETS, LABELS):
        for score, color, marker in [('Agreement', TEAL, 'o'), ('Composite', NAVY, 's')]:
            t = r[(r.network == net) & (r.score == score)].sort_values('coverage')
            ax.plot(100*t.coverage, 100*t.risk, color=color, marker=marker, label=score)
        ax.set_title(label, loc='left', fontweight='bold')
        ax.set_ylabel('Selective risk (%)')
        ax.set_xticks([50, 75, 90])
        ax.grid(alpha=.15)
        ax.legend(frameon=False, loc='best')
    axes[-1].set_xlabel('Coverage of eligible common proposals (%)')
    save(fig, out, 4)

    usecols = ['network', 'block', 'scenario', 'seed', 'controller', 'hour', 'vmax', 'command_mw', 'next_soc']
    selected = []
    for chunk in pd.read_csv(p / 'decisions.csv', usecols=usecols, chunksize=20000):
        rows = chunk[(chunk.network == 'SimBench assets') & (chunk.block == '2016-11-01 00:00:00')
                     & (chunk.scenario == 'Noisy') & (chunk.seed == 11)]
        if len(rows): selected.append(rows)
    trace = pd.concat(selected, ignore_index=True)
    trace.to_csv(out / 'Figure_5_trace.csv', index=False)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.3), sharex=True, layout='constrained')
    for mode, label, color in [('Ungated','Ungated',GRAY), ('Always half','Always half',TEAL),
                                ('Agreement ternary','Agreement',AMBER), ('Composite ternary','Composite',NAVY)]:
        t = trace[trace.controller == mode].sort_values('hour')
        axes[0].plot(t.hour, t.vmax, color=color, label=label, lw=1.25)
        axes[1].step(t.hour, t.command_mw, where='mid', color=color, lw=1.25)
        axes[2].plot(t.hour, t.next_soc, color=color, lw=1.25)
    axes[0].axhline(1.05,color='#A54739',ls='--',lw=1.)
    axes[0].set_ylabel('Maximum voltage (p.u.)')
    axes[0].legend(frameon=False, ncol=2, loc='best')
    axes[1].set_ylabel('Issued power (MW)')
    axes[2].set_ylabel('State of charge')
    axes[2].axhline(.15,color=GRAY,ls=':',lw=.8)
    axes[2].axhline(.90,color=GRAY,ls=':',lw=.8)
    axes[2].set_xlabel('Decision hour in the 72-hour block')
    for ax in axes: ax.grid(alpha=.15)
    save(fig, out, 5)
    print('Rendered Figures 2-5 from verified tables and per-hour records.')


if __name__ == '__main__':
    main()
