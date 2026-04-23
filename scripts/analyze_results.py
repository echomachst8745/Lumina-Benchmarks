"""Analyzes Google Benchmark JSON results."""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
NAME_RE = re.compile('(?:BM_|Benchmark)(?P<op>\\w+)<bench::(?P<lib>\\w+)Tag<(?P<scalar>\\w+)>,\\s*(?P<N>\\d+)>/repeats:(?P<reps>\\d+)_(?P<agg>mean|median|stddev|cv)')

def ParseOneJson(path: Path) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    rows: list[dict] = []
    for entry in payload.get('benchmarks', []):
        m = NAME_RE.match(entry['name'])
        if not m:
            continue
        unit = entry.get('time_unit', 'ns')
        time_to_ns = {'ns': 1.0, 'us': 1000.0, 'ms': 1000000.0, 's': 1000000000.0}[unit]
        real_ns = entry['real_time'] * time_to_ns
        cpu_ns = entry['cpu_time'] * time_to_ns
        rows.append({'source_file': path.name, 'op': m.group('op'), 'lib': m.group('lib'), 'scalar': m.group('scalar'), 'N': int(m.group('N')), 'repetitions': int(m.group('reps')), 'aggregate': m.group('agg'), 'real_time_ns': real_ns, 'cpu_time_ns': cpu_ns, 'iterations': entry.get('iterations'), 'items_per_second': entry.get('items_per_second'), 'bytes_per_second': entry.get('bytes_per_second')})
    return rows

def Collect(results_dir: Path) -> pd.DataFrame:
    json_files = sorted(results_dir.glob('*.json'))
    if not json_files:
        sys.exit(f'No *.json files found under {results_dir}')
    all_rows: list[dict] = []
    for jf in json_files:
        all_rows.extend(ParseOneJson(jf))
    if not all_rows:
        sys.exit('Parsed JSON files but no benchmark rows matched the expected name pattern.')
    return pd.DataFrame(all_rows)

def WriteLongCsv(df: pd.DataFrame, out_dir: Path) -> Path:
    out = out_dir / 'long.csv'
    df.sort_values(['op', 'scalar', 'N', 'lib', 'aggregate']).to_csv(out, index=False)
    return out

def WritePerOpPivots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(exist_ok=True)
    means = df[df['aggregate'] == 'mean']
    written: list[Path] = []
    for op, sub in means.groupby('op'):
        pivot = sub.pivot_table(index='N', columns=['scalar', 'lib'], values='real_time_ns', aggfunc='first').sort_index()
        path = out_dir / f'{op}.csv'
        pivot.to_csv(path)
        written.append(path)
    return written

def WriteSpeedupCsv(df: pd.DataFrame, out_dir: Path) -> Path:
    means = df[df['aggregate'] == 'mean'].copy()
    pivot = means.pivot_table(index=['op', 'scalar', 'N'], columns='lib', values='real_time_ns', aggfunc='first')
    if 'Lumina' not in pivot.columns:
        sys.exit('Lumina results missing -- cannot compute speedup baseline.')
    speedup = pivot.div(pivot['Lumina'], axis=0).rdiv(1.0)
    out = out_dir / 'speedup_vs_lumina.csv'
    speedup.to_csv(out)
    return out

def MakePlots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(exist_ok=True)
    medians = df[df['aggregate'] == 'median']
    written: list[Path] = []
    style = {'Lumina': ('o-', 'tab:blue'), 'Eigen': ('s-', 'tab:orange'), 'Arma': ('^-', 'tab:green')}
    for op, sub in medians.groupby('op'):
        for scalar, ssub in sub.groupby('scalar'):
            fig, ax = plt.subplots(figsize=(7.5, 5))
            for lib, lsub in ssub.groupby('lib'):
                lsub = lsub.sort_values('N')
                marker, colour = style.get(lib, ('x-', None))
                ax.plot(lsub['N'], lsub['real_time_ns'], marker, label=lib, color=colour, linewidth=1.5, markersize=6)
            sizes = sorted(ssub['N'].unique())
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xticks(sizes)
            ax.set_xticklabels([f'{n}x{n}' for n in sizes])
            ax.set_xlabel('Matrix size')
            ax.set_ylabel('Median wall time [ns]')
            ax.set_title(f'{op}  ({scalar})')
            ax.grid(True, which='both', linestyle=':', alpha=0.5)
            ax.legend()
            fig.tight_layout()
            path = out_dir / f'{op}_{scalar}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            written.append(path)
    return written

def Main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results_dir', type=Path, help='Directory containing the *.json files from one run.')
    args = ap.parse_args()
    if not args.results_dir.is_dir():
        sys.exit(f'Not a directory: {args.results_dir}')
    df = Collect(args.results_dir)
    print(f"Parsed {len(df)} aggregate rows across {df['op'].nunique()} ops, {df['lib'].nunique()} libs, {df['scalar'].nunique()} scalar types, {df['N'].nunique()} sizes.")
    analysis_dir = args.results_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    long_path = WriteLongCsv(df, analysis_dir)
    print(f'  long-format    -> {long_path}')
    pivot_paths = WritePerOpPivots(df, analysis_dir / 'by_op')
    print(f"  per-op pivots  -> {len(pivot_paths)} CSV(s) under {analysis_dir / 'by_op'}")
    speedup_path = WriteSpeedupCsv(df, analysis_dir)
    print(f'  speedup        -> {speedup_path}')
    plot_paths = MakePlots(df, analysis_dir / 'plots')
    print(f"  plots          -> {len(plot_paths)} PNG(s) under {analysis_dir / 'plots'}")
if __name__ == '__main__':
    Main()
