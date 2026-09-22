"""
Plots search efficiency for DEHB and Random Search across all nine
architecture-dataset pairs.

Each plot shows the best validation MAE found so far against the cumulative
number of training epochs allocated to the search. This allows both methods
to be compared using the same epoch budget.

Run with:
    python plots.py
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt

OWN = '/content/drive/MyDrive/AutoSTGNN/search_results/'
SH = '/content/drive/MyDrive/Shared Results/'

LABEL = {
    'agcrn': 'AGCRN',
    'stgcn': 'STCN',
    'graphwavenet': 'Graph WaveNet',
    'metrla': 'METR-LA',
    'pemsbay': 'PEMS-BAY',
    'electricity': 'Electricity'
}

MODELS = ['agcrn', 'stgcn', 'graphwavenet']
DATASETS = ['metrla', 'pemsbay', 'electricity']

# DEHB search logs for each model and dataset
DLOG = {
    ('agcrn', 'metrla'):             OWN + 'agcrn_metrla_dehb_20260905.json',
    ('agcrn', 'pemsbay'):            OWN + 'agcrn_pemsbay_dehb_20260907.json',
    ('agcrn', 'electricity'):        OWN + 'agcrn_electricity_dehb_20260909.json',
    ('stgcn', 'metrla'):             OWN + 'stgcn_metrla_dehb_20260904.json',
    ('stgcn', 'pemsbay'):            OWN + 'stgcn_pemsbay_dehb_20260907.json',
    ('stgcn', 'electricity'):        OWN + 'stgcn_electricity_dehb_20260909.json',
    ('graphwavenet', 'metrla'):      OWN + 'graphwavenet_metrla_dehb_20260909.json',
    ('graphwavenet', 'pemsbay'):     OWN + 'graphwavenet_pemsbay_dehb_20260910.json',
    ('graphwavenet', 'electricity'): OWN + 'graphwavenet_electricity_dehb_20260910.json',
}

# Random Search logs for each model and dataset
# Graph WaveNet results were saved as Excel files
RLOG = {
    ('agcrn', 'metrla'):             SH + 'Random Search/agcrn/metrla/agcrn_metrla_RS_20260819.json',
    ('agcrn', 'pemsbay'):            SH + 'Random Search/agcrn/pemsbay/agcrn_pemsbay_RS_20260820.json',
    ('agcrn', 'electricity'):        SH + 'Random Search/agcrn/electricity/agcrn_electricity_RS_20260909.json',
    ('stgcn', 'metrla'):             SH + 'Random Search/stgcn/metrla/stgcn_metrla_RS_20260819.json',
    ('stgcn', 'pemsbay'):            SH + 'Random Search/stgcn/pemsbay/stgcn_pemsbay_RS_20260819.json',
    ('stgcn', 'electricity'):        SH + 'Random Search/stgcn/electricity/stgcn_electricity_RS_20260908.json',
    ('graphwavenet', 'metrla'):      SH + 'Random Search/gwn/metrla/graphwavenet_metrla_RS_20260819.xlsx',
    ('graphwavenet', 'pemsbay'):     SH + 'Random Search/gwn/pemsbay/graphwavenet_pemsbay_RS_20260820.xlsx',
    ('graphwavenet', 'electricity'): SH + 'Random Search/gwn/electricity/graphwavenet_electricity_RS_20260908.xlsx',
}

# Number of Random Search trials and the fixed fidelity used for each pair
NTRIALS = {
    ('agcrn', 'metrla'): 25,
    ('stgcn', 'metrla'): 25,
    ('graphwavenet', 'metrla'): 25,
    ('agcrn', 'pemsbay'): 20,
    ('stgcn', 'pemsbay'): 20,
    ('graphwavenet', 'pemsbay'): 20,
    ('agcrn', 'electricity'): 20,
    ('stgcn', 'electricity'): 20,
    ('graphwavenet', 'electricity'): 20
}

MAXFID = {
    ('agcrn', 'metrla'): 11,
    ('stgcn', 'metrla'): 11,
    ('graphwavenet', 'metrla'): 12,
    ('agcrn', 'pemsbay'): 11,
    ('stgcn', 'pemsbay'): 11,
    ('graphwavenet', 'pemsbay'): 12,
    ('agcrn', 'electricity'): 12,
    ('stgcn', 'electricity'): 12,
    ('graphwavenet', 'electricity'): 12
}


def load_trials(path):
    """Loads Random Search trials from a JSON or Excel file."""

    if path.endswith('.xlsx'):
        df = pd.read_excel(path)

        # Summary rows are removed so only actual trials remain
        df = df[pd.to_numeric(df['trial'], errors='coerce').notna()].copy()
        df['trial'] = df['trial'].astype(int)

        return df.sort_values('trial').to_dict('records')

    d = json.load(open(path))
    d = [t for t in d if isinstance(t.get('trial'), int)]

    return sorted(d, key=lambda t: t['trial'])


def running_best(pairs):
    """Creates a best-so-far validation MAE curve."""

    pairs = sorted(pairs)
    xs, ys, b = [], [], float('inf')

    for e, v in pairs:
        b = min(b, v)
        xs.append(e)
        ys.append(b)

    return xs, ys


def main(out_path=OWN + 'efficiency_9cell.png'):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    for r, ds in enumerate(DATASETS):
        for c, m in enumerate(MODELS):
            ax = axes[r][c]
            n, fid = NTRIALS[(m, ds)], MAXFID[(m, ds)]

            # Random Search uses the same number of epochs for every trial
            rs = load_trials(RLOG[(m, ds)])[:n]

            x, y = running_best([
                ((i + 1) * fid, t['best_val_mae'])
                for i, t in enumerate(rs)
            ])

            ax.step(x, y, where='post', label='Random search')

            print(
                LABEL[ds].ljust(12),
                LABEL[m].ljust(14),
                'RS  ',
                len(rs),
                'trials, x ends',
                x[-1]
            )

            # DEHB uses different fidelities, so its recorded epoch total is used
            d = json.load(open(DLOG[(m, ds)]))

            pts = [
                (t['epochs_used_total'], t['val_mae'])
                for t in d if 'val_mae' in t
            ]

            x, y = running_best(pts)

            ax.step(x, y, where='post', label='DEHB')

            print(
                LABEL[ds].ljust(12),
                LABEL[m].ljust(14),
                'DEHB',
                len(d),
                'trials, x ends',
                x[-1]
            )

            ax.set_title(LABEL[m] + ', ' + LABEL[ds], fontsize=10)

            # Each plot ends at the allocated budget for that pair
            ax.set_xlim(0, n * fid)

            ax.grid(alpha=0.3)

            if r == 2:
                ax.set_xlabel('Cumulative allocated training epochs')

            if c == 0:
                ax.set_ylabel('Best-so-far validation MAE')

            if r == 0 and c == 0:
                ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.show()

    print('Saved ' + out_path)


if __name__ == "__main__":
    main()