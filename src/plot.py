"""
plot_convergence.py

Builds three figures comparing BOHB and Random Search (RS), each with one
subplot per dataset (3 datasets -> 1 figure with 3 panels):

  1. Search efficiency (primary): "best validation MAE found so far vs.
     wall-clock time" (a step curve against cumulative actual training
     time, i.e. sum of each trial's own trial_duration_sec — immune to
     idle gaps like a Colab disconnect) -> --out-efficiency
  2. Sample efficiency (secondary/supporting): "best validation MAE found
     so far vs. number of evaluations" (a step curve against trial count
     instead of wall-clock time or epochs) -> --out-sample
  3. Per-trial MAE by search method: "per-trial validation MAE across the
     search" (a scatter of each individual trial's own val MAE,
     unsmoothed, showing how noisy/exploratory each algorithm's search
     actually was; BOHB points are colored by training budget to show
     that low-budget trials are noisier proxies of true performance)
     -> --out-raw

--------------------------------------------------------------------------
EXPECTED INPUT FILES (one BOHB file + one RS file per dataset)
--------------------------------------------------------------------------
BOHB file   : produced by the BOHB run, one row per trial, must contain:
                  trial, cumulative_epochs, val_mae, trial_duration_sec
              (the aggregate "FINAL" row is automatically skipped)
              filename pattern that this script searches for:
                  bohb_*_<dataset>_*.xlsx        e.g. bohb_graphwavenet_metrla_seed42.xlsx

RS file     : produced by the Random Search run, one row per trial, must contain:
                  trial, best_val_mae, trial_duration_sec
              (RS logs don't record epochs_run per trial, so cumulative
              epochs are reconstructed as trial_index * epochs_per_trial,
              see RS_EPOCHS_PER_TRIAL below)
              filename pattern that this script searches for:
                  *_<dataset>_RS_*.xlsx           e.g. graphwavenet_metrla_RS_20260819.xlsx

--------------------------------------------------------------------------
FILE PATHS
--------------------------------------------------------------------------
File paths are looked up first in the FILE_PATHS dict below (edit it to
point at your actual files). If a dataset isn't in FILE_PATHS, or the
hardcoded path doesn't exist, the script falls back to globbing --data-dir
for a matching file. Any dataset still missing a BOHB or RS file just
shows up empty in its panel (with a note printed to the console) instead
of crashing.

The x-axis of the per-trial panel (figure 3) is capped at the point the
max training budget is reached (the smaller of the two algorithms' final
cumulative-epoch value), so the line doesn't trail off past where both
runs stop. The wall-clock and sample-efficiency panels are each capped
the same way, but on their own x-axis units (minutes / evaluation count).

# gwn #
    ''' # remove 
    "metrla": {
        "bohb": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Solvers/BOHB/gwn/metrla/bohb_graphwavenet_metrla_seed42.xlsx"),
        "rs": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Random Search/gwn/metrla/graphwavenet_metrla_RS_20260819.xlsx"),
    },
    "pemsbay": {
        "bohb": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Solvers/BOHB/gwn/pemsbay/bohb_graphwavenet_pemsbay_seed42.xlsx"),
        "rs": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Random Search/gwn/pemsbay/graphwavenet_pemsbay_RS_20260820.xlsx"),
    },
    "airquality": {
        "bohb": "",
        "rs": "",
    },
    ''' # remove 

    # agcrn #
        "metrla": {
            "bohb": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Solvers/BOHB/agcrn/metrla/bohb_agcrn_metrla_seed42.xlsx"),
            "rs": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Random Search/agcrn/metrla/agcrn_metrla_RS_20260819.xlsx"),
        },
        "pemsbay": {
            "bohb": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Solvers/BOHB/agcrn/pemsbay/bohb_agcrn_pemsbay_seed42.xlsx"),
            "rs": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Random Search/agcrn/pemsbay/agcrn_pemsbay_RS_20260820.xlsx"),
        },
        "airquality": {
            "bohb": "",
            "rs": "",
        },

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python plot_convergence.py \
        --data-dir /path/to/xlsx/files \
        --datasets metrla pemsbay airquality \
        --rs-epochs-per-trial 30 \
        --out-efficiency efficiency_wallclock_bohb_vs_rs.png \
        --out-sample sample_efficiency_bohb_vs_rs.png \
        --out-raw per_trial_mae_bohb_vs_rs.png
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------------------------------
# Hardcoded file paths
# --------------------------------------------------------------------------
# Map each dataset name to its BOHB and RS result files. Add an entry here
# for every dataset you have; datasets without an entry (or with a missing
# file) will just show an empty "No data found" panel.
#
# NOTE: do NOT wrap folder names in quotes here (e.g. "'Shared Results'").
# Quotes are only needed to escape spaces in a *shell* command; inside a
# Python string they become literal characters in the path, so Python ends
# up looking for a folder that actually has quote marks in its name, which
# doesn't exist. Spaces in folder names are fine as-is in a plain Python
# string — just type the real folder name with no extra quoting.
FILE_PATHS = {
    # stgcn #
    "metrla": {
        "bohb": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Solvers/BOHB/stgcn /metrla/bohb_stgcn_metrla_seed42.xlsx"),
        "rs": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Random Search/stgcn/metrla/stgcn_metrla_RS_20260819.xlsx"),
    },
    "pemsbay": {
        "bohb": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Solvers/BOHB/stgcn /pemsbay/bohb_stgcn_pemsbay_seed42.xlsx"),
        "rs": Path("/content/drive/MyDrive/AutoSTGNN/Shared Results/Random Search/stgcn/pemsbay/stgcn_pemsbay_RS_20260819.xlsx"),
    },
    "airquality": {
        "bohb": "",
        "rs": "",
    },
}













# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def find_file(data_dir: str, pattern: str) -> str | None:
    """Return the first file in data_dir matching a glob pattern, or None."""
    matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [warn] multiple files match '{pattern}', using {matches[0]}")
    return matches[0]


def diagnose_missing_path(path) -> None:
    """Walk up from `path` to the nearest folder that actually exists, and
    list what's really inside it. This turns a blind 'path not found' into
    something you can act on immediately — e.g. it'll show you that Drive
    isn't mounted, or that a folder is spelled/named differently than you
    typed it.
    """
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    print(f"    Closest existing ancestor: {p}")
    if not p.exists():
        return
    try:
        entries = sorted(os.listdir(p))
        print(f"    Contents of that folder ({len(entries)} items): {entries}")
    except Exception as e:
        print(f"    Could not list contents: {e}")


def resolve_path(data_dir: str, dataset: str, kind: str, pattern: str) -> str | None:
    """Resolve a file path for (dataset, kind) using, in order:
    1. The hardcoded FILE_PATHS table (if the path exists on disk).
    2. A glob search in data_dir matching `pattern`.

    Prints a specific message when a hardcoded path is set but doesn't
    exist, plus a diagnosis of where exactly the path breaks, so a typo'd
    path or an unmounted Drive doesn't just look like "no file found".
    """
    hardcoded = FILE_PATHS.get(dataset, {}).get(kind)
    if hardcoded:
        if os.path.exists(hardcoded):
            return hardcoded
        print(f"  [warn] hardcoded {kind} path for '{dataset}' does not exist: {hardcoded}")
        diagnose_missing_path(hardcoded)
    return find_file(data_dir, pattern)


def load_bohb(path: str) -> tuple[pd.DataFrame, float]:
    """Load a BOHB results file and compute the running-best val MAE.

    Returns (dataframe, max_budget_epochs) where max_budget_epochs is the
    largest `budget_epochs` value seen (the max fidelity BOHB trains at),
    used to cap the x-axis at the point the max budget is reached.

    The dataframe has one row per trial with columns:
        cumulative_epochs, trial_mae (that trial's own val_mae), best_so_far
        (running minimum of trial_mae up to and including that trial),
        budget_epochs (the fidelity/budget that trial was evaluated at,
        used to explain per-trial noise: low-budget trials are noisier,
        cheaper proxies of a config's true performance)
    """
    df = pd.read_excel(path)

    # Drop the aggregate "FINAL" summary row (trial == 'FINAL')
    df = df[pd.to_numeric(df["trial"], errors="coerce").notna()].copy()

    max_budget = df["budget_epochs"].max() if "budget_epochs" in df.columns else None

    df = df.dropna(subset=["cumulative_epochs", "val_mae"])
    df = df.sort_values("cumulative_epochs")
    df["trial_mae"] = df["val_mae"]
    df["best_so_far"] = df["trial_mae"].cummin()

    cols = ["cumulative_epochs", "trial_mae", "best_so_far"]
    if "budget_epochs" in df.columns:
        cols.append("budget_epochs")
    if "trial_duration_sec" in df.columns:
        cols.append("trial_duration_sec")

    return df[cols].reset_index(drop=True), max_budget


def load_rs(path: str, epochs_per_trial: int) -> pd.DataFrame:
    """Load a Random Search results file and compute the running-best val MAE.

    RS logs don't include a per-trial epoch count, so cumulative epochs are
    reconstructed assuming every trial trains for `epochs_per_trial` epochs
    (random search has no early-stopping / multi-fidelity budgeting).

    The dataframe has one row per trial with columns:
        cumulative_epochs, trial_mae (that trial's own val MAE, from the
        best_val_mae column), best_so_far (running minimum of trial_mae)
    """
    df = pd.read_excel(path)

    # Keep only real trial rows (drop blank rows / trailing "min test mae" summary row)
    df = df[pd.to_numeric(df["trial"], errors="coerce").notna()].copy()
    df["trial"] = df["trial"].astype(int)
    df = df.sort_values("trial").reset_index(drop=True)

    df = df.dropna(subset=["best_val_mae"])
    df["cumulative_epochs"] = (df.index + 1) * epochs_per_trial
    df["trial_mae"] = df["best_val_mae"]
    df["best_so_far"] = df["trial_mae"].cummin()

    cols = ["cumulative_epochs", "trial_mae", "best_so_far"]
    if "trial_duration_sec" in df.columns:
        cols.append("trial_duration_sec")

    return df[cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def compute_sample_series(df: pd.DataFrame) -> pd.DataFrame | None:
    """From a trial dataframe (already in trial-run order), build the
    running-best-so-far series against *number of evaluations* — i.e.
    trial count, regardless of how many epochs or how much wall-clock
    time each trial took. This isolates search-algorithm sample
    efficiency (how many configs it had to try) from training cost.
    """
    if df is None or df.empty:
        return None
    d = df.reset_index(drop=True).copy()
    d["evaluation"] = d.index + 1
    d["best_so_far_eval"] = d["trial_mae"].cummin()
    return d[["evaluation", "best_so_far_eval"]]


def plot_sample_panel(ax, dataset: str, bohb_eval: pd.DataFrame | None, rs_eval: pd.DataFrame | None,
                       x_max: float | None):
    """Step plot of best validation MAE found so far vs. number of
    evaluations (trials), so you can read off how many configs each
    method needed to try to reach a given quality level."""
    has_data = False

    if bohb_eval is not None and not bohb_eval.empty:
        ax.step(
            bohb_eval["evaluation"], bohb_eval["best_so_far_eval"],
            where="post", label="BOHB", color="#1f77b4", linewidth=3,
        )
        has_data = True

    if rs_eval is not None and not rs_eval.empty:
        ax.step(
            rs_eval["evaluation"], rs_eval["best_so_far_eval"],
            where="post", label="Random Search", color="#d62728",
            linewidth=3, linestyle="--",
        )
        has_data = True

    ax.set_title(dataset, fontsize=18, fontweight="bold")
    ax.set_xlabel("Number of evaluations (trials)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3)

    if x_max is not None:
        ax.set_xlim(right=x_max)

    if has_data:
        ax.legend(fontsize=13, loc="best")
    else:
        ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                 transform=ax.transAxes, color="gray", fontsize=14)


def plot_raw_panel(ax, dataset: str, bohb_df: pd.DataFrame | None, rs_df: pd.DataFrame | None,
                    x_max: float | None):
    """Scatter each trial's own val MAE (not the running best) against
    cumulative epochs, to show the raw exploration behavior of each search
    algorithm rather than the cleaned-up convergence curve.

    BOHB's points are colored by the training budget (epochs) each trial
    was evaluated at, since BOHB deliberately evaluates most configs at
    cheap, low-budget fidelities first — those low-budget val_mae values
    are noisier/less reliable estimates of true performance, which is a
    big part of why BOHB's per-trial curve looks noisier than RS's (RS
    always trains every trial to the same full budget).
    """
    has_data = False

    # Distinct color per BOHB budget level (sorted ascending: lowest budget
    # = lightest/most-noisy, highest budget = darkest/most-reliable).
    budget_palette = ["#a6cee3", "#1f78b4", "#08306b", "#6a3d9a", "#b15928"]

    if bohb_df is not None and not bohb_df.empty:
        if "budget_epochs" in bohb_df.columns and bohb_df["budget_epochs"].notna().any():
            budgets = sorted(bohb_df["budget_epochs"].dropna().unique())
            for i, b in enumerate(budgets):
                sub = bohb_df[bohb_df["budget_epochs"] == b]
                ax.scatter(
                    sub["cumulative_epochs"], sub["trial_mae"],
                    label=f"BOHB (budget={int(b)} epochs)",
                    color=budget_palette[i % len(budget_palette)],
                    marker="o", s=65, alpha=0.85,
                    edgecolors="white", linewidths=0.8,
                )
        else:
            # No budget info available - fall back to a single BOHB color.
            ax.scatter(
                bohb_df["cumulative_epochs"], bohb_df["trial_mae"],
                label="BOHB", color="#1f77b4", marker="o", s=60,
                alpha=0.8, edgecolors="white", linewidths=0.8,
            )
        has_data = True

    if rs_df is not None and not rs_df.empty:
        ax.scatter(
            rs_df["cumulative_epochs"], rs_df["trial_mae"],
            label="Random Search", color="#d62728", marker="^", s=60,
            alpha=0.8, edgecolors="white", linewidths=0.8,
        )
        has_data = True

    ax.set_title(dataset, fontsize=18, fontweight="bold")
    ax.set_xlabel("Cumulative epochs", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3)

    if x_max is not None:
        ax.set_xlim(right=x_max)

    if has_data:
        ax.legend(fontsize=11, loc="best")
    else:
        ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                 transform=ax.transAxes, color="gray", fontsize=14)


# --------------------------------------------------------------------------
# Efficiency: best MAE found so far vs. wall-clock time
# --------------------------------------------------------------------------

def compute_time_series(df: pd.DataFrame) -> pd.DataFrame | None:
    """From a trial dataframe with `trial_duration_sec` and `trial_mae`,
    build the running-best-so-far series against *cumulative actual
    training time* — the running sum of each trial's own training
    duration, in the order trials were run.

    This is deliberately built from trial_duration_sec.cumsum() rather
    than elapsed_since_start_sec: elapsed-since-start is a wall-clock
    timestamp, so it also counts any idle time between trials (e.g. a
    Colab disconnect/reconnect gap), which would make a run look slower
    than it actually was. Summing each trial's own duration only counts
    time actually spent training, so it's immune to those gaps.

    Returns columns: minutes (cumulative training time), best_so_far_time
    (running minimum of trial_mae in that order).
    """
    if df is None or df.empty or "trial_duration_sec" not in df.columns:
        return None

    d = df.dropna(subset=["trial_duration_sec", "trial_mae"]).copy()
    if d.empty:
        return None
    # Rows are already in trial-run order (sorted by cumulative_epochs for
    # BOHB, by trial number for RS), so cumsum here reflects actual elapsed
    # training time in the order trials were executed.
    d["minutes"] = d["trial_duration_sec"].cumsum() / 60.0
    d["best_so_far_time"] = d["trial_mae"].cummin()
    return d[["minutes", "best_so_far_time"]].reset_index(drop=True)


def plot_time_panel(ax, dataset: str, bohb_time: pd.DataFrame | None, rs_time: pd.DataFrame | None,
                     x_max: float | None):
    """Step plot of best validation MAE found so far vs. wall-clock time
    (minutes), so you can read off directly how long each method took to
    reach any given quality level."""
    has_data = False

    if bohb_time is not None and not bohb_time.empty:
        ax.step(
            bohb_time["minutes"], bohb_time["best_so_far_time"],
            where="post", label="BOHB", color="#1f77b4", linewidth=3,
        )
        has_data = True

    if rs_time is not None and not rs_time.empty:
        ax.step(
            rs_time["minutes"], rs_time["best_so_far_time"],
            where="post", label="Random Search", color="#d62728",
            linewidth=3, linestyle="--",
        )
        has_data = True

    ax.set_title(dataset, fontsize=18, fontweight="bold")
    ax.set_xlabel("Wall-clock time (minutes)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3)

    if x_max is not None:
        ax.set_xlim(right=x_max)

    if has_data:
        ax.legend(fontsize=13, loc="best")
    else:
        ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                 transform=ax.transAxes, color="gray", fontsize=14)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=".", help="Directory containing the xlsx result files")
    parser.add_argument("--datasets", nargs="+", default=["metrla", "pemsbay", "airquality"],
                         help="Dataset names to include, one panel each (default: metrla pemsbay airquality)")
    parser.add_argument("--rs-epochs-per-trial", type=int, default=30,
                         help="Epochs trained per RS trial, used to reconstruct cumulative epochs (default: 30)")
    parser.add_argument("--out-efficiency", default="efficiency_wallclock_bohb_vs_rs.png",
                         help="Search efficiency (wall-clock) step-plot output image path")
    parser.add_argument("--out-sample", default="sample_efficiency_bohb_vs_rs.png",
                         help="Sample efficiency (evaluations) step-plot output image path")
    parser.add_argument("--out-raw", default="per_trial_mae_bohb_vs_rs.png",
                         help="Per-trial MAE scatter-plot output image path")
    args = parser.parse_args()

    n = len(args.datasets)

    fig_eff, axes_eff = plt.subplots(1, n, figsize=(8 * n, 6.5), sharey=False)
    fig_sample, axes_sample = plt.subplots(1, n, figsize=(8 * n, 6.5), sharey=False)
    fig_raw, axes_raw = plt.subplots(1, n, figsize=(8 * n, 6.5), sharey=False)
    if n == 1:
        axes_eff = [axes_eff]
        axes_sample = [axes_sample]
        axes_raw = [axes_raw]

    for ax_eff, ax_sample, ax_raw, dataset in zip(axes_eff, axes_sample, axes_raw, args.datasets):
        print(f"Dataset: {dataset}")

        bohb_path = resolve_path(args.data_dir, dataset, "bohb", f"bohb_*_{dataset}_*.xlsx")
        rs_path = resolve_path(args.data_dir, dataset, "rs", f"*_{dataset}_RS_*.xlsx")

        bohb_df, rs_df = None, None
        max_bohb_budget = None
        x_max_candidates = []

        if bohb_path:
            print(f"  BOHB file: {bohb_path}")
            bohb_df, max_bohb_budget = load_bohb(bohb_path)
            if not bohb_df.empty:
                x_max_candidates.append(bohb_df["cumulative_epochs"].max())
        else:
            print(f"  [warn] no BOHB file found for '{dataset}' "
                  f"(expected path in FILE_PATHS, or pattern bohb_*_{dataset}_*.xlsx)")

        if rs_path:
            print(f"  RS file:   {rs_path}")
            rs_df = load_rs(rs_path, args.rs_epochs_per_trial)
            if not rs_df.empty:
                x_max_candidates.append(rs_df["cumulative_epochs"].max())
        else:
            print(f"  [warn] no RS file found for '{dataset}' "
                  f"(expected path in FILE_PATHS, or pattern *_{dataset}_RS_*.xlsx)")

        # Cap the x-axis at the point the max training budget is reached,
        # i.e. the smaller of the two runs' final cumulative-epoch value
        # (so the plot doesn't trail off with only one algorithm still
        # plotted), falling back to whichever run has data if only one exists.
        x_max = min(x_max_candidates) if len(x_max_candidates) == 2 else (
            x_max_candidates[0] if x_max_candidates else None
        )

        plot_raw_panel(ax_raw, dataset, bohb_df, rs_df, x_max)

        # Search efficiency: best-so-far MAE vs. cumulative actual training
        # time (sum of each trial's own trial_duration_sec), not raw
        # wall-clock elapsed-since-start — this stays accurate even if the
        # run had idle gaps (e.g. a Colab disconnect/reconnect).
        bohb_time = compute_time_series(bohb_df)
        rs_time = compute_time_series(rs_df)
        time_max_candidates = []
        if bohb_time is not None and not bohb_time.empty:
            time_max_candidates.append(bohb_time["minutes"].max())
        if rs_time is not None and not rs_time.empty:
            time_max_candidates.append(rs_time["minutes"].max())
        time_x_max = min(time_max_candidates) if len(time_max_candidates) == 2 else (
            time_max_candidates[0] if time_max_candidates else None
        )

        plot_time_panel(ax_eff, dataset, bohb_time, rs_time, time_x_max)

        # Sample efficiency: best-so-far MAE vs. number of evaluations
        # (trial count), independent of epochs or wall-clock cost.
        bohb_eval = compute_sample_series(bohb_df)
        rs_eval = compute_sample_series(rs_df)
        eval_max_candidates = []
        if bohb_eval is not None and not bohb_eval.empty:
            eval_max_candidates.append(bohb_eval["evaluation"].max())
        if rs_eval is not None and not rs_eval.empty:
            eval_max_candidates.append(rs_eval["evaluation"].max())
        eval_x_max = min(eval_max_candidates) if len(eval_max_candidates) == 2 else (
            eval_max_candidates[0] if eval_max_candidates else None
        )

        plot_sample_panel(ax_sample, dataset, bohb_eval, rs_eval, eval_x_max)

    fig_eff.suptitle("Search Efficiency: Best Validation MAE vs. Wall-Clock Time", fontsize=20, fontweight="bold")
    axes_eff[0].set_ylabel("Best validation MAE so far", fontsize=15)
    fig_eff.tight_layout(rect=[0, 0, 1, 0.95])
    fig_eff.savefig(args.out_efficiency, dpi=200)
    print(f"\nSaved search-efficiency figure to {args.out_efficiency}")

    fig_sample.suptitle("Sample Efficiency: Best Validation MAE vs. Number of Evaluations", fontsize=20, fontweight="bold")
    axes_sample[0].set_ylabel("Best validation MAE so far", fontsize=15)
    fig_sample.tight_layout(rect=[0, 0, 1, 0.95])
    fig_sample.savefig(args.out_sample, dpi=200)
    print(f"Saved sample-efficiency figure to {args.out_sample}")

    fig_raw.suptitle("Per-Trial Validation MAE Across the Search: BOHB vs. Random Search", fontsize=20, fontweight="bold", y=0.99)
    fig_raw.text(0.5, 0.935,
                 "BOHB colors = training budget per trial (epochs) — low-budget evals are cheaper but noisier proxies of true performance",
                 ha="center", fontsize=12, style="italic", color="dimgray")
    axes_raw[0].set_ylabel("Validation MAE (per trial)", fontsize=15)
    fig_raw.tight_layout(rect=[0, 0, 1, 0.90])
    fig_raw.savefig(args.out_raw, dpi=200)
    print(f"Saved per-trial figure to {args.out_raw}")


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# Debugging a bad path in Colab
# --------------------------------------------------------------------------
# If a hardcoded path in FILE_PATHS isn't being found, don't guess — list
# the actual folder contents to see the real names/spelling. Paste this
# into a Colab cell (after mounting Drive), adjusting the folder to inspect:
#
#   import os
#   folder = "/content/drive/MyDrive/AutoSTGNN/Shared Results"
#   for root, dirs, files in os.walk(folder):
#       for f in files:
#           if f.endswith(".xlsx"):
#               print(os.path.join(root, f))
#
# This prints every .xlsx file under `folder` with its exact path, so you
# can copy-paste the real paths into FILE_PATHS instead of typing them by
# hand.