#!/usr/bin/env python3
"""
plot_search_budget.py
=====================

Two figures comparing BOHB against random search (RS) over a HPO study of
spatio-temporal GNNs:

  1. epochs_per_fidelity   - every search trial as a point, coloured by the
                             fidelity (rung) it ran at, BOHB vs RS. Shows
                             where the multi-fidelity schedule actually put
                             the budget. Rungs are read per panel, never
                             assumed constant across architecture x dataset.
  2. anytime_performance   - incumbent validation MAE vs cumulative training
                             epochs, one panel per architecture x dataset,
                             with a horizontal reference line at the default
                             ("standard"/baseline) configuration's val MAE.

SEARCH ONLY. Final-retrain rows and "RS FINAL *" / "*final*" workbooks are
excluded throughout -- the curves describe the search phase, not the
post-search retrain.

Expected inputs (found recursively under --results-dir, extension .xlsx):

  BOHB     bohb_<arch>_<dataset>_seed42.xlsx
  RS       RS <ARCH> <DATASET>.xlsx
  default  <arch>_standard_<date>.xlsx        (one row per dataset)
  ignored  RS FINAL <ARCH> <DATASET>.xlsx, anything else matching *final*

Filename parsing is tolerant of case, spaces vs underscores, numeric upload
prefixes, " (1)"/" (2)" copy suffixes, and the aliases GWN -> graphwavenet,
METR-LA -> metrla, ELECTRCITY -> electricity.

Usage
-----
In Google Colab, no arguments are needed - Drive is mounted automatically,
PROJECT_ROOT below is used for input and output, and the figures are shown
inline:

    %run '/content/drive/MyDrive/AutoSTGNN/src/plot.py'

From a shell:

    python plot_search_budget.py --results-dir ./results
    python plot_search_budget.py --results-dir ./results --outdir ./figures \
        --formats png pdf --bohb-incumbent max-fidelity

Requires: pandas, numpy, matplotlib, openpyxl
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Environment - Colab does the tedious parts by itself
# --------------------------------------------------------------------------

__version__ = "1.4"

PROJECT_ROOT = Path("/content/drive/MyDrive/AutoSTGNN")
DISPLAY_WIDTH_PX = 1100          # inline preview width; files stay full-res


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def in_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").is_dir()


def mount_drive() -> None:
    """Mount Google Drive if we are on Colab and it is not mounted yet."""
    if not in_colab() or Path("/content/drive/MyDrive").is_dir():
        return
    try:
        from google.colab import drive
    except ImportError:
        return
    print("Mounting Google Drive ...")
    drive.mount("/content/drive")


def default_results_dir() -> Path:
    """First plausible results folder: the project root, then the cwd."""
    for candidate in (PROJECT_ROOT / "results", Path("results"),
                      Path("../results"), Path("/content/results")):
        if candidate.is_dir():
            return candidate
    return PROJECT_ROOT / "results" 


def default_outdir(results_dir: Path) -> Path:
    """Write figures beside the results folder, so they land in Drive too."""
    parent = results_dir.resolve().parent
    try:
        probe = parent / ".write_probe"
        probe.touch()
        probe.unlink()
        return parent / "figures"
    except OSError:
        return Path("figures")


def show_inline(paths: list) -> None:
    """Render the PNGs in the notebook output cell, scaled to fit."""
    if not in_notebook():
        return
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    for path in paths:
        if path.suffix.lower() == ".png":
            display(Image(filename=str(path), width=DISPLAY_WIDTH_PX))


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------

ARCH_ALIASES = {
    "graphwavenet": "graphwavenet",
    "graph_wavenet": "graphwavenet",
    "gwnet": "graphwavenet",
    "gwn": "graphwavenet",
    "agcrn": "agcrn",
    "stgcn": "stgcn",
}

DATASET_ALIASES = {
    "metrla": "metrla",
    "metr_la": "metrla",
    "pemsbay": "pemsbay",
    "pems_bay": "pemsbay",
    "bay": "pemsbay",
    "electricity": "electricity",
    "electrcity": "electricity",   # typo seen in the results folder
    "elecricity": "electricity",
    "elec": "electricity",
}

ARCH_LABEL = {"graphwavenet": "Graph WaveNet", "agcrn": "AGCRN", "stgcn": "STGCN"}
DATASET_LABEL = {"metrla": "METR-LA", "pemsbay": "PEMS-BAY", "electricity": "Electricity"}

ARCH_ORDER = ["graphwavenet", "agcrn", "stgcn"]
DATASET_ORDER = ["metrla", "pemsbay", "electricity"]

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

C_BOHB = "#1B7F5E"      # green
C_RS = "#C0255F"        # raspberry
C_DEFAULT = "#3F3F3F"
METHOD_COLOR = {"bohb": C_BOHB, "rs": C_RS}
METHOD_LABEL = {"bohb": "BOHB", "rs": "Random Search"}
METHOD_MARKER = {"bohb": "o", "rs": "^"}

# Colour encodes fidelity. Each method gets its own ramp so that the rung a
# trial ran at is readable, and RS is never left as an uncoloured outlier.
FIDELITY_RAMP = {
    "bohb": ["#BFE6D4", "#6FC4A1", "#1B7F5E", "#0A4F37"],
    "rs": ["#F5B9CD", "#E07095", "#C0255F", "#7C1340"],
}


def fidelity_shades(method: str, n: int) -> list:
    """n colours from a method's ramp, light (low fidelity) -> dark (high)."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(method, FIDELITY_RAMP[method])
    if n == 1:
        return [cmap(0.62)]
    return [cmap(v) for v in np.linspace(0.18, 0.95, n)]


def fidelity_label(method: str, fid: float) -> str:
    return f"{METHOD_LABEL[method]} (budget={fid:g} epochs)"


plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------


def normalise(stem: str) -> str:
    """Lowercase, strip copy-suffixes and upload prefixes, non-alnum -> '_'."""
    s = stem.lower()
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)          # trailing " (1)", " (2)"
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"^_*\d{6,}_*", "_", s)            # leading upload id
    return f"_{s.strip('_')}_"


def _match_alias(norm: str, aliases: dict[str, str]) -> str | None:
    for key in sorted(aliases, key=len, reverse=True):
        if key in norm:
            return aliases[key]
    return None


def classify(path: Path) -> tuple[str | None, str | None, str | None]:
    """Return (role, arch, dataset). role in {bohb, rs, default} or None."""
    norm = normalise(path.stem)
    arch = _match_alias(norm, ARCH_ALIASES)
    dataset = _match_alias(norm, DATASET_ALIASES)

    if "_standard_" in norm or "baseline" in norm or "_default_" in norm:
        return "default", arch, dataset
    if "final" in norm:
        return None, arch, dataset                # post-search retrain: skip
    if "bohb" in norm:
        return "bohb", arch, dataset
    if norm.startswith("_rs_") or "_random" in norm or "_randomsearch_" in norm:
        return "rs", arch, dataset
    return None, arch, dataset


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def first_col(df: pd.DataFrame, *names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def drop_junk_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing summary rows ('min test mae', blanks) and FINAL retrains."""
    if "trial" not in df.columns:
        return df.dropna(how="all")
    trial = df["trial"].astype("string").str.strip().str.lower()
    keep = trial.notna() & trial.str.fullmatch(r"\d+")
    return df.loc[keep].copy()


@dataclass
class Run:
    """One search run: per-trial epochs, fidelity and validation MAE."""

    method: str
    arch: str
    dataset: str
    source: Path
    epochs: np.ndarray          # epochs actually trained in each trial
    fidelity: np.ndarray        # rung / budget the trial was evaluated at
    val_mae: np.ndarray         # may contain NaN
    epochs_note: str = ""

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.epochs)

    @property
    def key(self) -> tuple[str, str]:
        return (self.arch, self.dataset)

    def incumbent(self, restrict_fidelity: float | None = None):
        """Step curve of best-so-far val MAE against cumulative epochs."""
        cum = self.cumulative
        ok = np.isfinite(self.val_mae)
        if restrict_fidelity is not None:
            ok &= self.fidelity == restrict_fidelity
        if not ok.any():
            return np.array([]), np.array([])
        return cum[ok], np.minimum.accumulate(self.val_mae[ok])


def load_bohb(path: Path, arch: str, dataset: str) -> Run:
    df = pd.read_excel(path)
    df = drop_junk_rows(df)                       # also removes the FINAL row

    ep_col = first_col(df, "epochs_run", "budget_epochs")
    fid_col = first_col(df, "budget_epochs", "epochs_run")
    val_col = first_col(df, "val_mae", "best_val_mae")
    if ep_col is None or val_col is None:
        raise ValueError(f"{path.name}: missing epochs/val_mae columns")

    epochs = pd.to_numeric(df[ep_col], errors="coerce").to_numpy(float)
    fidelity = pd.to_numeric(df[fid_col], errors="coerce").to_numpy(float)
    val = pd.to_numeric(df[val_col], errors="coerce").to_numpy(float)

    note = f"epochs from '{ep_col}', fidelity from '{fid_col}'"

    # Sanity-check the recorded cumulative column if the file carries one.
    cum_col = first_col(df, "cumulative_epochs")
    if cum_col is not None:
        recorded = pd.to_numeric(df[cum_col], errors="coerce").to_numpy(float)
        if np.isfinite(recorded).all() and not np.allclose(np.cumsum(epochs), recorded):
            print(f"  ! {path.name}: cumsum({ep_col}) disagrees with {cum_col}; "
                  f"using {cum_col}", file=sys.stderr)
            epochs = np.diff(np.concatenate([[0.0], recorded]))
            note = f"epochs from '{cum_col}' differences"

    return Run("bohb", arch, dataset, path, epochs, fidelity, val, note)


def load_rs(path: Path, arch: str, dataset: str, forced_epochs: float | None) -> Run:
    df = pd.read_excel(path)
    df = drop_junk_rows(df)

    val_col = first_col(df, "best_val_mae", "val_mae")
    if val_col is None:
        raise ValueError(f"{path.name}: no validation MAE column")
    val = pd.to_numeric(df[val_col], errors="coerce").to_numpy(float)
    n = len(val)

    ep_col = first_col(df, "epochs_run", "epochs", "budget_epochs", "retrain_epochs")
    if ep_col is not None:
        epochs = pd.to_numeric(df[ep_col], errors="coerce").to_numpy(float)
        note = f"epochs from '{ep_col}'"
    elif forced_epochs is not None:
        epochs = np.full(n, float(forced_epochs))
        note = f"{forced_epochs:g} epochs/trial (supplied)"
    else:
        epochs = np.full(n, np.nan)               # resolved later from BOHB rungs
        note = "epochs/trial pending"

    fidelity = epochs.copy()
    return Run("rs", arch, dataset, path, epochs, fidelity, val, note)


def infer_rs_epochs_from_paths(path: Path) -> float | None:
    """Fallback: largest 'best-epoch=NN' in the checkpoint paths, +1."""
    df = drop_junk_rows(pd.read_excel(path))
    col = first_col(df, "best_model_path", "checkpoint", "model_path")
    if col is None:
        return None
    found = [int(m.group(1))
             for s in df[col].dropna().astype(str)
             if (m := re.search(r"epoch[=_-](\d+)", s))]
    return float(max(found) + 1) if found else None


def load_defaults(path: Path, arch_hint: str | None) -> list[tuple[str, str, float]]:
    """Return (arch, dataset, val_mae) for each row of a *_standard_* workbook."""
    df = pd.read_excel(path).dropna(how="all")
    val_col = first_col(df, "val_mae", "best_val_mae")
    if val_col is None:
        return []
    out = []
    for _, row in df.iterrows():
        arch = str(row["model"]).strip().lower() if "model" in df.columns and pd.notna(row.get("model")) else arch_hint
        arch = ARCH_ALIASES.get(str(arch).replace("-", "").replace(" ", ""), arch)
        ds = row.get("dataset")
        ds = _match_alias(normalise(str(ds)), DATASET_ALIASES) if pd.notna(ds) else None
        val = pd.to_numeric(pd.Series([row[val_col]]), errors="coerce").iloc[0]
        if arch and ds and np.isfinite(val):
            out.append((arch, ds, float(val)))
    return out


@dataclass
class Study:
    runs: list[Run] = field(default_factory=list)
    defaults: dict[tuple[str, str], float] = field(default_factory=dict)

    def get(self, arch: str, dataset: str, method: str) -> Run | None:
        for r in self.runs:
            if r.arch == arch and r.dataset == dataset and r.method == method:
                return r
        return None

    @property
    def keys(self) -> list[tuple[str, str]]:
        present = {r.key for r in self.runs}
        ordered = [(a, d) for a in ARCH_ORDER for d in DATASET_ORDER if (a, d) in present]
        return ordered + sorted(present - set(ordered))


def collect(results_dir: Path, rs_epochs: float | None) -> Study:
    files = sorted(p for p in results_dir.rglob("*.xlsx") if not p.name.startswith("~$"))
    if not files:
        sys.exit(f"No .xlsx files found under {results_dir}")

    print(f"Scanning {len(files)} workbook(s) under {results_dir}\n")

    study = Study()
    candidates: dict[tuple[str, str, str], list[Path]] = {}

    for path in files:
        role, arch, dataset = classify(path)
        if role is None:
            print(f"  skip     {path.name}")
            continue
        if role == "default":
            rows = load_defaults(path, arch)
            for a, d, v in rows:
                study.defaults[(a, d)] = v
            print(f"  default  {path.name}  ->  {len(rows)} baseline row(s)")
            continue
        if arch is None or dataset is None:
            print(f"  skip     {path.name}  (could not parse arch/dataset)")
            continue
        candidates.setdefault((role, arch, dataset), []).append(path)

    # Resolve duplicate downloads: keep whichever workbook has the most trials.
    for (role, arch, dataset), paths in sorted(candidates.items()):
        if len(paths) > 1:
            # most trials wins; ties go to the shortest name, i.e. the
            # original rather than a " (1)" / " (2)" duplicate download.
            paths = sorted(paths, key=lambda p: (-len(drop_junk_rows(pd.read_excel(p))),
                                                 len(p.name), p.name))
            print(f"  ! {role} {arch}/{dataset}: {len(paths)} copies, using "
                  f"'{paths[0].name}' (most trials)", file=sys.stderr)
        path = paths[0]
        try:
            run = (load_bohb(path, arch, dataset) if role == "bohb"
                   else load_rs(path, arch, dataset, rs_epochs))
        except Exception as exc:                            # noqa: BLE001
            print(f"  ! failed on {path.name}: {exc}", file=sys.stderr)
            continue
        study.runs.append(run)
        print(f"  {role:<7}  {path.name}  ->  {len(run.epochs)} trial(s)")

    resolve_rs_epochs(study)
    return study


def resolve_rs_epochs(study: Study) -> None:
    """Fill in RS epochs/trial where the workbook does not record them.

    Preference: the matching BOHB run's highest *search* rung (RS trains each
    config at full fidelity, which is the same budget as BOHB's top rung),
    else the largest 'best-epoch=NN' seen in the checkpoint filenames + 1.
    """
    for run in study.runs:
        if run.method != "rs" or np.isfinite(run.epochs).all():
            continue
        bohb = study.get(run.arch, run.dataset, "bohb")
        val, src = None, ""
        if bohb is not None and np.isfinite(bohb.fidelity).any():
            val, src = float(np.nanmax(bohb.fidelity)), "BOHB max search rung"
        if val is None:
            val = infer_rs_epochs_from_paths(run.source)
            src = "max best-epoch in checkpoint paths + 1"
        if val is None:
            val, src = 1.0, "FALLBACK 1 epoch/trial - x-axis is trial count"
            print(f"  ! {run.source.name}: epochs/trial unknown; pass --rs-epochs",
                  file=sys.stderr)
        run.epochs = np.full(len(run.epochs), float(val))
        run.fidelity = run.epochs.copy()
        run.epochs_note = f"{val:g} epochs/trial ({src})"
        print(f"  ~ RS {run.arch}/{run.dataset}: assuming {run.epochs_note}")


# --------------------------------------------------------------------------
# Figure 1 - epochs per fidelity level
# --------------------------------------------------------------------------


def budget_table(study: Study) -> pd.DataFrame:
    rows = []
    for run in study.runs:
        ok = np.isfinite(run.epochs) & np.isfinite(run.fidelity)
        for fid in np.unique(run.fidelity[ok]):
            sel = ok & (run.fidelity == fid)
            rows.append(
                {
                    "architecture": run.arch,
                    "dataset": run.dataset,
                    "method": run.method,
                    "fidelity_epochs": float(fid),
                    "n_trials": int(sel.sum()),
                    "epochs_spent": float(run.epochs[sel].sum()),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    total = df.groupby(["architecture", "dataset", "method"])["epochs_spent"].transform("sum")
    df["share_of_budget"] = df["epochs_spent"] / total
    return df.sort_values(["architecture", "dataset", "method", "fidelity_epochs"])


def panel_grid(study: Study):
    """(archs, datasets, fig, axes) for a one-panel-per-arch-x-dataset grid."""
    keys = study.keys
    archs = [a for a in ARCH_ORDER if any(k[0] == a for k in keys)]
    archs += sorted({k[0] for k in keys} - set(archs))
    datasets = [d for d in DATASET_ORDER if any(k[1] == d for k in keys)]
    datasets += sorted({k[1] for k in keys} - set(datasets))
    fig, axes = plt.subplots(
        len(archs), len(datasets),
        figsize=(4.6 * len(datasets), 3.7 * len(archs)),
        squeeze=False,
        layout="constrained",
    )
    return archs, datasets, fig, axes


def plot_fidelity_scatter(study: Study, outdir: Path, formats, dpi):
    """Every trial as one point: colour = fidelity, marker = search method.

    Rungs are *not* assumed to be the same across panels - each architecture x
    dataset pair gets its own colour assignment (light = that pair's cheapest
    rung, dark = its most expensive) and its own legend stating the actual
    epoch budgets, so a pair running 5/15 is never mislabelled as 4/12.
    """
    archs, datasets, fig, axes = panel_grid(study)

    for r, arch in enumerate(archs):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            drawn, handles = False, []

            for method in ("bohb", "rs"):
                run = study.get(arch, ds, method)
                if run is None:
                    continue
                cum = run.cumulative
                ok = np.isfinite(run.val_mae) & np.isfinite(run.fidelity)
                if not ok.any():
                    continue
                drawn = True

                # Colours are resolved per panel, against this pair's rungs.
                rungs = sorted(np.unique(run.fidelity[ok]))
                shades = fidelity_shades(method, len(rungs))
                for fid, colour in zip(rungs, shades):
                    sel = ok & (run.fidelity == fid)
                    ax.scatter(cum[sel], run.val_mae[sel],
                               s=46, marker=METHOD_MARKER[method], color=colour,
                               edgecolors="white", linewidths=0.5,
                               alpha=0.9, zorder=3)
                    handles.append(
                        Line2D([], [], marker=METHOD_MARKER[method], ls="none",
                               ms=7, markerfacecolor=colour, markeredgecolor="white",
                               label=fidelity_label(method, fid))
                    )

            if not drawn:
                ax.set_axis_off()
                continue

            ax.legend(handles=handles, loc="best", fontsize=8,
                      frameon=True, framealpha=0.9, facecolor="white",
                      edgecolor="#DDDDDD", handletextpad=0.4,
                      borderpad=0.5, labelspacing=0.35)
            ax.set_title(f"{ARCH_LABEL.get(arch, arch)} \u00b7 {DATASET_LABEL.get(ds, ds)}",
                         pad=6)
            ax.set_axisbelow(True)
            ax.margins(x=0.04, y=0.12)
            if c == 0:
                ax.set_ylabel("Validation MAE")
            if r == len(archs) - 1:
                ax.set_xlabel("Cumulative training epochs")

    fig.suptitle("Trials by fidelity level: where the search budget was spent",
                 fontsize=13.5)
    return save(fig, outdir / "epochs_per_fidelity", formats, dpi)


# --------------------------------------------------------------------------
# Figure 2 - anytime performance
# --------------------------------------------------------------------------


def plot_anytime(study: Study, outdir: Path, formats, dpi, restrict: bool,
                 share_x: bool):
    if not study.keys:
        sys.exit("Nothing to plot.")
    archs, datasets, fig, axes = panel_grid(study)

    for r, arch in enumerate(archs):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            drawn, ys, ends = False, [], {}

            for method in ("bohb", "rs"):
                run = study.get(arch, ds, method)
                if run is None:
                    continue
                fid = None
                if restrict and method == "bohb" and np.isfinite(run.fidelity).any():
                    fid = float(np.nanmax(run.fidelity))
                x, y = run.incumbent(restrict_fidelity=fid)
                if x.size == 0:
                    continue
                drawn = True
                ys.append(y)
                ends[method] = (float(run.cumulative[-1]), float(y[-1]))
                ax.step(x, y, where="post", color=METHOD_COLOR[method], lw=2.0,
                        label=METHOD_LABEL[method], zorder=3)

            if not drawn:
                ax.set_axis_off()
                continue

            # Each curve simply stops at that method's last evaluation. The
            # x-axis still spans the larger of the two budgets, so a curve
            # ending early is visibly a search that ran out of budget first.
            xmax = max(x for x, _ in ends.values())
            for method, (x_end, y_end) in ends.items():
                ax.plot(x_end, y_end, "o", ms=4.5, color=METHOD_COLOR[method], zorder=4)

            base = study.defaults.get((arch, ds))
            if base is not None:
                ax.axhline(base, color=C_DEFAULT, ls="--", lw=1.3, zorder=2,
                           label="Default configuration")
                ax.annotate(f"default = {base:,.3g}", xy=(0.985, base),
                            xycoords=("axes fraction", "data"), ha="right", va="bottom",
                            fontsize=8, color=C_DEFAULT,
                            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))

            lo = min(y.min() for y in ys)
            hi = max(y[0] for y in ys)
            if base is not None:
                lo, hi = min(lo, base), max(hi, base)
            pad = 0.07 * max(hi - lo, 1e-9)
            ax.set_ylim(lo - pad, hi + 1.6 * pad)
            ax.set_xlim(0, xmax * 1.02)
            ax.set_title(f"{ARCH_LABEL.get(arch, arch)} · {DATASET_LABEL.get(ds, ds)}", pad=6)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel("Validation MAE (incumbent)")
            if r == len(archs) - 1:
                ax.set_xlabel("Cumulative training epochs")

    if share_x:
        live = [ax for row in axes for ax in row if ax.get_visible() and ax.has_data()]
        xmax = max(ax.get_xlim()[1] for ax in live)
        for ax in live:
            ax.set_xlim(0, xmax)

    handles = [Line2D([], [], color=C_BOHB, lw=2.1, label="BOHB"),
               Line2D([], [], color=C_RS, lw=2.1, label="Random Search"),
               Line2D([], [], color=C_DEFAULT, lw=1.3, ls="--", label="Default configuration")]
    note = " (incumbent over top-rung evaluations only)" if restrict else ""
    fig.suptitle("Anytime search performance: BOHB vs Random Search" + note,
                 fontsize=13.5)
    fig.legend(handles=handles, loc="outside lower center",
               ncol=min(len(handles), 4), fontsize=10.5,
               handletextpad=0.6, columnspacing=1.8)
    return save(fig, outdir / "anytime_performance", formats, dpi)


# --------------------------------------------------------------------------


def save(fig, stem: Path, formats, dpi) -> list:
    written = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi)
        written.append(out)
        print(f"  wrote {out}")
    plt.close(fig)
    return written


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fidelity-budget and anytime-performance plots for BOHB vs random search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=None,
                   help="root directory searched recursively for .xlsx workbooks "
                        f"(default: {PROJECT_ROOT / 'results'} on Colab, else ./results)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="where figures are written (default: a 'figures' folder "
                        "beside the results directory)")
    p.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int, default=400,
                   help="raster resolution; 400 gives roughly 5500px-wide PNGs")
    p.add_argument("--rs-epochs", type=float, default=None,
                   help="epochs per random-search trial, if the workbooks do not "
                        "record it (default: infer from the matching BOHB run's "
                        "highest search rung)")
    p.add_argument("--bohb-incumbent", choices=["all", "max-fidelity"], default="all",
                   help="'all' takes the running minimum over every BOHB evaluation; "
                        "'max-fidelity' counts only top-rung evaluations, which is the "
                        "like-for-like comparison against RS")
    p.add_argument("--share-x", action="store_true",
                   help="use one common epoch axis across all panels")
    p.add_argument("--no-display", action="store_true",
                   help="do not render the figures inline in a notebook")
    # %run passes the script path as argv[0] and nothing else, so an empty
    # argument list is the normal notebook case, not an error.
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Wrapper so a bad path prints one clear line instead of a traceback."""
    try:
        return _run(argv)
    except SystemExit as exc:
        if exc.code and not isinstance(exc.code, int):
            print(exc.code, file=sys.stderr)
            return 1
        return int(exc.code or 0)


def _run(argv=None) -> int:
    # Printed so a stale copy on Drive is obvious rather than mysterious.
    here = Path(__file__).resolve() if "__file__" in globals() else Path("<stdin>")
    print(f"plot.py v{__version__}  ({here})")

    args = parse_args(argv)

    # Colab: mount Drive before looking for anything on it.
    mount_drive()
    if args.results_dir is None:
        args.results_dir = default_results_dir()
        print(f"Results directory: {args.results_dir}")
    if args.outdir is None:
        args.outdir = default_outdir(args.results_dir)
        print(f"Output directory:  {args.outdir}\n")

    if not args.results_dir.is_dir():
        sys.exit(
            f"Results directory '{args.results_dir}' does not exist.\n"
            f"Edit PROJECT_ROOT at the top of this file, or pass "
            f"--results-dir /path/to/results."
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    study = collect(args.results_dir, args.rs_epochs)
    if not study.runs:
        sys.exit(f"No search runs were loaded from {args.results_dir}.")

    missing = [f"{a}/{d}" for a, d in study.keys if (a, d) not in study.defaults]
    if missing:
        print(f"\n  ! no default-config val MAE for: {', '.join(missing)} "
              f"(reference line omitted in those panels)", file=sys.stderr)

    table = budget_table(study)
    csv = args.outdir / "epochs_per_fidelity.csv"
    table.to_csv(csv, index=False)
    print(f"\nBudget breakdown -> {csv}")
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\nFigures")
    written = plot_fidelity_scatter(study, args.outdir, args.formats, args.dpi)
    written += plot_anytime(study, args.outdir, args.formats, args.dpi,
                            restrict=args.bohb_incumbent == "max-fidelity",
                            share_x=args.share_x)
    if not args.no_display:
        show_inline(written)
    return 0


if __name__ == "__main__":
    # Under %run, a bare SystemExit clutters the cell output, so only the
    # command-line path exits with a status code.
    if in_notebook():
        main()
    else:
        raise SystemExit(main())