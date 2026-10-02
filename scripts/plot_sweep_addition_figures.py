"""Figures for the altitude-sweep paper (paper/main.tex).

Reproduces the three result figures of the GLOBECOM submission with the
altitude-sweep heuristic added as a third series, keeping the original styling.

python scripts/plot_sweep_addition_figures.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

IEEE_W = 3.5
FONT_SZ, TICK_SZ, LW, MS = 9, 8, 1.4, 4.5

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": FONT_SZ, "axes.labelsize": FONT_SZ, "axes.titlesize": FONT_SZ,
    "xtick.labelsize": TICK_SZ, "ytick.labelsize": TICK_SZ,
    "legend.fontsize": TICK_SZ, "lines.linewidth": LW, "lines.markersize": MS,
    "axes.linewidth": 0.7, "grid.linewidth": 0.4, "grid.color": "#cccccc",
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False,
})

# Okabe-Ito, validated for CVD separation; each series also gets its own dash
# pattern and marker so the figures survive greyscale printing.
STYLE = {
    "kmeans":                dict(color="#0072B2", ls="-.", marker="s", label="K-Means"),
    "analytic":              dict(color="#D55E00", ls="--", marker="o", label="Analytic"),
    "kmeans_altitude_sweep": dict(color="#009E73", ls="-",  marker="^",
                                  label="Altitude Sweep"),
}
ORDER = ["kmeans", "analytic", "kmeans_altitude_sweep"]
DISTS = ["clustered", "hotspot", "uniform"]
DLABELS = ["Clustered", "Hotspot", "Uniform"]


def load_synthetic(results: Path) -> pd.DataFrame:
    """Published K-means / Analytic results merged with the new sweep run."""
    rows = []
    for f in glob.glob(str(results / "heuristic_50" / "task_*.json")):
        d = json.load(open(f))
        c = d["config"]
        for m, seeds in d["per_seed"].items():
            if m not in ("kmeans", "analytic"):
                continue
            for i, s in enumerate(seeds):
                rows.append(dict(dist=c["distribution"], n_drones=c["n_drones"],
                                 n_users=c["target_users"], method=m, seed=1000 + i,
                                 coverage_pct=s["coverage_pct"],
                                 mean_sinr_db=s["mean_sinr_db"]))
    pub = pd.DataFrame(rows)
    new = pd.read_csv(results / "sweep_addition" / "synthetic_sweep.csv")
    return pd.concat([pub, new[pub.columns.tolist()]], ignore_index=True)


def load_telecom(results: Path, telecom_run: str = "telecom_milan") -> pd.DataFrame:
    chunks = []
    for f in sorted(glob.glob(str(results / telecom_run / "main" / "*.csv"))):
        d = pd.read_csv(f)
        d = d[(d.n_users == 800) & (d.method.isin(ORDER)) & (d.n_drones <= 30)]
        if len(d):
            chunks.append(d)
    return pd.concat(chunks, ignore_index=True)


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> {out / name}.pdf")


def fig_coverage_by_dist(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(IEEE_W, 2.6))
    x = np.arange(len(DISTS))
    w = 0.26
    for k, m in enumerate(ORDER):
        vals = [df[(df.method == m) & (df.dist == d)].coverage_pct.mean() for d in DISTS]
        st = STYLE[m]
        ax.bar(x + (k - 1) * w, vals, width=w * 0.92, color=st["color"],
               label=st["label"], zorder=3)
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=TICK_SZ - 1.5,
                    color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(DLABELS)
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 112)
    ax.grid(True, axis="y", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3,
              columnspacing=1.2, handlelength=1.4)
    _save(fig, out, "fig_sweep_coverage_by_dist")


def fig_hotspot_vs_n(df: pd.DataFrame, out: Path):
    """(a) coverage and (b) mean user SINR vs fleet size on hotspot layouts.

    Panel (b) shows the interference mechanism behind panel (a): a user is
    covered iff its SINR clears gamma_r, drawn as a reference line.
    """
    sub = df[df.dist == "hotspot"]
    fig, (ax_c, ax_s) = plt.subplots(2, 1, figsize=(IEEE_W, 3.6), sharex=True)
    for ax, col in ((ax_c, "coverage_pct"), (ax_s, "mean_sinr_db")):
        for m in ORDER:
            d = sub[sub.method == m]
            # Caption claims +/-1 std across M, so average over seeds first and
            # take the spread across user counts -- not the pooled seed+M spread.
            per_m = d.groupby(["n_drones", "n_users"])[col].mean().unstack()
            mu, sd = per_m.mean(axis=1), per_m.std(axis=1)
            st = STYLE[m]
            ax.plot(mu.index, mu.values, color=st["color"], ls=st["ls"],
                    marker=st["marker"], markevery=7, label=st["label"], zorder=3)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=st["color"],
                            alpha=0.12, lw=0, zorder=2)
        ax.grid(True, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax_s.axhline(3.0, color="#555555", lw=0.8, ls=":", zorder=1)
    ax_s.text(50.5, 3.6, r"$\gamma_r = 3$ dB", ha="right", va="bottom",
              fontsize=TICK_SZ - 1, color="#555555")
    ax_c.set_ylabel("Coverage (%)")
    ax_c.set_ylim(0, 105)
    ax_c.legend(loc="upper right")
    ax_c.set_title("(a)", loc="left", fontsize=FONT_SZ)
    ax_s.set_ylabel("Mean user SINR (dB)")
    ax_s.set_title("(b)", loc="left", fontsize=FONT_SZ)
    ax_s.set_xlabel(r"Fleet size $N$")
    ax_s.set_xlim(0, 51)
    ax_s.xaxis.set_major_locator(ticker.MultipleLocator(10))
    fig.tight_layout(h_pad=0.6)
    _save(fig, out, "fig_sweep_hotspot_vs_N")


def fig_telecom(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(IEEE_W, 2.6))
    for m in ORDER:
        d = df[df.method == m]
        # Caption claims +/-1 sigma across snapshots: average over seeds within a
        # snapshot first, then take the spread across the 24 snapshots.
        per_snap = d.groupby(["n_drones", "snapshot_idx"]).coverage_pct.mean().unstack()
        mu, sd = per_snap.mean(axis=1), per_snap.std(axis=1)
        st = STYLE[m]
        ax.plot(mu.index, mu.values, color=st["color"], ls=st["ls"],
                marker=st["marker"], markevery=4, label=st["label"], zorder=3)
        ax.fill_between(mu.index, mu - sd, mu + sd, color=st["color"], alpha=0.12,
                        lw=0, zorder=2)
    ax.set_xlabel(r"Fleet size $N$")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(60, 95)
    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left")
    _save(fig, out, "fig_sweep_telecom_vs_N")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--telecom-run", default="telecom_milan",
                    help="results/<run>/main holds the telecom CSVs (telecom_v3 = old Trentino run)")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)

    if not glob.glob(str(res / "heuristic_50" / "task_*.json")):
        raise SystemExit(f"no synthetic results under {res}: run the experiments first (see README)")
    syn = load_synthetic(res)
    print(f"synthetic rows: {len(syn)}  methods: {sorted(syn.method.unique())}")
    fig_coverage_by_dist(syn, out)
    fig_hotspot_vs_n(syn, out)

    if not glob.glob(str(res / args.telecom_run / "main" / "*.csv")):
        print(f"no telecom results in {res / args.telecom_run / 'main'}: skipping the "
              "telecom figure (run scripts/run_telecom_statistical.py first, see README)")
        return
    tel = load_telecom(res, args.telecom_run)
    print(f"telecom rows: {len(tel)}")
    fig_telecom(tel, out)


if __name__ == "__main__":
    main()
