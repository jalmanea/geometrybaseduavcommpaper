"""Print every telecom number quoted in the ICC paper (paper/main.tex).

Reads results/<run>/main/*.csv written by scripts/run_telecom_statistical.py.
Conventions (as in the paper): seeds are averaged within each snapshot first;
"mean over N" averages N = 5..30 at M = 800; confidence intervals are paired
t-intervals over the 24 per-snapshot gains.

python scripts/telecom_numbers.py                     # Milan run
python scripts/telecom_numbers.py --run telecom_v3    # old Trentino run
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["kmeans", "analytic", "kmeans_altitude_sweep"]


def load(results: Path, run: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(results / run / "main" / "*.csv")))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d[d.method.isin(METHODS) & (d.n_drones <= 30)]


def per_snapshot(d: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged coverage, indexed by (n_drones, snapshot_idx), one column per method."""
    return d.groupby(["n_drones", "snapshot_idx", "method"]).coverage_pct.mean().unstack()


def paired_ci(g: pd.Series) -> tuple[float, float, float]:
    m, se = g.mean(), g.std(ddof=1) / np.sqrt(len(g))
    t = stats.t.ppf(0.975, len(g) - 1)
    return m, m - t * se, m + t * se


def report(d: pd.DataFrame, label: str) -> None:
    s = per_snapshot(d)
    byN = s.groupby("n_drones").mean()
    print(f"\n=== {label} ===")
    print("rows:", len(d), " snapshots:", d.snapshot_idx.nunique(), " seeds:", d.seed.nunique())
    print("mean coverage over N (sweep / kmeans / analytic): %.1f / %.1f / %.1f"
          % (byN.kmeans_altitude_sweep.mean(), byN.kmeans.mean(), byN.analytic.mean()))
    print("coverage at N=5 and N=30:")
    print(byN.loc[[5, 30], METHODS].round(1).to_string())

    gain = (s.kmeans_altitude_sweep - s.kmeans).unstack("snapshot_idx")
    print("sweep > kmeans on every snapshot at every N:", bool((gain > 0).all(axis=None)),
          " (min per-snapshot gain %.2f pp)" % gain.min().min())
    ci = pd.DataFrame([paired_ci(r) for _, r in gain.iterrows()],
                      index=gain.index, columns=["mean", "lo", "hi"])
    print("paired 95%% CI excludes zero at every N: %s" % bool((ci.lo > 0).all()))
    for n in (ci["mean"].idxmin(), ci["mean"].idxmax()):
        print("  N=%d: %+.2f pp  CI [%.2f, %.2f]" % (n, *ci.loc[n]))

    a = byN.analytic - byN.kmeans
    behind = list(a[a < 0].index)
    print("analytic - kmeans < 0 at N =", behind)
    print("  largest deficit %.1f pp at N=%d; at N=30 %+.1f pp"
          % (a.min(), a.idxmin(), a.loc[30]))
    print("sweep - analytic, mean over N: %+.1f pp" % (byN.kmeans_altitude_sweep - byN.analytic).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="telecom_milan")
    args = ap.parse_args()
    d = load(Path(args.results), args.run)
    report(d[d.n_users == 800], f"{args.run}, M = 800")
    pooled = d.groupby(["n_drones", "snapshot_idx", "method"]).coverage_pct.mean().unstack()
    byN = pooled.groupby("n_drones").mean()
    a = byN.analytic - byN.kmeans
    print(f"\n=== {args.run}, pooled over M ===")
    print("mean (sweep / kmeans / analytic): %.1f / %.1f / %.1f"
          % (byN.kmeans_altitude_sweep.mean(), byN.kmeans.mean(), byN.analytic.mean()))
    print("analytic - kmeans < 0 at N =", list(a[a < 0].index))


if __name__ == "__main__":
    main()
